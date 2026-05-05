import json
import os
import shutil
from time import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.cpp_extension import load

USE_CUDA_KERNEL = False   # toggled by --use_kernel CLI flag

_pe_cuda = None

def _load_pe_kernel():
    global _pe_cuda
    if _pe_cuda is None:
        src = Path(__file__).parent / 'nerf_cuda.cu'
        _pe_cuda = load(name='nerf_cuda', sources=[str(src)], verbose=False)
    return _pe_cuda


class _PEFn(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA positional-encoding kernels."""

    @staticmethod
    def forward(ctx, x, freqs):
        ctx.save_for_backward(x, freqs)
        return _load_pe_kernel().forward(x.float(), freqs).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, freqs = ctx.saved_tensors
        grad_x = _load_pe_kernel().backward(
            grad_out.float().contiguous(), x.float(), freqs
        ).to(x.dtype)
        return grad_x, None   # no gradient w.r.t. freqs


def fused_pe(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Positional encoding: fused CUDA kernel or pure-PyTorch fallback."""
    if USE_CUDA_KERNEL and x.is_cuda:
        return _PEFn.apply(x, freqs)
    # PyTorch fallback — identical numerical output
    x_freq = x.unsqueeze(-1) * freqs          # (N, C, L)
    enc    = torch.cat([torch.sin(x_freq),
                        torch.cos(x_freq)], dim=-1).reshape(x.shape[0], -1)
    return torch.cat([x, enc], dim=-1)


# ---------------------------------------------------------------------------
# Positional encoding module
# γ(x) = [x, sin(2⁰x), cos(2⁰x), …, sin(2^{L-1}x), cos(2^{L-1}x)]
# Original NeRF paper (Mildenhall et al., 2020): L = 10 for coordinates
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, L_freq: int, include_input: bool = True):
        super().__init__()
        self.L_freq        = L_freq
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(L_freq, dtype=torch.float32)  # 2^0 … 2^{L-1}
        self.register_buffer('freqs', freqs)

    def out_dim(self, in_features: int) -> int:
        d = 2 * self.L_freq * in_features
        if self.include_input:
            d += in_features
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_input:
            return fused_pe(x, self.freqs)
        # include_input=False: drop the passthrough prefix
        out = fused_pe(x, self.freqs)
        return out[:, x.shape[-1]:]


class NeRFModel(nn.Module):
    """
    MLP implicit neural representation with sinusoidal positional encoding.

    Architecture follows Mildenhall et al. (2020):
      - Positional encoding with L_freq frequency bands applied to input coordinates.
      - hidden_layers fully-connected ReLU layers of width hidden_features.
      - Skip connection: encoded input re-concatenated at layer skip_at (default 4).
      - Linear output layer — no output activation; values clipped to [0,1] at eval.

    Naming convention: n{hidden_layers}{hidden_features}_f{L_freq}
    e.g.  n8256_f10  →  8 layers, 256 neurons, L=10
    """

    def __init__(
        self,
        in_features:     int = 3,
        hidden_features: int = 256,
        hidden_layers:   int = 8,
        out_features:    int = 1,
        L_freq:          int = 10,
        skip_at:         int = 4,
    ):
        super().__init__()
        self.enc     = PositionalEncoding(L_freq, include_input=True)
        enc_dim      = self.enc.out_dim(in_features)
        self.skip_at = skip_at

        layers = []
        for i in range(hidden_layers):
            in_dim = enc_dim if i == 0 else hidden_features
            if i == skip_at:
                in_dim += enc_dim          # skip connection
            layers.append(nn.Linear(in_dim, hidden_features))

        self.layers       = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_features, out_features)
        self.activation   = nn.ReLU(inplace=True)

        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.enc(x)
        h   = enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, enc], dim=-1)
            h = self.activation(layer(h))
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Dataset  (same contract as SirenDataset)
# ---------------------------------------------------------------------------
class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, input_path: str):
        super().__init__()
        volume = tifffile.imread(input_path).astype(np.float32)
        vmin, vmax = volume.min(), volume.max()
        if vmax == vmin:
            raise ValueError("Input volume has zero variance.")
        volume = (volume - vmin) / (vmax - vmin)

        axes   = [np.linspace(-1, 1, s, dtype=np.float32) for s in volume.shape]
        grids  = np.meshgrid(*axes, indexing='ij')
        coords = np.stack([g.flatten() for g in grids], axis=-1)

        self.coords = torch.from_numpy(coords)
        self.gt     = torch.from_numpy(volume.flatten())

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return self.coords[idx], self.gt[idx]


# ---------------------------------------------------------------------------
# Training loop  (mirrors siren.py train() exactly)
# ---------------------------------------------------------------------------
def train(
    model, train_dataloader, epochs, lr,
    steps_til_summary, epochs_til_checkpoint, model_dir,
    loss_fn, summary_fn,
    val_dataloader=None, double_precision=False, clip_grad=False,
    loss_schedules=None, early_stopping_patience=20,
    use_amp=True, compile_model=False,
    wandb_run=None, run_config=None, resume=False,
):
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed ({e}), running in eager mode")

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optim  = torch.optim.Adam(lr=lr, params=model.parameters())

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    resume_path     = os.path.join(checkpoints_dir, 'resume.pth')

    start_epoch       = 0
    train_losses      = []
    best_loss         = float('inf')
    epochs_no_improve = 0

    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, weights_only=False)
        # Strip _orig_mod. prefix so the dict always has plain keys,
        # then load into the underlying module directly (bypasses the compile wrapper).
        plain = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
        getattr(model, '_orig_mod', model).load_state_dict(plain)
        optim.load_state_dict(ckpt['optim_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch       = ckpt['epoch'] + 1
        best_loss         = ckpt['best_loss']
        epochs_no_improve = ckpt['epochs_no_improve']
        train_losses      = ckpt['train_losses']
        print(f"Resumed from epoch {ckpt['epoch']}  best_loss={best_loss:.6f}")
    else:
        if resume:
            print(f"Warning: --resume set but no checkpoint found at {resume_path}. Starting fresh.")
        if not resume and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

    os.makedirs(os.path.join(model_dir, 'summaries'), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer      = SummaryWriter(os.path.join(model_dir, 'summaries'))
    model_name  = getattr(model, '_orig_mod', model).__class__.__name__
    run_ts      = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train_start = time()
    n_params    = sum(p.numel() for p in model.parameters())

    def ckpt_path(tag):
        return os.path.join(checkpoints_dir, f'{model_name}_{run_ts}_{tag}.pth')

    all_coords = train_dataloader.dataset.coords.cuda()
    all_gt     = train_dataloader.dataset.gt.cuda()
    if double_precision:
        all_coords = all_coords.double()
        all_gt     = all_gt.double()

    n_total         = len(all_gt)
    batch_size      = train_dataloader.batch_size
    steps_per_epoch = max(1, n_total // batch_size)
    best_path       = ckpt_path('best')
    total_steps     = start_epoch * steps_per_epoch

    with tqdm(total=steps_per_epoch * epochs, initial=total_steps) as pbar:
        for epoch in range(start_epoch, epochs):
            if epoch and not epoch % epochs_til_checkpoint:
                torch.save(model.state_dict(), ckpt_path(f'epoch_{epoch:04d}'))
                np.savetxt(
                    os.path.join(checkpoints_dir,
                                 f'{model_name}_{run_ts}_losses_epoch_{epoch:04d}.txt'),
                    np.array(train_losses),
                )
                torch.save({
                    'epoch':             epoch,
                    'model_state':       model.state_dict(),
                    'optim_state':       optim.state_dict(),
                    'scaler_state':      scaler.state_dict(),
                    'best_loss':         best_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'train_losses':      train_losses,
                }, resume_path)

            for step in range(steps_per_epoch):
                start_time  = time()
                idx         = torch.randint(n_total, (batch_size,), device='cuda')
                model_input = all_coords[idx]
                gt          = all_gt[idx]

                with torch.amp.autocast('cuda', enabled=use_amp):
                    model_output = model(model_input)
                    losses       = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if loss_schedules and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + '_weight',
                                          loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                optim.zero_grad()
                scaler.scale(train_loss).backward()
                if clip_grad:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1. if isinstance(clip_grad, bool) else clip_grad,
                    )
                scaler.step(optim)
                scaler.update()

                mse_val = train_loss.item()
                train_losses.append(mse_val)

                if not total_steps % steps_til_summary:
                    psnr_val = -10 * np.log10(mse_val + 1e-8)
                    writer.add_scalar('total_train_loss', mse_val, total_steps)
                    writer.add_scalar('psnr', psnr_val, total_steps)
                    torch.save(model.state_dict(), ckpt_path('current'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)
                    pbar.set_postfix(
                        ep=f'{epoch}/{epochs}',
                        loss=f'{mse_val:.4f}',
                        psnr=f'{psnr_val:.2f}dB',
                        es=f'{epochs_no_improve}/{early_stopping_patience}',
                        refresh=False,
                    )
                    tqdm.write("Epoch %d/%d  loss %.6f  PSNR %.2f dB  es %d/%d  iter %.3fs"
                               % (epoch, epochs, mse_val, psnr_val,
                                  epochs_no_improve, early_stopping_patience,
                                  time() - start_time))
                    if wandb_run is not None:
                        wandb_run.log(
                            {'loss': mse_val, 'psnr': psnr_val, 'epoch': epoch},
                            step=total_steps,
                        )

                pbar.update(1)
                total_steps += 1

            epoch_loss = np.mean(train_losses[-steps_per_epoch:])
            if epoch_loss < best_loss:
                best_loss         = epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                best_psnr = -10 * np.log10(best_loss + 1e-8)
                tqdm.write(f"Epoch {epoch}: best loss {best_loss:.6f}  "
                           f"PSNR {best_psnr:.2f} dB → {os.path.basename(best_path)}")
                if wandb_run is not None:
                    wandb_run.log(
                        {'best_loss': best_loss, 'best_psnr': best_psnr},
                        step=total_steps,
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    tqdm.write(f"Early stopping at epoch {epoch} "
                               f"(no improvement for {early_stopping_patience} epochs)")
                    break

            torch.save({
                'epoch':             epoch,
                'model_state':       model.state_dict(),
                'optim_state':       optim.state_dict(),
                'scaler_state':      scaler.state_dict(),
                'best_loss':         best_loss,
                'epochs_no_improve': epochs_no_improve,
                'train_losses':      train_losses,
            }, resume_path)

        torch.save(model.state_dict(), ckpt_path('final'))
        np.savetxt(
            os.path.join(checkpoints_dir,
                         f'{model_name}_{run_ts}_losses_final.txt'),
            np.array(train_losses),
        )

        report = {
            'config':         run_config or {},
            'n_params':       n_params,
            'best_psnr':      float(-10 * np.log10(best_loss + 1e-8)),
            'best_loss':      float(best_loss),
            'final_loss':     float(np.mean(train_losses[-1:])),
            'epochs_trained': epoch + 1,
            'train_sec':      round(time() - train_start, 1),
        }
        with open(os.path.join(model_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        tqdm.write(f"Report saved → {os.path.join(model_dir, 'report.json')}")

        if wandb_run is not None:
            wandb_run.summary.update({k: v for k, v in report.items() if k != 'config'})
            wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Train NeRF-MLP on a TIFF volume')
    parser.add_argument('--input',           type=str,   required=True)
    parser.add_argument('--model_dir',       type=str,   default='logs/nerf')
    parser.add_argument('--hidden_features', type=int,   required=True)
    parser.add_argument('--hidden_layers',   type=int,   required=True)
    parser.add_argument('--L_freq',          type=int,   default=10,
                        help='Positional-encoding frequency bands (NeRF paper default: 10)')
    parser.add_argument('--skip_at',         type=int,   default=4,
                        help='Layer index at which to inject the skip connection')
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--epochs',          type=int,   default=1000)
    parser.add_argument('--batch_size',      type=int,   default=2**20)
    parser.add_argument('--steps_til_summary',       type=int, default=20)
    parser.add_argument('--epochs_til_checkpoint',   type=int, default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--use_kernel',    action='store_true',
                        help='Use fused CUDA positional-encoding kernel')
    parser.add_argument('--no_amp',        action='store_true')
    parser.add_argument('--compile',       action='store_true')
    parser.add_argument('--wandb_project', type=str, default='nerf-ablation')
    parser.add_argument('--wandb_entity',  type=str, default=None)
    parser.add_argument('--no_wandb',      action='store_true')
    parser.add_argument('--resume',        action='store_true')
    args = parser.parse_args()

    import sys as _sys
    _self = _sys.modules[__name__]
    _self.USE_CUDA_KERNEL = args.use_kernel
    if args.use_kernel:
        print('Using fused CUDA positional-encoding kernel')
    else:
        print('Using PyTorch positional encoding (pass --use_kernel to enable fused kernel)')

    dataset     = NeRFDataset(args.input)
    in_features = dataset.coords.shape[-1]
    dataloader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=4,
    )

    model = NeRFModel(
        in_features=in_features,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
        L_freq=args.L_freq,
        skip_at=args.skip_at,
    ).cuda()

    def loss_fn(model_output, gt):
        return {'mse': nn.functional.mse_loss(model_output.squeeze(-1), gt)}

    def summary_fn(model, model_input, gt, model_output, writer, total_steps):
        writer.add_scalar('psnr', -10 * np.log10(
            nn.functional.mse_loss(model_output.squeeze(-1), gt).item() + 1e-8
        ), total_steps)

    arch_tag  = f'n{args.hidden_layers}{args.hidden_features}_f{args.L_freq}'
    model_dir = os.path.join(args.model_dir, arch_tag)

    run_config = {
        'hidden_layers':   args.hidden_layers,
        'hidden_features': args.hidden_features,
        'L_freq':          args.L_freq,
        'skip_at':         args.skip_at,
        'lr':              args.lr,
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'input':           args.input,
        'arch_tag':        arch_tag,
    }

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=arch_tag,
            config=run_config,
            dir='.',
        )

    train(
        model=model,
        train_dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        steps_til_summary=args.steps_til_summary,
        epochs_til_checkpoint=args.epochs_til_checkpoint,
        early_stopping_patience=args.early_stopping_patience,
        use_amp=not args.no_amp,
        compile_model=args.compile,
        model_dir=model_dir,
        loss_fn=loss_fn,
        summary_fn=summary_fn,
        wandb_run=wandb_run,
        run_config=run_config,
        resume=args.resume,
    )
