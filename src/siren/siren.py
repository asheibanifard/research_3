# SIREN (Sinusoidal Representation Networks) — Implementation Checklist
#
# Step 1: Define hyperparameters
#   - in_features: input coordinate dimensionality (e.g. 2 for images, 3 for volumes)
#   - hidden_features: width of each hidden layer
#   - hidden_layers: number of hidden layers
#   - out_features: output dimensionality (e.g. 1 for grayscale, 3 for RGB)
#   - omega_0: frequency scaling factor (default 30 for first layer)
#
# Step 2: Build SineLayer (single layer with sine activation)
#   - Linear(in_features, out_features, bias=True)
#   - Apply custom weight initialization (see Step 3)
#   - Forward: return sin(omega_0 * linear(x))
#   - Flag is_first to switch init scheme for the first layer
#
# Step 3: Weight initialization (critical for convergence)
#   - First layer:  Uniform(-1/fan_in, 1/fan_in)
#   - Hidden layers: Uniform(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
#   - Bias: zeros or small uniform (same scheme as weights)
#
# Step 4: Assemble full SIREN network
#   - First SineLayer: is_first=True, omega_0=omega_0
#   - Hidden SineLayers: is_first=False, omega_0=omega_0
#   - Final linear layer: nn.Linear(hidden_features, out_features), NO sine activation
#
# Step 5: Forward pass
#   - Pass input coordinates through each SineLayer sequentially
#   - Pass result through final linear layer
#   - Optionally apply output activation (e.g. sigmoid for image intensities)
#
# Step 6: Training loop
#   - Sample ground-truth (pixel values, SDF values, etc.) at input coordinates
#   - Compute loss (e.g. MSE between network output and ground-truth)
#   - Backpropagate and update with Adam (lr ~1e-4)
#   - Optionally supervise gradient outputs (e.g. for SDF eikonal term)

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
from torch.utils.cpp_extension import load
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

USE_CUDA_KERNEL = False  # toggled by --use-kernel CLI flag


_sine_cuda = None

def _load_cuda_kernel():
    global _sine_cuda
    if _sine_cuda is None:
        src = Path(__file__).parent / 'siren_cuda.cu'
        _sine_cuda = load(name='siren_cuda', sources=[str(src)], verbose=False)
    return _sine_cuda


class _SineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, omega_0):
        ctx.save_for_backward(x)
        ctx.omega_0 = omega_0
        return _load_cuda_kernel().forward(x.float(), omega_0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = _load_cuda_kernel().backward(
            grad_output.float().contiguous(), x.float(), ctx.omega_0
        ).to(x.dtype)
        return grad_x, None


def fused_sine(x: torch.Tensor, omega_0: float) -> torch.Tensor:
    if USE_CUDA_KERNEL and x.is_cuda:
        return _SineFn.apply(x, omega_0)
    return torch.sin((omega_0 * x).float()).to(x.dtype)
# Step 1: Data loading and preprocessing
#   - Load your dataset (e.g. images, 3D shapes, etc.) from aws
#   - Normalize input coordinates to a suitable range (e.g. [-1, 1])
#   - Create training dataloader that yields (input_coordinates, ground_truth) pairs
#   - Optionally create validation dataloader for monitoring generalization
class SirenDataset(torch.utils.data.Dataset):
    """
        A PyTorch Dataset for loading volumetric data from a TIFF file and preparing it for training a SIREN model.
        The dataset reads a 3D volume (or 2D image) from the specified input path, normalizes the pixel values to the range [0, 1], and constructs a coordinate grid corresponding to the spatial dimensions of the volume. Each sample returned by the dataset consists of a coordinate tuple (x, y, z) for 3D data or (x, y) for 2D data, along with the corresponding normalized pixel intensity value at that coordinate. The coordinates are normalized to the range [-1, 1] to facilitate training of the SIREN model.

        params:
        - input_path (str): The file path to the input TIFF volume. The volume should be a 3D array (D, H, W) or a 2D array (H, W) containing pixel intensity values.
        returns:
        - A PyTorch Dataset object that can be used with a DataLoader to yield (coordinates, pixel_value) pairs for training a SIREN model.
    """
    
    def __init__(self, input_path: str):
        super(SirenDataset, self).__init__()
        # Read the volume from the TIFF file
        volume = tifffile.imread(input_path).astype(np.float32)
        # Normalize the volume to [0, 1]
        vmin, vmax = volume.min(), volume.max()
        if vmax == vmin:
            raise ValueError("The input volume has zero variance (max equals min). Cannot normalize.")
        volume = (volume - vmin) / (vmax - vmin)
        shape = volume.shape  # (D, H, W) or (H, W)

        # Build normalized coordinate grid in [-1, 1] for each spatial dimension
        axes = [np.linspace(-1, 1, s, dtype=np.float32) for s in shape]
        grids = np.meshgrid(*axes, indexing='ij')
        coords = np.stack([g.flatten() for g in grids], axis=-1)  # (N, ndim)

        # Convert to PyTorch tensors
        self.coords = torch.from_numpy(coords)        # (N, ndim)
        self.gt = torch.from_numpy(volume.flatten())  # (N,)

    def __len__(self):
        # The length of the dataset is the number of coordinate-pixel pairs, which is equal to the total number of voxels in the volume.
        return len(self.gt)

    def __getitem__(self, idx):
        # Return the coordinate and corresponding pixel value for the given index. The coordinate is a tuple (x, y, z) for 3D data or (x, y) for 2D data, and the pixel value is the normalized intensity at that coordinate.
        return self.coords[idx], self.gt[idx]
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features, is_first):
        with torch.no_grad():
            if is_first:
                bound = 1 / in_features
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return fused_sine(self.linear(x), self.omega_0)


class SirenModel(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False,
          loss_schedules=None, early_stopping_patience=50, use_amp=True, compile_model=False,
          wandb_run=None, run_config=None, resume=False):

    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed ({e}), falling back to eager mode")

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optim  = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    resume_path     = os.path.join(checkpoints_dir, 'resume.pth')

    start_epoch       = 0
    train_losses      = []
    best_loss         = float('inf')
    epochs_no_improve = 0

    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()})
        optim.load_state_dict(ckpt['optim_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch       = ckpt['epoch'] + 1
        best_loss         = ckpt['best_loss']
        epochs_no_improve = ckpt['epochs_no_improve']
        train_losses      = ckpt['train_losses']
        print(f"Resumed from epoch {ckpt['epoch']}  best_loss={best_loss:.6f}")
    else:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    os.makedirs(os.path.join(model_dir, 'summaries'), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    writer = SummaryWriter(summaries_dir)

    model_name = getattr(model, '_orig_mod', model).__class__.__name__
    run_ts     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train_start = time()
    n_params = sum(p.numel() for p in model.parameters())

    def ckpt_path(tag):
        return os.path.join(checkpoints_dir, f'{model_name}_{run_ts}_{tag}.pth')

    # Move the full dataset to GPU once; sample random batches directly on GPU
    # to eliminate DataLoader IPC overhead on large volumes.
    all_coords = train_dataloader.dataset.coords.cuda()
    all_gt     = train_dataloader.dataset.gt.cuda()
    if double_precision:
        all_coords = all_coords.double()
        all_gt     = all_gt.double()
    n_total    = len(all_gt)
    batch_size = train_dataloader.batch_size

    steps_per_epoch = max(1, n_total // batch_size)
    best_path       = ckpt_path('best')
    total_steps     = start_epoch * steps_per_epoch

    with tqdm(total=steps_per_epoch * epochs, initial=total_steps) as pbar:
        for epoch in range(start_epoch, epochs):
            if epoch and not epoch % epochs_til_checkpoint:
                torch.save(model.state_dict(), ckpt_path(f'epoch_{epoch:04d}'))
                np.savetxt(os.path.join(checkpoints_dir, f'{model_name}_{run_ts}_losses_epoch_{epoch:04d}.txt'),
                           np.array(train_losses))
                torch.save({
                    'epoch':            epoch,
                    'model_state':      model.state_dict(),
                    'optim_state':      optim.state_dict(),
                    'scaler_state':     scaler.state_dict(),
                    'best_loss':        best_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'train_losses':     train_losses,
                }, resume_path)

            for step in range(steps_per_epoch):
                start_time = time()
                # randint is faster than randperm for large N (sampling with replacement)
                idx         = torch.randint(n_total, (batch_size,), device='cuda')
                model_input = all_coords[idx]
                gt          = all_gt[idx]

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                        train_loss = sum(l.mean() for l in losses.values())
                        scaler.scale(train_loss).backward()
                        return train_loss
                    optim.step(closure)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    model_output = model(model_input)
                    losses       = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if not use_lbfgs:
                    optim.zero_grad()
                    scaler.scale(train_loss).backward()

                    if clip_grad:
                        scaler.unscale_(optim)
                        max_norm = 1. if isinstance(clip_grad, bool) else clip_grad
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                    scaler.step(optim)
                    scaler.update()

                # .item() syncs GPU→CPU; do it once and reuse the value
                mse_val = train_loss.item()
                train_losses.append(mse_val)

                do_summary = not total_steps % steps_til_summary
                if do_summary:
                    psnr_val = -10 * np.log10(mse_val + 1e-8)
                    writer.add_scalar("total_train_loss", mse_val, total_steps)
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
                        wandb_run.log({'loss': mse_val, 'psnr': psnr_val, 'epoch': epoch}, step=total_steps)

                pbar.update(1)

                if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

            # --- per-epoch: best model & early stopping ---
            epoch_loss = np.mean(train_losses[-steps_per_epoch:])
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                best_psnr = -10 * np.log10(best_loss + 1e-8)
                tqdm.write(f"Epoch {epoch}: best loss {best_loss:.6f}  PSNR {best_psnr:.2f} dB → {os.path.basename(best_path)}")
                if wandb_run is not None:
                    wandb_run.log({'best_loss': best_loss, 'best_psnr': best_psnr}, step=total_steps)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                    break

        torch.save(model.state_dict(), ckpt_path('final'))
        np.savetxt(os.path.join(checkpoints_dir, f'{model_name}_{run_ts}_losses_final.txt'),
                   np.array(train_losses))

        report = {
            'config': run_config or {},
            'n_params': n_params,
            'best_psnr': float(-10 * np.log10(best_loss + 1e-8)),
            'best_loss': float(best_loss),
            'final_loss': float(np.mean(train_losses[-1:])),
            'epochs_trained': epoch + 1,
            'train_sec': round(time() - train_start, 1),
        }
        with open(os.path.join(model_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        tqdm.write(f"Report saved → {os.path.join(model_dir, 'report.json')}")

        if wandb_run is not None:
            wandb_run.summary.update({k: v for k, v in report.items() if k != 'config'})
            wandb_run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SIREN on a TIFF volume')
    parser.add_argument('--input',           type=str,   required=True,      help='Path to input TIFF file')
    parser.add_argument('--model_dir',       type=str,   default='logs/siren')
    parser.add_argument('--hidden_features', type=int,   required=True)
    parser.add_argument('--hidden_layers',   type=int,   required=True)
    parser.add_argument('--omega_0',         type=float, default=30.0)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--epochs',          type=int,   default=1000)
    parser.add_argument('--batch_size',      type=int,   default=2**20)   # 2**10
    parser.add_argument('--steps_til_summary',     type=int,  default=20)
    parser.add_argument('--epochs_til_checkpoint',   type=int,  default=20)
    parser.add_argument('--early_stopping_patience', type=int,  default=50, help='Stop after N epochs with no improvement')
    parser.add_argument('--use_kernel',              action='store_true', help='Use fused CUDA sine kernel')
    parser.add_argument('--no_amp',                  action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--compile',                 action='store_true', help='Enable torch.compile (requires working Triton/Inductor)')
    parser.add_argument('--wandb_project',  type=str, default='siren-ablation')
    parser.add_argument('--wandb_entity',   type=str, default=None)
    parser.add_argument('--no_wandb',       action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--resume',         action='store_true', help='Resume from checkpoints/resume.pth if it exists')
    args = parser.parse_args()

    import sys
    _self = sys.modules[__name__]
    _self.USE_CUDA_KERNEL = args.use_kernel
    if args.use_kernel:
        print('Using fused CUDA sine kernel')
    else:
        print('Using PyTorch sine (pass --use_kernel to enable fused kernel)')

    dataset = SirenDataset(args.input)
    in_features = dataset.coords.shape[-1]  # 2 for images, 3 for volumes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    model = SirenModel(
        in_features=in_features,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
        omega_0=args.omega_0,
    ).cuda()

    def loss_fn(model_output, gt):
        return {'mse': torch.nn.functional.mse_loss(model_output.squeeze(-1), gt)}

    def summary_fn(model, model_input, gt, model_output, writer, total_steps):
        writer.add_scalar('psnr', -10 * np.log10(
            torch.nn.functional.mse_loss(model_output.squeeze(-1), gt).item()
        ), total_steps)

    omega_tag = f'{args.omega_0:g}'.replace('.', 'p')
    arch_tag  = f's{args.hidden_layers}{args.hidden_features}_w{omega_tag}'
    model_dir = os.path.join(args.model_dir, arch_tag)

    run_config = {
        'hidden_layers':  args.hidden_layers,
        'hidden_features': args.hidden_features,
        'omega_0':        args.omega_0,
        'lr':             args.lr,
        'epochs':         args.epochs,
        'batch_size':     args.batch_size,
        'input':          args.input,
        'arch_tag':       arch_tag,
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
