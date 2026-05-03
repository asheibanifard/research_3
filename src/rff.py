"""
Random Fourier Feature (RFF) implicit neural representation.

Encoding — Tancik et al. (2020), Eq. 4:
    B ~ N(0, σ²I),  B ∈ ℝ^{m×C},  fixed (not trained)
    γ(v) = [ cos(2π B v),  sin(2π B v) ]  ∈ ℝ^{2m}          ← paper standard

With optional input passthrough (include_input=True, enabled here to match NeRF):
    γ(v) = [ v,  cos(2π B v),  sin(2π B v) ]  ∈ ℝ^{C + 2m}

σ is the Gaussian standard deviation in the paper's parameterisation.
To match NeRF L=10 (max freq 2⁹=512), choose σ ≈ 512/(2π) ≈ 81.

Unlike NeRF's deterministic geometric frequencies (2⁰, 2¹, …, 2^{L−1}),
RFF draws frequencies stochastically, making the encoding non-deterministic
between initialisations (but fixed within a single run via register_buffer).

Naming: n{hidden_layers}{hidden_features}_m{rff_features}_s{int(rff_scale)}
e.g.    n8256_m256_s10  →  8L · 256N · 256 RFF features · σ=10
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Reuse the architecture-agnostic training loop and dataset from nerf.py
from src.nerf import train, NeRFDataset

USE_CUDA_KERNEL = False   # toggled by --use_kernel CLI flag

_rff_cuda = None

def _load_rff_kernel():
    global _rff_cuda
    if _rff_cuda is None:
        src = Path(__file__).parent / 'rff_cuda.cu'
        _rff_cuda = load(name='rff_cuda', sources=[str(src)], verbose=False)
    return _rff_cuda


class _RFFFn(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA RFF kernels."""

    @staticmethod
    def forward(ctx, x, B):
        ctx.save_for_backward(x, B)
        # kernel always works in float32; cast back to original dtype
        return _load_rff_kernel().forward(x.float(), B.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, B = ctx.saved_tensors
        grad_x = _load_rff_kernel().backward(
            grad_out.float().contiguous(), x.float(), B.float()
        ).to(x.dtype)
        return grad_x, None   # no gradient w.r.t. B (it is fixed)


def fused_rff(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    RFF encoding: [cos(2πBx), sin(2πBx)] ∈ ℝ^{2m}  (no passthrough).
    Uses the fused CUDA kernel when USE_CUDA_KERNEL=True, else PyTorch fallback.
    Both paths produce identical numerical output.
    """
    if USE_CUDA_KERNEL and x.is_cuda:
        return _RFFFn.apply(x, B)
    proj = (2 * math.pi) * (x @ B.T)          # (N, m)
    return torch.cat([torch.cos(proj),
                      torch.sin(proj)], dim=-1)  # cos first (paper order)


# ── Encoding ──────────────────────────────────────────────────────────────────

class RandomFourierEncoding(nn.Module):
    """
    Fixed random Fourier feature map (Tancik et al., 2020).

    Samples B ~ N(0, σ²I) once at construction and registers it as a
    non-trainable buffer (frozen for the lifetime of the run).

    Paper encoding (Eq. 4):
        γ(v) = [ cos(2π B v),  sin(2π B v) ]
    With passthrough:
        γ(v) = [ v,  cos(2π B v),  sin(2π B v) ]
    """

    def __init__(
        self,
        in_features:   int,
        rff_features:  int,
        rff_scale:     float,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_features   = in_features
        self.rff_features  = rff_features
        self.rff_scale     = rff_scale
        self.include_input = include_input

        # B ~ N(0, σ²I) — paper parameterisation (σ = rff_scale)
        B = torch.randn(rff_features, in_features) * rff_scale
        self.register_buffer('B', B)   # fixed — never updated by optimiser

    def out_dim(self) -> int:
        d = 2 * self.rff_features
        if self.include_input:
            d += self.in_features
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = fused_rff(x, self.B)   # (N, 2m): [cos(2πBx), sin(2πBx)]
        if self.include_input:
            return torch.cat([x, enc], dim=-1)
        return enc


# ── Model ─────────────────────────────────────────────────────────────────────

class RFFModel(nn.Module):
    """
    ReLU MLP with Random Fourier Feature input encoding.

    Architecture is identical to NeRFModel — only the encoding differs:
      NeRF  : deterministic geometric frequencies  2⁰, 2¹, …, 2^{L−1}
      RFF   : stochastic Gaussian frequencies      B ~ N(0, σ²I)

    Skip connection re-injects the encoded input at layer `skip_at` (default 4),
    exactly as in NeRFModel, enabling direct comparison.
    """

    def __init__(
        self,
        in_features:     int   = 3,
        hidden_features: int   = 256,
        hidden_layers:   int   = 8,
        out_features:    int   = 1,
        rff_features:    int   = 256,
        rff_scale:       float = 10.0,
        skip_at:         int   = 4,
    ):
        super().__init__()
        self.enc     = RandomFourierEncoding(in_features, rff_features, rff_scale)
        enc_dim      = self.enc.out_dim()
        self.skip_at = skip_at

        layers = []
        for i in range(hidden_layers):
            in_dim = enc_dim if i == 0 else hidden_features
            if i == skip_at:
                in_dim += enc_dim            # skip: re-inject encoding
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train RFF-MLP on a TIFF volume')
    parser.add_argument('--input',           type=str,   required=True)
    parser.add_argument('--model_dir',       type=str,   default='logs/rff')
    parser.add_argument('--hidden_features', type=int,   required=True)
    parser.add_argument('--hidden_layers',   type=int,   required=True)
    parser.add_argument('--rff_features',    type=int,   default=256,
                        help='Number of random Fourier feature pairs (output dim = 2×rff_features)')
    parser.add_argument('--rff_scale',       type=float, default=10.0,
                        help='Std-dev σ of the Gaussian B ~ N(0, σ²I). Controls frequency bandwidth.')
    parser.add_argument('--skip_at',         type=int,   default=4)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--epochs',          type=int,   default=1000)
    parser.add_argument('--batch_size',      type=int,   default=2**20)
    parser.add_argument('--steps_til_summary',       type=int, default=20)
    parser.add_argument('--epochs_til_checkpoint',   type=int, default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--use_kernel',      action='store_true',
                        help='Use fused CUDA RFF kernel')
    parser.add_argument('--no_amp',          action='store_true')
    parser.add_argument('--compile',         action='store_true')
    parser.add_argument('--wandb_project',   type=str, default='rff-ablation')
    parser.add_argument('--wandb_entity',    type=str, default=None)
    parser.add_argument('--no_wandb',        action='store_true')
    parser.add_argument('--resume',          action='store_true')
    args = parser.parse_args()

    import sys as _sys
    _self = _sys.modules[__name__]
    _self.USE_CUDA_KERNEL = args.use_kernel
    if args.use_kernel:
        print('Using fused CUDA RFF kernel')
    else:
        print('Using PyTorch RFF encoding (pass --use_kernel to enable fused kernel)')

    dataset    = NeRFDataset(args.input)
    in_feat    = dataset.coords.shape[-1]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=4,
    )

    model = RFFModel(
        in_features=in_feat,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
        rff_features=args.rff_features,
        rff_scale=args.rff_scale,
        skip_at=args.skip_at,
    ).cuda()

    arch_tag  = (f'n{args.hidden_layers}{args.hidden_features}'
                 f'_m{args.rff_features}_s{int(args.rff_scale)}')
    model_dir = os.path.join(args.model_dir, arch_tag)

    run_config = {
        'hidden_layers':   args.hidden_layers,
        'hidden_features': args.hidden_features,
        'rff_features':    args.rff_features,
        'rff_scale':       args.rff_scale,
        'skip_at':         args.skip_at,
        'lr':              args.lr,
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'input':           args.input,
        'arch_tag':        arch_tag,
    }

    n_params = sum(p.numel() for p in model.parameters())
    enc_dim  = model.enc.out_dim()
    print(f'RFF Model  {arch_tag}')
    print(f'  Encoding dim : {enc_dim}  (in={in_feat}, m={args.rff_features}, σ={args.rff_scale})')
    print(f'  Parameters   : {n_params:,}')

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
        loss_fn=lambda out, gt: {'mse': nn.functional.mse_loss(out.squeeze(-1), gt)},
        summary_fn=lambda model, inp, gt, out, writer, step: writer.add_scalar(
            'psnr', -10 * np.log10(
                nn.functional.mse_loss(out.squeeze(-1), gt).item() + 1e-8
            ), step),
        wandb_run=wandb_run,
        run_config=run_config,
        resume=args.resume,
    )
