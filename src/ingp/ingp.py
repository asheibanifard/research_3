"""
Instant Neural Graphics Primitives — multi-resolution hash encoding.

Encoding (Müller et al., NeurIPS 2022):
    For each level l ∈ {0, …, L-1}:
        N_l = floor(N_min · b^l),   b = exp(log(N_max / N_min) / (L - 1))
        Hash the 8 voxel corners surrounding each query point:
            h(x, y, z) = (x  ⊕  π₁·y  ⊕  π₂·z)  &  (T - 1)
            π₁ = 2 654 435 761,   π₂ = 805 459 861   (Müller et al., 2022)
            T  = 2^log2_hashmap_size   (bitwise AND gives fast power-of-2 modulo)
        Trilinear interpolation of the 8 corner feature vectors from E_l ∈ ℝ^{T×F}
        Concatenate L levels → ℝ^{L·F}

MLP:  small ReLU network (hidden_layers layers of hidden_features neurons), NO skip
      connection — the multi-scale hash encoding already provides rich spatial context.

Arch tag:  h{L}f{F}T{log2T}_l{hidden_layers}n{hidden_features}
e.g.       h16f2T19_l2n64  →  16 levels · 2 feat/level · T=2¹⁹ · 2-layer · 64-neuron MLP
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from nerf.nerf import train, NeRFDataset

USE_CUDA_KERNEL = False   # toggled by --use_kernel CLI flag

# Spatial hashing primes (Müller et al., 2022)
_PI2 = 2_654_435_761
_PI3 =   805_459_861

# ── CUDA kernel (lazy-loaded on first use) ────────────────────────────────────

_ingp_cuda = None

def _load_ingp_kernel():
    global _ingp_cuda
    if _ingp_cuda is not None:
        return _ingp_cuda
    src = Path(__file__).parent / 'ingp_cuda.cu'
    _ingp_cuda = load(
        name='ingp_cuda',
        sources=[str(src)],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False,
    )
    return _ingp_cuda


class _HashEncFn(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA forward/backward kernels."""

    @staticmethod
    def forward(ctx, x, tables, resolutions):
        # resolutions: (n_levels,) int32 CUDA tensor
        out = _load_ingp_kernel().forward(x, tables, resolutions)
        ctx.save_for_backward(x, tables, resolutions)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, tables, resolutions = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        grad_tables = _load_ingp_kernel().backward(grad_out_c, x, tables, resolutions)
        return None, grad_tables, None   # no grad_x, no grad_resolutions


# ── Encoding ──────────────────────────────────────────────────────────────────

class MultiResHashEncoding(nn.Module):
    """
    Multi-resolution hash encoding (Müller et al., 2022).

    Hash tables stored as a single nn.Parameter of shape (n_levels, T, F).
    Forward dispatches to a fused CUDA kernel when running on GPU;
    falls back to a pure-PyTorch reference path on CPU.

    Output: (N, n_levels * n_features_per_level)
    """

    def __init__(
        self,
        in_features:          int   = 3,
        n_levels:             int   = 16,
        n_features_per_level: int   = 2,
        log2_hashmap_size:    int   = 19,
        base_resolution:      int   = 16,
        finest_resolution:    int   = 512,
    ):
        super().__init__()
        self.in_features          = in_features
        self.n_levels             = n_levels
        self.n_features_per_level = n_features_per_level
        self.T                    = 1 << log2_hashmap_size
        self.base_resolution      = base_resolution
        self.finest_resolution    = finest_resolution

        if n_levels > 1:
            b = math.exp(math.log(finest_resolution / base_resolution) / (n_levels - 1))
        else:
            b = 1.0
        self.resolutions = [
            max(1, int(math.floor(base_resolution * b ** l)))
            for l in range(n_levels)
        ]

        # Single contiguous parameter: (n_levels, T, F)
        self.tables = nn.Parameter(
            torch.empty(n_levels, self.T, n_features_per_level).uniform_(-1e-4, 1e-4)
        )

    def out_dim(self):
        return self.n_levels * self.n_features_per_level

    # ── PyTorch fallback (CPU or when CUDA kernel unavailable) ────────────────

    def _hash(self, x, y, z):
        pi2 = torch.tensor(_PI2, dtype=torch.int64, device=x.device)
        pi3 = torch.tensor(_PI3, dtype=torch.int64, device=x.device)
        return (x ^ (y * pi2) ^ (z * pi3)) & (self.T - 1)

    def _forward_pytorch(self, x):
        outs = []
        for l in range(self.n_levels):
            res = self.resolutions[l]
            xg  = (x + 1.0) * 0.5 * (res - 1)
            x0  = xg.long().clamp(0, res - 2)
            x1  = x0 + 1
            w   = (xg - x0.float()).clamp(0.0, 1.0)

            wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
            weights = torch.stack([
                (1-wx)*(1-wy)*(1-wz),  wx*(1-wy)*(1-wz),
                (1-wx)*   wy *(1-wz),  wx*   wy *(1-wz),
                (1-wx)*(1-wy)*   wz,   wx*(1-wy)*   wz,
                (1-wx)*   wy *   wz,   wx*   wy *   wz,
            ], dim=1)  # (N, 8)

            cx = torch.stack([x0[:,0], x1[:,0], x0[:,0], x1[:,0],
                               x0[:,0], x1[:,0], x0[:,0], x1[:,0]], dim=1)
            cy = torch.stack([x0[:,1], x0[:,1], x1[:,1], x1[:,1],
                               x0[:,1], x0[:,1], x1[:,1], x1[:,1]], dim=1)
            cz = torch.stack([x0[:,2], x0[:,2], x0[:,2], x0[:,2],
                               x1[:,2], x1[:,2], x1[:,2], x1[:,2]], dim=1)

            idx   = self._hash(cx.reshape(-1), cy.reshape(-1), cz.reshape(-1))
            # tables[l]: (T, F) — index by hashed corners
            feats = self.tables[l][idx.view(-1)]          # (N*8, F)
            feats = feats.view(x.shape[0], 8, -1)         # (N, 8, F)
            enc_l = (feats * weights.unsqueeze(-1)).sum(dim=1)
            outs.append(enc_l)

        return torch.cat(outs, dim=-1)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        x : (N, 3) float32, coordinates in [-1, 1]³
        Returns (N, n_levels * n_features_per_level)
        """
        if USE_CUDA_KERNEL and x.is_cuda:
            try:
                res_t = torch.tensor(
                    self.resolutions, dtype=torch.int32, device=x.device
                )
                return _HashEncFn.apply(
                    x.contiguous(), self.tables.contiguous(), res_t
                )
            except Exception:
                pass   # fall through to PyTorch path
        return self._forward_pytorch(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class InstantNGPModel(nn.Module):
    """
    Instant NGP: multi-resolution hash encoding + small ReLU MLP.

    Key difference from NeRF / RFF: no skip connection is required.
    The multi-scale hash encoding provides spatially structured features at every
    resolution, so the MLP only needs to decode — not encode — spatial information.
    Typical configuration: L=16, F=2, T=2¹⁹, 2-layer × 64-neuron MLP.
    """

    def __init__(
        self,
        in_features:          int   = 3,
        hidden_features:      int   = 64,
        hidden_layers:        int   = 2,
        out_features:         int   = 1,
        n_levels:             int   = 16,
        n_features_per_level: int   = 2,
        log2_hashmap_size:    int   = 19,
        base_resolution:      int   = 16,
        finest_resolution:    int   = 512,
    ):
        super().__init__()
        self.enc = MultiResHashEncoding(
            in_features, n_levels, n_features_per_level,
            log2_hashmap_size, base_resolution, finest_resolution,
        )
        enc_dim = self.enc.out_dim()

        layers = [nn.Linear(enc_dim, hidden_features), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(self.enc(x))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Instant NGP on a TIFF volume')
    parser.add_argument('--input',                 type=str,   required=True)
    parser.add_argument('--model_dir',             type=str,   default='logs/ingp')
    parser.add_argument('--n_levels',              type=int,   default=16,
                        help='Number of hash encoding resolution levels')
    parser.add_argument('--n_features_per_level',  type=int,   default=2,
                        help='Feature dimensionality per hash table entry')
    parser.add_argument('--log2_hashmap_size',     type=int,   default=19,
                        help='log2 of hash table size T  (T = 2^log2_hashmap_size)')
    parser.add_argument('--base_resolution',       type=int,   default=16,
                        help='Coarsest grid resolution N_min')
    parser.add_argument('--finest_resolution',     type=int,   default=512,
                        help='Finest grid resolution N_max')
    parser.add_argument('--hidden_layers',         type=int,   default=2)
    parser.add_argument('--hidden_features',       type=int,   default=64)
    parser.add_argument('--lr',                    type=float, default=1e-3)
    parser.add_argument('--epochs',                type=int,   default=500)
    parser.add_argument('--batch_size',            type=int,   default=2**20)
    parser.add_argument('--steps_til_summary',     type=int,   default=20)
    parser.add_argument('--epochs_til_checkpoint', type=int,   default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    parser.add_argument('--no_amp',    action='store_true')
    parser.add_argument('--use_kernel', action='store_true', help='Use fused CUDA hash-encoding kernel')
    parser.add_argument('--compile',   action='store_true')
    parser.add_argument('--wandb_project', type=str, default='ingp-ablation')
    parser.add_argument('--wandb_entity',  type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resume',   action='store_true')
    args = parser.parse_args()

    import sys as _sys
    _self = _sys.modules[__name__]
    _self.USE_CUDA_KERNEL = args.use_kernel
    if args.use_kernel:
        print('Using fused CUDA hash-encoding kernel')
    else:
        print('Using PyTorch hash encoding (pass --use_kernel to enable fused kernel)')

    dataset    = NeRFDataset(args.input)
    in_feat    = dataset.coords.shape[-1]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=4,
    )

    model = InstantNGPModel(
        in_features=in_feat,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
        n_levels=args.n_levels,
        n_features_per_level=args.n_features_per_level,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
    ).cuda()

    arch_tag  = (f'h{args.n_levels}f{args.n_features_per_level}'
                 f'T{args.log2_hashmap_size}_l{args.hidden_layers}n{args.hidden_features}')
    model_dir = os.path.join(args.model_dir, arch_tag)

    run_config = {
        'n_levels':             args.n_levels,
        'n_features_per_level': args.n_features_per_level,
        'log2_hashmap_size':    args.log2_hashmap_size,
        'base_resolution':      args.base_resolution,
        'finest_resolution':    args.finest_resolution,
        'hidden_layers':        args.hidden_layers,
        'hidden_features':      args.hidden_features,
        'lr':                   args.lr,
        'epochs':               args.epochs,
        'batch_size':           args.batch_size,
        'input':                args.input,
        'arch_tag':             arch_tag,
    }

    enc_dim  = model.enc.out_dim()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Instant NGP  {arch_tag}')
    print(f'  Levels       : {args.n_levels}  resolutions {model.enc.resolutions[0]}…{model.enc.resolutions[-1]}')
    print(f'  Encoding dim : {enc_dim}  (L={args.n_levels}, F={args.n_features_per_level})')
    print(f'  Hash table T : 2^{args.log2_hashmap_size} = {model.enc.T:,}  entries per level')
    print(f'  Parameters   : {n_params:,}  (hash tables: {model.enc.tables.numel():,},'
          f'  MLP: {sum(p.numel() for p in model.net.parameters()):,})')

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
