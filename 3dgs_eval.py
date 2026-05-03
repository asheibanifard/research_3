import argparse
import json
import numpy as np
import torch
import tifffile
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

from src._3dgs import GaussianCloud, AABB, VolumeDataset


RUN_DIR = Path('logs/3dgs/run')

# ── CUDA kernel (lazy-loaded on first use) ────────────────────────────────────
_eval_cuda = None

def _find_cuda_include() -> list:
    """Return an include path that has the full CUDA toolkit headers (cuda_runtime.h + nv/target).

    The base conda env installs nvcc but not the full header tree.  We walk
    sibling conda envs to find one that does; CUDA_HOME is checked first.
    """
    import os
    candidates = []
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        candidates.append(Path(cuda_home) / 'include')
    # Walk conda envs directory next to the active env
    conda_root = Path(__file__).parent  # fallback; real search below
    try:
        import sys as _sys
        conda_root = Path(_sys.executable).parent.parent
    except Exception:
        pass
    for inc in sorted((conda_root / 'envs').glob('*/include')):
        candidates.append(inc)
    # Also try the active env itself
    try:
        import sys as _sys
        candidates.insert(0, Path(_sys.prefix) / 'include')
    except Exception:
        pass
    for p in candidates:
        if (p / 'nv' / 'target').exists() and (p / 'cuda_runtime.h').exists():
            return [str(p)]
    return []


def _load_eval_kernel():
    global _eval_cuda
    if _eval_cuda is not None:
        return _eval_cuda
    from torch.utils.cpp_extension import load
    src = Path(__file__).parent / 'src' / '3dgs_eval_cuda.cu'
    _eval_cuda = load(
        name='3dgs_eval_cuda',
        sources=[str(src)],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        extra_include_paths=_find_cuda_include(),
        verbose=False,
    )
    return _eval_cuda


def load_config(run_dir: Path) -> SimpleNamespace:
    data = json.loads((run_dir / 'config.json').read_text())
    for key in ('volume', 'swc_path'):
        if data.get(key):
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = str(Path(__file__).parent / p)
    data.setdefault('use_kernel', False)
    data.setdefault('chunk_n', 1024)
    data.setdefault('scale_min_clamp', 1e-5)
    data.setdefault('mahal_max_clamp', 20.0)
    data.setdefault('eval_samples', 10000)
    return SimpleNamespace(**data)


def load_model(run_dir: Path, cfg: SimpleNamespace):
    for pattern in ('best_*.pth', 'best.pth', 'best.npz', 'final_*.pth', 'final.pth'):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            ckpt = hits[0]
            break
    else:
        raise FileNotFoundError(f'No checkpoint found in {run_dir}')

    device = torch.device(cfg.device)
    aabb   = AABB.unit()
    gc     = GaussianCloud.load(ckpt, aabb, device, cfg)
    print(f'Loaded {gc.N:,} Gaussians from {ckpt.name}')
    return gc, aabb


def reconstruct_volume(gc: GaussianCloud, dataset: VolumeDataset,
                       cfg: SimpleNamespace, aabb: AABB) -> np.ndarray:
    """Evaluate the Gaussian cloud at every voxel centre.

    On CUDA: single fused kernel launch — no Python loop, no pts allocation.
    On CPU:  chunked PyTorch fallback.
    """
    D, H, W = dataset.D, dataset.H, dataset.W

    if gc.device.type == 'cuda':
        kernel = _load_eval_kernel()
        lo = aabb.lo.cpu()
        hi = aabb.hi.cpu()
        with torch.inference_mode():
            flat = kernel.reconstruct_volume(
                gc.means.contiguous(),
                gc.log_s.contiguous(),
                gc.quats.contiguous(),
                gc.inten.contiguous(),
                float(lo[0]), float(hi[0]),
                float(lo[1]), float(hi[1]),
                float(lo[2]), float(hi[2]),
                D, H, W,
                float(gc.scale_min),
                float(gc.mahal_clamp),
            )
        vol = flat.reshape(D, H, W).cpu().numpy()
        torch.cuda.empty_cache()
        return vol

    # CPU fallback — chunked to keep memory bounded
    BATCH = 2 ** 14
    N     = D * H * W
    ids = torch.arange(D, dtype=torch.long).repeat_interleave(H * W)
    ihs = torch.arange(H, dtype=torch.long).repeat_interleave(W).tile(D)
    iws = torch.arange(W, dtype=torch.long).tile(D * H)
    flat = np.empty(N, dtype='float32')
    with torch.inference_mode():
        for s in tqdm(range(0, N, BATCH), desc='Reconstruct volume', leave=True):
            e         = min(s + BATCH, N)
            pts       = dataset._indices_to_pts(ids[s:e], ihs[s:e], iws[s:e], gc.device)
            flat[s:e] = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0, 1).numpy()
    return flat.reshape(D, H, W)


def compute_metrics(gt_vol: np.ndarray, recon_vol: np.ndarray) -> dict:
    mse  = float(np.mean((gt_vol - recon_vol) ** 2))
    psnr = float(-10 * np.log10(mse + 1e-8))
    return {'mse': round(mse, 6), 'psnr': round(psnr, 4)}


def plot_slices(gt_vol: np.ndarray, recon_vol: np.ndarray,
                out_prefix: str = None) -> None:
    """3×4 GT / Reconstruction / Difference panels for each of the three axes.

    Layout per figure:
        rows : GT  |  Reconstruction  |  Difference
        cols : 4 evenly-spaced (or user-specified) slice positions

    Parameters
    ----------
    gt_vol, recon_vol : (D, H, W) float32 arrays in [0, 1]
    out_prefix        : if given, save as  <prefix>_z.png / _y.png / _x.png
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    D, H, W = gt_vol.shape

    def _even4(n):
        step = n // 5
        return [step, 2 * step, 3 * step, 4 * step]

    axes_cfg = [
        # (label, slice indices, extractor)
        ('Z',  _even4(D),          lambda v, i: v[i, :, :]),
        ('Y',  [500, 650, 750, 900], lambda v, i: v[:, i, :]),
        ('X',  [500, 650, 750, 900], lambda v, i: v[:, :, i]),
    ]

    for axis, indices, get_sl in axes_cfg:
        # clip indices to valid range
        sizes = {'Z': D, 'Y': H, 'X': W}
        indices = [min(i, sizes[axis] - 1) for i in indices]

        # pre-fetch all slices to share colour scale
        gts   = [get_sl(gt_vol,    i) for i in indices]
        recs  = [get_sl(recon_vol, i) for i in indices]
        diffs = [g - r for g, r in zip(gts, recs)]
        dmax  = max(np.abs(d).max() for d in diffs)

        # figure height scales with the natural slice aspect ratio
        sample_sl = get_sl(gt_vol, indices[0])
        h_px, w_px = sample_sl.shape
        col_w = 4.0                          # inches per column
        row_h = col_w * (h_px / w_px)       # preserve pixel aspect
        fig, axs = plt.subplots(4, 4,
                                figsize=(col_w * 4 + 1.5, row_h * 4 + 1.5),
                                gridspec_kw={'wspace': 0.04, 'hspace': 0.10})

        for col, (idx, gt_sl, rec_sl, diff_sl) in enumerate(
                zip(indices, gts, recs, diffs)):

            kw_img  = dict(cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            kw_diff = dict(cmap='hot', vmin=0, vmax=dmax,
                           interpolation='nearest')

            axs[0, col].imshow(gt_sl,  **kw_img)
            axs[1, col].imshow(rec_sl, **kw_img)
            im_d = axs[2, col].imshow(np.abs(diff_sl), **kw_diff)
            # row 3: rec as background + diff overlay
            axs[3, col].imshow(rec_sl, **kw_img)
            axs[3, col].imshow(np.abs(diff_sl), **kw_diff, alpha=0.6)

            axs[0, col].set_title(f'{axis}={idx}', fontsize=9)
            axs[2, col].set_facecolor('black')
            for row in range(4):
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

        row_labels = ['GT', 'Recon', 'Diff', 'Recon + Diff']
        for row, lbl in enumerate(row_labels):
            axs[row, 0].set_ylabel(lbl, fontsize=10, rotation=90,
                                   labelpad=4, va='center')

        # shared colourbar for diff row
        cbar = fig.colorbar(im_d, ax=axs[2:, :], shrink=0.5, pad=0.01)
        cbar.set_label('|GT − Recon|', fontsize=8)

        fig.suptitle(f'{axis}-axis slices', fontsize=12, y=1.01)
        plt.tight_layout()

        if out_prefix:
            path = f'{out_prefix}_{axis.lower()}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved → {path}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a 3DGS run')
    parser.add_argument('--run_dir',  default=str(RUN_DIR))
    parser.add_argument('--out',      default=None, help='Save recon volume as .tif')
    parser.add_argument('--plot',     default=None, metavar='PREFIX',
                        help='Save slice plots as <PREFIX>_z.png / _y.png / _x.png')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg     = load_config(run_dir)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vol_raw    = tifffile.imread(cfg.volume).astype('float32')
    vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
    gt_vol     = (vol_raw - vmin) / (vmax - vmin) if vmax > vmin else vol_raw
    gt_vol_t   = torch.from_numpy(gt_vol)

    gc, aabb = load_model(run_dir, cfg)
    dataset  = VolumeDataset(gt_vol_t, aabb, cfg, swc_path=cfg.swc_path)

    recon_vol = reconstruct_volume(gc, dataset, cfg, aabb)
    metrics   = compute_metrics(gt_vol, recon_vol)
    print(f'PSNR = {metrics["psnr"]:.2f} dB   MSE = {metrics["mse"]:.2e}')

    if args.out:
        out_path = Path(args.out)
        out_arr  = (recon_vol * (vmax - vmin) + vmin).astype('float32')
        tifffile.imwrite(str(out_path), out_arr)
        print(f'Saved -> {out_path}')

    if args.plot is not None:
        plot_slices(gt_vol, recon_vol, out_prefix=args.plot or 'slices')
