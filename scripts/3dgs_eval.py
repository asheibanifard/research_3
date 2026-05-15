import argparse
import json
import sys
import numpy as np
import torch
import tifffile
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

# Add src/ to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from _3dgs._3dgs import GaussianCloud, AABB, VolumeDataset, render_splatted_mips


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
    src = Path(__file__).resolve().parent.parent / 'src' / '_3dgs' / '3dgs_eval_cuda.cu'
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
                data[key] = str(Path(__file__).resolve().parent.parent / p)
    data.setdefault('use_kernel', False)
    data.setdefault('chunk_n', 1024)
    data.setdefault('scale_min_clamp', 1e-5)
    data.setdefault('mahal_max_clamp', 20.0)
    data.setdefault('eval_samples', 10000)
    return SimpleNamespace(**data)


def load_model(run_dir: Path, cfg: SimpleNamespace, ckpt_path: Path = None):
    """Load model from checkpoint.
    
    Parameters
    ----------
    run_dir : directory containing config.json
    cfg : config SimpleNamespace
    ckpt_path : explicit checkpoint path (optional); if None, auto-finds in run_dir
    """
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = Path(ckpt_path)
    else:
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
        vol = flat.reshape(D, H, W).clamp(0, 1).cpu().numpy()
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
    import torch
    from skimage.metrics import structural_similarity
    from pytorch_msssim import ms_ssim
    import lpips

    # Ensure both volumes are in [0, 1] — guards against unclamped CUDA output
    gt_vol    = np.clip(gt_vol,    0.0, 1.0)
    recon_vol = np.clip(recon_vol, 0.0, 1.0)

    # ── PSNR — full volume, zonal breakdown, and log-domain ──────────────────
    diff2      = (gt_vol - recon_vol) ** 2
    mse_full   = float(np.mean(diff2))
    psnr_full  = float(-10 * np.log10(mse_full + 1e-10))

    # Zone masks (based on GT intensity after normalisation):
    #   background : gt < 0.05              (near-zero noise floor)
    #   dendrite   : 0.05 ≤ gt < 0.90      — honest signal region; no saturation
    #   soma       : gt ≥ 0.90              — likely clipped; diff artificially low
    #
    # PSNR_dendrite is the most diagnostic metric: it excludes both the
    # near-zero background (trivially easy) and the saturated soma (artificially
    # perfect).  This is where dendritic reconstruction quality is measured.
    bg_mask  = gt_vol <  0.05
    den_mask = (gt_vol >= 0.05) & (gt_vol < 0.90)
    soma_mask = gt_vol >= 0.90

    def _psnr_on(mask):
        if not mask.any():
            return float('nan')
        return float(-10 * np.log10(float(np.mean(diff2[mask])) + 1e-10))

    psnr_fg       = _psnr_on(~bg_mask)     # backward-compat field (non-background)
    psnr_dendrite = _psnr_on(den_mask)
    psnr_soma     = _psnr_on(soma_mask)

    # Log-domain PSNR: log1p(x) compresses bright soma, amplifies dim dendrites.
    # This is the perceptually-weighted metric — use it to track dendritic quality
    # without the soma swamping the gradient.
    log_gt    = np.log1p(gt_vol)    # in [0, log(2)] ≈ [0, 0.693]
    log_recon = np.log1p(recon_vol)
    mse_log   = float(np.mean((log_gt - log_recon) ** 2))
    psnr_log  = float(-10 * np.log10(mse_log + 1e-10))

    mse_fg = float(np.mean(diff2[~bg_mask])) if (~bg_mask).any() else float('nan')

    # ── XY MIP — used for all 2D perceptual metrics ───────────────────────────
    # Using the MIP avoids the empty-slice problem (most Z slices are background)
    # and keeps SSIM / MS-SSIM / LPIPS on the same 2-D projection.
    gt_mip  = np.max(gt_vol,    axis=0).astype('float32')   # (H, W)
    rec_mip = np.max(recon_vol, axis=0).astype('float32')

    # ── SSIM on XY MIP ───────────────────────────────────────────────────────
    ssim_val = float(structural_similarity(gt_mip, rec_mip, data_range=1.0))

    # ── MS-SSIM on XY MIP ────────────────────────────────────────────────────
    H, W = gt_mip.shape
    if H >= 160 and W >= 160:
        gt_t    = torch.from_numpy(gt_mip).unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
        recon_t = torch.from_numpy(rec_mip).unsqueeze(0).unsqueeze(0)
        ms_ssim_val = float(ms_ssim(gt_t, recon_t, data_range=1.0, size_average=True))
    else:
        ms_ssim_val = float('nan')

    # ── LPIPS on XY MIP ──────────────────────────────────────────────────────
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    def _to_lpips(arr):
        t = torch.from_numpy(arr)
        return t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = float(loss_fn(_to_lpips(gt_mip), _to_lpips(rec_mip)))

    return {
        'mse':           round(mse_full, 6),
        'psnr':          round(psnr_full, 4),
        'psnr_fg':       round(psnr_fg, 4),
        'psnr_dendrite': round(psnr_dendrite, 4),   # honest: excludes bg + saturated soma
        'psnr_soma':     round(psnr_soma, 4),        # soma-only (likely artefactually high)
        'psnr_log':      round(psnr_log, 4),          # log1p-domain: perceptual weighting
        'ssim':          round(ssim_val, 4),
        'ms_ssim':       round(ms_ssim_val, 4) if not np.isnan(ms_ssim_val) else None,
        'lpips':         round(lpips_val, 4),
    }


def save_metrics_txt(metrics: dict, path) -> None:
    lines = [
        f"PSNR (full vol)  : {metrics['psnr']:.4f} dB",
        f"PSNR (fg only)   : {metrics['psnr_fg']:.4f} dB",
        f"PSNR (dendrite)  : {metrics['psnr_dendrite']:.4f} dB   ← honest (gt in [0.05, 0.90])",
        f"PSNR (soma)      : {metrics['psnr_soma']:.4f} dB   (gt >= 0.90; likely saturated)",
        f"PSNR (log-domain): {metrics['psnr_log']:.4f} dB   (perceptual; log1p-weighted)",
        f"SSIM             : {metrics['ssim']:.4f}   (on XY MIP)",
        f"MS-SSIM          : {metrics['ms_ssim'] if metrics['ms_ssim'] is not None else 'N/A'}   (on XY MIP)",
        f"LPIPS            : {metrics['lpips']:.4f}   (AlexNet, on XY MIP)",
        f"MSE (full vol)   : {metrics['mse']:.6f}",
    ]
    Path(path).write_text('\n'.join(lines) + '\n')
    print(f'Saved metrics → {path}')


def render_mip(vol: np.ndarray) -> dict:
    """Render Maximum Intensity Projection along all three axes.
    
    Parameters
    ----------
    vol : (D, H, W) float32 array
    
    Returns
    -------
    dict with keys 'xy', 'xz', 'yz' — each is a 2D MIP projection
    """
    mips = {
        'xy': np.max(vol, axis=0),  # max along D axis (looking down)
        'xz': np.max(vol, axis=1),  # max along H axis (looking from side)
        'yz': np.max(vol, axis=2),  # max along W axis (looking from front)
    }
    return mips


def visualize_mip(gt_vol: np.ndarray, recon_vol: np.ndarray,
                  out_prefix: str = None) -> None:
    """Visualize MIP projections for GT and reconstructed volumes.
    
    Parameters
    ----------
    gt_vol, recon_vol : (D, H, W) float32 arrays in [0, 1]
    out_prefix        : if given, save as <prefix>_mip_xy.tif, etc.
    """
    import matplotlib.pyplot as plt
    
    gt_mips = render_mip(gt_vol)
    rec_mips = render_mip(recon_vol)
    
    axes_info = [
        ('xy', 'Z-axis (XY projection)'),
        ('xz', 'Y-axis (XZ projection)'),
        ('yz', 'X-axis (YZ projection)'),
    ]
    
    for axis_key, title in axes_info:
        gt_mip = gt_mips[axis_key]
        rec_mip = rec_mips[axis_key]
        diff_mip = np.abs(gt_mip - rec_mip)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # GT
        axs[0].imshow(gt_mip, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title('GT MIP', fontsize=12)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Reconstruction
        axs[1].imshow(rec_mip, cmap='gray', vmin=0, vmax=1)
        axs[1].set_title('Recon MIP', fontsize=12)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        # Difference
        im = axs[2].imshow(diff_mip, cmap='hot')
        axs[2].set_title('|Difference|', fontsize=12)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.colorbar(im, ax=axs[2])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if out_prefix:
            path = f'{out_prefix}_mip_{axis_key}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved → {path}')
        plt.show()


def save_mip_tif(vol: np.ndarray, out_prefix: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Save MIP projections as .tif files.
    
    Parameters
    ----------
    vol : (D, H, W) float32 array (values in [0, 1] after normalization)
    out_prefix : prefix for output files
    vmin, vmax : min/max values for denormalization
    """
    mips = render_mip(vol)
    
    axes_info = [
        ('xy', 'Z-axis projection'),
        ('xz', 'Y-axis projection'),
        ('yz', 'X-axis projection'),
    ]
    
    for axis_key, description in axes_info:
        mip = mips[axis_key]
        # Denormalize
        mip_denorm = (mip * (vmax - vmin) + vmin).astype('float32')
        path = f'{out_prefix}_mip_{axis_key}.tif'
        tifffile.imwrite(path, mip_denorm)
        print(f'Saved MIP ({description}) → {path}')


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
    parser.add_argument('--run_dir',     default=str(RUN_DIR))
    parser.add_argument('--ckpt',        default=None, help='Explicit checkpoint path')
    parser.add_argument('--out',         default=None, help='Save recon volume as .tif')
    parser.add_argument('--metrics_out', default=None, metavar='PATH',
                        help='Save PSNR/SSIM/MS-SSIM/LPIPS to a text file')
    parser.add_argument('--plot',     default=None, metavar='PREFIX',
                        help='Save slice plots as <PREFIX>_z.png / _y.png / _x.png')
    parser.add_argument('--mip',      action='store_true',
                        help='Render and visualize MIP projections (voxel-based)')
    parser.add_argument('--mip_out',  default=None, metavar='PREFIX',
                        help='Save MIP projections as <PREFIX>_mip_xy.tif, etc.')
    parser.add_argument('--splat_mip', action='store_true',
                        help='Render splatted MIP projections (Gaussian splatting-based)')
    parser.add_argument('--splat_mip_out', default=None, metavar='PREFIX',
                        help='Save splatted MIP projections as <PREFIX>_splat_mip_xy.png')
    parser.add_argument('--splat_depth_samples', type=int, default=32,
                        help='Number of depth slices to sample per ray (lower=faster)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg     = load_config(run_dir)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vol_raw    = tifffile.imread(cfg.volume).astype('float32')
    vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
    gt_vol     = (vol_raw - vmin) / (vmax - vmin) if vmax > vmin else vol_raw
    gt_vol_t   = torch.from_numpy(gt_vol)

    gc, aabb = load_model(run_dir, cfg, ckpt_path=args.ckpt)
    dataset  = VolumeDataset(gt_vol_t, aabb, cfg, swc_path=cfg.swc_path)

    recon_vol = reconstruct_volume(gc, dataset, cfg, aabb)
    metrics   = compute_metrics(gt_vol, recon_vol)
    print(f'PSNR (full vol)  = {metrics["psnr"]:.4f} dB')
    print(f'PSNR (fg only)   = {metrics["psnr_fg"]:.4f} dB')
    print(f'PSNR (dendrite)  = {metrics["psnr_dendrite"]:.4f} dB   ← honest (gt 0.05–0.90)')
    print(f'PSNR (soma)      = {metrics["psnr_soma"]:.4f} dB   (gt≥0.90; likely saturated)')
    print(f'PSNR (log-domain)= {metrics["psnr_log"]:.4f} dB   (perceptual)')
    print(f'SSIM             = {metrics["ssim"]:.4f}   (XY MIP)')
    print(f'MS-SSIM          = {metrics["ms_ssim"]}   (XY MIP)')
    print(f'LPIPS            = {metrics["lpips"]:.4f}   (XY MIP, AlexNet)')
    print(f'MSE (full vol)   = {metrics["mse"]:.2e}')

    metrics_path = args.metrics_out or (
        str(Path(args.out).with_suffix('').parent / 'metrics.txt') if args.out else None
    )
    if metrics_path:
        save_metrics_txt(metrics, metrics_path)

    if args.out:
        out_path = Path(args.out)
        out_arr  = (recon_vol * (vmax - vmin) + vmin).astype('float32')
        
        # Auto-detect format from extension
        if out_path.suffix == '.v3draw':
            # Vaa3D native format with proper header
            # Header: 24 bytes - endianness(4) + version(4) + dims(12 bytes: D,H,W each 4 bytes)
            import struct
            D, H, W = out_arr.shape
            with open(out_path, 'wb') as f:
                # Endianness marker: 0x01020304 = little-endian, 0x04030201 = big-endian
                f.write(struct.pack('<I', 0x01020304))  # little-endian marker
                f.write(struct.pack('<I', 2))           # version 2
                f.write(struct.pack('<III', D, H, W))   # dimensions
                f.write(struct.pack('<I', 4))           # datatype: 4=float32
                out_arr.tofile(f)
            print(f'Saved 3D volume {out_arr.shape} (Vaa3D .v3draw) -> {out_path}')
        elif out_path.suffix == '.raw':
            out_arr.tofile(str(out_path))
            print(f'Saved 3D volume {out_arr.shape} (raw binary) -> {out_path}')
        else:  # .tif default
            tifffile.imwrite(str(out_path), out_arr)
            print(f'Saved 3D volume {out_arr.shape} (TIFF) -> {out_path}')

    if args.mip:
        print('\n=== MIP Rendering (Voxel-based) ===')
        visualize_mip(gt_vol, recon_vol, out_prefix=args.mip_out)
    
    if args.mip_out:
        save_mip_tif(recon_vol, args.mip_out, vmin=vmin, vmax=vmax)

    if args.splat_mip:
        print('\n=== Splatted MIP Rendering (Gaussian Splatting) ===')
        splat_mips = render_splatted_mips(gc, dataset, cfg,
                                         depth_samples=args.splat_depth_samples)
        
        # Visualize splatted MIPs
        import matplotlib.pyplot as plt
        axes_info = [
            ('xy', 'Z-axis (XY splatted MIP)'),
            ('xz', 'Y-axis (XZ splatted MIP)'),
            ('yz', 'X-axis (YZ splatted MIP)'),
        ]
        
        for axis_key, title in axes_info:
            mip = splat_mips[axis_key].cpu().numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(mip, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{title} (Splatted)', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            if args.splat_mip_out:
                path = f'{args.splat_mip_out}_splat_mip_{axis_key}.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f'Saved → {path}')
            plt.show()

    if args.plot is not None:
        plot_slices(gt_vol, recon_vol, out_prefix=args.plot or 'slices')
