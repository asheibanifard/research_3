"""
render_inr.py — CLI entry point for the INR rendering paradigm.

Usage examples
──────────────
# Slice from best INGP model
python render_inr.py --method ingp --mode slice --axis 2 --pos 0.5

# MIP from best SIREN model
python render_inr.py --method siren --mode mip --axis 0

# DVR from 3DGS with fluorescence TF
python render_inr.py --method 3dgs --mode dvr --axis 2 --tf fluorescence

# Side-by-side comparison of all methods (same slice)
python render_inr.py --mode compare --axis 2 --pos 0.5

# Isosurface mesh from INGP
python render_inr.py --method ingp --mode isosurface --threshold 0.25

# Full progressive MIP panel (all axes, all methods)
python render_inr.py --mode panel

Run with --help for all options.
"""

import argparse
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


# ── load config ───────────────────────────────────────────────────────────────

def load_cfg():
    p = yaml.safe_load(open(ROOT / 'configs' / 'paths.yml'))
    r = yaml.safe_load(open(ROOT / 'configs' / 'renderer.yml'))
    return p, r


# ── build INRVolume from a pre-saved .npy reconstruction ─────────────────────

class _NpyVolume:
    """Lightweight adapter: serves a pre-computed numpy volume as an INRVolume.

    This lets the renderer work out-of-the-box with existing rec_best_vol/
    files before re-training the models with the new interface.
    """

    def __init__(self, npy_path: str, device='cpu'):
        import torch
        self._vol      = np.load(npy_path).astype(np.float32)   # (D, H, W)
        self.device    = torch.device(device)
        self.coord_range = 'unit'
        self.chunk     = 2 ** 20
        self.D, self.H, self.W = self._vol.shape

    def eval_slice(self, axis, position, res_a, res_b):
        from scipy.ndimage import zoom
        D, H, W = self.D, self.H, self.W
        if axis == 0:
            idx = round(position * (W - 1));  sl = self._vol[:, :, min(idx, W-1)]
        elif axis == 1:
            idx = round(position * (H - 1));  sl = self._vol[:, min(idx, H-1), :]
        else:
            idx = round(position * (D - 1));  sl = self._vol[min(idx, D-1), :, :]
        if sl.shape != (res_a, res_b):
            factors = (res_a / sl.shape[0], res_b / sl.shape[1])
            sl = zoom(sl, factors, order=1)
        return sl

    def eval_volume(self, D, H, W):
        from scipy.ndimage import zoom
        factors = (D / self.D, H / self.H, W / self.W)
        return zoom(self._vol, factors, order=1)

    def query(self, pts):
        import torch
        D, H, W = self.D, self.H, self.W
        pts_cpu = pts.cpu()
        idx_z = (pts_cpu[:, 2] * (D - 1)).long().clamp(0, D - 1)
        idx_y = (pts_cpu[:, 1] * (H - 1)).long().clamp(0, H - 1)
        idx_x = (pts_cpu[:, 0] * (W - 1)).long().clamp(0, W - 1)
        vol_t = torch.from_numpy(self._vol)
        return vol_t[idx_z, idx_y, idx_x].float().to(pts.device)


# ── volume factory ────────────────────────────────────────────────────────────

METHODS = ['siren', 'nerf', 'rff', 'ingp', '3dgs']

def build_volume(method: str, paths_cfg: dict, render_cfg: dict):
    """Return an INRVolume (or _NpyVolume) for the given method."""
    rec = paths_cfg['best_trained_dir']['rec_vol']
    key = '3dgs' if method == '3dgs' else method
    npy = ROOT / rec[key]
    if npy.exists():
        print(f'[{method}] loading pre-computed reconstruction: {npy.name}')
        return _NpyVolume(str(npy), device=render_cfg['device'])

    # fall back to live INR query (requires trained .pth checkpoint)
    best = paths_cfg['best_trained_dir']
    ckpt = ROOT / best[key]
    device = render_cfg['device']
    if method == 'siren':
        from src.renderer import SIRENVolume
        return SIRENVolume.from_checkpoint(ckpt, hidden_features=256,
                                           hidden_layers=4, device=device)
    if method in ('nerf', 'rff'):
        cls = __import__(f'src.renderer', fromlist=[method.capitalize()+'Volume'])
        V   = getattr(cls, method.capitalize() + 'Volume')
        return V.from_checkpoint(ckpt, device=device)
    if method == 'ingp':
        from src.renderer import INGPVolume
        return INGPVolume.from_checkpoint(ckpt, device=device)
    if method == '3dgs':
        from src.renderer import GaussianVolume
        return GaussianVolume.from_checkpoint(ckpt, device=device)
    raise ValueError(f'Unknown method: {method}')


# ── transfer function factory ─────────────────────────────────────────────────

def build_tf(name: str):
    from src.renderer.transfer_function import (
        tf_gray, tf_hot, tf_fluorescence, tf_cell_body, TransferFunction1D
    )
    return {'gray': tf_gray, 'hot': tf_hot,
            'fluorescence': tf_fluorescence,
            'cell_body': tf_cell_body}.get(name, tf_gray)()


# ── render modes ──────────────────────────────────────────────────────────────

def _save(img: np.ndarray, out_dir: Path, stem: str, dpi: int = 150):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{stem}.png'
    if img.ndim == 2:
        plt.imsave(str(path), img, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imsave(str(path), np.clip(img, 0, 1))
    print(f'Saved: {path}')


def render_slice(vol, args, rcfg, out_dir):
    from src.renderer import VolumeRenderer
    rdr = VolumeRenderer(vol)
    res = rcfg['render']['resolution']
    sl  = rdr.slice(args.axis, args.pos, res, res)
    _save(sl, out_dir, f'{args.method}_slice_ax{args.axis}_p{args.pos:.2f}')


def render_mip(vol, args, rcfg, out_dir):
    from src.renderer import VolumeRenderer
    rdr = VolumeRenderer(vol)
    res = rcfg['render']['resolution']
    n   = rcfg['render']['n_samples']
    img = rdr.mip(args.axis, res, res, n_samples=n)
    _save(img, out_dir, f'{args.method}_mip_ax{args.axis}')


def render_dvr(vol, args, rcfg, out_dir):
    from src.renderer import VolumeRenderer
    rdr = VolumeRenderer(vol)
    tf  = build_tf(rcfg['transfer_function'])
    res = rcfg['render']['resolution']
    n   = rcfg['render']['n_samples']
    bg  = rcfg['render']['background']
    img = rdr.dvr(args.axis, tf=tf, res_h=res, res_w=res, n_samples=n,
                  background=tuple(bg))
    _save(img, out_dir, f'{args.method}_dvr_ax{args.axis}_{rcfg["transfer_function"]}')


def render_isosurface(vol, args, rcfg, out_dir):
    from src.renderer import VolumeRenderer
    rdr   = VolumeRenderer(vol)
    verts, faces = rdr.isosurface(threshold=args.threshold, D=128, H=128, W=128)
    # simple matplotlib 3-D visualisation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.4)
    mesh.set_edgecolor('none');  mesh.set_facecolor('#4daf4a')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.set_zlim(0, 1)
    ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
    ax.set_title(f'{args.method} isosurface  threshold={args.threshold}')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{args.method}_isosurface_{args.threshold:.2f}.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def render_compare(args, pcfg, rcfg, out_dir):
    """Side-by-side comparison panel: all methods, same slice."""
    from src.renderer import VolumeRenderer
    res  = rcfg['render']['resolution']
    cols = []
    labels = []
    for method in METHODS:
        try:
            vol = build_volume(method, pcfg, rcfg)
            rdr = VolumeRenderer(vol)
            sl  = rdr.slice(args.axis, args.pos, res, res)
            cols.append(sl)
            labels.append(method.upper())
        except Exception as e:
            print(f'[{method}] skipped: {e}')

    if not cols:
        print('No volumes loaded.'); return

    fig, axs = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.5),
                            gridspec_kw={'wspace': 0.04})
    if len(cols) == 1:
        axs = [axs]
    for ax, sl, lbl in zip(axs, cols, labels):
        ax.imshow(sl, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(lbl, fontsize=10);  ax.axis('off')
    plt.suptitle(f'INR comparison — axis={args.axis}, pos={args.pos:.2f}',
                 fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'compare_ax{args.axis}_p{args.pos:.2f}.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def render_panel(args, pcfg, rcfg, out_dir):
    """MIP panel: 3 axes × all methods."""
    from src.renderer import VolumeRenderer
    res = rcfg['render']['resolution']
    n   = rcfg['render']['n_samples']
    methods_available = []
    vols = {}
    for method in METHODS:
        try:
            vols[method] = build_volume(method, pcfg, rcfg)
            methods_available.append(method)
        except Exception as e:
            print(f'[{method}] skipped: {e}')

    axes  = [0, 1, 2]
    nrow  = len(axes)
    ncol  = len(methods_available)
    fig, axs = plt.subplots(nrow, ncol,
                            figsize=(3.5 * ncol, 3.5 * nrow),
                            gridspec_kw={'wspace': 0.03, 'hspace': 0.06})
    if nrow == 1: axs = axs[None, :]
    if ncol == 1: axs = axs[:, None]

    AXIS_LABEL = ['MIP-X', 'MIP-Y', 'MIP-Z']
    for row, axis in enumerate(axes):
        for col, method in enumerate(methods_available):
            vol = vols[method]
            rdr = VolumeRenderer(vol)
            img = rdr.mip(axis, res, res, n_samples=n)
            axs[row, col].imshow(img, cmap='gray', vmin=0, vmax=1,
                                 interpolation='nearest')
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title(method.upper(), fontsize=9)
        axs[row, 0].set_ylabel(AXIS_LABEL[axis], fontsize=9, rotation=90,
                               labelpad=4, va='center')

    plt.suptitle('INR MIP comparison panel', fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'mip_panel_all.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description='INR Volume Renderer')
    ap.add_argument('--method',    default='ingp',
                    choices=METHODS, help='INR method to render')
    ap.add_argument('--mode',      default='slice',
                    choices=['slice', 'mip', 'dvr', 'isosurface',
                             'compare', 'panel'])
    ap.add_argument('--axis',      type=int,   default=2,
                    help='Projection / slice axis  0=X 1=Y 2=Z')
    ap.add_argument('--pos',       type=float, default=0.5,
                    help='Normalised slice position ∈ [0,1]')
    ap.add_argument('--tf',        default=None,
                    help='Transfer function override (gray|hot|fluorescence|cell_body)')
    ap.add_argument('--threshold', type=float, default=0.25,
                    help='Isosurface threshold ∈ [0,1]')
    ap.add_argument('--res',       type=int,   default=None,
                    help='Output resolution (overrides renderer.yml)')
    return ap.parse_args()


if __name__ == '__main__':
    args    = parse_args()
    pcfg, rcfg = load_cfg()

    if args.res:
        rcfg['render']['resolution'] = args.res
    if args.tf:
        rcfg['transfer_function'] = args.tf

    out_dir = ROOT / rcfg['output']['dir']

    if args.mode in ('compare', 'panel'):
        if args.mode == 'compare':
            render_compare(args, pcfg, rcfg, out_dir)
        else:
            render_panel(args, pcfg, rcfg, out_dir)
    else:
        vol = build_volume(args.method, pcfg, rcfg)
        dispatch = {'slice': render_slice, 'mip': render_mip,
                    'dvr': render_dvr, 'isosurface': render_isosurface}
        dispatch[args.mode](vol, args, rcfg, out_dir)
