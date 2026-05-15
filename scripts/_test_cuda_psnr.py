"""Force-enable the training CUDA kernel and compute sample PSNR."""
import sys, math, json, argparse, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / '_3dgs'))

import importlib.util
spec = importlib.util.spec_from_file_location("_3dgs_eval", Path(__file__).parent / "3dgs_eval.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

spec2 = importlib.util.spec_from_file_location("_3dgs_src", Path(__file__).parent.parent / "src/_3dgs/_3dgs.py")
_3dgs_src = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_3dgs_src)

# Force-enable the training CUDA kernel
_3dgs_src.USE_CUDA_KERNEL = True
kernel = _3dgs_src._load_3dgs_kernel()
print(f"Training CUDA kernel loaded: {kernel}")

RUN_DIR = Path('logs/3dgs/run13/20260514_135607')
CKPT    = RUN_DIR / 'best_20260514_135607.pth'

cfg = mod.load_config(RUN_DIR)
cfg.device = 'cuda'

import tifffile
vol_raw = tifffile.imread(cfg.volume).astype('float32')
vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
gt_vol = (vol_raw - vmin) / (vmax - vmin) if vmax > vmin else vol_raw
gt_vol_t = torch.from_numpy(gt_vol)

gc, aabb = mod.load_model(RUN_DIR, cfg, ckpt_path=CKPT)
dataset = mod.VolumeDataset(gt_vol_t, aabb, cfg, swc_path=cfg.swc_path)

# Test with PyTorch path (USE_CUDA_KERNEL=False on the GaussianCloud)
pts, gt_s = dataset.sample_uniform(50_000, gc.device)

# PyTorch path
with torch.no_grad():
    pred_pt = gc.forward(pts, chunk_n=cfg.chunk_n)
mse_pt = F.mse_loss(pred_pt.clamp(0,1), gt_s).item()
print(f"PyTorch path PSNR: {-10*math.log10(mse_pt+1e-10):.4f} dB  MSE={mse_pt:.4e}")
print(f"  pred range: [{pred_pt.min().item():.4f}, {pred_pt.max().item():.4f}]")

# Training CUDA path (manual)
inten_eval = F.softplus(gc.inten)
gain = torch.ones_like(inten_eval)
out = torch.zeros(pts.shape[0], device=gc.device, dtype=pts.dtype)
chunk_n = cfg.chunk_n
with torch.no_grad():
    for s in range(0, gc.means.shape[0], chunk_n):
        e = min(s + chunk_n, gc.means.shape[0])
        out = out + kernel.forward(
            pts.contiguous(),
            gc.means[s:e].contiguous(),
            gc.log_s[s:e].contiguous(),
            gc.quats[s:e].contiguous(),
            gain[s:e].contiguous(),
            inten_eval[s:e].contiguous(),
            gc.scale_min,
            gc.mahal_clamp,
        )
mse_cuda = F.mse_loss(out.clamp(0,1), gt_s).item()
print(f"Training CUDA path PSNR: {-10*math.log10(mse_cuda+1e-10):.4f} dB  MSE={mse_cuda:.4e}")
print(f"  pred range: [{out.min().item():.4f}, {out.max().item():.4f}]")

# Check if outputs are close
diff = (pred_pt - out).abs()
print(f"\nMax abs diff PyTorch vs CUDA kernel: {diff.max().item():.6f}")
print(f"Mean abs diff: {diff.mean().item():.6f}")
