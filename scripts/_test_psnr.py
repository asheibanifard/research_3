"""Quick sanity test: compare sample PSNR (PyTorch) vs full-volume PSNR (CUDA kernel)."""
import sys, math, json, argparse, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / '_3dgs'))

import importlib.util
spec = importlib.util.spec_from_file_location("_3dgs_eval", Path(__file__).parent / "3dgs_eval.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RUN_DIR = Path('logs/3dgs/run13/20260514_135607')
CKPT    = RUN_DIR / 'best_20260514_135607.pth'

cfg = mod.load_config(RUN_DIR)
cfg.device = 'cuda'

import tifffile
vol_raw = tifffile.imread(cfg.volume).astype('float32')
vmin, vmax = float(vol_raw.min()), float(vol_raw.max())
gt_vol = (vol_raw - vmin) / (vmax - vmin) if vmax > vmin else vol_raw
gt_vol_t = torch.from_numpy(gt_vol)

print(f"Volume shape: {gt_vol.shape}")
print(f"GT range: [{gt_vol.min():.5f}, {gt_vol.max():.5f}]  mean={gt_vol.mean():.5f}")
print(f"Fraction > 0.01: {(gt_vol > 0.01).mean():.4f}  > 0.05: {(gt_vol > 0.05).mean():.4f}")

gc, aabb = mod.load_model(RUN_DIR, cfg, ckpt_path=CKPT)
dataset = mod.VolumeDataset(gt_vol_t, aabb, cfg, swc_path=cfg.swc_path)

# 1. PyTorch sample PSNR (what training sees)
pts, gt_s = dataset.sample_uniform(50_000, gc.device)
with torch.no_grad():
    pred = gc.forward(pts, chunk_n=cfg.chunk_n)
mse_s = F.mse_loss(pred.clamp(0,1), gt_s).item()
psnr_s = -10 * math.log10(mse_s + 1e-10)
print(f"\n--- Sample PSNR (500k uniform, PyTorch forward) ---")
print(f"  PSNR = {psnr_s:.4f} dB  MSE = {mse_s:.4e}")
print(f"  GT mean = {gt_s.mean().item():.5f}  pred mean = {pred.mean().item():.5f}")

# 2. CUDA kernel full-volume PSNR
print(f"\n--- Full-volume PSNR (CUDA kernel) ---")
recon_vol = mod.reconstruct_volume(gc, dataset, cfg, aabb)
metrics = mod.compute_metrics(gt_vol, recon_vol)
print(f"  PSNR (full) = {metrics['psnr']:.4f} dB  MSE = {metrics['mse']:.4e}")
print(f"  PSNR (fg)   = {metrics['psnr_fg']:.4f} dB")
print(f"  Recon mean = {recon_vol.mean():.5f}  range = [{recon_vol.min():.4f}, {recon_vol.max():.4f}]")

# 3. Foreground breakdown
fg = gt_vol > 0.05
print(f"\n--- Foreground (gt>0.05) breakdown ---")
print(f"  fg voxel fraction: {fg.mean():.4f}")
diff2 = (gt_vol - recon_vol)**2
print(f"  MSE foreground:   {diff2[fg].mean():.4e}")
print(f"  MSE background:   {diff2[~fg].mean():.4e}")
print(f"  GT background mean: {gt_vol[~fg].mean():.5f}")
print(f"  Pred background mean: {recon_vol[~fg].mean():.5f}")
