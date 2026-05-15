from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable

import numpy as np
import tifffile
import torch
from tqdm import tqdm


def _load_volume(volume_path: str) -> tuple[torch.Tensor, float, float]:
    raw = tifffile.imread(volume_path)
    vol = raw.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return torch.from_numpy(vol), vmin, vmax


def train_impl(
    cfg: argparse.Namespace,
    *,
    aabb_cls: type,
    volume_dataset_cls: type,
    gaussian_cloud_cls: type,
    make_optimizer: Callable[[Any, argparse.Namespace], torch.optim.Optimizer],
    update_lr: Callable[[torch.optim.Optimizer, int, int, argparse.Namespace], None],
    compute_loss: Callable[..., tuple[torch.Tensor, dict]],
    evaluate_fields: Callable[..., dict],
):
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.out) / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading volume: {cfg.volume}")
    volume, vmin, vmax = _load_volume(cfg.volume)
    d, h, w = volume.shape
    print(f"  Shape     : {d} × {h} × {w}  ({d*h*w/1e6:.1f} M voxels)")
    print(f"  Intensity : [{vmin:.4f}, {vmax:.4f}] → [0, 1]")

    device = torch.device(cfg.device)
    aabb = aabb_cls.unit()
    dataset = volume_dataset_cls(volume, aabb, cfg, swc_path=cfg.swc_path)

    init_pts   = None
    init_quats = None
    if getattr(cfg, 'swc_init', True):
        if getattr(cfg, 'swc_oriented_init', False):
            pts, quats = dataset.swc_oriented_init_params()
            init_pts   = pts.to(device)
            init_quats = quats.to(device)
            print(f"  SWC init     : {init_pts.shape[0]} pts with oriented quats")
        else:
            init_pts = dataset.swc_init_points().to(device)

    # Extra Gaussians seeded inside bright interior voxels (soma fill).
    interior_n = int(getattr(cfg, 'interior_init_n', 0))
    if interior_n > 0:
        thresh  = float(getattr(cfg, 'interior_init_thresh', 0.3))
        int_pts = dataset.interior_init_points(interior_n, thresh).to(device)
        if int_pts.numel() > 0:
            if init_pts is not None and init_pts.numel() > 0:
                # identity quats for interior points
                int_q = torch.zeros(int_pts.shape[0], 4, device=device)
                int_q[:, 0] = 1.0
                init_pts   = torch.cat([init_pts, int_pts], dim=0)
                if init_quats is not None:
                    init_quats = torch.cat([init_quats, int_q], dim=0)
            else:
                init_pts = int_pts
            print(f"  Interior init: {int_pts.shape[0]} pts (thresh={thresh})")

    total_steps = cfg.epochs * cfg.steps_per_epoch
    gc = gaussian_cloud_cls(cfg.n_init, aabb, device, cfg,
                            init_pts=init_pts, init_quats=init_quats)
    optimizer = make_optimizer(gc, cfg)
    update_lr(optimizer, 0, total_steps, cfg)
    init_path  = out_dir / f"init_{run_stamp}.pth"
    best_path  = out_dir / f"best_{run_stamp}.pth"
    final_path = out_dir / f"final_{run_stamp}.pth"
    gc.save(init_path)

    log_entries = []
    best_psnr = -float('inf')
    bad_epochs = 0
    t0 = time()
    detail_interval = max(int(cfg.eval_detail_interval), 1)

    init_desc = 'SWC' if init_pts is not None and init_pts.numel() > 0 else 'random'
    print(f"  Gaussians : {gc.N} init from {init_desc}  (max {cfg.max_gaussians})")
    print(f"  Steps     : {total_steps}  ({cfg.epochs} ep × {cfg.steps_per_epoch})")

    step = 0
    last_psnr = float('nan')
    epoch_bar = tqdm(range(cfg.epochs), desc="epoch", unit="ep", dynamic_ncols=True)
    densify_until = getattr(cfg, 'densify_until_step', None)

    for epoch in epoch_bar:
        epoch_loss = torch.zeros((), device=device)

        for _ in range(cfg.steps_per_epoch):
            step += 1

            pts, gt = dataset.sample(cfg.batch, device)

            optimizer.zero_grad()
            pred = gc.forward(pts, chunk_n=cfg.chunk_n)
            loss, stats = compute_loss(pred, gt, gc, cfg, dataset, step=step)

            loss.backward()
            gc.accum_grads()

            if gc.means.grad is not None:
                torch.nn.utils.clip_grad_norm_([gc.means], max_norm=cfg.grad_clip_norm)

            optimizer.step()
            update_lr(optimizer, step, total_steps, cfg)
            gc.clamp_means()
            gc.clamp_scales(cfg.scale_max_hard, getattr(cfg, 'scale_min_hard', None))

            epoch_loss = epoch_loss + stats['loss']

            if (
                step >= cfg.densify_from_step
                and (densify_until is None or step <= densify_until)
                and step % cfg.densify_interval == 0
            ):
                n_pruned, n_added = gc.densify_and_prune(cfg)
                optimizer = make_optimizer(gc, cfg)
                update_lr(optimizer, step, total_steps, cfg)
                tqdm.write(
                    f"  [step {step:6d}] densify — "
                    f"pruned {n_pruned:5d}  added {n_added:5d}  total {gc.N:6d}"
                )

            if cfg.ckpt_interval > 0 and step % cfg.ckpt_interval == 0:
                gc.save(out_dir / f"ckpt_{run_stamp}_{step:07d}.pth")

        avg_loss = (epoch_loss / cfg.steps_per_epoch).item()
        detail_eval = ((epoch + 1) % detail_interval == 0) or (epoch == cfg.epochs - 1)
        eval_metrics = evaluate_fields(gc, dataset, cfg, detail=detail_eval)
        last_psnr = eval_metrics['psnr']
        s_mean, s_max = gc.scale_stats()
        elapsed = time() - t0
        inten_mean = gc.intensity().mean().item()

        log_entries.append({
            'epoch': epoch + 1,
            'step': step,
            'loss': round(avg_loss, 6),
            'psnr': round(eval_metrics['psnr'], 4),
            'vol_psnr': None if math.isnan(eval_metrics['vol_psnr']) else round(eval_metrics['vol_psnr'], 4),
            'n_gauss': gc.N,
            's_mean': round(s_mean, 5),
            's_max': round(s_max, 5),
            'inten_mean': round(inten_mean, 4),
            'elapsed_s': round(elapsed, 1),
        })

        epoch_bar.set_postfix(
            loss=f'{avg_loss:.4f}',
            psnr=f'{last_psnr:.2f}',
            N=gc.N,
            t=f'{elapsed/60:.1f}m',
        )

        if last_psnr > best_psnr:
            best_psnr = last_psnr
            bad_epochs = 0
            gc.save(best_path)
            tqdm.write(
                f"  ★ new best  epoch {epoch+1:4d}  "
                f"PSNR={last_psnr:.2f} dB  N={gc.N}  "
                f"t={elapsed/60:.1f} min"
            )
        else:
            bad_epochs += 1
            if cfg.early_stop_patience is not None and bad_epochs >= cfg.early_stop_patience:
                tqdm.write(
                    f"  [epoch {epoch+1:4d}] early stopping after {bad_epochs} stagnant epochs "
                    f"(best PSNR={best_psnr:.2f} dB)"
                )
                break

    epoch_bar.close()

    gc.save(final_path)

    with open(out_dir / "log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    print(f"\nDone. Best PSNR = {best_psnr:.2f} dB  →  {best_path}")
    return gc, log_entries
