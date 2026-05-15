"""
inr_volume.py — Unified query interface for all trained INR models.

Every concrete subclass exposes a single method:
    query(pts) -> intensity
where pts is (N, 3) float32 in normalised coords [-1, 1]³ or [0, 1]³
(controlled by self.coord_range) and the return is (N,) float32 in [0, 1].

Adapter subclasses wrap the project's existing model classes so the renderer
never needs to know which backend is running.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch
import torch.nn as nn


# ── Abstract base ─────────────────────────────────────────────────────────────

class INRVolume(ABC):
    """Continuous scalar field I: ℝ³ → [0, 1] backed by a trained INR."""

    device: torch.device
    chunk: int           # max points per forward pass (memory budget)
    coord_range: str     # 'unit' = [0,1]³  |  'symmetric' = [-1,1]³

    @abstractmethod
    def _forward_chunk(self, pts: torch.Tensor) -> torch.Tensor:
        """Evaluate the INR on a single chunk.  pts: (N,3) → (N,)."""

    @torch.inference_mode()
    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """Query intensity at arbitrary 3-D coordinates.

        pts : (N, 3) float32  — coordinates in self.coord_range
        Returns (N,) float32 clipped to [0, 1].
        """
        pts = pts.to(self.device)
        out = torch.empty(pts.shape[0], device=self.device, dtype=torch.float32)
        for s in range(0, pts.shape[0], self.chunk):
            e = min(s + self.chunk, pts.shape[0])
            out[s:e] = self._forward_chunk(pts[s:e]).clamp(0., 1.)
        return out

    def query_numpy(self, pts: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(pts.astype(np.float32))
        return self.query(t).cpu().numpy()

    # ── convenience: dense grid evaluation ───────────────────────────────────
    @torch.inference_mode()
    def eval_volume(self, D: int, H: int, W: int) -> np.ndarray:
        """Evaluate on a regular D×H×W grid covering [0,1]³."""
        iz = torch.arange(D, dtype=torch.float32) / max(D - 1, 1)
        ih = torch.arange(H, dtype=torch.float32) / max(H - 1, 1)
        iw = torch.arange(W, dtype=torch.float32) / max(W - 1, 1)
        zz, yy, xx = torch.meshgrid(iz, ih, iw, indexing='ij')  # D×H×W each
        pts = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  # (N,3)
        if self.coord_range == 'symmetric':
            pts = pts * 2. - 1.
        vol = self.query(pts).cpu().numpy().reshape(D, H, W)
        return vol

    def eval_slice(self, axis: int, index: float,
                   res_a: int, res_b: int) -> np.ndarray:
        """Evaluate a 2-D slice at normalised position index ∈ [0,1] along axis."""
        a = torch.linspace(0., 1., res_a)
        b = torch.linspace(0., 1., res_b)
        ga, gb = torch.meshgrid(a, b, indexing='ij')
        const = torch.full_like(ga, index)
        if axis == 0:   # YZ slice  (x = const)
            pts = torch.stack([const, ga, gb], dim=-1).reshape(-1, 3)
        elif axis == 1:  # XZ slice  (y = const)
            pts = torch.stack([ga, const, gb], dim=-1).reshape(-1, 3)
        else:            # XY slice  (z = const)
            pts = torch.stack([ga, gb, const], dim=-1).reshape(-1, 3)
        if self.coord_range == 'symmetric':
            pts = pts * 2. - 1.
        return self.query(pts).cpu().numpy().reshape(res_a, res_b)


# ── SIREN adapter ─────────────────────────────────────────────────────────────

class SIRENVolume(INRVolume):
    """Wraps a trained SIREN network."""

    coord_range = 'symmetric'   # SIREN is trained on [-1,1]³

    def __init__(self, model: nn.Module, device='cuda', chunk: int = 2**18):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.chunk  = chunk

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path,
                        hidden_features: int, hidden_layers: int,
                        omega_0: float = 30., device='cuda') -> 'SIRENVolume':
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))
        from src.siren.siren import Siren
        model = Siren(in_features=3, out_features=1,
                      hidden_features=hidden_features,
                      hidden_layers=hidden_layers,
                      omega_0=omega_0, outermost_linear=True)
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        return cls(model, device=device)

    def _forward_chunk(self, pts):
        coords = pts.unsqueeze(0)       # (1, N, 3)
        out, _ = self.model(coords)
        return out.squeeze(-1).squeeze(0)  # (N,)


# ── NeRF (PE) adapter ─────────────────────────────────────────────────────────

class NeRFVolume(INRVolume):
    coord_range = 'unit'

    def __init__(self, model: nn.Module, device='cuda', chunk: int = 2**18):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.chunk  = chunk

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device='cuda') -> 'NeRFVolume':
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))
        from src.nerf.nerf import NeRF
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        cfg  = ckpt.get('config', {})
        model = NeRF(**cfg)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        return cls(model, device=device)

    def _forward_chunk(self, pts):
        return self.model(pts).squeeze(-1)


# ── RFF adapter ───────────────────────────────────────────────────────────────

class RFFVolume(INRVolume):
    coord_range = 'unit'

    def __init__(self, model: nn.Module, device='cuda', chunk: int = 2**18):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.chunk  = chunk

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device='cuda') -> 'RFFVolume':
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))
        from src.rff.rff import RFFNet
        ckpt  = torch.load(str(ckpt_path), map_location='cpu')
        cfg   = ckpt.get('config', {})
        model = RFFNet(**cfg)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        return cls(model, device=device)

    def _forward_chunk(self, pts):
        return self.model(pts).squeeze(-1)


# ── INGP adapter ──────────────────────────────────────────────────────────────

class INGPVolume(INRVolume):
    coord_range = 'unit'

    def __init__(self, model: nn.Module, device='cuda', chunk: int = 2**20):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.chunk  = chunk

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device='cuda') -> 'INGPVolume':
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))
        from src.ingp.ingp import INGP
        ckpt  = torch.load(str(ckpt_path), map_location='cpu')
        cfg   = ckpt.get('config', {})
        model = INGP(**cfg)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        return cls(model, device=device)

    def _forward_chunk(self, pts):
        return self.model(pts).squeeze(-1)


# ── 3DGS adapter ──────────────────────────────────────────────────────────────

class GaussianVolume(INRVolume):
    """3D Gaussian Splatting as a continuous volume field.

    Uses the fused CUDA kernel when available, falls back to Python loop.
    """
    coord_range = 'unit'

    def __init__(self, gc, aabb, device='cuda', chunk: int = 2**16,
                 use_kernel: bool = True):
        self.gc      = gc
        self.aabb    = aabb
        self.device  = torch.device(device)
        self.chunk   = chunk
        self._kernel = None
        if use_kernel and device != 'cpu':
            self._kernel = self._load_kernel()

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path,
                        device='cuda', use_kernel=True) -> 'GaussianVolume':
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(root))
        from src._3dgs._3dgs import GaussianCloud, AABB
        from src._3dgs._3dgs import _default_cfg
        aabb = AABB.unit()
        cfg  = _default_cfg()
        gc   = GaussianCloud.load(str(ckpt_path), aabb, torch.device(device), cfg)
        return cls(gc, aabb, device=device, use_kernel=use_kernel)

    def _load_kernel(self):
        import os
        from pathlib import Path as P
        from torch.utils.cpp_extension import load
        root = P(__file__).resolve().parent.parent.parent
        src  = root / 'src' / '_3dgs' / '3dgs_eval_cuda.cu'
        cands = []
        try:
            r = P(__file__).parent.parent
            for p in sorted((r/'envs').glob('*/include')): cands.append(p)
            import sys
            cands.insert(0, P(sys.prefix)/'include')
        except Exception:
            pass
        inc = [str(p) for p in cands
               if (p/'nv'/'target').exists() and (p/'cuda_runtime.h').exists()]
        return load(name='3dgs_eval_cuda_rdr',
                    sources=[str(src)],
                    extra_cuda_cflags=['-O3','--use_fast_math'],
                    extra_include_paths=inc, verbose=False)

    def _forward_chunk(self, pts):
        # pts: (N,3) in [0,1]³  →  map to AABB
        lo = self.aabb.lo.to(self.device)
        hi = self.aabb.hi.to(self.device)
        world = lo + pts * (hi - lo)
        gc = self.gc
        # build Σ⁻¹ inline for arbitrary points (mirrors GaussianCloud.forward)
        return gc.forward(world, chunk_n=len(world)).clamp(0., 1.)

    # fast full-volume path using the dedicated kernel
    @torch.inference_mode()
    def eval_volume(self, D: int, H: int, W: int) -> np.ndarray:
        if self._kernel is not None:
            gc = self.gc
            lo = self.aabb.lo.cpu()
            hi = self.aabb.hi.cpu()
            flat = self._kernel.reconstruct_volume(
                gc.means.contiguous(), gc.log_s.contiguous(),
                gc.quats.contiguous(), gc.inten.contiguous(),
                float(lo[0]), float(hi[0]),
                float(lo[1]), float(hi[1]),
                float(lo[2]), float(hi[2]),
                D, H, W, float(gc.scale_min), float(gc.mahal_clamp))
            return flat.reshape(D, H, W).cpu().numpy()
        return super().eval_volume(D, H, W)
