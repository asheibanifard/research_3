"""
volume_renderer.py — Rendering modes for any INRVolume.

Rendering modes
───────────────
  slice_xy / slice_xz / slice_yz   : orthographic 2-D cross-sections
  mip                               : maximum intensity projection
  dvr                               : direct volume rendering (front-to-back)
  isosurface                        : marching-cubes mesh on the INR field

All methods return numpy arrays.  GPU tensors are produced internally and
moved to CPU only for the final output, keeping GPU memory usage low.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch

from .inr_volume import INRVolume
from .transfer_function import TransferFunction, TransferFunction1D, tf_gray


# ── Camera ────────────────────────────────────────────────────────────────────

class OrthographicCamera:
    """Axis-aligned orthographic projection for volume rendering."""

    def __init__(self, axis: int = 2, img_h: int = 512, img_w: int = 512,
                 near: float = 0., far: float = 1., n_samples: int = 256):
        """
        axis      : projection axis  0=X, 1=Y, 2=Z  (rays travel along this axis)
        near/far  : ray start/end in [0,1]³ along axis
        n_samples : number of sample points per ray
        """
        self.axis      = axis
        self.img_h     = img_h
        self.img_w     = img_w
        self.near      = near
        self.far       = far
        self.n_samples = n_samples

    def generate_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (origins, step) both (H*W, 3) and () tensors.

        origins : (H*W, 3) — ray start points on the near plane in [0,1]³
        step    : (3,)    — displacement per sample along the ray
        """
        h, w = self.img_h, self.img_w
        a = torch.linspace(0., 1., h)
        b = torch.linspace(0., 1., w)
        ga, gb = torch.meshgrid(a, b, indexing='ij')   # H×W
        flat_a = ga.reshape(-1)                         # H*W
        flat_b = gb.reshape(-1)

        if self.axis == 0:   # rays along X
            o = torch.stack([torch.full_like(flat_a, self.near), flat_a, flat_b], dim=-1)
            d = torch.tensor([1., 0., 0.])
        elif self.axis == 1:  # rays along Y
            o = torch.stack([flat_a, torch.full_like(flat_a, self.near), flat_b], dim=-1)
            d = torch.tensor([0., 1., 0.])
        else:                 # rays along Z  (default top-down)
            o = torch.stack([flat_a, flat_b, torch.full_like(flat_a, self.near)], dim=-1)
            d = torch.tensor([0., 0., 1.])

        step_size = (self.far - self.near) / max(self.n_samples - 1, 1)
        return o, d * step_size


# ── Main renderer ─────────────────────────────────────────────────────────────

class VolumeRenderer:
    """All rendering operations on an INRVolume."""

    def __init__(self, volume: INRVolume):
        self.vol = volume

    # ── 2-D slice ─────────────────────────────────────────────────────────────

    def slice(self, axis: int, position: float,
              res_h: int = 512, res_w: int = 512) -> np.ndarray:
        """Axis-aligned 2-D cross-section.

        axis     : 0=YZ, 1=XZ, 2=XY
        position : normalised slice position ∈ [0, 1]
        Returns  : (res_h, res_w) float32 in [0, 1]
        """
        return self.vol.eval_slice(axis, position, res_h, res_w)

    def slice_rgb(self, axis: int, position: float,
                  tf: TransferFunction = None,
                  res_h: int = 512, res_w: int = 512) -> np.ndarray:
        """Colourised 2-D slice using a transfer function.
        Returns (res_h, res_w, 3) uint8.
        """
        if tf is None:
            tf = tf_gray()
        sl   = self.slice(axis, position, res_h, res_w)         # (H, W)
        rgba = tf(sl)                                            # (H, W, 4)
        rgb  = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        return rgb

    # ── MIP ───────────────────────────────────────────────────────────────────

    def mip(self, axis: int, res_h: int = 512, res_w: int = 512,
            n_samples: int = 256) -> np.ndarray:
        """Maximum intensity projection along axis.
        Returns (res_h, res_w) float32 in [0, 1].
        """
        cam = OrthographicCamera(axis=axis, img_h=res_h, img_w=res_w,
                                 n_samples=n_samples)
        origins, step = cam.generate_rays()
        device = self.vol.device
        origins = origins.to(device)
        step    = step.to(device)

        HW = res_h * res_w
        max_img = torch.zeros(HW, device=device)
        for i in range(n_samples):
            pts = origins + step * i           # (HW, 3)
            val = self.vol.query(pts)          # (HW,)
            max_img = torch.maximum(max_img, val)
        return max_img.cpu().numpy().reshape(res_h, res_w)

    # ── DVR (front-to-back alpha compositing) ─────────────────────────────────

    def dvr(self, axis: int,
            tf: Optional[TransferFunction] = None,
            res_h: int = 512, res_w: int = 512,
            n_samples: int = 256,
            background: Tuple[float, float, float] = (0., 0., 0.)) -> np.ndarray:
        """Direct volume rendering with front-to-back alpha compositing.

        Returns (res_h, res_w, 3) float32 RGB in [0, 1].
        """
        if tf is None:
            tf = tf_gray()

        cam = OrthographicCamera(axis=axis, img_h=res_h, img_w=res_w,
                                 n_samples=n_samples)
        origins, step = cam.generate_rays()
        device = self.vol.device
        origins = origins.to(device)
        step    = step.to(device)

        HW = res_h * res_w
        rgb_acc = torch.zeros(HW, 3, device=device)
        T_acc   = torch.ones(HW,    device=device)   # accumulated transmittance

        for i in range(n_samples):
            pts = origins + step * i              # (HW, 3)
            val = self.vol.query(pts).cpu().numpy()  # (HW,) on CPU for TF

            rgba = tf(val)                        # (HW, 4)
            a    = torch.from_numpy(rgba[:, 3]).to(device).clamp(0., 1.)
            col  = torch.from_numpy(rgba[:, :3]).to(device).clamp(0., 1.)

            # front-to-back: C += T * a * col;  T *= (1 - a)
            rgb_acc += (T_acc * a).unsqueeze(-1) * col
            T_acc   *= (1. - a)

        # composite over background
        bg  = torch.tensor(background, device=device)
        img = rgb_acc + T_acc.unsqueeze(-1) * bg
        return img.clamp(0., 1.).cpu().numpy().reshape(res_h, res_w, 3)

    # ── isosurface (marching cubes on a sampled grid) ─────────────────────────

    def isosurface(self, threshold: float = 0.3,
                   D: int = 128, H: int = 128, W: int = 128):
        """Extract an isosurface mesh from the INR field.

        Returns (vertices, faces) as numpy arrays.
        Requires skimage.measure.marching_cubes.
        """
        from skimage.measure import marching_cubes
        vol = self.vol.eval_volume(D, H, W)
        verts, faces, normals, _ = marching_cubes(vol, level=threshold)
        # normalise vertices to [0, 1]³
        verts /= np.array([D - 1, H - 1, W - 1], dtype=np.float32)
        return verts, faces

    # ── comparison panel ──────────────────────────────────────────────────────

    def comparison_panel(self, volumes: dict,
                         axis: int = 2, position: float = 0.5,
                         res: int = 256) -> np.ndarray:
        """Render the same slice across multiple INRVolumes side-by-side.

        volumes : {'label': INRVolume, ...}
        Returns : (res, n_volumes * res) float32 panel
        """
        import copy
        panels = []
        orig = self.vol
        for label, vol in volumes.items():
            self.vol = vol
            panels.append(self.slice(axis, position, res, res))
        self.vol = orig
        return np.concatenate(panels, axis=1)

    # ── progressive refinement renderer ───────────────────────────────────────

    def progressive_slice(self, axis: int, position: float,
                          res_final: int = 1024,
                          levels: int = 4):
        """Generator: yield progressively refined slices (coarse→fine).

        Yields (res, res) float32 arrays at increasing resolutions.
        Useful for interactive viewers — display each as it arrives.
        """
        for lvl in range(levels):
            res = res_final // (2 ** (levels - 1 - lvl))
            sl  = self.slice(axis, position, res, res)
            yield lvl, res, sl
