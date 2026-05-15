"""
Gaussian Field Fitting — Volumetric Regression
==============================================
Fits a mixture of anisotropic 3-D Gaussians to a voxel grid by minimising MSE
between the Gaussian field evaluated at sampled coordinates and the
corresponding ground-truth voxel intensities.

Model:
    f(x) = Σ_k  v_k · exp(−½ (x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k))

where v_k = softplus(raw_inten_k), and
Σ_k = R_k diag(s_k²) R_kᵀ  (from quaternion + log-scale parameterisation).

All numerical constants are sourced from the config (YAML or CLI).
No default values appear in function signatures.

Usage
-----
python src/3dgs.py --config configs/3dgs.yml [--volume override.tif]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Training loop lives in a separate module so it can be unit-tested and reused
# without importing the full model stack.  This file wires all components
# together and passes them in via dependency injection.
from _3dgs._3dgs_training import train_impl as _train_impl


# ── CUDA kernel singletons ────────────────────────────────────────────────────
# USE_CUDA_KERNEL is toggled by --use_kernel at startup.  Keeping it as a
# module-level flag avoids threading a boolean through every call site while
# still allowing a clean CPU fallback path.
USE_CUDA_KERNEL = False
_3dgs_cuda      = None
_eval_cuda      = None   # 3dgs_eval_cuda: reconstruct_volume + splat_mip


def _find_cuda_include() -> list:
    """Return an include path that has the full CUDA toolkit headers (cuda_runtime.h + nv/target)."""
    import os
    candidates = []
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        candidates.append(Path(cuda_home) / 'include')
    try:
        import sys as _sys
        conda_root = Path(_sys.executable).parent.parent
        for inc in sorted((conda_root / 'envs').glob('*/include')):
            candidates.append(inc)
        candidates.insert(0, Path(_sys.prefix) / 'include')
    except Exception:
        pass
    for p in candidates:
        if (p / 'nv' / 'target').exists() and (p / 'cuda_runtime.h').exists():
            return [str(p)]
    return []


def _load_eval_kernel():
    """Lazily compile and cache 3dgs_eval_cuda (reconstruct_volume + splat_mip)."""
    global _eval_cuda
    if _eval_cuda is None:
        src = Path(__file__).parent / "3dgs_eval_cuda.cu"
        extra_inc = _find_cuda_include()
        extra_flags = ["-O3", "--use_fast_math"] + [f"-I{p}" for p in extra_inc]
        _eval_cuda = load(
            name="3dgs_eval_cuda",
            sources=[str(src)],
            extra_cuda_cflags=extra_flags,
            extra_include_paths=extra_inc,
            verbose=False,
        )
    return _eval_cuda


def _load_3dgs_kernel():
    """Lazily compile and cache the fused CUDA Gaussian-field extension."""
    global _3dgs_cuda
    if _3dgs_cuda is None:
        src        = Path(__file__).parent / "3dgs_cuda.cu"
        extra_inc  = _find_cuda_include()
        extra_flags = ["-O3", "--use_fast_math"] + [f"-I{p}" for p in extra_inc]
        _3dgs_cuda = load(
            name="gaussian_3dgs_cuda",
            sources=[str(src)],
            extra_cuda_cflags=extra_flags,
            extra_include_paths=extra_inc,
            verbose=False,
        )
    return _3dgs_cuda


# ─────────────────────────────────────────────────────────────────────────────
# AABB — Axis-Aligned Bounding Box
# ─────────────────────────────────────────────────────────────────────────────
# Why a dedicated class?  Every coordinate in the pipeline lives in the
# canonical normalised cube [-1,1]³ that matches PyTorch's F.grid_sample
# convention.  Wrapping lo/hi in an AABB makes the coordinate contract
# explicit and prevents silent mismatches between voxel-index space and the
# normalised training space.
#%%
class AABB:
    """Axis-Aligned Bounding Box used as the canonical training coordinate space.

    Why [-1,1]³?  PyTorch's F.grid_sample maps -1 to the first voxel and +1
    to the last.  Keeping Gaussian means and sample coordinates in the same
    space lets us call grid_sample directly for trilinear GT lookups without
    any extra coordinate transform.
    """

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor):
        self.lo     = lo
        self.hi     = hi
        self.center = 0.5 * (lo + hi)
        self.extent = hi - lo

    @classmethod
    def unit(cls):
        """Return the canonical unit cube [-1,1]³ used throughout training."""
        return cls(torch.full((3,), -1.0), torch.full((3,), 1.0))

    def to(self, device):
        """Move lo/hi to device; returns a new AABB (tensors are not mutated)."""
        return AABB(self.lo.to(device), self.hi.to(device))

    def contains(self, pts: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True for each row of pts that lies inside [lo, hi].

        Used to detect Gaussians that have drifted out-of-volume and should
        be pruned during adaptive density control.
        """
        lo = self.lo.to(pts.device)
        hi = self.hi.to(pts.device)
        return ((pts >= lo) & (pts <= hi)).all(-1)

    def clamp(self, pts: torch.Tensor) -> torch.Tensor:
        """Project pts onto the bounding box surface.

        Applied to Gaussian means every gradient step to prevent them from
        drifting off-volume where they contribute nothing to the in-volume loss.
        """
        lo = self.lo.to(pts.device)
        hi = self.hi.to(pts.device)
        return pts.clamp(lo, hi)

    def random_pts(self, n: int, device) -> torch.Tensor:
        """Sample n points uniformly at random from the interior.

        Used by VolumeDataset.sample_uniform() to generate training query
        points in a single vectorised call instead of looping over voxel indices.
        """
        u = torch.rand(n, 3, device=device)
        return u * self.extent.to(device) + self.lo.to(device)

    def is_empty(self) -> bool:
        return bool((self.hi <= self.lo).any().item())

    def volume(self) -> torch.Tensor:
        return self.extent.clamp_min(0).prod()

    def intersects(self, other: "AABB") -> bool:
        return bool(((self.hi >= other.lo) & (other.hi >= self.lo)).all().item())

    def intersection(self, other: "AABB") -> "AABB":
        return AABB(torch.maximum(self.lo, other.lo), torch.minimum(self.hi, other.hi))

    def intersection_volume(self, other: "AABB") -> torch.Tensor:
        return (torch.minimum(self.hi, other.hi) - torch.maximum(self.lo, other.lo)).clamp_min(0).prod()


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / covariance utilities
# ─────────────────────────────────────────────────────────────────────────────
# Why quaternions?  They live on the 4-D unit sphere, have no gimbal-lock
# singularity, and admit a closed-form gradient.  Gradient descent on the
# raw 4-vector followed by F.normalize stays on SO(3) without constrained
# optimisation — simpler and faster than Lie-group exponential maps.
#
# Why log-scale?  Optimising log(s) ∈ ℝ removes the s > 0 constraint.
# Any real value maps to a valid scale via exp, and the gradient never
# vanishes near zero (unlike clamped-scale approaches).
#%%
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to 3×3 rotation matrices.

    Uses the closed-form Rodrigues formula expanded component-wise:
        R_ij = f(w, x, y, z)
    Avoids materialising the skew-symmetric cross-product matrix.

    Parameters
    ----------
    q : (N, 4) float  [w, x, y, z]; need not be normalised on input.

    Returns
    -------
    R : (N, 3, 3) rotation matrices in SO(3).
    """
    q = F.normalize(q, dim=-1)      # enforce unit norm before formula
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_sigma_inv(log_scales: torch.Tensor,
                    quats:      torch.Tensor,
                    scale_min:  float) -> torch.Tensor:
    """Build the inverse covariance matrix Σ⁻¹ for each Gaussian.

    Derivation (avoids explicit matrix inversion):
        Σ   = R diag(s²) Rᵀ
        Σ⁻¹ = R diag(1/s²) Rᵀ
            = (R · diag(1/s)) · (R · diag(1/s))ᵀ   [let RD = R * (1/s)]
            = RD @ RDᵀ

    Parameters
    ----------
    log_scales : (N, 3)  log of per-axis standard deviations.
    quats      : (N, 4)  rotation quaternions [w, x, y, z].
    scale_min  : float   lower clamp on s before inversion (prevents 1/~0).

    Returns
    -------
    (N, 3, 3) symmetric positive-definite inverse covariance matrices.
    """
    s  = torch.exp(log_scales).clamp(min=scale_min)
    R  = quat_to_rotmat(quats)
    RD = R * (1.0 / s).unsqueeze(1)    # (N,3,3) * (N,1,3) → (N,3,3)
    return RD @ RD.transpose(-1, -2)


# ─────────────────────────────────────────────────────────────────────────────
# SWC skeleton utilities
# ─────────────────────────────────────────────────────────────────────────────
# Why initialise from SWC skeletons?  Random initialisation forces Gaussians
# to migrate across empty space before they can cover the thin neuron branches.
# Seeding directly on the skeleton shortens this drift phase and typically
# halves the epochs needed to reach a given PSNR.
#%%
def load_swc(path: str) -> np.ndarray:
    """Parse an SWC morphology file and return an (N, 5) float32 array.

    SWC column layout:  id  type  x  y  z  radius  parent
    We keep [x, y, z, radius, parent] per node (columns 2-6).

    Blank lines and '#' comment lines are skipped.
    Rows with fewer than 7 columns are silently dropped (malformed nodes).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append([
                float(parts[2]),        # x (voxel index)
                float(parts[3]),        # y
                float(parts[4]),        # z
                float(parts[5]),        # radius
                int(float(parts[6])),   # parent id
            ])
    if not rows:
        raise ValueError(f'No SWC points found in {path}')
    return np.asarray(rows, dtype=np.float32)


def load_swc_full(path: str) -> np.ndarray:
    """Parse an SWC file and return (N, 6) float32: [node_id, x, y, z, radius, parent_id].

    Unlike load_swc, the node ID is preserved as column 0 so that parent-child
    direction vectors can be computed for oriented Gaussian initialisation.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append([
                int(float(parts[0])),   # node id
                float(parts[2]),        # x (voxel index)
                float(parts[3]),        # y
                float(parts[4]),        # z
                float(parts[5]),        # radius
                int(float(parts[6])),   # parent id (-1 = root)
            ])
    if not rows:
        raise ValueError(f'No SWC points found in {path}')
    return np.asarray(rows, dtype=np.float32)


def swc_points_to_unit_aabb(swc_xyz: np.ndarray, volume_shape: Tuple[int, int, int]) -> np.ndarray:
    """Map SWC voxel-index coordinates into the normalised [-1,1]³ AABB.

    SWC files store coordinates in voxel-index space (origin at corner,
    extents given by volume_shape).  Gaussian means live in [-1,1]³ so
    that F.grid_sample can look up GT intensities at their positions.

    Linear map: index 0 → -1,  index (size-1) → +1.

    Parameters
    ----------
    swc_xyz      : (N, 3) array of [x, y, z] voxel-index coordinates.
    volume_shape : (D, H, W) shape of the raw voxel volume.

    Returns
    -------
    (N, 3) float32 array in [-1, 1]³.
    """
    d, h, w = volume_shape
    xyz = swc_xyz.copy().astype(np.float32)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x = (x / max(w - 1, 1)) * 2.0 - 1.0
    y = (y / max(h - 1, 1)) * 2.0 - 1.0
    z = (z / max(d - 1, 1)) * 2.0 - 1.0
    return np.stack([x, y, z], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernel wrapper — _GaussianFieldFn
# ─────────────────────────────────────────────────────────────────────────────
# Why a custom autograd.Function?  The fused CUDA kernel computes both the
# forward pass and analytic gradients in one launch, avoiding the O(N × M)
# intermediate tensors that the pure-PyTorch path materialises.
# autograd.Function slots the kernel into the autograd tape so that
# loss.backward() works transparently without rewriting the training loop.
#%%
class _GaussianFieldFn(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA Gaussian-field kernel.

    Forward:  evaluates f(x) = Σ_k softplus(inten_k) · exp(-½ dᵀ Σ_k⁻¹ d)
              via kernel.forward (chunked over Gaussians to cap VRAM).

    Backward: delegates to _chunked_cuda_backward which applies the chain-rule
              through softplus and uses OOM-adaptive chunk halving so the
              backward never crashes on large models.

    Note: gain is computed internally as ones (kept for kernel ABI compat).
    The saved tensors are the raw inten params; softplus is recomputed in
    backward to chain through the activation.
    """

    @staticmethod
    def forward(ctx, pts, means, log_s, quats, inten, scale_min, mahal_clamp, chunk_n):
        ctx.save_for_backward(pts, means, log_s, quats, inten)
        ctx.scale_min   = float(scale_min)
        ctx.mahal_clamp = float(mahal_clamp)
        ctx.chunk_n     = int(chunk_n)

        kernel     = _load_3dgs_kernel()
        pts_c      = pts.contiguous()
        inten_eval = F.softplus(inten)          # v_k = softplus(raw), always > 0
        gain       = torch.ones_like(inten_eval) # kernel ABI requires a gain tensor
        out = torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)
        for s in range(0, means.shape[0], ctx.chunk_n):
            e = min(s + ctx.chunk_n, means.shape[0])
            out = out + kernel.forward(
                pts_c,
                means[s:e].contiguous(),
                log_s[s:e].contiguous(),
                quats[s:e].contiguous(),
                gain[s:e].contiguous(),
                inten_eval[s:e].contiguous(),
                ctx.scale_min,
                ctx.mahal_clamp,
            )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        pts, means, log_s, quats, inten = ctx.saved_tensors
        kernel = _load_3dgs_kernel()
        grads = _chunked_cuda_backward(
            kernel, grad_out, pts, means, log_s, quats, inten,
            ctx.scale_min, ctx.mahal_clamp, ctx.chunk_n,
        )
        # Return None for non-differentiable args: pts, scale_min, mahal_clamp, chunk_n
        return None, *grads, None, None, None


def _chunked_cuda_backward(kernel, grad_out, pts, means, log_s, quats, inten,
                           scale_min: float, mahal_clamp: float, chunk_n: int):
    """Run the CUDA backward pass with OOM-adaptive chunk halving.

    Why adaptive chunk halving?  The backward kernel allocates intermediate
    buffers proportional to chunk_n × batch_size.  On machines with limited
    VRAM, a fixed chunk_n may cause an OOM on the first backward call.
    Rather than forcing the user to tune chunk_n manually, we catch
    torch.OutOfMemoryError and halve the chunk until it succeeds or hits 1.

    Why chain through softplus here?  kernel.backward returns gradients with
    respect to the post-softplus intensity (g_inten_eval = ∂L/∂v).  We need
    ∂L/∂raw_inten = ∂L/∂v · ∂v/∂raw = g_inten_eval · sigmoid(raw_inten)
    because the saved tensor is raw_inten, not v.
    """
    grad_out_c   = grad_out.contiguous()
    pts_c        = pts.contiguous()
    means_c      = means.contiguous()
    log_s_c      = log_s.contiguous()
    quats_c      = quats.contiguous()
    inten_c      = inten.contiguous()
    inten_eval_c = F.softplus(inten_c)
    gain_c       = torch.ones_like(inten_eval_c)  # kernel ABI

    n_gauss = means_c.shape[0]
    chunk   = n_gauss if chunk_n <= 0 else min(int(chunk_n), n_gauss)

    while True:
        try:
            if chunk >= n_gauss:
                # Single-shot: process all Gaussians in one kernel call
                g_means, g_log_s, g_quats, _g_gain, g_inten_eval = kernel.backward(
                    grad_out_c, pts_c, means_c, log_s_c, quats_c,
                    gain_c, inten_eval_c, scale_min, mahal_clamp,
                )
                # Chain through softplus: ∂L/∂raw = ∂L/∂v · sigmoid(raw)
                return g_means, g_log_s, g_quats, g_inten_eval * torch.sigmoid(inten_c)

            grad_means = torch.zeros_like(means_c)
            grad_log_s = torch.zeros_like(log_s_c)
            grad_quats = torch.zeros_like(quats_c)
            grad_inten = torch.zeros_like(inten_c)

            for s in range(0, n_gauss, chunk):
                e = min(s + chunk, n_gauss)
                g_means, g_log_s, g_quats, _g_gain, g_inten_eval = kernel.backward(
                    grad_out_c, pts_c,
                    means_c[s:e], log_s_c[s:e], quats_c[s:e],
                    gain_c[s:e], inten_eval_c[s:e],
                    scale_min, mahal_clamp,
                )
                grad_means[s:e].copy_(g_means)
                grad_log_s[s:e].copy_(g_log_s)
                grad_quats[s:e].copy_(g_quats)
                grad_inten[s:e].copy_(g_inten_eval * torch.sigmoid(inten_c[s:e]))

            return grad_means, grad_log_s, grad_quats, grad_inten

        except torch.OutOfMemoryError:
            if chunk == 1:
                raise   # can't go lower; propagate the OOM
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)


# ─────────────────────────────────────────────────────────────────────────────
# GaussianCloud — the learnable model
# ─────────────────────────────────────────────────────────────────────────────
# Why plain tensors instead of nn.Module?  Adaptive density control changes N
# (the number of Gaussians) at every densification step.  nn.Module stores
# parameters as nn.Parameter objects with fixed shapes; replacing them requires
# re-registering every parameter and rebuilding the optimizer regardless.
# Plain requires_grad tensors make the shape mutation explicit and keep
# make_optimizer() as the single source of truth for parameter groups.
#%%
class GaussianCloud:
    """
    Mutable set of 3-D Gaussians stored as plain (requires_grad) tensors.

    Why not nn.Module?
    ------------------
    Adaptive density control changes N at every densification step.  Using
    plain tensors lets densify_and_prune() swap them for new tensors of a
    different shape without fighting Module's parameter registry.

    Parameters (all in AABB space = [-1,1]³)
    -----------------------------------------
    means      : (N, 3)  Gaussian centres.
    log_scales : (N, 3)  log of per-axis standard deviations.
                         Log parameterisation keeps s > 0 without constraints.
    quats      : (N, 4)  rotation quaternions [w, x, y, z].
    inten      : (N,)    raw intensity params; v_k = softplus(inten_k) > 0.
                         Softplus avoids the dead-gradient problem of plain clamp.

    Numerical constants (scale_min, mahal_clamp) are stored as attributes
    so no magic numbers appear elsewhere in the class body.
    """

    def __init__(self, n_init: int, aabb: AABB, device, cfg: argparse.Namespace,
                 init_pts: torch.Tensor | None = None,
                 init_quats: torch.Tensor | None = None):
        self.aabb        = aabb
        self.device      = device
        self.scale_min   = cfg.scale_min_clamp
        self.mahal_clamp = cfg.mahal_max_clamp

        lo = aabb.lo.to(device)
        ex = aabb.extent.to(device)

        # ── Initialise means (and optional oriented quats) ────────────────────
        # SWC skeleton coordinates give a warm start near the neuron structure.
        # init_quats (if provided) carries branch-direction quaternions that are
        # selected with the same permutation as the means.
        if init_pts is not None and init_pts.numel() > 0:
            init_pts = init_pts.to(device)
            M = init_pts.shape[0]
            if M >= n_init:
                sel = torch.randperm(M, device=device)[:n_init]
            else:
                extra = torch.randint(0, M, (n_init - M,), device=device)
                sel   = torch.cat([torch.arange(M, device=device), extra])
            means = init_pts[sel]

            if init_quats is not None and init_quats.numel() > 0:
                iq = init_quats.to(device)
                quats = iq[sel % iq.shape[0]]
            else:
                quats = torch.zeros(n_init, 4, device=device)
                quats[:, 0] = 1.0
        else:
            means = lo + torch.rand(n_init, 3, device=device) * ex
            quats = torch.zeros(n_init, 4, device=device)
            quats[:, 0] = 1.0

        # ── Initialise shape: PSF-aware anisotropic scales ────────────────────
        # In the unit AABB all axes span [-1,1], but the voxel spacing differs:
        #   z step = 2/(D-1),   x/y step = 2/(W-1)
        # A physically-isotropic Gaussian must have s_z/s_xy = (W-1)/(D-1).
        # init_scale_z_factor encodes this ratio (default 1.0 = isotropic).
        # Setting it to ~(W-1)/(D-1) gives Gaussians that are spherical in
        # physical voxel space at initialisation, matching the confocal PSF.
        log_s = torch.full((n_init, 3), math.log(cfg.init_scale), device=device)
        z_factor = max(float(getattr(cfg, 'init_scale_z_factor', 1.0)), 1e-3)
        if z_factor != 1.0:
            log_s[:, 2] = log_s[:, 2] + math.log(z_factor)

        # ── Initialise intensity via softplus inverse ─────────────────────────
        # We optimise raw_inten where v = softplus(raw_inten).
        # Invert: raw = log(exp(v) - 1) so that softplus(raw) ≈ cfg.init_inten.
        init_inten = max(float(cfg.init_inten), 1.0e-6)
        inten = torch.full((n_init,), math.log(math.expm1(init_inten)), device=device)

        self.means  = means.requires_grad_(True)
        self.log_s  = log_s.requires_grad_(True)
        self.quats  = quats.requires_grad_(True)
        self.inten  = inten.requires_grad_(True)

        # Running mean |∂L/∂μ| — drives clone/split decisions in densification
        self._grad_acc   = torch.zeros(n_init, device=device)
        self._grad_count = torch.zeros(n_init, device=device)

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def N(self) -> int:
        """Current number of Gaussians (changes after every densification)."""
        return self.means.shape[0]

    def intensity(self) -> torch.Tensor:
        """Per-Gaussian intensity v_k = softplus(inten_k).

        Softplus maps any real number to (0, ∞) with non-zero gradient
        everywhere — avoids the dead-unit problem of ReLU or hard clamping.
        """
        return F.softplus(self.inten)

    def all_params(self):
        """Return all optimisable tensors (for optimizer construction)."""
        return [self.means, self.log_s, self.quats, self.inten]

    # ── forward evaluation ─────────────────────────────────────────────────────
    def forward(self, pts: torch.Tensor, chunk_n: int) -> torch.Tensor:
        """Evaluate the Gaussian mixture at query points.

        Two execution paths (selected by the module-level USE_CUDA_KERNEL flag):

        CUDA path  — delegates to _GaussianFieldFn, which calls the fused
                     kernel.  Faster; uses OOM-adaptive chunking in backward.

        PyTorch path — chunked matrix operations with full autograd support.
                       Portable; used when the CUDA kernel is unavailable.

        Both paths produce identical results up to floating-point order.

        Parameters
        ----------
        pts    : (M, 3) query coordinates in [-1,1]³ AABB space.
        chunk_n: Gaussians evaluated per chunk; trades speed vs peak VRAM.

        Returns
        -------
        (M,) predicted intensity at each query point.
        """
        if USE_CUDA_KERNEL:
            return _GaussianFieldFn.apply(
                pts.contiguous(),
                self.means, self.log_s, self.quats, self.inten,
                self.scale_min, self.mahal_clamp, chunk_n,
            )

        inten = self.intensity()
        out   = torch.zeros(pts.shape[0], device=self.device)

        for s in range(0, self.N, chunk_n):
            e   = min(s + chunk_n, self.N)
            si  = build_sigma_inv(self.log_s[s:e], self.quats[s:e], self.scale_min)
            v   = inten[s:e]

            # diff: (M, chunk, 3) — displacement from each query to each centre
            diff = pts.unsqueeze(1) - self.means[s:e].unsqueeze(0)
            # xS:  (M, chunk, 3) — diff pre-multiplied by Σ⁻¹
            xS   = (diff.unsqueeze(-2) @ si.unsqueeze(0)).squeeze(-2)
            # mah: (M, chunk) — Mahalanobis²; skip distant Gaussians (matches CUDA kernel)
            mah  = (xS * diff).sum(-1)
            w    = torch.where(mah < self.mahal_clamp,
                               torch.exp(-0.5 * mah.clamp(max=self.mahal_clamp)),
                               torch.zeros_like(mah))
            out  = out + (v * w).sum(-1)        # accumulate over chunk

        return out

    # ── gradient accumulation for density control ──────────────────────────────
    def accum_grads(self):
        """Accumulate |∂L/∂μ| for the adaptive density controller.

        The running mean gradient magnitude over a window of steps is used to
        decide which Gaussians to clone (under-reconstructed) or split (too
        large).  Must be called after loss.backward() and before the optimizer
        clears .grad via zero_grad().
        """
        if self.means.grad is not None:
            g = self.means.grad.detach().norm(dim=-1)
            self._grad_acc   = self._grad_acc   + g
            self._grad_count = self._grad_count + 1.0

    def reset_grad_acc(self):
        """Zero the gradient accumulator after each densification step."""
        self._grad_acc   = torch.zeros(self.N, device=self.device)
        self._grad_count = torch.zeros(self.N, device=self.device)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _slice(self, idx):
        """Clone detached parameter slices by index mask or index tensor.

        Returns bare tensors (no grad_fn) so that densification can safely
        concatenate and reassign them without creating a second autograd graph.
        """
        return (
            self.means[idx].detach().clone(),
            self.log_s[idx].detach().clone(),
            self.quats[idx].detach().clone(),
            self.inten[idx].detach().clone(),
        )

    def _assign(self, means, log_s, quats, inten):
        """Replace all parameter tensors and re-enable gradients.

        Called at the end of densify_and_prune() with the newly assembled
        tensors of the updated shape.  The optimizer must be rebuilt after
        this call because the tensor identities change.
        """
        self.means  = means.requires_grad_(True)
        self.log_s  = log_s.requires_grad_(True)
        self.quats  = quats.requires_grad_(True)
        self.inten  = inten.requires_grad_(True)

    # ── densification ──────────────────────────────────────────────────────────
    def densify_and_prune(self, cfg: argparse.Namespace) -> Tuple[int, int]:
        """Adaptive density control (clone / split / prune).

        Motivation
        ----------
        Gaussians with large positional gradients |∂L/∂μ| are under-fitting
        their region.  Instead of just moving them (slow), we spawn new
        Gaussians to cover under-represented space.

        Rules (following 3DGS paper heuristics)
        ----------------------------------------
        Clone — avg_grad > thresh AND max_scale < densify_max_scale
                Duplicate + perturb by ≈ σ.
                Use: small Gaussian in high-gradient region needs a neighbour.

        Split — avg_grad > thresh AND max_scale ≥ densify_max_scale
                Replace with 2 smaller daughters offset ±noise, shrunken by
                split_scale_divisor.
                Use: large Gaussian spanning high-frequency detail needs finer
                     resolution.

        Prune — centre lies outside the AABB → removed immediately.
                These have drifted off-volume and contribute nothing.

        Everything runs under torch.no_grad() because this is parameter
        mutation, not loss computation.  The optimizer must be rebuilt after
        this call because tensor shapes change.

        Returns
        -------
        (n_pruned, n_added) counts for logging.
        """
        device      = self.device
        grad_thresh = cfg.densify_grad_thresh
        max_scale   = cfg.densify_max_scale
        max_n       = cfg.max_gaussians
        divisor     = cfg.split_scale_divisor
        log_floor   = cfg.log_scale_floor

        with torch.no_grad():
            avg_g  = self._grad_acc / self._grad_count.clamp(min=1.0)
            curr_s = torch.exp(self.log_s).max(-1).values
            in_box = self.aabb.to(device).contains(self.means)

            high_g = avg_g  > grad_thresh
            small  = curr_s < max_scale
            # A Gaussian is dead if it left the volume OR its intensity is
            # negligible (pruned by lambda_count driving softplus(inten) → 0).
            dim    = F.softplus(self.inten) < cfg.prune_inten_thresh
            dead   = ~in_box | dim

            keep     = ~dead
            n_pruned = self.N - keep.sum().item()

            clone_idx = (high_g & small  & keep).nonzero(as_tuple=True)[0]
            split_idx = (high_g & ~small & keep).nonzero(as_tuple=True)[0]

            m, ls, q, iv = self._slice(keep)
            parts_m  = [m];  parts_ls = [ls]; parts_q = [q]; parts_iv = [iv]
            n_added  = 0

            # ── Clone ────────────────────────────────────────────────────────
            budget = max_n - m.shape[0]
            if len(clone_idx) > 0 and budget > 0:
                k         = min(len(clone_idx), budget)
                clone_idx = clone_idx[:k]
                cm, cls_, cq, civ = self._slice(clone_idx)
                # Offset the clone by ≈ 1σ so it doesn't perfectly overlap
                perturb   = torch.randn_like(cm) * torch.exp(cls_).mean(-1, keepdim=True)
                parts_m.append(cm + perturb)
                parts_ls.append(cls_)
                parts_q.append(cq)
                parts_iv.append(civ)
                n_added += k
                budget  -= k

            # ── Split ────────────────────────────────────────────────────────
            if len(split_idx) > 0 and budget >= 2:
                k         = min(len(split_idx), budget // 2)
                split_idx = split_idx[:k]
                sm, sls, sq, siv = self._slice(split_idx)
                s_s    = torch.exp(sls)
                noise  = torch.randn_like(sm) * s_s
                # Shrink daughters so they fit inside the parent's footprint
                new_ls = torch.log(s_s / divisor).clamp(min=log_floor)
                for sign in (+1, -1):
                    parts_m.append(sm + sign * noise)
                    parts_ls.append(new_ls)
                    parts_q.append(sq.clone())
                    parts_iv.append(siv.clone())
                n_added += 2 * k

                # Remove split parents from the kept base (replaced by daughters)
                split_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
                split_mask[split_idx] = True
                m2, ls2, q2, iv2 = self._slice(keep & ~split_mask)
                parts_m[0] = m2;  parts_ls[0] = ls2
                parts_q[0] = q2;  parts_iv[0] = iv2

            # ── Assemble and assign ───────────────────────────────────────────
            self._assign(
                torch.cat(parts_m),  torch.cat(parts_ls),
                torch.cat(parts_q),  torch.cat(parts_iv),
            )
            self.reset_grad_acc()

        return int(n_pruned), int(n_added)

    def clamp_means(self):
        """Project Gaussian centres back into the AABB after each gradient step.

        Out-of-box Gaussians contribute nothing to the in-volume loss but still
        consume optimizer state and pollute the gradient accumulator used for
        density control.
        """
        with torch.no_grad():
            self.means.data.copy_(self.aabb.clamp(self.means.data))

    # ── stats ──────────────────────────────────────────────────────────────────
    def scale_stats(self) -> Tuple[float, float]:
        """Return (mean, max) of the per-Gaussian maximum axis scale.

        Used for logging and early detection of scale blow-up during training.
        """
        s = torch.exp(self.log_s.detach()).max(-1).values
        if s.numel() == 0:
            return float('nan'), float('nan')
        return s.mean().item(), s.max().item()

    def clamp_scales(self, scale_max_hard: float | None = None,
                     scale_min_hard: float | None = None):
        """Hard-clamp log_scales to keep Gaussians within a valid range.

        Why clamp in log space?  The optimizer works on log_s, so clamping
        log_s directly avoids a round-trip through exp/log and the associated
        floating-point error.  The ceiling is applied every gradient step so
        the soft scale_ceiling_reg loss term has a gradient even when the hard
        clamp is active.

        scale_min_hard: meaningful lower bound (e.g. 0.003) that prevents the
        aniso penalty from collapsing s_min to a needle.  Falls back to the
        numerical-stability floor (scale_min_clamp) when not set.
        """
        with torch.no_grad():
            if scale_max_hard is not None and scale_max_hard > 0:
                self.log_s.data.clamp_(max=math.log(scale_max_hard))
            # Prefer the meaningful floor; always at least the stability floor
            min_floor = max(scale_min_hard or 0.0, self.scale_min)
            self.log_s.data.clamp_(min=math.log(min_floor))

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, path):
        """Save all parameter tensors to a `.pth` checkpoint.

        The 'inten_param' field records the activation convention so that
        load() can correctly invert it even if the convention changes in
        future versions.
        """
        torch.save(
            {
                "means": self.means.detach().cpu(),
                "log_scales": self.log_s.detach().cpu(),
                "quats": self.quats.detach().cpu(),
                "intensities": self.inten.detach().cpu(),
                "inten_param": "softplus",
            },
            str(path),
        )

    @classmethod
    def load(cls, path, aabb: AABB, device, cfg: argparse.Namespace) -> "GaussianCloud":
        """Restore a GaussianCloud from a saved checkpoint.

        Handles two legacy intensity conventions so that old checkpoints can
        be resumed:
        1. 'softplus' (current): stored = raw_inten; v = softplus(raw).
        2. Sigmoid (old): stored v ∈ [0,1]; inverted via log(expm1(v)).
        Inversion lets the optimizer resume from the same loss landscape.
        """
        path = Path(path)
        if path.suffix == ".npz":
            d = np.load(str(path))

            def _fetch(key):
                return d[key]

            inten_param = d['inten_param'].item() if 'inten_param' in d.files else None
            inten_key = 'intensities' if 'intensities' in d else 'opacities'
        else:
            d = torch.load(str(path), map_location="cpu")

            def _fetch(key):
                value = d[key]
                return value.cpu().numpy() if isinstance(value, torch.Tensor) else value

            inten_param = d.get('inten_param')
            inten_key = 'intensities' if 'intensities' in d else 'opacities'

        obj = cls.__new__(cls)
        obj.aabb        = aabb
        obj.device      = device
        obj.scale_min   = cfg.scale_min_clamp
        obj.mahal_clamp = cfg.mahal_max_clamp

        def _p(key):
            return torch.tensor(_fetch(key), device=device).requires_grad_(True)

        obj.means  = _p('means')
        obj.log_s  = _p('log_scales')
        obj.quats  = _p('quats')
        inten_raw  = torch.tensor(_fetch(inten_key), device=device)
        if str(inten_param) == 'softplus':
            obj.inten = inten_raw.requires_grad_(True)
        elif inten_raw.numel() > 0 and float(inten_raw.min()) >= 0.0 and float(inten_raw.max()) <= 1.0:
            obj.inten = torch.log(torch.expm1(inten_raw.clamp_min(1.0e-6))).requires_grad_(True)
        else:
            obj.inten = inten_raw.requires_grad_(True)
        obj.reset_grad_acc()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
# Why separate learning rates per parameter group?
#   means  — need a fine spatial LR with exponential decay (coarse-to-fine).
#   log_s, quats, inten — converge faster; cosine annealing gives a smooth
#   warm-to-cold schedule without the hard cliff of step decay.
# Why rebuild the optimizer after every densification?
#   The tensor shapes change (N grows or shrinks), making the old Adam momentum
#   buffers stale.  Rebuilding is the only correct approach; the overhead is
#   negligible because densification happens every few hundred steps.
#%%
def make_optimizer(gc: GaussianCloud, cfg: argparse.Namespace) -> torch.optim.Adam:
    """Build a per-parameter-group Adam optimizer for all trainable tensors.

    Each parameter group gets its own initial LR (set here) and name (used by
    update_lr() to select the right schedule formula).  adam_eps is set very
    small (1e-15) because the loss landscape is well-conditioned and a larger
    eps would over-smooth the adaptive step sizes.
    """
    return torch.optim.Adam([
        {'params': [gc.means],  'lr': cfg.lr_means,  'name': 'means'},
        {'params': [gc.log_s],  'lr': cfg.lr_scales, 'name': 'scales'},
        {'params': [gc.quats],  'lr': cfg.lr_quats,  'name': 'quats'},
        {'params': [gc.inten],  'lr': cfg.lr_inten,  'name': 'inten'},
    ], eps=cfg.adam_eps)


def _lr_warmup_scale(step: int, total_steps: int, cfg: argparse.Namespace) -> float:
    """Compute a linear warmup scale factor ∈ [lr_warmup_init_factor, 1.0].

    Why warmup?  At initialisation the Gaussian means are far from their
    optimal positions.  Large LRs early on cause chaotic jumps; a short
    linear ramp from a fraction of the target LR stabilises the first few
    hundred steps without meaningfully slowing overall convergence.

    Returns 1.0 immediately if warmup_steps ≤ 1 (warmup disabled).
    """
    warmup_steps = min(max(int(cfg.lr_warmup_steps), 0), max(total_steps, 1))
    if warmup_steps <= 1:
        return 1.0

    start       = min(max(float(cfg.lr_warmup_init_factor), 0.0), 1.0)
    warmup_step = min(max(step, 0), warmup_steps - 1)
    t           = warmup_step / (warmup_steps - 1)
    return start + (1.0 - start) * t


def update_lr(optimizer: torch.optim.Adam, step: int,
              total_steps: int, cfg: argparse.Namespace):
    """Update per-group learning rates according to the training schedule.

    means  — exponential decay:
        lr(t) = lr_means_final · (lr_means / lr_means_final)^(1−t)
        Linear in log-LR space; gives large steps early (global placement)
        and tiny steps late (sub-voxel precision).

    others — cosine annealing:
        lr(t) = lr_min + 0.5·(lr_max − lr_min)·(1 + cos(π·t))
        where lr_min = lr_max · lr_final_fraction.
        Smooth warm-to-cold decay.

    all    — multiplied by linear warmup scale (ramps from lr_warmup_init_factor
             to 1.0 over the first lr_warmup_steps steps).

    Parameters
    ----------
    step        : global gradient step index (1-based).
    total_steps : epochs × steps_per_epoch.
    """
    step   = min(max(step, 0), max(total_steps - 1, 0))
    t      = step / max(total_steps - 1, 1)
    frac   = cfg.lr_final_fraction
    warmup = _lr_warmup_scale(step, total_steps, cfg)

    for g in optimizer.param_groups:
        name = g['name']
        if name == 'means':
            lr = cfg.lr_means_final * (cfg.lr_means / cfg.lr_means_final) ** (1 - t)
        else:
            lr_max = getattr(cfg, f'lr_{name}')
            lr_min = lr_max * frac
            lr     = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))
        g['lr'] = lr * warmup


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
# Objective: reconstruct the neuron volume with the fewest ellipsoids at the
# highest fidelity.  The full loss is:
#
#   L = L1(pred, gt) + λ_ssim · SSIM_slice(pred, gt)    [reconstruction — GaussianPile]
#     + λ_scale    · (1/N) Σ_k  s_max,k²               [bloat penalty]
#     + λ_ceiling  · (1/N) Σ_k  relu(s_max,k − cap)    [hard ceiling]
#     + λ_outlier  · (1/N) Σ_k  relu(s_max,k − τ)      [outlier suppression]
#     + λ_sparsity · (1/N) Σ_k  v_k · (1 − GT(μ_k))   [background → zero]
#     + λ_aniso    · (1/N) Σ_k  s_min,k²               [thin axes → elongated]
#     + λ_count    · (1/N) Σ_k  σ(raw_inten_k)         [soft-L0 count penalty]
#     + λ_L1       · (1/N) Σ_k  v_k                    [L1 intensity → kill zombies]
#     + λ_coverage · (1/N) Σ_k  −log(s_max,k / s_ref)  [coverage reward → bigger ellipsoids]
#
# where s_max,k / s_min,k are the longest / shortest axes of ellipsoid k,
# v_k = softplus(raw_inten_k), σ = sigmoid, τ = median + 3·MAD,
# s_ref = init_scale (the initialisation scale, typically 0.05).
#
# λ_L1 and λ_coverage are the two terms that address run6's failure modes:
#
#   λ_L1 adds a steady pull on every Gaussian toward v_k = 0.  The sigmoid
#        count gradient σ(1−σ) → 0 for already-dead Gaussians, leaving a
#        zombie population with v ≈ 0.005–0.01.  L1's gradient is constant
#        (= sigmoid(raw)), so it finishes the job: those zombies collapse.
#
#   λ_coverage rewards each surviving Gaussian for having a large s_max.
#        −log(s_max/s_ref) < 0 when s_max > s_ref, directly reducing the
#        loss.  Together with λ_aniso (collapse s_min), this drives a
#        sphere → cigar transition: one long ellipsoid covers an entire
#        neurite segment instead of several small spheres.
#
# Why a dispatch table (_LOSS_TERM_SPECS)?
#   It separates term definition from the summation loop, making it trivial
#   to add or remove a term without touching the loop logic.
#%%
class _LossContext(NamedTuple):
    """Frozen bundle of all quantities needed by loss term functions.

    Computed once per step by _make_loss_context() and shared across all
    term functions to avoid redundant computation.
    """
    pred:    torch.Tensor       # (M,) model prediction
    gt:      torch.Tensor       # (M,) ground-truth intensities
    gc:      "GaussianCloud"
    cfg:     argparse.Namespace
    dataset: "VolumeDataset"
    s_max:   torch.Tensor       # (N,) per-Gaussian max axis scale
    s_min:   torch.Tensor       # (N,) per-Gaussian min axis scale
    step:    int = 0            # current global optimizer step (for conditional losses)


def _ssim_2d(pred: torch.Tensor, gt: torch.Tensor, window: int = 11) -> torch.Tensor:
    """SSIM loss on (1, 1, H, W) float32 tensors. Returns 1 − SSIM ∈ [0, 2].

    Uses a separable Gaussian window (σ=1.5) to estimate local statistics.
    The 1− convention makes it a loss (0 = perfect match).
    """
    sigma  = 1.5
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    gk     = torch.exp(-coords.pow(2) / (2.0 * sigma ** 2))
    gk     = gk / gk.sum()
    filt   = (gk.unsqueeze(0) * gk.unsqueeze(1)).reshape(1, 1, window, window)

    pad    = window // 2
    mu1    = F.conv2d(pred,      filt, padding=pad)
    mu2    = F.conv2d(gt,        filt, padding=pad)
    sig1   = F.conv2d(pred*pred, filt, padding=pad) - mu1 * mu1
    sig2   = F.conv2d(gt*gt,     filt, padding=pad) - mu2 * mu2
    sig12  = F.conv2d(pred*gt,   filt, padding=pad) - mu1 * mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2.0*mu1*mu2 + c1) * (2.0*sig12 + c2)) / \
               ((mu1*mu1 + mu2*mu2 + c1) * (sig1 + sig2 + c2))
    return 1.0 - ssim_map.mean()


def _ssim_slice_loss(gc: "GaussianCloud", dataset: "VolumeDataset",
                     cfg: argparse.Namespace) -> torch.Tensor:
    """Render a random Z-slice crop and return the SSIM loss against GT.

    Crop size is cfg.ssim_crop (default 64).  At 64×64 = 4 096 query points
    this adds roughly 2× the per-step forward cost — acceptable overhead.
    The rendered crop is differentiable w.r.t. all Gaussian parameters.
    """
    device = gc.device
    D, H, W = dataset.D, dataset.H, dataset.W
    crop    = int(getattr(cfg, 'ssim_crop', 64))
    crop    = min(crop, H, W)

    z  = int(torch.randint(0, D,         (1,)).item())
    r0 = int(torch.randint(0, H - crop + 1, (1,)).item())
    c0 = int(torch.randint(0, W - crop + 1, (1,)).item())

    lo, hi = gc.aabb.lo, gc.aabb.hi
    ih_idx = torch.arange(r0, r0 + crop)
    iw_idx = torch.arange(c0, c0 + crop)
    ih_g, iw_g = torch.meshgrid(ih_idx, iw_idx, indexing='ij')   # (crop, crop)

    zn = float(lo[2] + (z / max(D - 1, 1)) * (hi[2] - lo[2]))
    yn = lo[1] + (ih_g.float() / max(H - 1, 1)) * (hi[1] - lo[1])
    xn = lo[0] + (iw_g.float() / max(W - 1, 1)) * (hi[0] - lo[0])
    pts = torch.stack([xn.reshape(-1), yn.reshape(-1),
                       torch.full((crop * crop,), zn)], dim=-1).to(device)

    pred_flat = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
    gt_flat   = dataset.vol[z, r0:r0+crop, c0:c0+crop].reshape(-1).to(device)

    pred_2d = pred_flat.reshape(1, 1, crop, crop)
    gt_2d   = gt_flat.reshape(1, 1, crop, crop)
    return _ssim_2d(pred_2d, gt_2d)


def _loss_term_mse(ctx: _LossContext) -> torch.Tensor:
    """Reconstruction loss: foreground-weighted L1 + λ_ssim · SSIM slice loss.

    SSIM is gated by cfg.ssim_start_step (default 0 = always active).  Set
    ssim_start_step = densify_until_step to prevent SSIM positional gradients
    from inflating _grad_acc during the densification phase.

    L1 vs MSE (GaussianPile):
      L1 penalises residuals linearly — fairer to thin bright neurites where
      a single voxel of error matters as much as a large background region.
      MSE's quadratic scaling over-weights a few large errors and under-weights
      the many small structural errors along neurite edges.

    SSIM slice term:
      Rendered on a random 64×64 Z-crop each step.  Captures local contrast
      and structural continuity that pixel-wise L1 misses — specifically
      discontinuities where a neurite branch disappears.
      Weight: cfg.lambda_ssim (default 0.2, matching GaussianPile).
    """
    if getattr(ctx.cfg, 'log_intensity_loss', False):
        # log1p-domain L1: compresses bright soma, amplifies dim dendrite signal.
        # ∂(log1p(x))/∂x = 1/(1+x) → gradient at soma (x≈1) is half that at
        # dendrites (x≈0.2), naturally rebalancing attention.
        pred_l = torch.log1p(ctx.pred.clamp(0, 1))
        gt_l   = torch.log1p(ctx.gt.clamp(0, 1))
        l1 = F.l1_loss(pred_l, gt_l)
    else:
        l1 = F.l1_loss(ctx.pred, ctx.gt)
    ssim_start = int(getattr(ctx.cfg, 'ssim_start_step', 0))
    if ctx.step >= ssim_start:
        lssim = _ssim_slice_loss(ctx.gc, ctx.dataset, ctx.cfg)
        l1 = l1 + getattr(ctx.cfg, 'lambda_ssim', 0.2) * lssim
    return l1


def _loss_term_scale_reg(ctx: _LossContext) -> torch.Tensor:
    """L2 penalty on max per-axis scale.

    Discourages a few large blobs from dominating the fit.
    Weight: cfg.lambda_scale.
    """
    return ctx.s_max.pow(2).mean()


def _loss_term_scale_ceiling(ctx: _LossContext) -> torch.Tensor:
    """Soft ReLU penalty for scales exceeding scale_max_hard.

    Provides a differentiable gradient toward the hard cap applied by
    clamp_scales() each step — without it, Gaussians pushed against the
    ceiling would have zero gradient and stall.
    Weight: cfg.lambda_scale_ceiling.
    """
    if ctx.cfg.scale_max_hard is not None and ctx.cfg.scale_max_hard > 0:
        return F.relu(ctx.s_max - ctx.cfg.scale_max_hard).mean()
    return ctx.s_max.new_zeros(())


def _loss_term_scale_outlier(ctx: _LossContext) -> torch.Tensor:
    """Penalise scales beyond median + 3·MAD.

    More robust than a fixed threshold — the threshold adapts to the
    current scale distribution without manual tuning.  The threshold is
    computed under no_grad to avoid second-order gradients through the
    median/MAD operations.
    Weight: cfg.lambda_scale_outlier.
    """
    with torch.no_grad():
        med            = ctx.s_max.median()
        mad            = (ctx.s_max - med).abs().median()
        outlier_thresh = med + 3.0 * mad
    return F.relu(ctx.s_max - outlier_thresh).mean()


def _loss_term_sparsity(ctx: _LossContext) -> torch.Tensor:
    """Sparsity penalty: intensity × (1 − GT_at_mean).

    Encourages Gaussians in dark background regions to reduce their
    intensity rather than needing high scale to 'reach' a bright voxel
    far away.  Delegates to loss_sparsity_intensity().
    Weight: cfg.lambda_sparsity.
    """
    return loss_sparsity_intensity(ctx.gc, ctx.dataset, ctx.cfg)


def _loss_term_anisotropy(ctx: _LossContext) -> torch.Tensor:
    """Anisotropy penalty: L2 on the shortest axis of each ellipsoid.

    Penalising s_min² while leaving s_max free collapses the short axes
    toward zero, turning spheres into cigars aligned with the neurite.
    One elongated ellipsoid can then cover an entire branch segment,
    reducing the total count needed for a given fidelity.
    Weight: cfg.lambda_aniso.
    """
    return ctx.s_min.pow(2).mean()


def _loss_term_count(ctx: _LossContext) -> torch.Tensor:
    """Soft-L0 count penalty: mean sigmoid of raw intensity parameters.

    sigmoid(raw_inten) ≈ 1 for active Gaussians, ≈ 0 for dead ones.
    The mean is therefore a differentiable proxy for the active fraction.
    Gradient ∂σ/∂raw = σ(1−σ) peaks in the transition zone, gently
    pulling borderline Gaussians toward extinction rather than hard-pruning.
    Weight: cfg.lambda_count.
    """
    return torch.sigmoid(ctx.gc.inten).mean()


def _loss_term_L1_intensity(ctx: _LossContext) -> torch.Tensor:
    """L1 penalty on per-Gaussian intensity: mean(softplus(raw_inten)).

    Unlike the sigmoid count term whose gradient σ(1−σ) → 0 for very
    negative raw_inten (already-dead Gaussians), the L1 gradient is a
    constant sigmoid(raw_inten) that never fully vanishes.  This gives a
    steady pull toward v_k = 0 for every Gaussian, finishing off the
    zombie population that the count term leaves behind.
    Weight: cfg.lambda_L1.
    """
    return F.softplus(ctx.gc.inten).mean()


def _loss_term_coverage(ctx: _LossContext) -> torch.Tensor:
    """Coverage reward: penalise Gaussians smaller than s_ref.

    −log(s_max / s_ref) is negative when s_max > s_ref, reducing total
    loss and rewarding each surviving Gaussian for stretching to cover
    more of a neurite segment.  Combined with the anisotropy term this
    drives Gaussians toward long cigars rather than many small spheres.
    s_ref is fixed at cfg.init_scale (the initialisation scale).
    Weight: cfg.lambda_coverage.
    """
    # Clamp at scale_max_hard so reward saturates before the hard cap —
    # prevents the term from dominating MSE as Gaussians press against the ceiling.
    s_ref   = max(float(getattr(ctx.cfg, 'init_scale', 0.05)), 1e-4)
    cap     = float(ctx.cfg.scale_max_hard) if getattr(ctx.cfg, 'scale_max_hard', None) else 1.0
    s_ratio = (ctx.s_max / s_ref).clamp(min=1e-4, max=cap / s_ref)
    return -torch.log(s_ratio).mean()


def _loss_term_gradient(ctx: _LossContext) -> torch.Tensor:
    """Gradient sharpness loss on a random Z-slice crop.

    Computes central-difference gradients (dx, dy) of both the predicted and
    GT slices, then returns their L1 difference.  This directly penalises
    blurry soma/dendrite edges where the intensity transition should be sharp.
    Weight: cfg.lambda_grad.
    """
    import random as _random
    gc      = ctx.gc
    dataset = ctx.dataset
    cfg     = ctx.cfg
    D, H, W = dataset.D, dataset.H, dataset.W
    device  = gc.device
    crop    = min(int(getattr(cfg, 'ssim_crop', 64)), H, W)

    z  = _random.randint(0, D - 1)
    r0 = _random.randint(0, max(0, H - crop))
    c0 = _random.randint(0, max(0, W - crop))

    lo, hi = gc.aabb.lo, gc.aabb.hi
    ih_idx = torch.arange(r0, r0 + crop, device=device)
    iw_idx = torch.arange(c0, c0 + crop, device=device)
    ih_g, iw_g = torch.meshgrid(ih_idx, iw_idx, indexing='ij')
    zn  = float(lo[2] + (z / max(D - 1, 1)) * (hi[2] - lo[2]))
    yn  = lo[1] + (ih_g.float() / max(H - 1, 1)) * (hi[1] - lo[1])
    xn  = lo[0] + (iw_g.float() / max(W - 1, 1)) * (hi[0] - lo[0])
    pts = torch.stack([xn.reshape(-1), yn.reshape(-1),
                       torch.full((crop * crop,), zn, device=device)], dim=-1)

    pred = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0, 1).reshape(crop, crop)
    gt   = dataset.vol[z, r0:r0 + crop, c0:c0 + crop].to(device)

    # Central difference on interior pixels only (avoids boundary artefacts)
    pred_dx = pred[:, 2:] - pred[:, :-2]
    gt_dx   = gt[:, 2:]   - gt[:, :-2]
    pred_dy = pred[2:, :] - pred[:-2, :]
    gt_dy   = gt[2:, :]   - gt[:-2, :]

    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


# Dispatch table: (name, weight_cfg_attr, term_fn)
# weight_cfg_attr=None means the term is added unweighted (MSE).
_LOSS_TERM_SPECS = (
    ("mse",               None,                    _loss_term_mse),
    ("scale_reg",         "lambda_scale",           _loss_term_scale_reg),
    ("scale_ceiling_reg", "lambda_scale_ceiling",   _loss_term_scale_ceiling),
    ("scale_outlier_reg", "lambda_scale_outlier",   _loss_term_scale_outlier),
    ("sparsity",          "lambda_sparsity",        _loss_term_sparsity),
    ("anisotropy",        "lambda_aniso",           _loss_term_anisotropy),
    ("count",             "lambda_count",           _loss_term_count),
    ("L1_intensity",      "lambda_L1",              _loss_term_L1_intensity),
    ("coverage",          "lambda_coverage",        _loss_term_coverage),
    ("gradient",          "lambda_grad",            _loss_term_gradient),
)


def _make_loss_context(pred, gt, gc, cfg, dataset, step: int = 0) -> _LossContext:
    """Pre-compute shared quantities and pack them into a _LossContext.

    Both s_max and s_min are derived from the same exp(log_s) call so the
    tensor is only materialised once.
    """
    s = torch.exp(gc.log_s)
    return _LossContext(
        pred    = pred,
        gt      = gt,
        gc      = gc,
        cfg     = cfg,
        dataset = dataset,
        s_max   = s.max(-1).values,
        s_min   = s.min(-1).values,
        step    = step,
    )


def _compute_loss_terms(ctx: _LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum all loss terms according to _LOSS_TERM_SPECS.

    Iterates the dispatch table, calling each term function and multiplying
    by its weight (or adding directly for unweighted terms).  Returns the
    total scalar and a dict of individual terms for logging.
    """
    total = ctx.pred.new_zeros(())
    terms = {}

    for name, weight_attr, term_fn in _LOSS_TERM_SPECS:
        term        = term_fn(ctx)
        terms[name] = term
        if weight_attr is None:
            total = total + term
        else:
            total = total + getattr(ctx.cfg, weight_attr) * term

    return total, terms


def compute_loss(pred, gt, gc, cfg, dataset, step: int = 0):
    """Compute the total training loss and return individual term values.

    Public entry point used by the training loop.  Builds the context,
    runs the dispatch table, and appends 'loss' (the total) to the stats dict.

    Returns
    -------
    total : scalar tensor for loss.backward().
    stats : dict mapping term name → scalar tensor, for logging.
    """
    total, terms = _compute_loss_terms(_make_loss_context(pred, gt, gc, cfg, dataset, step=step))
    terms['loss'] = total
    return total, terms


def loss_sparsity_intensity(gc: "GaussianCloud", dataset: "VolumeDataset",
                            cfg: argparse.Namespace) -> torch.Tensor:
    """Penalise bright Gaussians sitting in dark (background) regions.

    For each Gaussian, look up the GT voxel intensity at its centre:
        sparsity = mean_k [ softplus(inten_k) · (1 − GT(μ_k)) ]

    A Gaussian with high intensity in a voxel where GT ≈ 0 incurs a large
    penalty and is encouraged to either move toward the foreground or reduce
    its intensity.

    gc.means is already in [-1,1]³, matching F.grid_sample's convention,
    so no coordinate transformation is needed before the lookup.
    """
    means_norm = gc.means   # (N, 3) already in [-1,1]³

    grid = means_norm.view(1, 1, 1, -1, 3)     # (1,1,1,N,3) for grid_sample 5-D API

    vol         = dataset.vol.unsqueeze(0).unsqueeze(0).to(gc.device)  # (1,1,D,H,W)
    gt_at_means = F.grid_sample(
        vol, grid, mode='bilinear', align_corners=True
    ).view(-1)                                  # (N,)

    inten    = gc.intensity()                   # softplus → always > 0
    sparsity = (inten * (1.0 - gt_at_means)).mean()
    return sparsity


# ─────────────────────────────────────────────────────────────────────────────
# Volume dataset — continuous sampling with trilinear interpolation
# ─────────────────────────────────────────────────────────────────────────────
# Why continuous sampling instead of integer-index sampling?
#   Gaussian centres are continuous coordinates in [-1,1]³.  Snapping GT
#   lookups to the nearest voxel index introduces a ½-voxel quantisation
#   error that limits PSNR at convergence.  F.grid_sample with mode='bilinear'
#   gives an exact trilinear interpolant at any sub-voxel coordinate,
#   matching what the Gaussian field itself produces.
#%%
class VolumeDataset:
    """Continuous-coordinate sampler for a (D, H, W) float32 voxel volume.

    Provides random query points in [-1,1]³ together with their trilinearly
    interpolated GT intensities via F.grid_sample.

    Parameters
    ----------
    volume   : (D, H, W) float32 tensor, normalised to [0, 1].
    aabb     : AABB defining the query domain (always AABB.unit() in practice).
    cfg      : config namespace (currently unused; kept for API uniformity).
    swc_path : optional path to SWC file; loads skeleton points used for
               Gaussian initialisation.
    """

    def __init__(self, volume: torch.Tensor, aabb: AABB, cfg: argparse.Namespace,
                 swc_path: str | None = None):
        assert volume.dim() == 3, "Volume must be (D, H, W)"
        self.vol           = volume
        self.D, self.H, self.W = volume.shape
        self.aabb          = aabb

        if swc_path is not None:
            swc_full = load_swc_full(swc_path)
            self.swc_unit = torch.from_numpy(
                swc_points_to_unit_aabb(swc_full[:, 1:4], volume.shape)
            ).float()
            self.swc_ids        = torch.from_numpy(swc_full[:, 0]).long()
            self.swc_parent_ids = torch.from_numpy(swc_full[:, 5]).long()
        else:
            self.swc_unit       = None
            self.swc_ids        = None
            self.swc_parent_ids = None

        # Importance sampling CDF: mix of intensity + gradient magnitude.
        # grad_sample_weight (default 0) = fraction of probability mass given to edges.
        # At 0.0 this is the original intensity-only CDF; at 0.5 it samples edges
        # as often as bright foreground, which improves soma boundary reconstruction.
        flat = volume.reshape(-1).float()
        grad_w = float(getattr(cfg, 'grad_sample_weight', 0.0))
        if grad_w > 0.0:
            # Central-difference gradient magnitude, zero-padded at boundaries.
            gx = torch.zeros_like(volume)
            gy = torch.zeros_like(volume)
            gz = torch.zeros_like(volume)
            gx[:, :, 1:-1] = (volume[:, :, 2:] - volume[:, :, :-2]) * 0.5
            gy[:, 1:-1, :] = (volume[:, 2:, :] - volume[:, :-2, :]) * 0.5
            gz[1:-1, :, :] = (volume[2:, :, :] - volume[:-2, :, :]) * 0.5
            grad_mag = (gx.pow(2) + gy.pow(2) + gz.pow(2)).sqrt().reshape(-1)
            probs = (1.0 - grad_w) * (flat + 0.1) + grad_w * (grad_mag + 0.01)
        else:
            probs = flat + 0.1
        self._sample_cdf = torch.cumsum(probs / probs.sum(), dim=0).cpu()

    def _indices_to_pts(self, id_, ih, iw, device) -> torch.Tensor:
        """Convert integer voxel indices (z, y, x) into AABB coordinates.

        Used by vol_psnr() to evaluate the field at every voxel centre
        without holding the full D×H×W coordinate grid in memory.
        """
        lo = self.aabb.lo
        hi = self.aabb.hi
        z  = lo[2] + (id_.float() / (self.D - 1)) * (hi[2] - lo[2])
        y  = lo[1] + (ih.float() / (self.H - 1)) * (hi[1] - lo[1])
        x  = lo[0] + (iw.float() / (self.W - 1)) * (hi[0] - lo[0])
        return torch.stack([x, y, z], dim=-1).to(device)

    def sample_uniform(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n query points uniformly from the AABB with trilinear GT.

        The (1,1,1,n,3) grid layout is required by the 5-D grid_sample API
        (batch, channel, depth, height → here collapsed to 1, width = n).
        """
        pts  = self.aabb.random_pts(n, device)                     # (n, 3)
        vol  = self.vol.unsqueeze(0).unsqueeze(0).to(device)       # (1,1,D,H,W)
        grid = pts.view(1, 1, 1, n, 3)                             # (1,1,1,n,3)
        gt   = F.grid_sample(vol, grid, mode='bilinear',
                             align_corners=True).view(n)
        return pts, gt

    def sample_importance(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n points biased toward bright voxels using the precomputed CDF.

        Uses inverse-CDF sampling on the flattened volume.  Each drawn index
        maps to the centre of a voxel, so GT is a direct lookup (no interpolation
        error).  Bright voxels are sampled ~10x more often than dark background.
        """
        u    = torch.rand(n)
        idxs = torch.searchsorted(self._sample_cdf, u).clamp(0, self._sample_cdf.numel() - 1)
        iz   = (idxs // (self.H * self.W)).long()
        ih   = ((idxs % (self.H * self.W)) // self.W).long()
        iw   = (idxs % self.W).long()
        pts  = self._indices_to_pts(iz, ih, iw, device)
        gt   = self.vol[iz, ih, iw].to(device)
        return pts, gt

    def sample(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main sampling interface called by the training loop each step."""
        return self.sample_importance(n, device)

    def swc_init_points(self) -> torch.Tensor:
        """Return SWC skeleton points in [-1,1]³, or an empty (0,3) tensor."""
        if self.swc_unit is None:
            return torch.empty(0, 3)
        return self.swc_unit

    def swc_oriented_init_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (positions, quats) with Gaussian x-axis aligned to the branch direction.

        For each SWC node that has a parent, the initial quaternion rotates [1,0,0]
        to the parent→child direction in unit-AABB space.  Root nodes (parent_id < 0)
        receive identity quaternions.

        Returns
        -------
        pts   : (N, 3)  positions in [-1, 1]³
        quats : (N, 4)  initial orientations [w, x, y, z]
        """
        if self.swc_unit is None:
            return torch.empty(0, 3), torch.empty(0, 4)

        pts  = self.swc_unit          # (N, 3)
        N    = pts.shape[0]
        ids  = self.swc_ids           # (N,) int64 node IDs
        pids = self.swc_parent_ids    # (N,) int64 parent IDs

        id_to_row = {int(ids[i].item()): i for i in range(N)}

        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0            # default: identity [1,0,0,0]

        for i in range(N):
            pid = int(pids[i].item())
            if pid < 0 or pid not in id_to_row:
                continue             # root node
            pr = id_to_row[pid]
            d  = pts[i] - pts[pr]   # parent→child in AABB coords
            ln = d.norm().item()
            if ln < 1e-8:
                continue             # coincident nodes
            d = d / ln
            dx, dy, dz = d[0].item(), d[1].item(), d[2].item()

            if dx > 0.9999:          # already along +x → identity
                pass
            elif dx < -0.9999:       # antiparallel → 180° around z-axis
                quats[i] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            else:
                # q = normalize([1+dx, 0, -dz, dy]): rotates [1,0,0] → d
                q = torch.tensor([1.0 + dx, 0.0, -dz, dy])
                quats[i] = F.normalize(q, dim=-1)

        return pts, quats

    def interior_init_points(self, n: int, thresh: float = 0.3) -> torch.Tensor:
        """Sample n random points from interior voxels (gt > thresh) in [-1,1]³.

        Used to seed Gaussians inside the soma volume, which the SWC centerline
        does not cover. thresh=0.3 targets the bright soma interior specifically.
        """
        flat   = self.vol.reshape(-1)
        mask   = (flat >= thresh).nonzero(as_tuple=False).squeeze(1)
        if mask.numel() == 0:
            return torch.empty(0, 3)
        idx    = mask[torch.randint(0, mask.numel(), (min(n, mask.numel()),))]
        D, H, W = self.D, self.H, self.W
        iz = (idx // (H * W)).long()
        ih = ((idx % (H * W)) // W).long()
        iw = (idx % W).long()
        lo, hi = self.aabb.lo, self.aabb.hi
        x = lo[0] + (iw.float() / max(W - 1, 1)) * (hi[0] - lo[0])
        y = lo[1] + (ih.float() / max(H - 1, 1)) * (hi[1] - lo[1])
        z = lo[2] + (iz.float() / max(D - 1, 1)) * (hi[2] - lo[2])
        return torch.stack([x, y, z], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────
# Why multiple PSNR estimators?
#   psnr_on_samples — fast per-epoch estimate on a random subset.
#   vol_psnr        — exact full-volume PSNR, slice-by-slice to avoid OOM;
#                     only run every detail_interval epochs.
#   evaluate_fields — dispatches to the right estimator based on a detail flag.
#   eval_slice      — renders a single Z-slice for visual inspection.
#%%
@torch.no_grad()
def psnr_on_samples(gc: GaussianCloud, dataset: VolumeDataset,
                    cfg: argparse.Namespace) -> float:
    """Estimate PSNR from a uniform random sample of eval_samples points.

    Cheap enough to run every epoch.  Predictions clamped to [0,1] before
    MSE to match the GT range.  Returns inf on perfect reconstruction.
    """
    pts, gt = dataset.sample_uniform(cfg.eval_samples, gc.device)
    pred    = gc.forward(pts, chunk_n=cfg.chunk_n)
    mse     = F.mse_loss(pred.clamp(0.0, 1.0), gt)
    return float('inf') if mse == 0 else -10.0 * math.log10(mse.item())


def vol_psnr(gc: GaussianCloud, dataset: VolumeDataset,
             cfg: argparse.Namespace) -> float:
    """Compute full-volume PSNR iterating over Z-slices to avoid OOM.

    Evaluates all D×H×W voxels when the volume is small enough (≤
    eval_full_max_voxels), otherwise falls back to random sampling.
    Slice-by-slice iteration keeps peak VRAM proportional to H×W×chunk_n
    rather than D×H×W.
    """
    D, H, W   = dataset.D, dataset.H, dataset.W
    total_vox = int(D) * int(H) * int(W)
    device    = gc.device

    if getattr(cfg, 'eval_full', False) or total_vox <= getattr(cfg, 'eval_full_max_voxels', 5_000_000):
        pts_list = []
        gt_list  = []
        for z in range(D):
            pts = dataset._indices_to_pts(
                torch.full((H*W,), z, dtype=torch.long),
                torch.arange(H, dtype=torch.long).repeat_interleave(W),
                torch.arange(W, dtype=torch.long).tile(H),
                device,
            )
            gt   = dataset.vol[z].reshape(-1).to(device)
            pred = gc.forward(pts, chunk_n=cfg.chunk_n)
            pts_list.append(pred)
            gt_list.append(gt)
        pred_all = torch.cat(pts_list)
        gt_all   = torch.cat(gt_list)
    else:
        pts, gt  = dataset.sample_uniform(cfg.eval_samples, device)
        pred_all = gc.forward(pts, chunk_n=cfg.chunk_n)
        gt_all   = gt

    mse = F.mse_loss(pred_all.clamp(0.0, 1.0), gt_all)
    return float('inf') if mse == 0 else -10.0 * math.log10(mse.item())


@torch.no_grad()
def evaluate_fields(gc: GaussianCloud, dataset: VolumeDataset,
                    cfg: argparse.Namespace, detail: bool) -> dict:
    """Dispatch to fast or detailed PSNR estimator depending on detail flag.

    detail=True triggers vol_psnr (slow, exact).
    detail=False skips it and returns NaN to keep per-epoch overhead low.
    """
    metrics = {'psnr': psnr_on_samples(gc, dataset, cfg)}
    if detail:
        metrics['vol_psnr'] = vol_psnr(gc, dataset, cfg)
    else:
        metrics['vol_psnr'] = float('nan')
    return metrics


@torch.no_grad()
def eval_slice(gc: GaussianCloud, dataset: VolumeDataset,
               z_idx: int, cfg: argparse.Namespace) -> torch.Tensor:
    """Render one Z-slice of the Gaussian field for visual inspection.

    Creates a dense (H, W) grid of query points at depth z_idx, evaluates
    the field, and returns the result clamped to [0,1] as a CPU tensor.
    Used for qualitative comparison with the GT slice during analysis.
    """
    H, W   = dataset.H, dataset.W
    lo, hi = gc.aabb.lo, gc.aabb.hi
    yg, xg = torch.meshgrid(
        torch.linspace(float(lo[1]), float(hi[1]), H),
        torch.linspace(float(lo[0]), float(hi[0]), W),
        indexing='ij',
    )
    z_val = float(lo[2] + (z_idx / (dataset.D - 1)) * (hi[2] - lo[2]))
    pts   = torch.stack([xg, yg, torch.full_like(xg, z_val)], dim=-1)
    pts   = pts.reshape(-1, 3).to(gc.device)
    pred  = gc.forward(pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
    return pred.reshape(H, W).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Splatting-based rendering with MIP projection
# ─────────────────────────────────────────────────────────────────────────────
# Why splatting?  Different from voxel-based volumetric reconstruction,
# splatting projects 3D Gaussians onto 2D image planes and accumulates their
# contributions per-pixel.  This enables view-dependent rendering and MIP
# visualization from arbitrary viewpoints.
#
# MIP (Maximum Intensity Projection) finds the maximum intensity along each
# depth ray, giving a natural way to visualise thin structures (neurites)
# without explicit rendering of intermediate slices.
#%%
@torch.no_grad()
def splat_to_image(gc: GaussianCloud, dataset: VolumeDataset,
                   cfg: argparse.Namespace, view_axis: str = 'z',
                   img_h: int = 256, img_w: int = 256,
                   depth_samples: int = 32) -> torch.Tensor:
    """Project 3D Gaussians onto a 2D image via efficient splatting.

    Samples depths uniformly and evaluates Gaussian field at each depth.
    Uses chunking to manage memory; compatible with large volumes.

    Parameters
    ----------
    gc          : GaussianCloud with learned Gaussian parameters.
    dataset     : VolumeDataset containing volume dimensions and AABB.
    cfg         : config namespace.
    view_axis   : 'x', 'y', or 'z' — which axis to view down (depth direction).
    img_h, img_w: output image resolution.
    depth_samples: number of depths to sample (default 32 for speed).

    Returns
    -------
    (depth_samples, img_h, img_w) tensor of Gaussian field evaluations.
    """
    device = gc.device
    lo, hi = gc.aabb.lo.to(device), gc.aabb.hi.to(device)
    
    # Build coordinate grids for the output image plane
    if view_axis == 'z':
        # Looking down Z; project onto XY plane
        yg, xg = torch.meshgrid(
            torch.linspace(float(lo[1]), float(hi[1]), img_h),
            torch.linspace(float(lo[0]), float(hi[0]), img_w),
            indexing='ij',
        )
        yg, xg = yg.to(device), xg.to(device)
        zg_vals = torch.linspace(float(lo[2]), float(hi[2]), depth_samples)
        image_stack = []
        
        for z_val in zg_vals:
            pts = torch.stack([xg, yg, torch.full_like(xg, z_val)], dim=-1)
            pts = pts.reshape(-1, 3)
            
            # Batch evaluate with chunking
            chunk_size = 32768
            slice_result = []
            for s in range(0, pts.shape[0], chunk_size):
                e = min(s + chunk_size, pts.shape[0])
                chunk_pts = pts[s:e]
                intensities = gc.forward(chunk_pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
                slice_result.append(intensities)
            
            full_slice = torch.cat(slice_result, dim=0).reshape(img_h, img_w)
            image_stack.append(full_slice)
        return torch.stack(image_stack, dim=0)
    
    elif view_axis == 'y':
        # Looking down Y; project onto XZ plane
        zg, xg = torch.meshgrid(
            torch.linspace(float(lo[2]), float(hi[2]), img_h),
            torch.linspace(float(lo[0]), float(hi[0]), img_w),
            indexing='ij',
        )
        zg, xg = zg.to(device), xg.to(device)
        yg_vals = torch.linspace(float(lo[1]), float(hi[1]), depth_samples)
        image_stack = []
        
        for y_val in yg_vals:
            pts = torch.stack([xg, torch.full_like(xg, y_val), zg], dim=-1)
            pts = pts.reshape(-1, 3)
            
            chunk_size = 32768
            slice_result = []
            for s in range(0, pts.shape[0], chunk_size):
                e = min(s + chunk_size, pts.shape[0])
                chunk_pts = pts[s:e]
                intensities = gc.forward(chunk_pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
                slice_result.append(intensities)
            
            full_slice = torch.cat(slice_result, dim=0).reshape(img_h, img_w)
            image_stack.append(full_slice)
        return torch.stack(image_stack, dim=0)
    
    else:  # view_axis == 'x'
        # Looking down X; project onto YZ plane
        zg, yg = torch.meshgrid(
            torch.linspace(float(lo[2]), float(hi[2]), img_h),
            torch.linspace(float(lo[1]), float(hi[1]), img_w),
            indexing='ij',
        )
        zg, yg = zg.to(device), yg.to(device)
        xg_vals = torch.linspace(float(lo[0]), float(hi[0]), depth_samples)
        image_stack = []
        
        for x_val in xg_vals:
            pts = torch.stack([torch.full_like(yg, x_val), yg, zg], dim=-1)
            pts = pts.reshape(-1, 3)
            
            chunk_size = 32768
            slice_result = []
            for s in range(0, pts.shape[0], chunk_size):
                e = min(s + chunk_size, pts.shape[0])
                chunk_pts = pts[s:e]
                intensities = gc.forward(chunk_pts, chunk_n=cfg.chunk_n).clamp(0.0, 1.0)
                slice_result.append(intensities)
            
            full_slice = torch.cat(slice_result, dim=0).reshape(img_h, img_w)
            image_stack.append(full_slice)
        return torch.stack(image_stack, dim=0)


@torch.no_grad()
def compute_mip_from_stack(image_stack: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute Maximum Intensity Projection from a depth stack.

    Takes the maximum along the depth dimension, returning a single 2D image
    where each pixel holds the brightest voxel along the corresponding ray.

    Parameters
    ----------
    image_stack : (D, H, W) or (H, W, D) tensor depending on axis.
    axis        : which axis to max over (0 for depth-first layout).

    Returns
    -------
    (H, W) 2D MIP image.
    """
    return torch.max(image_stack, dim=axis).values


@torch.no_grad()
def render_splatted_mips(gc: GaussianCloud, dataset: VolumeDataset,
                         cfg: argparse.Namespace,
                         depth_samples: int = 32) -> dict:
    """Render MIP projections from three orthogonal viewpoints using splatting.

    Output dimensions match the volume exactly (no downsampling):
      'xy' : (H, W)  — looking down Z
      'xz' : (D, W)  — looking down Y
      'yz' : (D, H)  — looking down X

    On CUDA: uses the fused splat_mip kernel (single launch per view).
    On CPU:  falls back to the chunked Python implementation.

    Parameters
    ----------
    depth_samples : depth positions sampled per ray (trades speed vs accuracy).
    """
    D, H, W = dataset.D, dataset.H, dataset.W
    mips    = {}

    if gc.device.type == 'cuda':
        kernel = _load_eval_kernel()
        lo = gc.aabb.lo.cpu()
        hi = gc.aabb.hi.cpu()
        lo_x, hi_x = float(lo[0]), float(hi[0])
        lo_y, hi_y = float(lo[1]), float(hi[1])
        lo_z, hi_z = float(lo[2]), float(hi[2])
        means_c    = gc.means.contiguous()
        log_s_c    = gc.log_s.contiguous()
        quats_c    = gc.quats.contiguous()
        inten_c    = gc.inten.contiguous()
        scale_min  = float(gc.scale_min)
        mahal_clamp = float(gc.mahal_clamp)

        print(f'  Rendering XY MIP ({depth_samples} depth samples at {H}×{W}) [CUDA]...')
        flat = kernel.splat_mip(means_c, log_s_c, quats_c, inten_c,
                                lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
                                H, W, depth_samples, 0, scale_min, mahal_clamp)
        mips['xy'] = flat.reshape(H, W)

        print(f'  Rendering XZ MIP ({depth_samples} depth samples at {D}×{W}) [CUDA]...')
        flat = kernel.splat_mip(means_c, log_s_c, quats_c, inten_c,
                                lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
                                D, W, depth_samples, 1, scale_min, mahal_clamp)
        mips['xz'] = flat.reshape(D, W)

        print(f'  Rendering YZ MIP ({depth_samples} depth samples at {D}×{H}) [CUDA]...')
        flat = kernel.splat_mip(means_c, log_s_c, quats_c, inten_c,
                                lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
                                D, H, depth_samples, 2, scale_min, mahal_clamp)
        mips['yz'] = flat.reshape(D, H)

        torch.cuda.empty_cache()

    else:
        print(f'  Rendering XY MIP ({depth_samples} depth samples at {H}×{W}) [CPU]...')
        stack_xy = splat_to_image(gc, dataset, cfg, view_axis='z',
                                  img_h=H, img_w=W, depth_samples=depth_samples)
        mips['xy'] = compute_mip_from_stack(stack_xy, axis=0)

        print(f'  Rendering XZ MIP ({depth_samples} depth samples at {D}×{W}) [CPU]...')
        stack_xz = splat_to_image(gc, dataset, cfg, view_axis='y',
                                  img_h=D, img_w=W, depth_samples=depth_samples)
        mips['xz'] = compute_mip_from_stack(stack_xz, axis=0)

        print(f'  Rendering YZ MIP ({depth_samples} depth samples at {D}×{H}) [CPU]...')
        stack_yz = splat_to_image(gc, dataset, cfg, view_axis='x',
                                  img_h=D, img_w=H, depth_samples=depth_samples)
        mips['yz'] = compute_mip_from_stack(stack_yz, axis=0)

    return mips


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
# Why is the training loop in a separate module (_3dgs_training)?
#   Separating the loop from the model components allows unit-testing the loop
#   logic (epoch structure, densification scheduling, checkpointing) against
#   mock components without spinning up a real GaussianCloud or GPU.
#   This file wires all real components together and injects them via keyword
#   arguments, matching the interface that _train_impl expects.
#%%
def train(cfg: argparse.Namespace):
    """Wire all components and delegate to the training loop implementation.

    All model classes and functions are passed by dependency injection so that
    _train_impl remains importable and testable without this file's global state.
    """
    return _train_impl(
        cfg,
        aabb_cls           = AABB,
        volume_dataset_cls = VolumeDataset,
        gaussian_cloud_cls = GaussianCloud,
        make_optimizer     = make_optimizer,
        update_lr          = update_lr,
        compute_loss       = compute_loss,
        evaluate_fields    = evaluate_fields,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — argument parsing
# ─────────────────────────────────────────────────────────────────────────────
# Why a two-pass parse?
#   We want YAML as a config file with CLI overrides, but argparse processes
#   the full argv in one shot.  The trick: first parse_known_args extracts
#   only --config (ignoring unknown flags), load the YAML, set those values
#   as argparse defaults, then run the full parse — CLI flags silently win.
#%%
def _load_yaml_config(config_path: str, parser: argparse.ArgumentParser) -> dict:
    """Load and validate a YAML config file against the known CLI arguments.

    Why validate against known keys?  Unknown keys in the YAML are almost
    always typos.  Failing loudly here prevents silent no-ops where the user
    thinks they've set a parameter but the training loop ignored it.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        parser.error(f"--config must point to a YAML mapping: {config_path}")

    valid_keys   = {action.dest for action in parser._actions if action.dest != "help"}
    unknown_keys = sorted(set(data) - valid_keys)
    if unknown_keys:
        parser.error(
            f"Unknown keys in config file {config_path}: {', '.join(unknown_keys)}"
        )

    return data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with optional YAML config file.

    Priority (highest to lowest):
        1. Explicit CLI flags  (e.g. --epochs 1000).
        2. YAML config values  (e.g. epochs: 500 in 3dgs.yml).
        3. argparse defaults   (fallback if neither of the above set it).

    Two-pass strategy: parse_known_args → load YAML → set_defaults → full parse.
    This gives clean YAML-as-config semantics without a custom config system.
    """
    p = argparse.ArgumentParser(
        description="Gaussian field fitting — volumetric regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None,
                   help="YAML config file; CLI flags override YAML values")
    p.add_argument("--use_kernel",    dest="use_kernel", action="store_true",
                   help="Use fused CUDA kernel for Gaussian field evaluation")
    p.add_argument("--no_use_kernel", dest="use_kernel", action="store_false",
                   help="Disable fused CUDA kernel even if enabled in config")
    p.set_defaults(use_kernel=False)

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--volume",  default=None)
    p.add_argument("--out",     default="logs/3dgs/run")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--n_init",        type=int,   default=10_000)
    p.add_argument("--max_gaussians", type=int,   default=50_000)
    p.add_argument("--init_scale",    type=float, default=0.05)
    p.add_argument("--init_inten",    type=float, default=0.1,
                   help="Initial intensity amplitude")

    # ── Training schedule ─────────────────────────────────────────────────────
    p.add_argument("--epochs",              type=int,   default=1000)
    p.add_argument("--steps_per_epoch",     type=int,   default=50)
    p.add_argument("--batch",              type=int,   default=2048)
    p.add_argument("--chunk_n",            type=int,   default=1024)
    p.add_argument("--early_stop_patience", type=int,  default=None,
                   help="Stop after this many epochs without PSNR improvement")

    # ── Numerical stability ────────────────────────────────────────────────────
    p.add_argument("--scale_min_clamp", type=float, default=1e-5,
                   help="Minimum Gaussian scale (prevents singular covariance)")
    p.add_argument("--mahal_max_clamp", type=float, default=20.0,
                   help="Mahalanobis distance clamp (prevents exp underflow)")
    p.add_argument("--grad_clip_norm",  type=float, default=1.0,
                   help="Max gradient norm for Gaussian centres")

    # ── Learning rates ─────────────────────────────────────────────────────────
    p.add_argument("--lr_means",              type=float, default=1.6e-4)
    p.add_argument("--lr_means_final",        type=float, default=1.6e-6)
    p.add_argument("--lr_scales",             type=float, default=5e-3)
    p.add_argument("--lr_quats",              type=float, default=1e-3)
    p.add_argument("--lr_inten",              type=float, default=1e-2)
    p.add_argument("--lr_final_fraction",     type=float, default=0.01,
                   help="Cosine annealing floor as fraction of initial LR")
    p.add_argument("--lr_warmup_steps",       type=int,   default=100,
                   help="Linear warmup length in optimizer steps")
    p.add_argument("--lr_warmup_init_factor", type=float, default=0.1,
                   help="Initial LR fraction applied during warmup")
    p.add_argument("--adam_eps",              type=float, default=1e-15)

    # ── Regularisation ─────────────────────────────────────────────────────────
    p.add_argument("--lambda_scale",         type=float, default=1e-4)
    p.add_argument("--lambda_sparsity",      type=float, default=1e-3)
    p.add_argument("--lambda_scale_ceiling", type=float, default=1e-3)
    p.add_argument("--lambda_scale_outlier", type=float, default=5e-4)
    p.add_argument("--scale_max_hard",       type=float, default=None)
    p.add_argument("--scale_min_hard",       type=float, default=None)
    p.add_argument("--lambda_aniso",  type=float, default=0.0,
                   help="L2 penalty on min per-axis scale; encourages elongated ellipsoids")
    p.add_argument("--lambda_count",  type=float, default=0.0,
                   help="Soft-L0 count penalty (mean sigmoid of raw intensities)")
    p.add_argument("--lambda_L1",     type=float, default=0.0,
                   help="L1 intensity penalty (mean softplus(raw)); finishes off zombie Gaussians")
    p.add_argument("--lambda_coverage", type=float, default=0.0,
                   help="Coverage reward: -log(s_max/init_scale); rewards large elongated ellipsoids")
    p.add_argument("--lambda_ssim",   type=float, default=0.2,
                   help="Weight of SSIM slice loss in the reconstruction term (GaussianPile)")
    p.add_argument("--lambda_grad",   type=float, default=0.0,
                   help="Weight of gradient sharpness loss on Z-slice crops (edge preservation)")
    p.add_argument("--grad_sample_weight", type=float, default=0.0,
                   help="Fraction of importance-sampling CDF mass given to high-gradient voxels (0=off, 0.5=equal mix)")
    p.add_argument("--interior_init_n", type=int, default=0,
                   help="Extra Gaussians seeded at bright interior voxels to fill the soma (0=off)")
    p.add_argument("--interior_init_thresh", type=float, default=0.3,
                   help="Intensity threshold for interior_init_n (voxels with gt > thresh are eligible)")
    p.add_argument("--swc_oriented_init",    dest="swc_oriented_init", action="store_true",
                   help="Align initial Gaussian x-axis with the SWC branch direction at each node")
    p.add_argument("--no_swc_oriented_init", dest="swc_oriented_init", action="store_false")
    p.set_defaults(swc_oriented_init=False)
    p.add_argument("--log_intensity_loss", dest="log_intensity_loss", action="store_true",
                   help="Compute L1 in log1p space to de-emphasise bright soma, amplify dim dendrites")
    p.add_argument("--no_log_intensity_loss", dest="log_intensity_loss", action="store_false")
    p.set_defaults(log_intensity_loss=False)
    p.add_argument("--ssim_crop",       type=int,   default=64,
                   help="Side length of random Z-crop used to compute SSIM each step")
    p.add_argument("--ssim_start_step", type=int,   default=0,
                   help="Activate SSIM loss only after this many optimizer steps; "
                        "set to densify_until_step to prevent SSIM from inflating _grad_acc")
    p.add_argument("--init_scale_z_factor", type=float, default=1.0,
                   help="Multiply z-axis init_scale by this factor (PSF anisotropy correction; "
                        "set to (W-1)/(D-1) for confocal volumes)")
    p.add_argument("--prune_inten_thresh", type=float, default=1e-3,
                   help="Remove Gaussians with softplus(inten) below this during densify")

    # ── Adaptive density control ───────────────────────────────────────────────
    p.add_argument("--densify_from_step",   type=int,   default=500)
    p.add_argument("--densify_until_step",  type=int,   default=None,
                   help="Stop densification after this optimizer step; unset keeps it active to the end")
    p.add_argument("--densify_interval",    type=int,   default=200)
    p.add_argument("--densify_grad_thresh", type=float, default=2e-4)
    p.add_argument("--densify_max_scale",   type=float, default=0.04)
    p.add_argument("--split_scale_divisor", type=float, default=1.6,
                   help="Scale shrink factor applied to split daughters")
    p.add_argument("--log_scale_floor",     type=float, default=-6.0,
                   help="Minimum log-scale after split")

    # ── Evaluation / logging ───────────────────────────────────────────────────
    p.add_argument("--eval_samples",         type=int,  default=200_000)
    p.add_argument("--log_interval",         type=int,  default=10)
    p.add_argument("--eval_detail_interval", type=int,  default=5)
    p.add_argument("--swc_path",             type=str,  default=None)
    p.add_argument("--swc_init",    dest="swc_init", action="store_true")
    p.add_argument("--no_swc_init", dest="swc_init", action="store_false")
    p.set_defaults(swc_init=True)
    p.add_argument("--ckpt_interval", type=int, default=2000,
                   help="Save ckpt_STEP.pth every N optimizer steps; 0 disables periodic checkpoints")

    # ── Two-pass YAML + CLI merge ──────────────────────────────────────────────
    pre, _ = p.parse_known_args()
    if pre.config is not None:
        p.set_defaults(**_load_yaml_config(pre.config, p))

    args = p.parse_args()
    if args.volume is None:
        p.error("--volume is required (on CLI or in config file)")

    return args


if __name__ == "__main__":
    cfg = parse_args()
    USE_CUDA_KERNEL = cfg.use_kernel
    if USE_CUDA_KERNEL:
        print("Using fused CUDA Gaussian field kernel")
    else:
        print("Using PyTorch Gaussian field implementation (pass --use_kernel to enable fused kernel)")
    print(f"Device : {cfg.device}")
    print(f"Config : {cfg.config}")
    print(f"Output : {cfg.out}")
    train(cfg)
