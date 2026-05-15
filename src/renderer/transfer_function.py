"""
transfer_function.py — 1-D and 2-D transfer functions for direct volume rendering.

A transfer function maps scalar intensity I ∈ [0,1]  →  RGBA colour.
The renderer queries tf(I) at each sample point along a ray.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RGBAPoint:
    intensity: float          # control point in [0, 1]
    r: float; g: float; b: float; a: float   # colour + opacity


class TransferFunction:
    """Base class — subclass and override __call__."""

    def __call__(self, intensities: np.ndarray) -> np.ndarray:
        """intensities : (N,) float32 → (N, 4) float32 RGBA in [0, 1]."""
        raise NotImplementedError


class TransferFunction1D(TransferFunction):
    """Piecewise-linear 1-D transfer function defined by RGBA control points.

    Usage
    -----
    tf = TransferFunction1D()
    tf.add(0.00, r=0,   g=0,   b=0,   a=0.0)   # fully transparent background
    tf.add(0.15, r=0.2, g=0.2, b=0.8, a=0.2)   # faint blue for low signal
    tf.add(0.50, r=0.8, g=0.4, b=0.0, a=0.6)   # orange for mid-range
    tf.add(1.00, r=1.0, g=1.0, b=1.0, a=1.0)   # white for bright structures
    """

    def __init__(self):
        self._points: list[RGBAPoint] = []

    def add(self, intensity: float,
            r: float, g: float, b: float, a: float) -> 'TransferFunction1D':
        self._points.append(RGBAPoint(intensity, r, g, b, a))
        self._points.sort(key=lambda p: p.intensity)
        return self

    def __call__(self, intensities: np.ndarray) -> np.ndarray:
        pts = self._points
        if not pts:
            rgba = np.zeros((*intensities.shape, 4), dtype=np.float32)
            rgba[..., 3] = intensities          # default: opacity = intensity
            return rgba

        xs = np.array([p.intensity for p in pts], dtype=np.float32)
        ys = np.array([[p.r, p.g, p.b, p.a] for p in pts], dtype=np.float32)
        i  = np.searchsorted(xs, intensities, side='right').clip(1, len(xs) - 1)
        t  = ((intensities - xs[i - 1]) / np.maximum(xs[i] - xs[i - 1], 1e-8))
        t  = t.clip(0., 1.)[..., None]
        return (1. - t) * ys[i - 1] + t * ys[i]


# ── preset transfer functions ──────────────────────────────────────────────────

def tf_fluorescence() -> TransferFunction1D:
    """Green fluorescence — transparent background, bright green for signal."""
    return (TransferFunction1D()
            .add(0.00, r=0.0, g=0.0, b=0.0, a=0.0)
            .add(0.05, r=0.0, g=0.2, b=0.0, a=0.0)
            .add(0.20, r=0.0, g=0.6, b=0.2, a=0.2)
            .add(0.60, r=0.2, g=1.0, b=0.2, a=0.7)
            .add(1.00, r=1.0, g=1.0, b=0.8, a=1.0))

def tf_hot() -> TransferFunction1D:
    """Hot colourmap — black → red → yellow → white."""
    return (TransferFunction1D()
            .add(0.00, r=0.0, g=0.0, b=0.0, a=0.0)
            .add(0.10, r=0.1, g=0.0, b=0.0, a=0.1)
            .add(0.40, r=0.8, g=0.0, b=0.0, a=0.5)
            .add(0.70, r=1.0, g=0.8, b=0.0, a=0.8)
            .add(1.00, r=1.0, g=1.0, b=1.0, a=1.0))

def tf_gray() -> TransferFunction1D:
    """Grayscale — transparent background, white signal."""
    return (TransferFunction1D()
            .add(0.00, r=0.0, g=0.0, b=0.0, a=0.0)
            .add(0.05, r=0.0, g=0.0, b=0.0, a=0.0)
            .add(1.00, r=1.0, g=1.0, b=1.0, a=1.0))

def tf_cell_body() -> TransferFunction1D:
    """Highlights cell body (high intensity) in white, processes in blue."""
    return (TransferFunction1D()
            .add(0.00, r=0.0, g=0.0, b=0.0, a=0.0)
            .add(0.08, r=0.0, g=0.0, b=0.5, a=0.1)
            .add(0.30, r=0.2, g=0.3, b=0.9, a=0.4)
            .add(0.70, r=0.8, g=0.8, b=1.0, a=0.8)
            .add(1.00, r=1.0, g=1.0, b=1.0, a=1.0))
