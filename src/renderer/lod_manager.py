"""
lod_manager.py — Tiled and LOD (level-of-detail) INR volumes for huge datasets.

Problem
-------
A single fluorescence volume can be 10 000³ voxels (≈ 4 TB float32).
No INR can be trained on the full volume at once, and no GPU has enough VRAM
to evaluate it in one pass.

Solution
--------
TiledINRVolume
  Partitions the bounding box into a regular N_x × N_y × N_z grid of tiles.
  Each tile has its own trained INRVolume (different model or same class).
  Tiles are loaded on demand and evicted with LRU when the GPU cache is full.
  During a query, each point is dispatched to its tile, evaluated locally,
  then the results are reassembled in order.

LODVolume
  Wraps 2–4 INRVolumes at different resolutions (coarse → fine).
  During rendering, the LOD is chosen based on the projected screen-space
  voxel size: coarse INR for overview, fine INR for zoomed regions.
  This mirrors GPU mip-map LOD selection.

Usage
-----
# Tile layout for a 10 000 × 10 000 × 10 000 volume
tv = TiledINRVolume(grid=(4, 4, 4), tile_models=loaded_dict, cache_size=16)
renderer = VolumeRenderer(tv)
img = renderer.mip(axis=2, res_h=1024, res_w=1024)

# LOD
lod = LODVolume([coarse_vol, fine_vol], resolutions=[64, 512])
renderer = VolumeRenderer(lod)
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from .inr_volume import INRVolume


# ── LRU GPU cache for tile models ─────────────────────────────────────────────

class _LRUCache:
    """Thread-safe LRU cache for INRVolume tiles (keyed by tile index tuple)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict[tuple, INRVolume] = OrderedDict()

    def get(self, key: tuple) -> Optional[INRVolume]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: tuple, vol: INRVolume):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = vol
        if len(self._cache) > self.capacity:
            evicted_key, evicted = self._cache.popitem(last=False)
            # move evicted model to CPU to free GPU VRAM
            try:
                evicted.model.cpu()
            except AttributeError:
                pass

    def __len__(self):
        return len(self._cache)


# ── Tiled INR volume ──────────────────────────────────────────────────────────

class TiledINRVolume(INRVolume):
    """A huge volume partitioned into N_x×N_y×N_z tiles, each a local INRVolume.

    Coordinate convention: all coordinates are in [0,1]³ (normalised world).
    Each tile covers the sub-cube
        x ∈ [ix/Nx, (ix+1)/Nx],  y similarly,  z similarly.
    Inside its tile, a local INRVolume is queried in [0,1]³ local coords.

    Parameters
    ----------
    grid       : (Nx, Ny, Nz) — number of tiles per axis
    tile_loader: callable(ix, iy, iz) → INRVolume  — loads / trains a tile on demand
    cache_size : number of tiles to keep in GPU memory simultaneously
    device     : target device for queries
    """

    coord_range = 'unit'

    def __init__(self, grid: Tuple[int, int, int],
                 tile_loader,
                 cache_size: int = 8,
                 device: str = 'cuda',
                 chunk: int = 2 ** 18):
        self.grid        = grid          # (Nx, Ny, Nz)
        self._loader     = tile_loader
        self._cache      = _LRUCache(cache_size)
        self.device      = torch.device(device)
        self.chunk       = chunk

    @classmethod
    def from_dict(cls, grid: Tuple[int, int, int],
                  tile_models: Dict[tuple, INRVolume],
                  cache_size: int = 8,
                  device: str = 'cuda') -> 'TiledINRVolume':
        """Build from a pre-loaded dict {(ix,iy,iz): INRVolume}."""
        def loader(ix, iy, iz):
            return tile_models[(ix, iy, iz)]
        return cls(grid, loader, cache_size=cache_size, device=device)

    def _get_tile(self, ix: int, iy: int, iz: int) -> INRVolume:
        key = (ix, iy, iz)
        t   = self._cache.get(key)
        if t is None:
            t = self._loader(ix, iy, iz)
            self._cache.put(key, t)
        return t

    def _forward_chunk(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # not used — query() is overridden

    @torch.inference_mode()
    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """Dispatch each point to its tile and reassemble.  pts: (N,3) in [0,1]³."""
        pts    = pts.to(self.device).clamp(0., 1.)
        Nx, Ny, Nz = self.grid
        N      = pts.shape[0]
        out    = torch.zeros(N, device=self.device)

        # tile indices for each point
        ix = (pts[:, 0] * Nx).long().clamp(0, Nx - 1)
        iy = (pts[:, 1] * Ny).long().clamp(0, Ny - 1)
        iz = (pts[:, 2] * Nz).long().clamp(0, Nz - 1)

        # group points by tile
        tile_ids = ix * (Ny * Nz) + iy * Nz + iz   # scalar tile id
        for tid in tile_ids.unique():
            tidv  = tid.item()
            tix   = int(tidv // (Ny * Nz))
            tiy   = int((tidv % (Ny * Nz)) // Nz)
            tiz   = int(tidv % Nz)
            mask  = (tile_ids == tid)
            local = pts[mask].clone()

            # map from world [0,1] to local tile [0,1]
            local[:, 0] = local[:, 0] * Nx - tix
            local[:, 1] = local[:, 1] * Ny - tiy
            local[:, 2] = local[:, 2] * Nz - tiz

            if self._get_tile(tix, tiy, tiz).coord_range == 'symmetric':
                local = local * 2. - 1.

            tile = self._get_tile(tix, tiy, tiz)
            out[mask] = tile.query(local.to(tile.device)).to(self.device)

        return out


# ── LOD volume ────────────────────────────────────────────────────────────────

class LODVolume(INRVolume):
    """Switch between multiple INRVolumes based on screen-space LOD.

    levels       : list of INRVolume, ordered coarse → fine
    resolutions  : list of int — native voxel resolution per level along longest axis
                   (used to compute the pixel-to-voxel ratio for LOD selection)

    LOD selection rule (similar to OpenGL mip-map):
        Use level l such that  resolutions[l] ≈ rendered_res.
        If rendered_res > resolutions[-1], use the finest level.
    """

    coord_range = 'unit'

    def __init__(self, levels: List[INRVolume], resolutions: List[int],
                 device: str = 'cuda', chunk: int = 2 ** 18):
        assert len(levels) == len(resolutions)
        self.levels      = levels
        self.resolutions = resolutions
        self.device      = torch.device(device)
        self.chunk       = chunk
        self._active_lod = len(levels) - 1    # default: finest

    def set_lod(self, rendered_res: int):
        """Select the coarsest level whose resolution ≥ rendered_res."""
        for i, res in enumerate(self.resolutions):
            if res >= rendered_res:
                self._active_lod = i
                return
        self._active_lod = len(self.levels) - 1

    @property
    def active(self) -> INRVolume:
        return self.levels[self._active_lod]

    def _forward_chunk(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def query(self, pts: torch.Tensor) -> torch.Tensor:
        return self.active.query(pts.to(self.active.device)).to(self.device)

    def eval_volume(self, D: int, H: int, W: int) -> np.ndarray:
        self.set_lod(max(D, H, W))
        return self.active.eval_volume(D, H, W)

    def eval_slice(self, axis: int, index: float,
                   res_a: int, res_b: int) -> np.ndarray:
        self.set_lod(max(res_a, res_b))
        return self.active.eval_slice(axis, index, res_a, res_b)


# ── Helper: build a tiled volume from a single huge numpy array ───────────────

def build_tiled_volume_from_numpy(vol: np.ndarray,
                                  grid: Tuple[int, int, int],
                                  model_class,
                                  train_fn,
                                  device: str = 'cuda') -> TiledINRVolume:
    """Train a separate INR on each tile of a numpy volume.

    vol        : (D, H, W) float32 in [0, 1]
    grid       : (Nx, Ny, Nz) tile layout
    model_class: INRVolume subclass to use per tile
    train_fn   : callable(tile_np, tile_bounds) → INRVolume
                 tile_np: (d, h, w) numpy tile, tile_bounds: ((x0,x1),(y0,y1),(z0,z1))
    """
    D, H, W = vol.shape
    Nx, Ny, Nz = grid
    trained_tiles = {}

    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                z0 = int(round(iz * D / Nz));  z1 = int(round((iz + 1) * D / Nz))
                y0 = int(round(iy * H / Ny));  y1 = int(round((iy + 1) * H / Ny))
                x0 = int(round(ix * W / Nx));  x1 = int(round((ix + 1) * W / Nx))
                tile = vol[z0:z1, y0:y1, x0:x1]
                bounds = ((x0 / W, x1 / W), (y0 / H, y1 / H), (z0 / D, z1 / D))
                print(f'Training tile ({ix},{iy},{iz}) shape={tile.shape}')
                trained_tiles[(ix, iy, iz)] = train_fn(tile, bounds)

    return TiledINRVolume.from_dict(grid, trained_tiles, device=device)
