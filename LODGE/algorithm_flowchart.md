# LODGE: Level-of-Detail Large-Scale Gaussian Splatting
> Paper: arxiv.org/abs/2505.23158

---

## Algorithm Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT                                  │
│   Multi-view images  +  Camera poses (intrinsics/extrinsics)    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 1: BASE 3DGS TRAINING                       │
│  • Initialize 3D Gaussians from SfM point cloud                 │
│  • Optimize: position μ, covariance Σ, opacity α, SH color c    │
│  • Densification (clone/split) + pruning via opacity threshold  │
│  • Loss: L = L_rgb (L1 + SSIM) + λ · L_regularization          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│          STAGE 2: HIERARCHICAL LOD CONSTRUCTION                 │
│                                                                 │
│   Full Gaussian set G₀  (finest level)                          │
│        │                                                        │
│        ├──► Compute camera-distance-based selection criterion   │
│        │    for each Gaussian g_i:                              │
│        │       score(g_i) = importance(g_i) / dist(g_i, cam)²  │
│        │                                                        │
│        ▼                                                        │
│   For each LOD level L = 1, 2, ..., N:                         │
│   ┌─────────────────────────────────────────────────────┐      │
│   │  2a. DEPTH-AWARE 3D SMOOTHING FILTER                │      │
│   │      • Aggregate neighboring Gaussians              │      │
│   │      • Smooth opacity and color using depth-         │      │
│   │        weighted kernel to preserve boundaries        │      │
│   │                                                     │      │
│   │  2b. IMPORTANCE-BASED PRUNING                       │      │
│   │      • Rank Gaussians by visual contribution score  │      │
│   │      • Discard low-importance primitives            │      │
│   │      • Retain top-k% → Gaussian subset G_L          │      │
│   │                                                     │      │
│   │  2c. FINE-TUNING                                    │      │
│   │      • Re-optimize G_L on training views at this   │      │
│   │        scale to recover fidelity lost via pruning   │      │
│   │      • Loss: same photometric + regularization      │      │
│   └──────────────────┬──────────────────────────────────┘      │
│                      │  repeat for next LOD level               │
│                      ▼                                          │
│   Output: LOD pyramid {G₀ ⊃ G₁ ⊃ G₂ ⊃ ... ⊃ G_N}             │
│           (G₀ finest, G_N coarsest)                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 3: DYNAMIC SPATIAL CHUNKING                     │
│                                                                 │
│  • Partition scene bounding volume into spatial chunks C_k      │
│  • Assign each Gaussian in {G₀,...,G_N} to its chunk           │
│  • Build per-chunk LOD index for fast lookup at render time     │
│  • Store chunks on disk; only load relevant ones into VRAM      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RENDERING PIPELINE                            │
│                                                                 │
│  Given: camera pose P, frustum F, device memory budget M        │
│                                                                 │
│  Step R1 │ FRUSTUM CULLING                                      │
│          │  Discard chunks with no overlap with F               │
│          ▼                                                      │
│  Step R2 │ DYNAMIC CHUNK LOADING                                │
│          │  For each visible chunk C_k:                         │
│          │    dist_k = distance(cam_center, C_k)                │
│          │    Select LOD level l_k = f(dist_k, screen_cov_k)   │
│          │    Load G_{l_k} ∩ C_k into GPU if within budget M   │
│          ▼                                                      │
│  Step R3 │ OPACITY-BLENDING AT CHUNK BOUNDARIES                 │
│          │  At borders between adjacent chunks:                 │
│          │    blend α using distance-weighted opacity mask      │
│          │    to suppress seam artifacts                        │
│          ▼                                                      │
│  Step R4 │ DEPTH SORT + SPLATTING                               │
│          │  Sort active Gaussians back-to-front                 │
│          │  Rasterize via differentiable Gaussian splatting     │
│          ▼                                                      │
│  Step R5 │ ALPHA COMPOSITING                                    │
│          │  C = Σ_i  c_i · α_i · Π_{j<i}(1 - α_j)            │
│          │  Early termination when Σα ≥ threshold T            │
│          ▼                                                      │
│        OUTPUT: Rendered novel-view image                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Summary

| Module | Input | Output | Key Idea |
|--------|-------|--------|----------|
| Base 3DGS Training | Images + poses | Dense Gaussians G₀ | Standard 3DGS optimization |
| Depth-Aware Smoothing | G_L | Smoothed G_L | Depth-weighted kernel merging |
| Importance Pruning | Smoothed G_L | Sparse G_{L+1} | Rank by visual contribution |
| Fine-Tuning | Sparse G_{L+1} | Refined G_{L+1} | Photometric re-optimization |
| Spatial Chunking | LOD pyramid | Chunk index | Spatial partitioning for streaming |
| Dynamic Loading | Camera pose | Active Gaussians | Distance-based LOD + memory budget |
| Opacity Blending | Chunk boundaries | Seamless render | Distance-weighted α masking |
| Splatting & Compositing | Active Gaussians | Rendered image | 3DGS rasterization + early stop |

---

## Loss Functions

**Training Loss (all stages):**
```
L_total = (1 - λ) · L1(I_render, I_gt) + λ · L_SSIM(I_render, I_gt) + β · L_reg
```

Where:
- `L1` — pixel-wise absolute error
- `L_SSIM` — structural similarity loss
- `L_reg` — regularization (opacity sparsity / scale penalty)
- `λ = 0.2`, `β` tuned per stage

---

## LOD Selection Criterion (at render time)

```
For chunk C_k with camera at distance d:
    screen_coverage(k) = projected_area(C_k) / total_pixels
    l_k = argmin_l { |G_l ∩ C_k| : screen_coverage(k) < threshold(l) }
```

Coarser LOD (higher `l`) is selected for:
- Large viewing distance `d`
- Low screen coverage
- Limited VRAM budget `M`
