#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────────────────────────────────────
// Multi-resolution hash encoding — fused CUDA kernels
//
// Implements Müller et al. (2022) Instant NGP encoding:
//   For each level l and query point n:
//     1. Scale coordinate to grid:  xg = (x + 1) / 2 * (res_l - 1)
//     2. Floor lower-left corner:   x0 = clamp(floor(xg), 0, res_l - 2)
//     3. Fractional weights:        w  = clamp(xg - x0, 0, 1)
//     4. Hash 8 voxel corners:      h  = (cx ⊕ π₁·cy ⊕ π₂·cz) & (T-1)
//        π₁ = 2 654 435 761,  π₂ = 805 459 861  (Müller et al., 2022)
//        T  = 2^log2T  →  bitwise AND replaces expensive modulo
//     5. Trilinear interpolation of 8 hash-table feature vectors
//
// Hash tables are stored as a single (n_levels, T, F) float tensor.
//
// Thread assignment: one thread per (n, l) pair.
//   gridDim  = ceil(N * n_levels / blockDim)
//   blockDim = 256
//
// Backward: recomputes corners and scatter-adds grad via atomicAdd.
//   No grad w.r.t. x (coordinates are non-trainable dataset samples).
// ─────────────────────────────────────────────────────────────────────────────

#define PI2 2654435761ULL
#define PI3  805459861ULL

// Spatial hash: h = (cx ⊕ π₁·cy ⊕ π₂·cz) & (T-1).
// Uses uint64 arithmetic to avoid overflow before masking.
__device__ __forceinline__ uint32_t hash3d(
    uint32_t cx, uint32_t cy, uint32_t cz, uint32_t T_mask
) {
    uint64_t h = (uint64_t)cx
               ^ ((uint64_t)cy * PI2)
               ^ ((uint64_t)cz * PI3);
    return (uint32_t)(h & (uint64_t)T_mask);
}


// ── Forward ──────────────────────────────────────────────────────────────────

__global__ void ingp_forward_kernel(
    const float* __restrict__ x,           // (N, 3)  coords in [-1, 1]
    const float* __restrict__ tables,      // (n_levels, T, F)
    const int*   __restrict__ resolutions, // (n_levels,)
    float*       __restrict__ out,         // (N, n_levels * F)
    int N, int n_levels, int F, uint32_t T, uint32_t T_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_levels) return;

    int n = idx / n_levels;
    int l = idx % n_levels;

    float px = x[n * 3 + 0];
    float py = x[n * 3 + 1];
    float pz = x[n * 3 + 2];

    int res = resolutions[l];

    // Scale to grid [0, res-1]
    float xg = (px + 1.0f) * 0.5f * (float)(res - 1);
    float yg = (py + 1.0f) * 0.5f * (float)(res - 1);
    float zg = (pz + 1.0f) * 0.5f * (float)(res - 1);

    // Floor lower corner, clamped to [0, res-2]
    int x0 = min(max((int)floorf(xg), 0), res - 2);
    int y0 = min(max((int)floorf(yg), 0), res - 2);
    int z0 = min(max((int)floorf(zg), 0), res - 2);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    // Fractional parts (trilinear weights)
    float wx = fminf(fmaxf(xg - (float)x0, 0.0f), 1.0f);
    float wy = fminf(fmaxf(yg - (float)y0, 0.0f), 1.0f);
    float wz = fminf(fmaxf(zg - (float)z0, 0.0f), 1.0f);

    // 8 corner weights (order: 000,100,010,110,001,101,011,111)
    float w[8] = {
        (1-wx)*(1-wy)*(1-wz),
           wx *(1-wy)*(1-wz),
        (1-wx)*   wy *(1-wz),
           wx *   wy *(1-wz),
        (1-wx)*(1-wy)*   wz,
           wx *(1-wy)*   wz,
        (1-wx)*   wy *   wz,
           wx *   wy *   wz,
    };

    // Corner (cx, cy, cz) coordinates
    uint32_t cx_[8] = {(uint32_t)x0,(uint32_t)x1,(uint32_t)x0,(uint32_t)x1,
                       (uint32_t)x0,(uint32_t)x1,(uint32_t)x0,(uint32_t)x1};
    uint32_t cy_[8] = {(uint32_t)y0,(uint32_t)y0,(uint32_t)y1,(uint32_t)y1,
                       (uint32_t)y0,(uint32_t)y0,(uint32_t)y1,(uint32_t)y1};
    uint32_t cz_[8] = {(uint32_t)z0,(uint32_t)z0,(uint32_t)z0,(uint32_t)z0,
                       (uint32_t)z1,(uint32_t)z1,(uint32_t)z1,(uint32_t)z1};

    // Hash table for this level: tables + l * T * F
    const float* tbl = tables + (uint64_t)l * T * F;
    // Output slice for this (n, l): out + (n * n_levels + l) * F
    float* out_nl = out + ((uint64_t)n * n_levels + l) * F;

    // Initialise output
    for (int f = 0; f < F; f++) out_nl[f] = 0.0f;

    // Accumulate trilinearly interpolated features
    for (int c = 0; c < 8; c++) {
        uint32_t h = hash3d(cx_[c], cy_[c], cz_[c], T_mask);
        const float* feat = tbl + (uint64_t)h * F;
        float wc = w[c];
        for (int f = 0; f < F; f++)
            out_nl[f] += wc * feat[f];
    }
}


// ── Backward ─────────────────────────────────────────────────────────────────
// Recomputes corner hashes and scatter-adds gradient to grad_tables.
// atomicAdd is required because multiple (n, c) pairs may share the same hash.
// No grad w.r.t. x (coordinates are fixed dataset samples, not trainable).

__global__ void ingp_backward_kernel(
    const float* __restrict__ grad_out,    // (N, n_levels * F)
    const float* __restrict__ x,           // (N, 3)
    float*                    grad_tables, // (n_levels, T, F)  ← modified atomically
    const int*   __restrict__ resolutions, // (n_levels,)
    int N, int n_levels, int F, uint32_t T, uint32_t T_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_levels) return;

    int n = idx / n_levels;
    int l = idx % n_levels;

    float px = x[n * 3 + 0];
    float py = x[n * 3 + 1];
    float pz = x[n * 3 + 2];

    int res = resolutions[l];

    float xg = (px + 1.0f) * 0.5f * (float)(res - 1);
    float yg = (py + 1.0f) * 0.5f * (float)(res - 1);
    float zg = (pz + 1.0f) * 0.5f * (float)(res - 1);

    int x0 = min(max((int)floorf(xg), 0), res - 2);
    int y0 = min(max((int)floorf(yg), 0), res - 2);
    int z0 = min(max((int)floorf(zg), 0), res - 2);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    float wx = fminf(fmaxf(xg - (float)x0, 0.0f), 1.0f);
    float wy = fminf(fmaxf(yg - (float)y0, 0.0f), 1.0f);
    float wz = fminf(fmaxf(zg - (float)z0, 0.0f), 1.0f);

    float w[8] = {
        (1-wx)*(1-wy)*(1-wz),
           wx *(1-wy)*(1-wz),
        (1-wx)*   wy *(1-wz),
           wx *   wy *(1-wz),
        (1-wx)*(1-wy)*   wz,
           wx *(1-wy)*   wz,
        (1-wx)*   wy *   wz,
           wx *   wy *   wz,
    };

    uint32_t cx_[8] = {(uint32_t)x0,(uint32_t)x1,(uint32_t)x0,(uint32_t)x1,
                       (uint32_t)x0,(uint32_t)x1,(uint32_t)x0,(uint32_t)x1};
    uint32_t cy_[8] = {(uint32_t)y0,(uint32_t)y0,(uint32_t)y1,(uint32_t)y1,
                       (uint32_t)y0,(uint32_t)y0,(uint32_t)y1,(uint32_t)y1};
    uint32_t cz_[8] = {(uint32_t)z0,(uint32_t)z0,(uint32_t)z0,(uint32_t)z0,
                       (uint32_t)z1,(uint32_t)z1,(uint32_t)z1,(uint32_t)z1};

    // Gradient of output for this (n, l)
    const float* gout = grad_out + ((uint64_t)n * n_levels + l) * F;
    // Gradient table for level l
    float* gtbl = grad_tables + (uint64_t)l * T * F;

    // Scatter-add: grad_table[h, f] += w[c] * grad_out[n, l*F + f]
    for (int c = 0; c < 8; c++) {
        uint32_t h = hash3d(cx_[c], cy_[c], cz_[c], T_mask);
        float wc = w[c];
        float* dst = gtbl + (uint64_t)h * F;
        for (int f = 0; f < F; f++)
            atomicAdd(&dst[f], wc * gout[f]);
    }
}


// ── Python-visible entry points ───────────────────────────────────────────────

torch::Tensor ingp_forward(
    torch::Tensor x,           // (N, 3) float32, CUDA
    torch::Tensor tables,      // (n_levels, T, F) float32, CUDA
    torch::Tensor resolutions  // (n_levels,) int32, CUDA
) {
    TORCH_CHECK(x.is_cuda()          && x.is_contiguous(),        "x must be contiguous CUDA");
    TORCH_CHECK(tables.is_cuda()     && tables.is_contiguous(),    "tables must be contiguous CUDA");
    TORCH_CHECK(resolutions.is_cuda(), "resolutions must be CUDA");
    TORCH_CHECK(x.scalar_type()      == torch::kFloat, "x must be float32");
    TORCH_CHECK(tables.scalar_type() == torch::kFloat, "tables must be float32");

    int N        = x.size(0);
    int n_levels = tables.size(0);
    int T        = tables.size(1);
    int F        = tables.size(2);
    uint32_t T_mask = (uint32_t)(T - 1);

    auto out = torch::zeros({N, n_levels * F}, x.options());

    const int threads = 256;
    const int blocks  = (N * n_levels + threads - 1) / threads;

    ingp_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        tables.data_ptr<float>(),
        resolutions.data_ptr<int>(),
        out.data_ptr<float>(),
        N, n_levels, F, (uint32_t)T, T_mask
    );
    return out;
}

torch::Tensor ingp_backward(
    torch::Tensor grad_out,    // (N, n_levels * F) float32, CUDA
    torch::Tensor x,           // (N, 3) float32, CUDA
    torch::Tensor tables,      // (n_levels, T, F) float32, CUDA  — for shape
    torch::Tensor resolutions  // (n_levels,) int32, CUDA
) {
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous(), "grad_out must be contiguous CUDA");
    TORCH_CHECK(x.is_cuda()        && x.is_contiguous(),        "x must be contiguous CUDA");
    TORCH_CHECK(tables.is_cuda()   && tables.is_contiguous(),    "tables must be contiguous CUDA");
    TORCH_CHECK(resolutions.is_cuda(), "resolutions must be CUDA");

    int N        = x.size(0);
    int n_levels = tables.size(0);
    int T        = tables.size(1);
    int F        = tables.size(2);
    uint32_t T_mask = (uint32_t)(T - 1);

    // Gradient w.r.t. hash tables, initialised to zero (accumulated via atomicAdd)
    auto grad_tables = torch::zeros_like(tables);

    const int threads = 256;
    const int blocks  = (N * n_levels + threads - 1) / threads;

    ingp_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        grad_tables.data_ptr<float>(),
        resolutions.data_ptr<int>(),
        N, n_levels, F, (uint32_t)T, T_mask
    );
    return grad_tables;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &ingp_forward,
          "Fused Instant NGP hash encoding forward  (CUDA)");
    m.def("backward", &ingp_backward,
          "Fused Instant NGP hash encoding backward (CUDA, atomicAdd to hash tables)");
}
