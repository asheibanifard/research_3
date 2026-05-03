/*
 * 3dgs_eval_cuda.cu — fused inference kernel for Gaussian-cloud volume evaluation.
 *
 * Replaces the Python slice-by-slice loop in 3dgs_eval.py with a single
 * CUDA launch.  Each thread owns one voxel and accumulates contributions
 * from every Gaussian.  Gaussians are loaded into shared memory in tiles
 * of TILE to exploit L1/shared-mem bandwidth.
 *
 * Exposed function (via pybind11 / torch.utils.cpp_extension):
 *
 *   reconstruct_volume(means, log_s, quats, inten,
 *                      lo_x, hi_x, lo_y, hi_y, lo_z, hi_z,
 *                      D, H, W, scale_min, mahal_clamp)
 *     -> float32 cuda tensor, shape (D*H*W,), values in [0, 1]
 *
 * Coordinate convention (matches VolumeDataset._indices_to_pts):
 *   flat index  n = iz*(H*W) + ih*W + iw
 *   x = lo_x + (iw / (W-1)) * (hi_x - lo_x)   pts[:,0]
 *   y = lo_y + (ih / (H-1)) * (hi_y - lo_y)   pts[:,1]
 *   z = lo_z + (iz / (D-1)) * (hi_z - lo_z)   pts[:,2]
 *
 * Math (matches GaussianCloud.forward / build_sigma_inv):
 *   s    = exp(log_s).clamp(min=scale_min)
 *   R    = quat_to_rotmat(q)
 *   RD   = R * diag(1/s)          column-wise
 *   SI   = RD @ RD^T              inverse covariance
 *   mah2 = d @ SI @ d             d = pt - mean
 *   out += softplus(inten) * exp(-0.5 * clamp(mah2, max=mahal_clamp))
 */

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>

// ── tuning constants ──────────────────────────────────────────────────────────
// BLOCK: threads per block = voxels processed in parallel per SM invocation.
// TILE:  Gaussians loaded into shared memory per round.
// Shared-mem per block = TILE * (3+3+4+1) * 4  =  TILE * 44 bytes.
// With TILE=256: 11 264 B — well within the 48 KB limit.
#define BLOCK 256
#define TILE  256

// ── kernel ────────────────────────────────────────────────────────────────────
__global__ void eval_volume_kernel(
        const float* __restrict__ means,   // (N, 3)  Gaussian centres
        const float* __restrict__ log_s,   // (N, 3)  log per-axis scales
        const float* __restrict__ quats,   // (N, 4)  [w, x, y, z]
        const float* __restrict__ inten,   // (N,)    raw (pre-softplus) intensity
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int D, int H, int W, int N,
        float scale_min, float mahal_clamp,
        float* __restrict__ out)           // (D*H*W,)
{
    // shared-memory tile for one batch of TILE Gaussians
    __shared__ float s_mu[TILE][3];
    __shared__ float s_ls[TILE][3];
    __shared__ float s_qu[TILE][4];
    __shared__ float s_iv[TILE];       // post-softplus intensity

    const int n   = blockIdx.x * BLOCK + threadIdx.x;
    const int DHW = D * H * W;

    // ── compute this thread's 3-D query coordinate ────────────────────────────
    float cx = 0.f, cy = 0.f, cz = 0.f;
    bool active = (n < DHW);
    if (active) {
        int iz = n / (H * W);
        int ih = (n % (H * W)) / W;
        int iw = n % W;
        // matches VolumeDataset._indices_to_pts: x←iw/W, y←ih/H, z←iz/D
        cx = lo_x + (W > 1 ? (float)iw / (float)(W - 1) : 0.5f) * (hi_x - lo_x);
        cy = lo_y + (H > 1 ? (float)ih / (float)(H - 1) : 0.5f) * (hi_y - lo_y);
        cz = lo_z + (D > 1 ? (float)iz / (float)(D - 1) : 0.5f) * (hi_z - lo_z);
    }

    float acc = 0.f;

    // ── tile loop over all N Gaussians ────────────────────────────────────────
    for (int t0 = 0; t0 < N; t0 += TILE) {
        const int tn = min(TILE, N - t0);   // Gaussians in this tile

        // cooperatively load tile from global → shared memory
        for (int i = threadIdx.x; i < tn; i += BLOCK) {
            const int g = t0 + i;
            s_mu[i][0] = means[g * 3 + 0];
            s_mu[i][1] = means[g * 3 + 1];
            s_mu[i][2] = means[g * 3 + 2];
            s_ls[i][0] = log_s[g * 3 + 0];
            s_ls[i][1] = log_s[g * 3 + 1];
            s_ls[i][2] = log_s[g * 3 + 2];
            s_qu[i][0] = quats[g * 4 + 0];   // w
            s_qu[i][1] = quats[g * 4 + 1];   // x
            s_qu[i][2] = quats[g * 4 + 2];   // y
            s_qu[i][3] = quats[g * 4 + 3];   // z
            // softplus(r) = log(1+exp(r)); numerically stable variant
            float r = inten[g];
            s_iv[i] = (r > 20.f) ? r : log1pf(__expf(r));
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; i++) {

                // ── quaternion → rotation matrix ─────────────────────────────
                float qw = s_qu[i][0], qx = s_qu[i][1];
                float qy = s_qu[i][2], qz = s_qu[i][3];
                float ni = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
                qw *= ni;  qx *= ni;  qy *= ni;  qz *= ni;

                float R00 = 1.f - 2.f*(qy*qy + qz*qz);
                float R01 =       2.f*(qx*qy - qw*qz);
                float R02 =       2.f*(qx*qz + qw*qy);
                float R10 =       2.f*(qx*qy + qw*qz);
                float R11 = 1.f - 2.f*(qx*qx + qz*qz);
                float R12 =       2.f*(qy*qz - qw*qx);
                float R20 =       2.f*(qx*qz - qw*qy);
                float R21 =       2.f*(qy*qz + qw*qx);
                float R22 = 1.f - 2.f*(qx*qx + qy*qy);

                // ── per-axis scales ───────────────────────────────────────────
                float sx = fmaxf(__expf(s_ls[i][0]), scale_min);
                float sy = fmaxf(__expf(s_ls[i][1]), scale_min);
                float sz = fmaxf(__expf(s_ls[i][2]), scale_min);
                float isx = 1.f / sx,  isy = 1.f / sy,  isz = 1.f / sz;

                // ── RD = R * diag(1/s)  (scale each column) ──────────────────
                float D00 = R00*isx,  D01 = R01*isy,  D02 = R02*isz;
                float D10 = R10*isx,  D11 = R11*isy,  D12 = R12*isz;
                float D20 = R20*isx,  D21 = R21*isy,  D22 = R22*isz;

                // ── SI = RD @ RD^T  (6 unique elements of symmetric 3×3) ─────
                float S00 = D00*D00 + D01*D01 + D02*D02;
                float S01 = D00*D10 + D01*D11 + D02*D12;
                float S02 = D00*D20 + D01*D21 + D02*D22;
                float S11 = D10*D10 + D11*D11 + D12*D12;
                float S12 = D10*D20 + D11*D21 + D12*D22;
                float S22 = D20*D20 + D21*D21 + D22*D22;

                // ── Mahalanobis²  = d @ SI @ d ────────────────────────────────
                float d0 = cx - s_mu[i][0];
                float d1 = cy - s_mu[i][1];
                float d2 = cz - s_mu[i][2];
                float mah = d0*(S00*d0 + S01*d1 + S02*d2)
                          + d1*(S01*d0 + S11*d1 + S12*d2)
                          + d2*(S02*d0 + S12*d1 + S22*d2);
                mah = fminf(fmaxf(mah, 0.f), mahal_clamp);

                acc += s_iv[i] * __expf(-0.5f * mah);
            }
        }
        __syncthreads();
    }

    if (active)
        out[n] = fminf(fmaxf(acc, 0.f), 1.f);
}

// ── C++ / Python binding ──────────────────────────────────────────────────────
torch::Tensor reconstruct_volume(
        torch::Tensor means,      // (N, 3) float32 cuda contiguous
        torch::Tensor log_s,      // (N, 3)
        torch::Tensor quats,      // (N, 4)
        torch::Tensor inten,      // (N,)
        float lo_x, float hi_x,
        float lo_y, float hi_y,
        float lo_z, float hi_z,
        int D, int H, int W,
        float scale_min, float mahal_clamp)
{
    TORCH_CHECK(means.is_cuda(),        "means must be a CUDA tensor");
    TORCH_CHECK(means.is_contiguous(),  "means must be contiguous");
    TORCH_CHECK(log_s.is_contiguous(),  "log_s must be contiguous");
    TORCH_CHECK(quats.is_contiguous(),  "quats must be contiguous");
    TORCH_CHECK(inten.is_contiguous(),  "inten must be contiguous");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");

    const int N   = static_cast<int>(means.size(0));
    const int DHW = D * H * W;
    auto out = torch::empty({DHW}, means.options());

    const int blocks = (DHW + BLOCK - 1) / BLOCK;
    eval_volume_kernel<<<blocks, BLOCK>>>(
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        inten.data_ptr<float>(),
        lo_x, hi_x,
        lo_y, hi_y,
        lo_z, hi_z,
        D, H, W, N,
        scale_min, mahal_clamp,
        out.data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return out;    // (DHW,) float32, values in [0,1]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reconstruct_volume", &reconstruct_volume,
          "Evaluate Gaussian cloud at every voxel centre. "
          "Returns (D*H*W,) float32 CUDA tensor in [0,1].");
}
