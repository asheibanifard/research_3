/*
 * 3dgs_cuda.cu — Fused CUDA kernel for 3-D Gaussian field evaluation
 *
 * Implements forward and backward passes for:
 *   f(x) = Σ_k  gain_k · inten_k · exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))
 *
 * where inten is passed as the post-softplus value (softplus applied in Python
 * before calling kernel.forward), so the kernel treats it as a plain positive float.
 * The softplus chain-rule is applied in Python after kernel.backward returns.
 *
 * Covariance:  Σ_k = R_k diag(s_k²) R_kᵀ,   parameterised by [w,x,y,z] + log_s.
 *
 * Both kernels are one-thread-per-sample-point. Gradients w.r.t. Gaussian
 * parameters are accumulated via atomicAdd (safe for concurrent thread access
 * since multiple sample points may hit the same Gaussian).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ─── Device-side helper functions ────────────────────────────────────────────

/* Normalize a raw quaternion q → qn and store inv_norm = 1/||q||.
 * inv_norm is saved so that the gradient through normalisation can be
 * computed without a second rsqrt in the backward pass. */
__device__ __forceinline__ void normalize_quat(
        const float* __restrict__ q,
        float* qn,
        float& inv_norm)
{
    float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    inv_norm  = rsqrtf(fmaxf(n2, 1e-12f));
    qn[0] = q[0] * inv_norm;
    qn[1] = q[1] * inv_norm;
    qn[2] = q[2] * inv_norm;
    qn[3] = q[3] * inv_norm;
}

/* Closed-form Rodrigues formula: unit quaternion [w,x,y,z] → 3×3 rotation.
 * Row-major flat layout: R[i*3+j] = R_{ij}. */
__device__ __forceinline__ void quat_to_rotmat(const float* qn, float* R)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    R[0] = 1.f - 2.f*(y*y + z*z);
    R[1] = 2.f*(x*y - w*z);
    R[2] = 2.f*(x*z + w*y);
    R[3] = 2.f*(x*y + w*z);
    R[4] = 1.f - 2.f*(x*x + z*z);
    R[5] = 2.f*(y*z - w*x);
    R[6] = 2.f*(x*z - w*y);
    R[7] = 2.f*(y*z + w*x);
    R[8] = 1.f - 2.f*(x*x + y*y);
}

/* u = Rᵀ v   (transposed matrix–vector product) */
__device__ __forceinline__ void mat_t_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[3]*v[1] + R[6]*v[2];
    u[1] = R[1]*v[0] + R[4]*v[1] + R[7]*v[2];
    u[2] = R[2]*v[0] + R[5]*v[1] + R[8]*v[2];
}

/* u = R v */
__device__ __forceinline__ void mat_vec(
        const float* __restrict__ R,
        const float* __restrict__ v,
        float* u)
{
    u[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    u[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    u[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

/* Backprop through quat_to_rotmat (Rodrigues) and through normalize_quat.
 *
 * Given:
 *   grad_R[9]  — ∂L/∂R in flat row-major layout
 *   qn[4]      — normalised quaternion [w,x,y,z]
 *   inv_norm   — 1/||q_raw|| from normalize_quat
 *
 * Computes:
 *   grad_qraw[4] — ∂L/∂q_raw
 *
 * Two steps:
 *   1. ∂L/∂qn  by summing ∂R[i,j]/∂qn · grad_R[i,j] over all (i,j).
 *   2. Backprop through normalisation: qn = q_raw·inv_norm
 *      ∂L/∂q_raw = inv_norm · (∂L/∂qn  −  qn · (qnᵀ · ∂L/∂qn))
 */
__device__ __forceinline__ void quat_grad_from_rot_grad(
        const float* __restrict__ grad_R,
        const float* __restrict__ qn,
        float inv_norm,
        float* grad_qraw)
{
    const float w=qn[0], x=qn[1], y=qn[2], z=qn[3];
    /* Index helpers: g(i,j) = grad_R[i*3+j] */
    #define G(i,j) grad_R[(i)*3+(j)]

    float gw =  2.f*( -z*G(0,1) + y*G(0,2) + z*G(1,0) - x*G(1,2) - y*G(2,0) + x*G(2,1) );
    float gx =  2.f*(  y*G(0,1) + z*G(0,2) + y*G(1,0) - 2.f*x*G(1,1) - w*G(1,2) + z*G(2,0) + w*G(2,1) - 2.f*x*G(2,2) );
    float gy =  2.f*( -2.f*y*G(0,0) + x*G(0,1) + w*G(0,2) + x*G(1,0) + z*G(1,2) - w*G(2,0) + z*G(2,1) - 2.f*y*G(2,2) );
    float gz =  2.f*( -2.f*z*G(0,0) - w*G(0,1) + x*G(0,2) + w*G(1,0) - 2.f*z*G(1,1) + y*G(1,2) + x*G(2,0) + y*G(2,1) );
    #undef G

    /* Backprop through normalisation: remove the component along qn. */
    float dot = gw*w + gx*x + gy*y + gz*z;
    grad_qraw[0] = inv_norm * (gw - w*dot);
    grad_qraw[1] = inv_norm * (gx - x*dot);
    grad_qraw[2] = inv_norm * (gy - y*dot);
    grad_qraw[3] = inv_norm * (gz - z*dot);
}


// ─── Forward kernel (shared-memory tiled) ────────────────────────────────────
/*
 * Each block of BLOCK_FWD threads handles BLOCK_FWD sample points.
 * Gaussians are loaded cooperatively into shared memory TILE_FWD at a time,
 * reducing global-memory reads by a factor of BLOCK_FWD vs the naïve kernel.
 *
 * Shared-mem per block = TILE_FWD × (3+3+4+1+1) × 4 = TILE_FWD × 48 B.
 * With TILE_FWD=256: 12 288 B — well within the 48 KB limit.
 */
#define BLOCK_FWD 256
#define TILE_FWD  256

__global__ void gaussian_forward_kernel(
        const float* __restrict__ pts,
        const float* __restrict__ means,
        const float* __restrict__ log_s,
        const float* __restrict__ quats,
        const float* __restrict__ gain,
        const float* __restrict__ inten,
        float scale_min, float mahal_clamp,
        int M, int N,
        float* __restrict__ out)
{
    __shared__ float s_mu[TILE_FWD][3];
    __shared__ float s_ls[TILE_FWD][3];
    __shared__ float s_qu[TILE_FWD][4];
    __shared__ float s_ga[TILE_FWD];
    __shared__ float s_iv[TILE_FWD];

    const int m      = blockIdx.x * BLOCK_FWD + threadIdx.x;
    const bool active = (m < M);

    float px=0.f, py=0.f, pz=0.f;
    if (active) { px = pts[m*3+0]; py = pts[m*3+1]; pz = pts[m*3+2]; }
    float acc = 0.f;

    for (int t0 = 0; t0 < N; t0 += TILE_FWD) {
        const int tn = min(TILE_FWD, N - t0);

        /* cooperative load: each thread loads one Gaussian (if in range) */
        for (int i = threadIdx.x; i < tn; i += BLOCK_FWD) {
            const int k = t0 + i;
            s_mu[i][0] = means[k*3+0]; s_mu[i][1] = means[k*3+1]; s_mu[i][2] = means[k*3+2];
            s_ls[i][0] = log_s[k*3+0]; s_ls[i][1] = log_s[k*3+1]; s_ls[i][2] = log_s[k*3+2];
            s_qu[i][0] = quats[k*4+0]; s_qu[i][1] = quats[k*4+1];
            s_qu[i][2] = quats[k*4+2]; s_qu[i][3] = quats[k*4+3];
            s_ga[i] = gain[k];
            s_iv[i] = inten[k];
        }
        __syncthreads();

        if (active) {
            for (int i = 0; i < tn; i++) {
                float qn[4], R[9], inv_norm;
                normalize_quat(s_qu[i], qn, inv_norm);
                quat_to_rotmat(qn, R);

                const float s0 = fmaxf(expf(s_ls[i][0]), scale_min);
                const float s1 = fmaxf(expf(s_ls[i][1]), scale_min);
                const float s2 = fmaxf(expf(s_ls[i][2]), scale_min);

                const float diff[3] = {px-s_mu[i][0], py-s_mu[i][1], pz-s_mu[i][2]};
                float u[3];
                mat_t_vec(R, diff, u);

                const float mahal = u[0]*u[0]/(s0*s0) + u[1]*u[1]/(s1*s1) + u[2]*u[2]/(s2*s2);
                if (mahal >= mahal_clamp) continue;

                acc += s_ga[i] * s_iv[i] * expf(-0.5f * mahal);
            }
        }
        __syncthreads();
    }

    if (active) out[m] = acc;
}


// ─── Backward kernel (transposed: one thread per Gaussian, no atomicAdd) ─────
/*
 * Transposed layout eliminates all atomicAdd contention.
 *
 * Original layout:  M threads × N Gaussians → M×N×11 atomic writes.
 * Transposed layout: N threads × M points  → each thread writes its own
 *                    Gaussian gradient exactly once at the end.
 *
 * Each thread k:
 *   1. Loads its Gaussian parameters once (R, scales, gain, inten).
 *   2. Loops over all M sample points accumulating gradient contributions
 *      into local registers (gm, gls, gR, …).
 *   3. Writes the accumulated gradients to global memory once — no atomics.
 *
 * quat_grad_from_rot_grad is linear in grad_R, so accumulating grad_R across
 * all M points and applying it once is mathematically identical to applying it
 * per-point and summing (which the original kernel did via atomicAdd).
 */
__global__ void gaussian_backward_kernel(
        const float* __restrict__ grad_out,
        const float* __restrict__ pts,
        const float* __restrict__ means,
        const float* __restrict__ log_s,
        const float* __restrict__ quats,
        const float* __restrict__ gain,
        const float* __restrict__ inten,
        float scale_min, float mahal_clamp,
        int M, int N,
        float* __restrict__ grad_means,
        float* __restrict__ grad_log_s,
        float* __restrict__ grad_quats,
        float* __restrict__ grad_gain,
        float* __restrict__ grad_inten)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    /* ── Load this Gaussian's data once ──────────────────────────────────── */
    float qn[4], R[9], inv_norm;
    normalize_quat(quats + k*4, qn, inv_norm);
    quat_to_rotmat(qn, R);

    const float rs0 = expf(log_s[k*3+0]);
    const float rs1 = expf(log_s[k*3+1]);
    const float rs2 = expf(log_s[k*3+2]);
    const bool  cl0 = rs0 <= scale_min;
    const bool  cl1 = rs1 <= scale_min;
    const bool  cl2 = rs2 <= scale_min;
    const float s0  = fmaxf(rs0, scale_min);
    const float s1  = fmaxf(rs1, scale_min);
    const float s2  = fmaxf(rs2, scale_min);
    const float is2_0 = 1.f/(s0*s0);
    const float is2_1 = 1.f/(s1*s1);
    const float is2_2 = 1.f/(s2*s2);
    const float g_k = gain[k];
    const float v_k = inten[k];
    const float mx  = means[k*3+0], my = means[k*3+1], mz = means[k*3+2];

    /* ── Local accumulators (registers, no atomics) ───────────────────────── */
    float gm0=0.f, gm1=0.f, gm2=0.f;
    float gls0=0.f, gls1=0.f, gls2=0.f;
    float gR[9] = {0.f,0.f,0.f, 0.f,0.f,0.f, 0.f,0.f,0.f};
    float g_gain_acc=0.f, g_inten_acc=0.f;

    /* ── Loop over all M sample points ───────────────────────────────────── */
    for (int m = 0; m < M; ++m) {
        const float g_out = grad_out[m];
        if (g_out == 0.f) continue;

        const float diff[3] = {pts[m*3+0]-mx, pts[m*3+1]-my, pts[m*3+2]-mz};
        float u[3];
        mat_t_vec(R, diff, u);

        const float mahal = u[0]*u[0]*is2_0 + u[1]*u[1]*is2_1 + u[2]*u[2]*is2_2;
        if (mahal >= mahal_clamp) continue;

        const float w           = expf(-0.5f * mahal);
        const float grad_factor = g_out * g_k * v_k * w;

        g_gain_acc  += g_out * v_k * w;
        g_inten_acc += g_out * g_k * w;

        const float tmp[3] = {u[0]*is2_0, u[1]*is2_1, u[2]*is2_2};
        float adiff[3];
        mat_vec(R, tmp, adiff);

        gm0 += grad_factor * adiff[0];
        gm1 += grad_factor * adiff[1];
        gm2 += grad_factor * adiff[2];

        if (!cl0) gls0 += grad_factor * u[0]*u[0]*is2_0;
        if (!cl1) gls1 += grad_factor * u[1]*u[1]*is2_1;
        if (!cl2) gls2 += grad_factor * u[2]*u[2]*is2_2;

        /* accumulate ∂L/∂R across all points; apply quat backprop once below */
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                gR[i*3+j] -= grad_factor * diff[i] * tmp[j];
    }

    /* ── Write results once — no atomics ─────────────────────────────────── */
    grad_means[k*3+0] = gm0;
    grad_means[k*3+1] = gm1;
    grad_means[k*3+2] = gm2;
    grad_log_s[k*3+0] = gls0;
    grad_log_s[k*3+1] = gls1;
    grad_log_s[k*3+2] = gls2;
    grad_gain[k]  = g_gain_acc;
    grad_inten[k] = g_inten_acc;

    float grad_qraw[4];
    quat_grad_from_rot_grad(gR, qn, inv_norm, grad_qraw);
    grad_quats[k*4+0] = grad_qraw[0];
    grad_quats[k*4+1] = grad_qraw[1];
    grad_quats[k*4+2] = grad_qraw[2];
    grad_quats[k*4+3] = grad_qraw[3];
}


// ─── Python-visible entry points ──────────────────────────────────────────────

torch::Tensor gaussian_forward(
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp)
{
    TORCH_CHECK(pts.is_cuda()   && pts.is_contiguous(),   "pts must be contiguous CUDA float32");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be contiguous CUDA float32");
    TORCH_CHECK(log_s.is_cuda() && log_s.is_contiguous(), "log_s must be contiguous CUDA float32");
    TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(), "quats must be contiguous CUDA float32");
    TORCH_CHECK(gain.is_cuda()  && gain.is_contiguous(),  "gain must be contiguous CUDA float32");
    TORCH_CHECK(inten.is_cuda() && inten.is_contiguous(), "inten must be contiguous CUDA float32");
    TORCH_CHECK(pts.scalar_type()   == torch::kFloat32, "pts must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(log_s.scalar_type() == torch::kFloat32, "log_s must be float32");
    TORCH_CHECK(quats.scalar_type() == torch::kFloat32, "quats must be float32");
    TORCH_CHECK(gain.scalar_type()  == torch::kFloat32, "gain must be float32");
    TORCH_CHECK(inten.scalar_type() == torch::kFloat32, "inten must be float32");
    TORCH_CHECK(pts.dim() == 2   && pts.size(1) == 3,   "pts must be (M, 3)");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
    TORCH_CHECK(log_s.dim() == 2 && log_s.size(1) == 3, "log_s must be (N, 3)");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be (N, 4)");
    TORCH_CHECK(gain.dim()  == 1, "gain must be (N,)");
    TORCH_CHECK(inten.dim() == 1, "inten must be (N,)");

    const int M = static_cast<int>(pts.size(0));
    const int N = static_cast<int>(means.size(0));
    auto out = torch::zeros({M}, pts.options());

    const int blocks = (M + BLOCK_FWD - 1) / BLOCK_FWD;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_forward_kernel<<<blocks, BLOCK_FWD, 0, stream>>>(
        pts.data_ptr<float>(),
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        gain.data_ptr<float>(),
        inten.data_ptr<float>(),
        scale_min, mahal_clamp, M, N,
        out.data_ptr<float>()
    );
    return out;
}


py::tuple gaussian_backward(
        torch::Tensor grad_out,
        torch::Tensor pts,
        torch::Tensor means,
        torch::Tensor log_s,
        torch::Tensor quats,
        torch::Tensor gain,
        torch::Tensor inten,
        float scale_min,
        float mahal_clamp)
{
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous(), "grad_out must be contiguous CUDA");
    TORCH_CHECK(pts.is_cuda()   && pts.is_contiguous(),   "pts must be contiguous CUDA");
    TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be contiguous CUDA");
    TORCH_CHECK(log_s.is_cuda() && log_s.is_contiguous(), "log_s must be contiguous CUDA");
    TORCH_CHECK(quats.is_cuda() && quats.is_contiguous(), "quats must be contiguous CUDA");
    TORCH_CHECK(gain.is_cuda()  && gain.is_contiguous(),  "gain must be contiguous CUDA");
    TORCH_CHECK(inten.is_cuda() && inten.is_contiguous(), "inten must be contiguous CUDA");

    const int M = static_cast<int>(pts.size(0));
    const int N = static_cast<int>(means.size(0));

    auto grad_means = torch::zeros_like(means);
    auto grad_log_s = torch::zeros_like(log_s);
    auto grad_quats = torch::zeros_like(quats);
    auto grad_gain  = torch::zeros_like(gain);
    auto grad_inten = torch::zeros_like(inten);

    /* Transposed kernel: one thread per Gaussian (N), not per point (M). */
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_out.data_ptr<float>(),
        pts.data_ptr<float>(),
        means.data_ptr<float>(),
        log_s.data_ptr<float>(),
        quats.data_ptr<float>(),
        gain.data_ptr<float>(),
        inten.data_ptr<float>(),
        scale_min, mahal_clamp, M, N,
        grad_means.data_ptr<float>(),
        grad_log_s.data_ptr<float>(),
        grad_quats.data_ptr<float>(),
        grad_gain.data_ptr<float>(),
        grad_inten.data_ptr<float>()
    );

    return py::make_tuple(grad_means, grad_log_s, grad_quats, grad_gain, grad_inten);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &gaussian_forward,  "Gaussian splatting forward (CUDA)");
    m.def("backward", &gaussian_backward, "Gaussian splatting backward (CUDA)");
}
