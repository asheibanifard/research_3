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


// ─── Forward kernel ───────────────────────────────────────────────────────────
/*
 * One thread per sample point m ∈ [0, M).
 * For each m, accumulates contributions from all N Gaussians:
 *   out[m] = Σ_k  gain[k] · inten[k] · exp(-½ · mahal_k)
 * where mahal_k = uᵀ diag(1/s²) u,  u = Rᵀ (pts[m] - means[k]).
 * Gaussians with mahal_k >= mahal_clamp are skipped (exp ≈ 0).
 */
__global__ void gaussian_forward_kernel(
        const float* __restrict__ pts,      /* (M,3)  query coordinates      */
        const float* __restrict__ means,    /* (N,3)  Gaussian centres        */
        const float* __restrict__ log_s,    /* (N,3)  log per-axis std-devs   */
        const float* __restrict__ quats,    /* (N,4)  quaternions [w,x,y,z]   */
        const float* __restrict__ gain,     /* (N,)   amplitude gate (=1)     */
        const float* __restrict__ inten,    /* (N,)   softplus intensity       */
        float scale_min,
        float mahal_clamp,
        int M, int N,
        float* __restrict__ out             /* (M,)   output field values      */
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const float px = pts[m*3+0];
    const float py = pts[m*3+1];
    const float pz = pts[m*3+2];
    float acc = 0.f;

    for (int k = 0; k < N; ++k) {
        float qn[4], R[9];
        float inv_norm;
        normalize_quat(quats + k*4, qn, inv_norm);
        quat_to_rotmat(qn, R);

        const float s0 = fmaxf(expf(log_s[k*3+0]), scale_min);
        const float s1 = fmaxf(expf(log_s[k*3+1]), scale_min);
        const float s2 = fmaxf(expf(log_s[k*3+2]), scale_min);

        const float diff[3] = {px - means[k*3+0], py - means[k*3+1], pz - means[k*3+2]};
        float u[3];
        mat_t_vec(R, diff, u);  /* u = Rᵀ diff */

        const float mahal = u[0]*u[0]/(s0*s0) + u[1]*u[1]/(s1*s1) + u[2]*u[2]/(s2*s2);
        if (mahal >= mahal_clamp) continue;

        acc += gain[k] * inten[k] * expf(-0.5f * mahal);
    }
    out[m] = acc;
}


// ─── Backward kernel ──────────────────────────────────────────────────────────
/*
 * One thread per sample point m ∈ [0, M).
 * For each m and each Gaussian k, accumulates:
 *
 *   grad_inten[k] += g_out[m] · gain[k] · w          (∂L/∂inten, post-softplus)
 *   grad_gain[k]  += g_out[m] · inten[k] · w
 *
 *   let grad_factor = g_out[m] · gain[k] · inten[k] · w
 *   let tmp[i] = u[i] / s[i]²                         (Σ⁻¹ direction in local frame)
 *   let adiff  = R · tmp                               (Σ⁻¹ diff in world frame)
 *
 *   grad_means[k]  += grad_factor · adiff
 *   grad_log_s[k,i]+= grad_factor · u[i]² / s[i]²    (if not clamped)
 *   grad_R[i,j]    = -grad_factor · diff[i] · tmp[j]
 *   grad_quats[k]  += backprop(grad_R through quat_to_rotmat + normalise)
 *
 * atomicAdd is required because multiple threads (sample points) may
 * simultaneously update the same Gaussian's gradient.
 */
__global__ void gaussian_backward_kernel(
        const float* __restrict__ grad_out,  /* (M,)   upstream gradient       */
        const float* __restrict__ pts,       /* (M,3)                          */
        const float* __restrict__ means,     /* (N,3)                          */
        const float* __restrict__ log_s,     /* (N,3)                          */
        const float* __restrict__ quats,     /* (N,4)                          */
        const float* __restrict__ gain,      /* (N,)                           */
        const float* __restrict__ inten,     /* (N,)   softplus value          */
        float scale_min,
        float mahal_clamp,
        int M, int N,
        float* __restrict__ grad_means,      /* (N,3)                          */
        float* __restrict__ grad_log_s,      /* (N,3)                          */
        float* __restrict__ grad_quats,      /* (N,4)                          */
        float* __restrict__ grad_gain,       /* (N,)                           */
        float* __restrict__ grad_inten       /* (N,)   w.r.t. softplus value   */
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const float g_out = grad_out[m];
    if (g_out == 0.f) return;

    const float px = pts[m*3+0];
    const float py = pts[m*3+1];
    const float pz = pts[m*3+2];

    for (int k = 0; k < N; ++k) {
        float qn[4], R[9];
        float inv_norm;
        normalize_quat(quats + k*4, qn, inv_norm);
        quat_to_rotmat(qn, R);

        const float rs0 = expf(log_s[k*3+0]);
        const float rs1 = expf(log_s[k*3+1]);
        const float rs2 = expf(log_s[k*3+2]);
        /* Track whether each axis was clamped (no gradient through clamp). */
        const bool cl0 = rs0 <= scale_min;
        const bool cl1 = rs1 <= scale_min;
        const bool cl2 = rs2 <= scale_min;
        const float s0 = fmaxf(rs0, scale_min);
        const float s1 = fmaxf(rs1, scale_min);
        const float s2 = fmaxf(rs2, scale_min);
        const float is2_0 = 1.f/(s0*s0);
        const float is2_1 = 1.f/(s1*s1);
        const float is2_2 = 1.f/(s2*s2);

        const float diff[3] = {px - means[k*3+0], py - means[k*3+1], pz - means[k*3+2]};
        float u[3];
        mat_t_vec(R, diff, u);  /* u = Rᵀ diff */

        const float mahal = u[0]*u[0]*is2_0 + u[1]*u[1]*is2_1 + u[2]*u[2]*is2_2;
        if (mahal >= mahal_clamp) continue;

        const float g    = gain[k];
        const float v    = inten[k];
        const float w    = expf(-0.5f * mahal);

        /* Chain: ∂L/∂w = g_out · g · v,  ∂w/∂mahal = -0.5·w,
         * ∂mahal/∂diff = 2·Σ⁻¹·diff  →  grad_factor absorbs -0.5·(-2) = 1. */
        const float grad_factor = g_out * g * v * w;

        /* ── ∂L/∂inten and ∂L/∂gain ───────────────────────────────────────── */
        atomicAdd(&grad_inten[k], g_out * g * w);   /* w.r.t. post-softplus v */
        atomicAdd(&grad_gain[k],  g_out * v * w);

        /* ── ∂L/∂means ────────────────────────────────────────────────────── */
        const float tmp[3] = {u[0]*is2_0, u[1]*is2_1, u[2]*is2_2};
        float adiff[3];
        mat_vec(R, tmp, adiff);  /* adiff = R tmp = Σ⁻¹ diff (world frame) */

        atomicAdd(&grad_means[k*3+0], grad_factor * adiff[0]);
        atomicAdd(&grad_means[k*3+1], grad_factor * adiff[1]);
        atomicAdd(&grad_means[k*3+2], grad_factor * adiff[2]);

        /* ── ∂L/∂log_s ────────────────────────────────────────────────────── */
        if (!cl0) atomicAdd(&grad_log_s[k*3+0], grad_factor * u[0]*u[0]*is2_0);
        if (!cl1) atomicAdd(&grad_log_s[k*3+1], grad_factor * u[1]*u[1]*is2_1);
        if (!cl2) atomicAdd(&grad_log_s[k*3+2], grad_factor * u[2]*u[2]*is2_2);

        /* ── ∂L/∂quats (via ∂L/∂R) ─────────────────────────────────────────
         * grad_R[i,j] = ∂L/∂R[i,j] = -grad_factor · diff[i] · tmp[j]      */
        float grad_R[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                grad_R[i*3+j] = -grad_factor * diff[i] * tmp[j];

        float grad_qraw[4];
        quat_grad_from_rot_grad(grad_R, qn, inv_norm, grad_qraw);

        atomicAdd(&grad_quats[k*4+0], grad_qraw[0]);
        atomicAdd(&grad_quats[k*4+1], grad_qraw[1]);
        atomicAdd(&grad_quats[k*4+2], grad_qraw[2]);
        atomicAdd(&grad_quats[k*4+3], grad_qraw[3]);
    }
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

    const int threads = 256;
    const int blocks  = (M + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gaussian_forward_kernel<<<blocks, threads, 0, stream>>>(
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

    const int threads = 256;
    const int blocks  = (M + threads - 1) / threads;
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
