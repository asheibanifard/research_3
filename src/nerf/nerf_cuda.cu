#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────────────────────────────────────
// Fused positional-encoding forward
//
// Computes γ(x) = [x, sin(f₀·x), cos(f₀·x), …, sin(f_{L-1}·x), cos(f_{L-1}·x)]
// in a single pass over global memory instead of the four separate ops
// (unsqueeze, mul, sin+cos cat, reshape, cat) that PyTorch would issue.
//
// Output layout (per row, matches Python PositionalEncoding):
//   [x₀, x₁, x₂,                          ← C passthrough values
//    sin(f₀x₀), …, sin(f_{L-1}x₀),         ← L sin values for coord 0
//    cos(f₀x₀), …, cos(f_{L-1}x₀),         ← L cos values for coord 0
//    sin(f₀x₁), …, sin(f_{L-1}x₁),         ← coord 1
//    cos(f₀x₁), …, cos(f_{L-1}x₁),
//    sin(f₀x₂), …, sin(f_{L-1}x₂),         ← coord 2
//    cos(f₀x₂), …, cos(f_{L-1}x₂)]
//
// Each thread writes one output element.
// Frequency array (L ≤ 32 floats) is cached in shared memory.
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void pe_forward_kernel(
    const scalar_t* __restrict__ x,      // (N, C)
    const float*    __restrict__ freqs,  // (L,)   f_l = 2^l
    scalar_t*       __restrict__ out,    // (N, out_dim)  out_dim = C + C*2L
    int N, int C, int L
) {
    // Cache frequencies in shared memory (L ≤ 32 → at most 128 bytes)
    extern __shared__ float s_freqs[];
    if (threadIdx.x < L)
        s_freqs[threadIdx.x] = freqs[threadIdx.x];
    __syncthreads();

    int out_dim = C + C * 2 * L;
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_dim) return;

    int n   = idx / out_dim;   // sample index
    int pos = idx % out_dim;   // output channel index

    if (pos < C) {
        // passthrough: copy raw coordinate
        out[idx] = x[n * C + pos];
    } else {
        // encoding region: pos ∈ [C, C + C*2L)
        int enc_pos = pos - C;          // 0 … C*2L - 1
        int c       = enc_pos / (2*L);  // which input coordinate
        int fl      = enc_pos % (2*L);  // index within [sin₀…sin_{L-1}, cos₀…cos_{L-1}]
        int l       = fl % L;           // frequency band index
        float val   = (float)x[n * C + c] * s_freqs[l];
        out[idx]    = (scalar_t)(fl < L ? sinf(val) : cosf(val));
    }
}

// double specialisation (keeps full precision)
template <>
__global__ void pe_forward_kernel<double>(
    const double* __restrict__ x,
    const float*  __restrict__ freqs,
    double*       __restrict__ out,
    int N, int C, int L
) {
    extern __shared__ float s_freqs[];
    if (threadIdx.x < L)
        s_freqs[threadIdx.x] = freqs[threadIdx.x];
    __syncthreads();

    int out_dim = C + C * 2 * L;
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_dim) return;

    int n   = idx / out_dim;
    int pos = idx % out_dim;

    if (pos < C) {
        out[idx] = x[n * C + pos];
    } else {
        int enc_pos = pos - C;
        int c       = enc_pos / (2*L);
        int fl      = enc_pos % (2*L);
        int l       = fl % L;
        double val  = x[n * C + c] * (double)freqs[l];
        out[idx]    = (fl < L) ? sin(val) : cos(val);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Fused positional-encoding backward
//
// grad_x[n,c] =  grad_out[n,c]                                (passthrough)
//              + Σ_l  grad_out[n, C + c*2L + l]    ·  cos(f_l·x[n,c]) · f_l   (∂sin/∂x)
//              + Σ_l  grad_out[n, C + c*2L + L + l] · (−sin(f_l·x[n,c])) · f_l (∂cos/∂x)
//
// Each thread computes the full gradient for one (n, c) pair by iterating
// over all L frequency bands — no atomics needed.
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void pe_backward_kernel(
    const scalar_t* __restrict__ grad_out,  // (N, out_dim)
    const scalar_t* __restrict__ x,         // (N, C)
    const float*    __restrict__ freqs,     // (L,)
    scalar_t*       __restrict__ grad_x,    // (N, C)
    int N, int C, int L
) {
    extern __shared__ float s_freqs[];
    if (threadIdx.x < L)
        s_freqs[threadIdx.x] = freqs[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one (n,c) pair
    if (idx >= N * C) return;

    int n       = idx / C;
    int c       = idx % C;
    int out_dim = C + C * 2 * L;

    float x_val = (float)x[idx];
    // passthrough gradient
    float grad  = (float)grad_out[n * out_dim + c];

    // accumulate gradients from the sin and cos encoding channels
    int base = n * out_dim + C + c * 2 * L;
    for (int l = 0; l < L; ++l) {
        float f   = s_freqs[l];
        float val = x_val * f;
        grad += (float)grad_out[base + l]     *  cosf(val) * f;  // d sin / d x
        grad += (float)grad_out[base + L + l] * -sinf(val) * f;  // d cos / d x
    }

    grad_x[idx] = (scalar_t)grad;
}

template <>
__global__ void pe_backward_kernel<double>(
    const double* __restrict__ grad_out,
    const double* __restrict__ x,
    const float*  __restrict__ freqs,
    double*       __restrict__ grad_x,
    int N, int C, int L
) {
    extern __shared__ float s_freqs[];
    if (threadIdx.x < L)
        s_freqs[threadIdx.x] = freqs[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n       = idx / C;
    int c       = idx % C;
    int out_dim = C + C * 2 * L;

    double x_val = x[idx];
    double grad  = grad_out[n * out_dim + c];

    int base = n * out_dim + C + c * 2 * L;
    for (int l = 0; l < L; ++l) {
        double f   = (double)freqs[l];
        double val = x_val * f;
        grad += grad_out[base + l]     *  cos(val) * f;
        grad += grad_out[base + L + l] * -sin(val) * f;
    }

    grad_x[idx] = grad;
}


// ── Python-visible entry points ───────────────────────────────────────────────

torch::Tensor pe_forward(
    torch::Tensor x,      // (N, C) float/double, CUDA
    torch::Tensor freqs   // (L,)   float,         CUDA
) {
    TORCH_CHECK(x.is_cuda(),     "x must be a CUDA tensor");
    TORCH_CHECK(freqs.is_cuda(), "freqs must be a CUDA tensor");

    int N = x.size(0);
    int C = x.size(1);
    int L = freqs.size(0);

    auto out = torch::empty({N, C + C * 2 * L}, x.options());

    const int threads  = 256;
    const int out_size = N * (C + C * 2 * L);
    const int blocks   = (out_size + threads - 1) / threads;
    const int smem     = L * sizeof(float);

    // freqs is always float32 regardless of x dtype
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pe_forward", ([&] {
        pe_forward_kernel<scalar_t><<<blocks, threads, smem>>>(
            x.data_ptr<scalar_t>(),
            freqs.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            N, C, L
        );
    }));
    return out;
}

torch::Tensor pe_backward(
    torch::Tensor grad_out,  // (N, out_dim)
    torch::Tensor x,         // (N, C)
    torch::Tensor freqs      // (L,)
) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");

    int N = x.size(0);
    int C = x.size(1);
    int L = freqs.size(0);

    auto grad_x = torch::empty_like(x);

    const int threads = 256;
    const int blocks  = (N * C + threads - 1) / threads;
    const int smem    = L * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pe_backward", ([&] {
        pe_backward_kernel<scalar_t><<<blocks, threads, smem>>>(
            grad_out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            freqs.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            N, C, L
        );
    }));
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &pe_forward,  "Fused positional-encoding forward  (CUDA)");
    m.def("backward", &pe_backward, "Fused positional-encoding backward (CUDA)");
}
