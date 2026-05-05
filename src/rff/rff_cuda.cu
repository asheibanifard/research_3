#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ─────────────────────────────────────────────────────────────────────────────
// Fused Random Fourier Feature forward
//
// Computes the RFF encoding (Tancik et al., 2020, Eq. 4):
//   γ(x) = [ cos(2π·B[0]·x), …, cos(2π·B[m-1]·x),
//             sin(2π·B[0]·x), …, sin(2π·B[m-1]·x) ]  ∈ ℝ^{2m}
//
// where B ∈ ℝ^{m×C} is the fixed frequency matrix (B ~ N(0,σ²I)).
// Passthrough concatenation  [x, enc]  is handled in Python.
//
// Input / output shapes:
//   x   : (N, C)   float or double, CUDA
//   B   : (m, C)   float32,         CUDA  (always float — σ is a global scale)
//   out : (N, 2m)  same dtype as x
//
// Output layout per row:
//   [cos(2π·B[0]·x), …, cos(2π·B[m-1]·x),     ← m cos values
//    sin(2π·B[0]·x), …, sin(2π·B[m-1]·x)]      ← m sin values
//
// Thread assignment: one thread per output element.
// The full B matrix (m×C floats) is cached in shared memory.
//   m=512, C=3 → 6 144 bytes — well within the 48 KB limit.
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void rff_forward_kernel(
    const scalar_t* __restrict__ x,    // (N, C)
    const float*    __restrict__ B,    // (m, C)
    scalar_t*       __restrict__ out,  // (N, 2m)
    int N, int C, int m
) {
    extern __shared__ float s_B[];
    // Load B into shared memory (each thread loads multiple elements if needed)
    for (int i = threadIdx.x; i < m * C; i += blockDim.x)
        s_B[i] = B[i];
    __syncthreads();

    int out_dim = 2 * m;
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_dim) return;

    int n   = idx / out_dim;   // sample index
    int pos = idx % out_dim;   // output channel: [0, m) = cos, [m, 2m) = sin

    int j       = pos % m;     // frequency feature index
    bool is_sin = pos >= m;

    // Dot product:  2π · B[j] · x[n]
    float dot = 0.0f;
    for (int c = 0; c < C; ++c)
        dot += s_B[j * C + c] * (float)x[n * C + c];
    dot *= 2.0f * M_PI;

    out[idx] = (scalar_t)(is_sin ? sinf(dot) : cosf(dot));
}

// Double specialisation (keeps full precision for both trig and dot product)
template <>
__global__ void rff_forward_kernel<double>(
    const double* __restrict__ x,
    const float*  __restrict__ B,
    double*       __restrict__ out,
    int N, int C, int m
) {
    extern __shared__ float s_B[];
    for (int i = threadIdx.x; i < m * C; i += blockDim.x)
        s_B[i] = B[i];
    __syncthreads();

    int out_dim = 2 * m;
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_dim) return;

    int n       = idx / out_dim;
    int pos     = idx % out_dim;
    int j       = pos % m;
    bool is_sin = pos >= m;

    double dot = 0.0;
    for (int c = 0; c < C; ++c)
        dot += (double)s_B[j * C + c] * x[n * C + c];
    dot *= 2.0 * M_PI;

    out[idx] = is_sin ? sin(dot) : cos(dot);
}


// ─────────────────────────────────────────────────────────────────────────────
// Fused Random Fourier Feature backward
//
// Given grad_out ∈ ℝ^{N×2m} (gradient w.r.t. the encoding output),
// computes grad_x ∈ ℝ^{N×C}:
//
//   ∂L/∂x[n,c] = Σ_j  grad_out[n,j]   · (−sin(2π·B[j]·x[n])) · 2π·B[j,c]
//              + Σ_j  grad_out[n,m+j]  ·    cos(2π·B[j]·x[n])  · 2π·B[j,c]
//
// The passthrough gradient (identity) is added by PyTorch's autograd
// when include_input=True (the cat operation).
//
// Each thread handles one (n, c) pair and iterates over all m features.
// No atomics needed — each output is independent.
// ─────────────────────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void rff_backward_kernel(
    const scalar_t* __restrict__ grad_out,  // (N, 2m)
    const scalar_t* __restrict__ x,         // (N, C)
    const float*    __restrict__ B,          // (m, C)
    scalar_t*       __restrict__ grad_x,    // (N, C)
    int N, int C, int m
) {
    extern __shared__ float s_B[];
    for (int i = threadIdx.x; i < m * C; i += blockDim.x)
        s_B[i] = B[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one (n, c) pair
    if (idx >= N * C) return;

    int n = idx / C;
    int c = idx % C;

    float grad = 0.0f;

    for (int j = 0; j < m; ++j) {
        // Recompute dot product  2π · B[j] · x[n]
        float dot = 0.0f;
        for (int c2 = 0; c2 < C; ++c2)
            dot += s_B[j * C + c2] * (float)x[n * C + c2];
        dot *= 2.0f * M_PI;

        float twopi_Bjc = 2.0f * M_PI * s_B[j * C + c];
        // grad from cos channel:  d/dx cos(2πBx) = -sin(2πBx) · 2π·B[j,c]
        grad += (float)grad_out[n * 2 * m + j]     * (-sinf(dot)) * twopi_Bjc;
        // grad from sin channel:  d/dx sin(2πBx) =  cos(2πBx) · 2π·B[j,c]
        grad += (float)grad_out[n * 2 * m + m + j] *   cosf(dot)  * twopi_Bjc;
    }

    grad_x[idx] = (scalar_t)grad;
}

template <>
__global__ void rff_backward_kernel<double>(
    const double* __restrict__ grad_out,
    const double* __restrict__ x,
    const float*  __restrict__ B,
    double*       __restrict__ grad_x,
    int N, int C, int m
) {
    extern __shared__ float s_B[];
    for (int i = threadIdx.x; i < m * C; i += blockDim.x)
        s_B[i] = B[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;
    int c = idx % C;

    double grad = 0.0;

    for (int j = 0; j < m; ++j) {
        double dot = 0.0;
        for (int c2 = 0; c2 < C; ++c2)
            dot += (double)s_B[j * C + c2] * x[n * C + c2];
        dot *= 2.0 * M_PI;

        double twopi_Bjc = 2.0 * M_PI * (double)s_B[j * C + c];
        grad += grad_out[n * 2 * m + j]     * (-sin(dot)) * twopi_Bjc;
        grad += grad_out[n * 2 * m + m + j] *   cos(dot)  * twopi_Bjc;
    }

    grad_x[idx] = grad;
}


// ── Python-visible entry points ───────────────────────────────────────────────

torch::Tensor rff_forward(
    torch::Tensor x,   // (N, C) float/double, CUDA
    torch::Tensor B    // (m, C) float32,       CUDA — fixed frequency matrix
) {
    TORCH_CHECK(x.is_cuda(),     "x must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),     "B must be a CUDA tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat, "B must be float32");

    int N = x.size(0);
    int C = x.size(1);
    int m = B.size(0);

    auto out = torch::empty({N, 2 * m}, x.options());

    const int threads  = 256;
    const int blocks   = (N * 2 * m + threads - 1) / threads;
    const int smem     = m * C * sizeof(float);   // B in shared mem

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rff_forward", ([&] {
        rff_forward_kernel<scalar_t><<<blocks, threads, smem>>>(
            x.data_ptr<scalar_t>(),
            B.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            N, C, m
        );
    }));
    return out;
}

torch::Tensor rff_backward(
    torch::Tensor grad_out,  // (N, 2m)
    torch::Tensor x,         // (N, C)
    torch::Tensor B          // (m, C)
) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),        "B must be a CUDA tensor");

    int N = x.size(0);
    int C = x.size(1);
    int m = B.size(0);

    auto grad_x = torch::empty_like(x);

    const int threads = 256;
    const int blocks  = (N * C + threads - 1) / threads;
    const int smem    = m * C * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rff_backward", ([&] {
        rff_backward_kernel<scalar_t><<<blocks, threads, smem>>>(
            grad_out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            B.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            N, C, m
        );
    }));
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &rff_forward,
          "Fused RFF forward  (CUDA): [cos(2πBx), sin(2πBx)]");
    m.def("backward", &rff_backward,
          "Fused RFF backward (CUDA)");
}
