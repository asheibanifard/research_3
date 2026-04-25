#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Fused scale-and-sine: out[i] = sin(omega_0 * in[i])
// Avoids the extra global-memory round-trip of doing omega*x then sin(x) separately.
template <typename scalar_t>
__global__ void sine_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ output,
    float omega_0,
    int   n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = sinf(omega_0 * input[i]);
}

template <>
__global__ void sine_forward_kernel<double>(
    const double* __restrict__ input,
    double*       __restrict__ output,
    float omega_0,
    int   n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = sin((double)omega_0 * input[i]);
}

// Backward: d_input[i] = d_output[i] * omega_0 * cos(omega_0 * input[i])
template <typename scalar_t>
__global__ void sine_backward_kernel(
    const scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ input,
    scalar_t*       __restrict__ d_input,
    float omega_0,
    int   n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_input[i] = d_output[i] * omega_0 * cosf(omega_0 * input[i]);
}

template <>
__global__ void sine_backward_kernel<double>(
    const double* __restrict__ d_output,
    const double* __restrict__ input,
    double*       __restrict__ d_input,
    float omega_0,
    int   n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_input[i] = d_output[i] * (double)omega_0 * cos((double)omega_0 * input[i]);
}

// ── Python-visible entry points ──────────────────────────────────────────────

torch::Tensor sine_forward(torch::Tensor input, float omega_0) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sine_forward", ([&] {
        sine_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            omega_0, n
        );
    }));
    return output;
}

torch::Tensor sine_backward(torch::Tensor d_output, torch::Tensor input, float omega_0) {
    auto d_input = torch::empty_like(input);
    int n = input.numel();
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sine_backward", ([&] {
        sine_backward_kernel<scalar_t><<<blocks, threads>>>(
            d_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            d_input.data_ptr<scalar_t>(),
            omega_0, n
        );
    }));
    return d_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &sine_forward,  "Fused sine forward  (CUDA)");
    m.def("backward", &sine_backward, "Fused sine backward (CUDA)");
}
