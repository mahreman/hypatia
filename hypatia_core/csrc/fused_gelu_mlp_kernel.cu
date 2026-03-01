// hypatia_core/csrc/fused_gelu_mlp_kernel.cu
// Fused GELU MLP: cuBLAS GEMM + Custom GELU kernel + cuBLAS GEMM
//
// Optimizations over sequential PyTorch ops:
// 1. Single CUDA stream, no Python round-trips between ops
// 2. In-place GELU avoids intermediate tensor allocation
// 3. cuBLAS GEMM via at::addmm/mm (same as PyTorch, but fused in one call)
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <cmath>

namespace hypatia {

// GELU activation kernel (in-place, exact tanh formula)
// Uses fast __tanhf intrinsic for GPU (hardware-accelerated on SM_75+)
template <typename scalar_t>
__global__ void gelu_inplace_kernel(
    scalar_t* __restrict__ data,
    int64_t size) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const scalar_t x = data[idx];
        // GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const scalar_t sqrt_2_pi = scalar_t(0.7978845608);
        const scalar_t c = scalar_t(0.044715);
        const scalar_t inner = sqrt_2_pi * (x + c * x * x * x);
        data[idx] = scalar_t(0.5) * x * (scalar_t(1.0) + tanh(inner));
    }
}

// Fused forward: Linear(w1,b1) → GELU → Linear(w2,b2)
// All operations on the same CUDA stream, minimal allocations.
at::Tensor fused_gelu_mlp_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& w1,
    const c10::optional<at::Tensor>& b1_opt,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2_opt) {

    TORCH_CHECK(input.is_cuda(), "fused_gelu_mlp: input must be CUDA tensor");
    TORCH_CHECK(w1.is_cuda(), "fused_gelu_mlp: w1 must be CUDA tensor");
    TORCH_CHECK(w2.is_cuda(), "fused_gelu_mlp: w2 must be CUDA tensor");

    const auto in_features = w1.size(1);
    const auto hidden = w1.size(0);
    const auto out_features = w2.size(0);

    // Flatten batch dimensions
    const auto input_sizes = input.sizes().vec();
    const int64_t batch_size = input.numel() / in_features;
    auto input_2d = input.view({batch_size, in_features});

    // Layer 1: cuBLAS GEMM — h = input @ w1^T + b1
    at::Tensor h;
    if (b1_opt.has_value()) {
        h = at::addmm(b1_opt.value(), input_2d, w1.t());
    } else {
        h = at::mm(input_2d, w1.t());
    }

    // In-place GELU activation (no extra allocation)
    {
        const int64_t total = h.numel();
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            h.scalar_type(),
            "gelu_inplace_kernel",
            [&] {
                gelu_inplace_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    h.data_ptr<scalar_t>(),
                    total
                );
            }
        );
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                     "gelu_inplace_kernel launch failed");
    }

    // Layer 2: cuBLAS GEMM — output = h @ w2^T + b2
    at::Tensor output;
    if (b2_opt.has_value()) {
        output = at::addmm(b2_opt.value(), h, w2.t());
    } else {
        output = at::mm(h, w2.t());
    }

    // Restore batch shape
    auto output_sizes = input_sizes;
    output_sizes.back() = out_features;
    return output.view(output_sizes);
}

} // namespace hypatia
