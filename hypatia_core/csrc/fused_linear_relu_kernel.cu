// hypatia_core/csrc/fused_linear_relu_kernel.cu
// PLAN C: ATen-based implementation (cuBLAS via at::linear)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

namespace hypatia {
namespace fused_linear_relu {

// ✅ PLAN C: Use ATen's at::linear (cuBLAS) + at::relu_
// No custom GEMM kernel - let PyTorch handle it
at::Tensor fused_linear_relu_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {

    TORCH_CHECK(input.is_cuda(),  "fused_linear_relu: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "fused_linear_relu: weight must be CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "fused_linear_relu: bias must be CUDA tensor");
    }

    // ✅ Use ATen's linear (cuBLAS / cuBLASLt under the hood)
    // This is 50-100x faster than naive GEMM
    at::Tensor output = at::linear(input, weight, bias_opt);

    // ✅ In-place ReLU (no extra allocation)
    at::relu_(output);

    return output;
}

} // namespace fused_linear_relu
} // namespace hypatia
