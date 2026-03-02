// hypatia_core/csrc/fused_layernorm.cpp
// PyBind11 interface for fused LayerNorm CUDA kernel
//
// Supports: y = gamma * (x - mean) / sqrt(var + eps) + beta
// with warp-reduction for mean/variance computation.

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace hypatia {

// CUDA forward declaration
at::Tensor fused_layernorm_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    double eps);

at::Tensor fused_layernorm_forward(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    double eps) {

    if (input.is_cuda()) {
        return fused_layernorm_forward_cuda(input, gamma, beta, eps);
    }

    // CPU fallback: use PyTorch's native layer_norm
    auto normalized_shape = gamma.sizes().vec();
    return at::layer_norm(input, normalized_shape, gamma, beta, eps);
}

} // namespace hypatia

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &hypatia::fused_layernorm_forward,
          "Hypatia fused LayerNorm forward (mean+var reduction + affine)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5);
}
