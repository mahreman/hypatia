// hypatia_core/csrc/fused_gelu_mlp.cpp
// PyBind11 interface for fused GELU MLP CUDA kernel
// Pattern: Linear(w1,b1) → GELU → Linear(w2,b2)

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace hypatia {

// CUDA forward declaration
at::Tensor fused_gelu_mlp_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& w1,
    const c10::optional<at::Tensor>& b1,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2);

at::Tensor fused_gelu_mlp_forward(
    const at::Tensor& input,
    const at::Tensor& w1,
    const c10::optional<at::Tensor>& b1,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2) {

    if (input.is_cuda()) {
        return fused_gelu_mlp_forward_cuda(input, w1, b1, w2, b2);
    }

    // CPU fallback: standard PyTorch ops
    auto h = at::linear(input, w1, b1.has_value() ? b1.value() : at::Tensor());
    h = at::gelu(h);
    return at::linear(h, w2, b2.has_value() ? b2.value() : at::Tensor());
}

} // namespace hypatia

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &hypatia::fused_gelu_mlp_forward,
          "Hypatia fused GELU MLP forward (Linear→GELU→Linear)",
          py::arg("input"), py::arg("w1"), py::arg("b1"),
          py::arg("w2"), py::arg("b2"));
}
