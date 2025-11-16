// hypatia_core/csrc/fused_linear_relu.cpp

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace hypatia {
namespace fused_linear_relu {

// CUDA forward deklarasyonu (kernel dosyasından)
at::Tensor fused_linear_relu_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

at::Tensor fused_linear_relu_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {

    TORCH_CHECK(input.device().is_cuda(),
                "fused_linear_relu_forward: only CUDA is implemented in this extension");
    return fused_linear_relu_forward_cuda(input, weight, bias_opt);
}

} // namespace fused_linear_relu
} // namespace hypatia

// PyBind11 kayıt
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &hypatia::fused_linear_relu::fused_linear_relu_forward,
        "Hypatia fused Linear+ReLU forward (CUDA)"
    );
}
