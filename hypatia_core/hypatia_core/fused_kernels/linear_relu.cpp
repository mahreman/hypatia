// C++ interface for fused Linear+ReLU CUDA kernels
#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA functions
torch::Tensor fused_linear_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

std::vector<torch::Tensor> fused_linear_relu_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output);

// C++ interface wrappers
torch::Tensor fused_linear_relu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Input validation
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(-1) == weight.size(1),
                "Input features must match weight columns");
    TORCH_CHECK(weight.size(0) == bias.size(0),
                "Weight rows must match bias size");

    // Device check
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    // Dtype check - for now only support fp32
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only FP32 supported for now");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only FP32 supported for now");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Only FP32 supported for now");

    return fused_linear_relu_forward_cuda(input, weight, bias);
}

std::vector<torch::Tensor> fused_linear_relu_backward(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output) {

    // Input validation
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");

    return fused_linear_relu_backward_cuda(grad_out, input, weight, bias, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_linear_relu_forward,
          "Fused Linear+ReLU forward (CUDA)");
    m.def("backward", &fused_linear_relu_backward,
          "Fused Linear+ReLU backward (CUDA)");
}
