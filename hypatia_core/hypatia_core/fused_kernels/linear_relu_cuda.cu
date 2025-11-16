// CUDA implementation of fused Linear+ReLU
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

// Simple ReLU kernel to apply after GEMM
// For v1, we'll use at::addmm for GEMM then apply ReLU in a separate kernel
// (still much faster than Python overhead, single function call from Python)
// Future versions will do true single-kernel fusion
template <typename scalar_t>
__global__ void relu_kernel(
    scalar_t* __restrict__ data,
    const int64_t size) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0 ? data[idx] : 0;
    }
}

// Backward: compute grad_input, grad_weight, grad_bias
// grad_input = grad_out * (output > 0) @ weight
// grad_weight = (grad_out * (output > 0))^T @ input
// grad_bias = sum(grad_out * (output > 0), dim=0)
template <typename scalar_t>
__global__ void relu_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ output,
    scalar_t* __restrict__ grad_input_buffer,
    const int64_t size) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU gradient: pass through if output > 0, else zero
        grad_input_buffer[idx] = output[idx] > 0 ? grad_out[idx] : 0;
    }
}

} // namespace

// Forward pass: y = relu(x @ W^T + b)
torch::Tensor fused_linear_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Get input dimensions
    auto input_sizes = input.sizes().vec();
    const int64_t in_features = weight.size(1);
    const int64_t out_features = weight.size(0);

    // Compute output shape: (..., out_features)
    auto output_sizes = input_sizes;
    output_sizes.back() = out_features;

    // Reshape input to 2D for GEMM: (batch, in_features)
    const int64_t batch_size = input.numel() / in_features;
    auto input_2d = input.reshape({batch_size, in_features});

    // Perform Linear: output = input @ weight^T + bias
    // Using PyTorch's optimized GEMM for now (v1 prototype)
    // Future: custom CUDA GEMM kernel with ReLU fused
    auto output = at::addmm(bias, input_2d, weight.t());

    // Apply ReLU in-place
    const int64_t total_size = output.numel();
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "relu_forward_cuda", ([&] {
        relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            total_size
        );
    }));

    // Reshape back to original batch dimensions
    return output.reshape(output_sizes);
}

// Backward pass
std::vector<torch::Tensor> fused_linear_relu_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output) {

    const int64_t in_features = weight.size(1);
    const int64_t out_features = weight.size(0);
    const int64_t batch_size = input.numel() / in_features;

    // Reshape tensors to 2D
    auto input_2d = input.reshape({batch_size, in_features});
    auto grad_out_2d = grad_out.reshape({batch_size, out_features});
    auto output_2d = output.reshape({batch_size, out_features});

    // Apply ReLU backward: grad_output_masked = grad_out * (output > 0)
    auto grad_relu = torch::zeros_like(grad_out_2d);
    const int64_t total_size = grad_out_2d.numel();
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "relu_backward_cuda", ([&] {
        relu_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out_2d.data_ptr<scalar_t>(),
            output_2d.data_ptr<scalar_t>(),
            grad_relu.data_ptr<scalar_t>(),
            total_size
        );
    }));

    // Compute gradients using PyTorch operations
    // grad_input = grad_relu @ weight
    auto grad_input = at::mm(grad_relu, weight);

    // grad_weight = grad_relu^T @ input
    auto grad_weight = at::mm(grad_relu.t(), input_2d);

    // grad_bias = sum(grad_relu, dim=0)
    auto grad_bias = grad_relu.sum(0);

    // Reshape grad_input back to original input shape
    grad_input = grad_input.reshape(input.sizes());

    return {grad_input, grad_weight, grad_bias};
}
