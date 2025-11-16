// hypatia_core/csrc/fused_linear_relu_kernel.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

namespace hypatia {
namespace fused_linear_relu {

template <typename scalar_t>
__global__ void fused_linear_relu_forward_kernel(
    const scalar_t* __restrict__ input,   // [N, in_features]
    const scalar_t* __restrict__ weight,  // [out_features, in_features]
    const scalar_t* __restrict__ bias,    // [out_features] or nullptr
    scalar_t* __restrict__ output,        // [N, out_features]
    int64_t N,
    int64_t in_features,
    int64_t out_features) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y; // batch idx
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // out_feature idx

    if (row >= N || col >= out_features) {
        return;
    }

    const scalar_t* x_row = input + row * in_features;
    const scalar_t* w_row = weight + col * in_features;

    scalar_t acc = bias ? bias[col] : scalar_t(0);

    // Naive GEMM inner loop (research prototype, not cuBLAS-fast)
    for (int64_t k = 0; k < in_features; ++k) {
        acc += x_row[k] * w_row[k];
    }

    // ReLU
    output[row * out_features + col] = acc > scalar_t(0) ? acc : scalar_t(0);
}

at::Tensor fused_linear_relu_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {

    TORCH_CHECK(input.is_cuda(),  "fused_linear_relu: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "fused_linear_relu: weight must be CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "fused_linear_relu: bias must be CUDA tensor");
    }

    TORCH_CHECK(input.dim() == 2,  "fused_linear_relu: input must be 2D [N, in_features]");
    TORCH_CHECK(weight.dim() == 2, "fused_linear_relu: weight must be 2D [out_features, in_features]");

    const auto N = input.size(0);
    const auto in_features = input.size(1);
    const auto out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features,
                "fused_linear_relu: weight.shape[1] must match input.shape[1]");

    at::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.dim() == 1, "fused_linear_relu: bias must be 1D [out_features]");
        TORCH_CHECK(bias.size(0) == out_features,
                    "fused_linear_relu: bias.size(0) must match weight.size(0)");
    }

    auto input_contig  = input.contiguous();
    auto weight_contig = weight.contiguous();
    at::Tensor output  = at::empty({N, out_features}, input.options());

    const int threads_x = 16;
    const int threads_y = 16;
    dim3 block(threads_x, threads_y);
    dim3 grid(
        (out_features + threads_x - 1) / threads_x,
        (N           + threads_y - 1) / threads_y
    );

    const auto stream = at::cuda::getCurrentCUDAStream();

    // Ensure bias is contiguous if present
    const at::Tensor bias_c = bias_opt.has_value()
        ? bias_opt.value().contiguous()
        : at::Tensor();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contig.scalar_type(),
        "fused_linear_relu_forward_kernel",
        [&] {
            const scalar_t* input_ptr  = input_contig.data_ptr<scalar_t>();
            const scalar_t* weight_ptr = weight_contig.data_ptr<scalar_t>();
            const scalar_t* bias_ptr   = bias_opt.has_value() ? bias_c.data_ptr<scalar_t>() : nullptr;
            scalar_t* output_ptr       = output.data_ptr<scalar_t>();

            fused_linear_relu_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                N,
                in_features,
                out_features
            );
        }
    );

    // CUDA error check (debugging)
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_linear_relu_forward_kernel launch failed");

    return output;
}

} // namespace fused_linear_relu
} // namespace hypatia
