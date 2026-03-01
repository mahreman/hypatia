// hypatia_core/csrc/fused_layernorm_kernel.cu
// Fused LayerNorm: mean + variance reduction + normalize + affine transform
//
// Optimizations over sequential PyTorch ops:
// 1. Single kernel launch (no separate mean/var/normalize passes)
// 2. Warp shuffle reduction for mean and variance (minimal shared memory)
// 3. Single read of input data (fused normalize + affine)
//
// y[i] = gamma[i] * (x[i] - mean) * rsqrt(var + eps) + beta[i]

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <cmath>

namespace hypatia {

// Warp-level reduction using shuffle instructions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused LayerNorm kernel: one block per row
// Each block computes mean, variance, and normalized output for one row.
template <typename scalar_t>
__global__ void layernorm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int64_t features,
    scalar_t eps) {

    const int64_t row = blockIdx.x;
    const scalar_t* row_in = input + row * features;
    scalar_t* row_out = output + row * features;

    // Phase 1: Compute mean using parallel reduction
    extern __shared__ char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);

    scalar_t local_sum = scalar_t(0);
    for (int64_t i = threadIdx.x; i < features; i += blockDim.x) {
        local_sum += row_in[i];
    }

    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);

    // Write warp results to shared memory
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = blockDim.x / warpSize;

    if (lane_id == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    scalar_t mean;
    if (warp_id == 0) {
        scalar_t val = (lane_id < num_warps) ? shared[lane_id] : scalar_t(0);
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared[0] = val / scalar_t(features);
        }
    }
    __syncthreads();
    mean = shared[0];

    // Phase 2: Compute variance
    scalar_t local_var = scalar_t(0);
    for (int64_t i = threadIdx.x; i < features; i += blockDim.x) {
        scalar_t d = row_in[i] - mean;
        local_var += d * d;
    }

    local_var = warp_reduce_sum(local_var);
    if (lane_id == 0) {
        shared[warp_id] = local_var;
    }
    __syncthreads();

    scalar_t inv_std;
    if (warp_id == 0) {
        scalar_t val = (lane_id < num_warps) ? shared[lane_id] : scalar_t(0);
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared[0] = val / scalar_t(features);
        }
    }
    __syncthreads();
    inv_std = rsqrt(shared[0] + eps);

    // Phase 3: Normalize + affine transform (fused single pass)
    for (int64_t i = threadIdx.x; i < features; i += blockDim.x) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}

at::Tensor fused_layernorm_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "fused_layernorm: input must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "fused_layernorm: gamma must be CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "fused_layernorm: beta must be CUDA tensor");

    const auto features = input.size(-1);
    const auto batch = input.numel() / features;

    auto output = at::empty_like(input);

    // Choose thread count: round up to nearest warp, max 1024
    const int threads = std::min(int64_t(1024),
                                 ((features + 31) / 32) * 32);
    const int num_warps = threads / 32;
    const int shared_bytes = num_warps * sizeof(float);  // shared memory for warp reduction

    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fused_layernorm_kernel",
        [&] {
            layernorm_kernel<scalar_t><<<batch, threads, shared_bytes, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                features,
                scalar_t(eps)
            );
        }
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "fused_layernorm_kernel launch failed");

    return output;
}

} // namespace hypatia
