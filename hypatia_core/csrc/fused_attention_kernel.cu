// hypatia_core/csrc/fused_attention_kernel.cu
// Fused multi-head causal self-attention
//
// Fuses: Q/K/V projections → reshape → scaled dot-product → causal mask
//        → softmax → V aggregation → output projection
//
// All cuBLAS GEMM calls on same stream, minimal intermediate allocations.
// Uses PyTorch's optimized at::bmm for batched GEMM (cuBLAS under the hood).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <cmath>

namespace hypatia {

// Fused causal mask + softmax kernel
// Applies upper-triangular mask (future tokens → -inf) and row-wise softmax
// in a single pass. Avoids creating a full mask tensor.
template <typename scalar_t>
__global__ void causal_mask_softmax_kernel(
    scalar_t* __restrict__ scores,
    int64_t seq_len,
    int64_t total_heads) {

    // Each block handles one (head, row) pair
    const int64_t head_row = blockIdx.x;
    if (head_row >= total_heads * seq_len) return;

    const int64_t row = head_row % seq_len;
    scalar_t* row_data = scores + head_row * seq_len;

    // Apply causal mask: set future positions to -inf
    for (int64_t j = threadIdx.x + row + 1; j < seq_len; j += blockDim.x) {
        row_data[j] = scalar_t(-1e9);
    }
    __syncthreads();

    // Compute max for numerical stability (reduction)
    __shared__ scalar_t shared_max;
    if (threadIdx.x == 0) {
        scalar_t max_val = scalar_t(-1e30);
        for (int64_t j = 0; j <= row; j++) {
            if (row_data[j] > max_val) max_val = row_data[j];
        }
        shared_max = max_val;
    }
    __syncthreads();

    // Compute exp and sum
    __shared__ scalar_t shared_sum;
    if (threadIdx.x == 0) {
        scalar_t sum = scalar_t(0);
        for (int64_t j = 0; j <= row; j++) {
            scalar_t e = exp(row_data[j] - shared_max);
            row_data[j] = e;
            sum += e;
        }
        shared_sum = sum;
    }
    __syncthreads();

    // Normalize
    scalar_t inv_sum = scalar_t(1.0) / shared_sum;
    for (int64_t j = threadIdx.x; j <= row; j += blockDim.x) {
        row_data[j] *= inv_sum;
    }
    // Zero out masked positions
    for (int64_t j = threadIdx.x + row + 1; j < seq_len; j += blockDim.x) {
        row_data[j] = scalar_t(0);
    }
}

at::Tensor fused_attention_forward_cuda(
    const at::Tensor& input,         // [batch*seq_len, hidden]
    const at::Tensor& wq, const c10::optional<at::Tensor>& bq_opt,
    const at::Tensor& wk, const c10::optional<at::Tensor>& bk_opt,
    const at::Tensor& wv, const c10::optional<at::Tensor>& bv_opt,
    const at::Tensor& wo, const c10::optional<at::Tensor>& bo_opt,
    int64_t n_heads) {

    TORCH_CHECK(input.is_cuda(), "fused_attention: input must be CUDA tensor");

    auto empty = at::Tensor();
    const auto total_rows = input.size(0);
    const auto hidden = wq.size(0);
    const auto head_dim = hidden / n_heads;

    // Q, K, V projections via cuBLAS
    auto q = at::linear(input, wq, bq_opt.has_value() ? bq_opt.value() : empty);
    auto k = at::linear(input, wk, bk_opt.has_value() ? bk_opt.value() : empty);
    auto v = at::linear(input, wv, bv_opt.has_value() ? bv_opt.value() : empty);

    // Reshape: [total_rows, hidden] → [total_rows, n_heads, head_dim]
    //        → [n_heads, total_rows, head_dim] for batched GEMM
    q = q.view({total_rows, n_heads, head_dim}).permute({1, 0, 2}).contiguous();
    k = k.view({total_rows, n_heads, head_dim}).permute({1, 0, 2}).contiguous();
    v = v.view({total_rows, n_heads, head_dim}).permute({1, 0, 2}).contiguous();

    // Scaled dot-product: scores = (Q @ K^T) / sqrt(d_k)
    // [n_heads, total_rows, head_dim] @ [n_heads, head_dim, total_rows]
    // → [n_heads, total_rows, total_rows]
    const auto scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto scores = at::bmm(q, k.transpose(1, 2)) * scale;

    // Fused causal mask + softmax (custom kernel)
    {
        const int64_t seq_len = total_rows;  // For single-batch, total_rows = seq_len
        const int64_t total_head_rows = n_heads * seq_len;
        const int threads = std::min(int64_t(256), seq_len);
        const auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            scores.scalar_type(),
            "causal_mask_softmax",
            [&] {
                causal_mask_softmax_kernel<scalar_t><<<total_head_rows, threads, 0, stream>>>(
                    scores.data_ptr<scalar_t>(),
                    seq_len,
                    n_heads
                );
            }
        );
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                     "causal_mask_softmax kernel launch failed");
    }

    // Attention output: scores @ V
    // [n_heads, total_rows, total_rows] @ [n_heads, total_rows, head_dim]
    // → [n_heads, total_rows, head_dim]
    auto attn_out = at::bmm(scores, v);

    // Reshape back: [n_heads, total_rows, head_dim] → [total_rows, hidden]
    attn_out = attn_out.permute({1, 0, 2}).contiguous().view({total_rows, hidden});

    // Output projection via cuBLAS
    return at::linear(attn_out, wo, bo_opt.has_value() ? bo_opt.value() : empty);
}

} // namespace hypatia
