// hypatia_core/csrc/fused_attention.cpp
// PyBind11 interface for fused multi-head causal self-attention CUDA kernel

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace hypatia {

// CUDA forward declaration
at::Tensor fused_attention_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& wq, const c10::optional<at::Tensor>& bq,
    const at::Tensor& wk, const c10::optional<at::Tensor>& bk,
    const at::Tensor& wv, const c10::optional<at::Tensor>& bv,
    const at::Tensor& wo, const c10::optional<at::Tensor>& bo,
    int64_t n_heads);

at::Tensor fused_attention_forward(
    const at::Tensor& input,
    const at::Tensor& wq, const c10::optional<at::Tensor>& bq,
    const at::Tensor& wk, const c10::optional<at::Tensor>& bk,
    const at::Tensor& wv, const c10::optional<at::Tensor>& bv,
    const at::Tensor& wo, const c10::optional<at::Tensor>& bo,
    int64_t n_heads) {

    if (input.is_cuda()) {
        return fused_attention_forward_cuda(input, wq, bq, wk, bk, wv, bv, wo, bo, n_heads);
    }

    // CPU fallback using PyTorch ops
    auto empty = at::Tensor();
    auto q = at::linear(input, wq, bq.has_value() ? bq.value() : empty);
    auto k = at::linear(input, wk, bk.has_value() ? bk.value() : empty);
    auto v = at::linear(input, wv, bv.has_value() ? bv.value() : empty);

    auto batch_seq = input.size(0);
    auto hidden = wq.size(0);
    auto head_dim = hidden / n_heads;

    // Reshape to [batch*seq, n_heads, head_dim] -> [n_heads, batch*seq, head_dim]
    q = q.view({batch_seq, n_heads, head_dim}).permute({1, 0, 2});
    k = k.view({batch_seq, n_heads, head_dim}).permute({1, 0, 2});
    v = v.view({batch_seq, n_heads, head_dim}).permute({1, 0, 2});

    // Scaled dot-product attention
    auto scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto scores = at::bmm(q, k.transpose(1, 2)) * scale;

    // Causal mask (upper triangular = -inf)
    auto mask = at::ones({batch_seq, batch_seq}, scores.options()).triu(1) * (-1e9);
    scores = scores + mask.unsqueeze(0);
    scores = at::softmax(scores, -1);

    // Attention output
    auto attn_out = at::bmm(scores, v); // [n_heads, batch*seq, head_dim]
    attn_out = attn_out.permute({1, 0, 2}).contiguous().view({batch_seq, hidden});

    // Output projection
    return at::linear(attn_out, wo, bo.has_value() ? bo.value() : empty);
}

} // namespace hypatia

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &hypatia::fused_attention_forward,
          "Hypatia fused multi-head causal self-attention forward",
          py::arg("input"),
          py::arg("wq"), py::arg("bq"),
          py::arg("wk"), py::arg("bk"),
          py::arg("wv"), py::arg("bv"),
          py::arg("wo"), py::arg("bo"),
          py::arg("n_heads"));
}
