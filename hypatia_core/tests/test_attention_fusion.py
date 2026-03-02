"""
Attention Fusion Tests for Hypatia

Tests:
  1. Rust native fused_attention_forward binding
  2. dispatch_fused_attention Python auto-dispatch
  3. E-graph attention-full-fusion rewrite rule
  4. Numerical correctness vs PyTorch reference
  5. FusedAttention module (from fused_modules)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def attention_params():
    """Create Q/K/V/O weight+bias tensors for 64-dim, 4-head attention."""
    torch.manual_seed(42)
    hidden = 64
    n_heads = 4
    wq = torch.randn(hidden, hidden) * 0.1
    bq = torch.randn(hidden) * 0.01
    wk = torch.randn(hidden, hidden) * 0.1
    bk = torch.randn(hidden) * 0.01
    wv = torch.randn(hidden, hidden) * 0.1
    bv = torch.randn(hidden) * 0.01
    wo = torch.randn(hidden, hidden) * 0.1
    bo = torch.randn(hidden) * 0.01
    return dict(
        hidden=hidden, n_heads=n_heads,
        wq=wq, bq=bq, wk=wk, bk=bk,
        wv=wv, bv=bv, wo=wo, bo=bo,
    )


@pytest.fixture
def attention_input(attention_params):
    """Random input for attention: [seq_len, hidden]."""
    torch.manual_seed(99)
    seq_len = 8
    hidden = attention_params["hidden"]
    return torch.randn(seq_len, hidden)


def pytorch_reference_attention(x, wq, bq, wk, bk, wv, bv, wo, bo, n_heads):
    """Reference multi-head self-attention using pure PyTorch."""
    total_rows = x.size(0)
    hidden = x.size(1)
    head_dim = hidden // n_heads

    q = F.linear(x, wq, bq)
    k = F.linear(x, wk, bk)
    v = F.linear(x, wv, bv)

    q = q.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
    k = k.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
    v = v.view(total_rows, n_heads, head_dim).permute(1, 0, 2)

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale

    # Causal mask
    mask = torch.ones(total_rows, total_rows, dtype=x.dtype).triu(1) * (-1e9)
    scores = scores + mask.unsqueeze(0)
    scores = F.softmax(scores, dim=-1)

    attn_out = torch.bmm(scores, v)
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(total_rows, hidden)
    return F.linear(attn_out, wo, bo)


# ============================================================================
# Test 1: Rust Native fused_attention_forward
# ============================================================================

class TestRustFusedAttention:
    def test_basic_forward(self, attention_params, attention_input):
        """Rust native attention should produce valid output."""
        from _hypatia_core import fused_attention_forward

        p = attention_params
        x_np = attention_input.numpy()
        seq_len = x_np.shape[0]

        result = fused_attention_forward(
            x_np,
            p["wq"].numpy(), p["bq"].numpy(),
            p["wk"].numpy(), p["bk"].numpy(),
            p["wv"].numpy(), p["bv"].numpy(),
            p["wo"].numpy(), p["bo"].numpy(),
            1, seq_len, p["n_heads"],
        )
        out = np.array(result)
        assert out.shape == (seq_len, p["hidden"]), f"Expected ({seq_len}, {p['hidden']}), got {out.shape}"
        assert np.all(np.isfinite(out)), "Output contains NaN/Inf"

    def test_no_bias(self, attention_params, attention_input):
        """Attention without bias should also work."""
        from _hypatia_core import fused_attention_forward

        p = attention_params
        x_np = attention_input.numpy()
        seq_len = x_np.shape[0]

        result = fused_attention_forward(
            x_np,
            p["wq"].numpy(), None,
            p["wk"].numpy(), None,
            p["wv"].numpy(), None,
            p["wo"].numpy(), None,
            1, seq_len, p["n_heads"],
        )
        out = np.array(result)
        assert out.shape == (seq_len, p["hidden"])
        assert np.all(np.isfinite(out))

    def test_batch_support(self, attention_params):
        """Multiple batches should be handled correctly."""
        from _hypatia_core import fused_attention_forward

        p = attention_params
        batch = 2
        seq_len = 4
        torch.manual_seed(77)
        x = torch.randn(batch * seq_len, p["hidden"])

        result = fused_attention_forward(
            x.numpy(),
            p["wq"].numpy(), p["bq"].numpy(),
            p["wk"].numpy(), p["bk"].numpy(),
            p["wv"].numpy(), p["bv"].numpy(),
            p["wo"].numpy(), p["bo"].numpy(),
            batch, seq_len, p["n_heads"],
        )
        out = np.array(result)
        assert out.shape == (batch * seq_len, p["hidden"])
        assert np.all(np.isfinite(out))

    def test_different_head_counts(self):
        """Different head counts (1, 2, 8) should all work."""
        from _hypatia_core import fused_attention_forward

        hidden = 32
        seq_len = 4
        torch.manual_seed(42)
        x = torch.randn(seq_len, hidden).numpy()

        for n_heads in [1, 2, 4, 8]:
            wq = torch.randn(hidden, hidden).numpy() * 0.1
            wk = torch.randn(hidden, hidden).numpy() * 0.1
            wv = torch.randn(hidden, hidden).numpy() * 0.1
            wo = torch.randn(hidden, hidden).numpy() * 0.1

            result = fused_attention_forward(
                x, wq, None, wk, None, wv, None, wo, None,
                1, seq_len, n_heads,
            )
            out = np.array(result)
            assert out.shape == (seq_len, hidden), f"Failed for n_heads={n_heads}"
            assert np.all(np.isfinite(out)), f"NaN/Inf for n_heads={n_heads}"

    def test_dimension_mismatch_error(self):
        """Invalid dimensions should raise an error."""
        from _hypatia_core import fused_attention_forward

        x = np.zeros((4, 32), dtype=np.float32)
        w = np.zeros((32, 32), dtype=np.float32)

        with pytest.raises(Exception):
            # batch*seq_len != total_rows
            fused_attention_forward(
                x, w, None, w, None, w, None, w, None,
                3, 3, 4,  # 3*3=9 != 4
            )

    def test_non_divisible_heads_error(self):
        """hidden not divisible by n_heads should raise an error."""
        from _hypatia_core import fused_attention_forward

        x = np.zeros((4, 32), dtype=np.float32)
        w = np.zeros((32, 32), dtype=np.float32)

        with pytest.raises(Exception):
            fused_attention_forward(
                x, w, None, w, None, w, None, w, None,
                1, 4, 3,  # 32 % 3 != 0
            )


# ============================================================================
# Test 2: dispatch_fused_attention
# ============================================================================

class TestDispatchFusedAttention:
    def test_cpu_dispatch(self, attention_params, attention_input):
        """dispatch_fused_attention should work on CPU."""
        from hypatia_core.fused_modules import dispatch_fused_attention

        p = attention_params
        result = dispatch_fused_attention(
            attention_input, p["wq"], p["bq"], p["wk"], p["bk"],
            p["wv"], p["bv"], p["wo"], p["bo"], p["n_heads"],
        )
        assert result.shape == attention_input.shape
        assert torch.all(torch.isfinite(result))

    def test_dispatch_matches_pytorch(self, attention_params, attention_input):
        """Dispatch result should be close to PyTorch reference."""
        from hypatia_core.fused_modules import dispatch_fused_attention

        p = attention_params
        with torch.no_grad():
            ref = pytorch_reference_attention(
                attention_input, p["wq"], p["bq"], p["wk"], p["bk"],
                p["wv"], p["bv"], p["wo"], p["bo"], p["n_heads"],
            )
            dispatched = dispatch_fused_attention(
                attention_input, p["wq"], p["bq"], p["wk"], p["bk"],
                p["wv"], p["bv"], p["wo"], p["bo"], p["n_heads"],
            )

        # Allow some numerical tolerance (Rust uses slightly different GEMM)
        max_diff = (ref - dispatched).abs().max().item()
        assert max_diff < 0.1, f"Max diff from PyTorch reference: {max_diff}"

    def test_dispatch_no_bias(self, attention_params, attention_input):
        """dispatch should handle None biases."""
        from hypatia_core.fused_modules import dispatch_fused_attention

        p = attention_params
        result = dispatch_fused_attention(
            attention_input, p["wq"], None, p["wk"], None,
            p["wv"], None, p["wo"], None, p["n_heads"],
        )
        assert result.shape == attention_input.shape
        assert torch.all(torch.isfinite(result))


# ============================================================================
# Test 3: E-graph Attention Fusion Rewrite Rules
# ============================================================================

class TestEGraphAttentionFusion:
    def test_attention_full_fusion_rule(self):
        """E-graph should fuse linear(wo,bo,attention(linear(wq,bq,x),...)) → fused_attention."""
        from _hypatia_core import optimize_ast

        expr = "(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))"
        result = optimize_ast(expr)
        assert "fused_attention" in result, \
            f"Expected fused_attention in result, got: {result}"

    def test_fusion_preserves_variables(self):
        """Fused expression should contain all original weight/bias variables."""
        from _hypatia_core import optimize_ast

        expr = "(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))"
        result = optimize_ast(expr)

        for var in ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "x"]:
            assert var in result, f"Variable '{var}' missing from fused result: {result}"

    def test_partial_pattern_no_fusion(self):
        """Attention without output linear should NOT trigger full fusion."""
        from _hypatia_core import optimize_ast

        # Just attention node without wrapping linear
        expr = "(attention (linear wq bq x) (linear wk bk x) (linear wv bv x))"
        result = optimize_ast(expr)
        # Should NOT have fused_attention (since there's no output projection linear)
        assert "fused_attention" not in result, \
            f"Partial pattern should not fuse, got: {result}"

    def test_different_inputs_no_fusion(self):
        """Q/K/V from different inputs should NOT fuse (requires same x)."""
        from _hypatia_core import optimize_ast

        # Different variables for Q, K, V inputs
        expr = "(linear wo bo (attention (linear wq bq x1) (linear wk bk x2) (linear wv bv x3)))"
        result = optimize_ast(expr)
        # The pattern requires ?x to be the same for all three projections
        assert "fused_attention" not in result, \
            f"Different inputs should not fuse, got: {result}"

    def test_nested_attention_in_residual(self):
        """Attention inside residual connection: (add x (linear wo bo (attention ...)))."""
        from _hypatia_core import optimize_ast

        expr = "(add x (linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x))))"
        result = optimize_ast(expr)
        # The attention part should still be fused
        assert "fused_attention" in result, \
            f"Attention in residual should still fuse, got: {result}"


# ============================================================================
# Test 4: Numerical Correctness
# ============================================================================

class TestNumericalCorrectness:
    def test_rust_vs_pytorch_small(self):
        """Rust native attention should be close to PyTorch reference (small model)."""
        from _hypatia_core import fused_attention_forward

        torch.manual_seed(42)
        hidden = 16
        n_heads = 2
        seq_len = 4

        wq = torch.randn(hidden, hidden) * 0.1
        bq = torch.randn(hidden) * 0.01
        wk = torch.randn(hidden, hidden) * 0.1
        bk = torch.randn(hidden) * 0.01
        wv = torch.randn(hidden, hidden) * 0.1
        bv = torch.randn(hidden) * 0.01
        wo = torch.randn(hidden, hidden) * 0.1
        bo = torch.randn(hidden) * 0.01

        x = torch.randn(seq_len, hidden)

        # PyTorch reference
        with torch.no_grad():
            ref = pytorch_reference_attention(x, wq, bq, wk, bk, wv, bv, wo, bo, n_heads)

        # Rust native
        result = fused_attention_forward(
            x.numpy(), wq.numpy(), bq.numpy(), wk.numpy(), bk.numpy(),
            wv.numpy(), bv.numpy(), wo.numpy(), bo.numpy(),
            1, seq_len, n_heads,
        )
        rust_out = torch.from_numpy(np.array(result))

        max_diff = (ref - rust_out).abs().max().item()
        # Tolerance: float32 GEMM differences between PyTorch and matrixmultiply
        assert max_diff < 0.05, f"Max diff Rust vs PyTorch: {max_diff}"

    def test_rust_vs_pytorch_larger(self):
        """Larger model: 128-dim, 8 heads, 16 tokens."""
        from _hypatia_core import fused_attention_forward

        torch.manual_seed(123)
        hidden = 128
        n_heads = 8
        seq_len = 16

        wq = torch.randn(hidden, hidden) * 0.05
        bq = torch.zeros(hidden)
        wk = torch.randn(hidden, hidden) * 0.05
        bk = torch.zeros(hidden)
        wv = torch.randn(hidden, hidden) * 0.05
        bv = torch.zeros(hidden)
        wo = torch.randn(hidden, hidden) * 0.05
        bo = torch.zeros(hidden)

        x = torch.randn(seq_len, hidden) * 0.1

        with torch.no_grad():
            ref = pytorch_reference_attention(x, wq, bq, wk, bk, wv, bv, wo, bo, n_heads)

        result = fused_attention_forward(
            x.numpy(), wq.numpy(), bq.numpy(), wk.numpy(), bk.numpy(),
            wv.numpy(), bv.numpy(), wo.numpy(), bo.numpy(),
            1, seq_len, n_heads,
        )
        rust_out = torch.from_numpy(np.array(result))

        max_diff = (ref - rust_out).abs().max().item()
        mean_diff = (ref - rust_out).abs().mean().item()
        assert max_diff < 0.1, f"Max diff: {max_diff}, mean diff: {mean_diff}"

    def test_causal_mask_effect(self):
        """Earlier tokens should not attend to later tokens (causal)."""
        from _hypatia_core import fused_attention_forward

        torch.manual_seed(42)
        hidden = 16
        n_heads = 2
        seq_len = 4

        # Identity-ish weights to make attention behavior clear
        wq = torch.eye(hidden) * 0.5
        wk = torch.eye(hidden) * 0.5
        wv = torch.eye(hidden) * 0.5
        wo = torch.eye(hidden) * 0.5

        x = torch.randn(seq_len, hidden)

        # First token (position 0) output should not depend on tokens 1,2,3
        result_full = fused_attention_forward(
            x.numpy(), wq.numpy(), None, wk.numpy(), None,
            wv.numpy(), None, wo.numpy(), None,
            1, seq_len, n_heads,
        )
        full_out = np.array(result_full)

        # Run with only first token
        result_single = fused_attention_forward(
            x[:1].numpy(), wq.numpy(), None, wk.numpy(), None,
            wv.numpy(), None, wo.numpy(), None,
            1, 1, n_heads,
        )
        single_out = np.array(result_single)

        # First row should be identical (causal: position 0 only sees itself)
        diff = np.abs(full_out[0] - single_out[0]).max()
        assert diff < 1e-5, f"Causal mask not working: diff={diff}"


# ============================================================================
# Test 5: FusedAttention Module
# ============================================================================

class TestFusedAttentionModule:
    def test_module_forward(self):
        """FusedAttention module should produce correct output shape."""
        from hypatia_core.fused_modules import FusedAttention

        hidden = 64
        n_heads = 4
        seq_len = 8

        attn = FusedAttention(hidden, n_heads)
        attn.eval()

        x = torch.randn(seq_len, hidden)
        with torch.no_grad():
            out = attn(x)

        assert out.shape == (seq_len, hidden)
        assert torch.all(torch.isfinite(out))

    def test_module_no_bias(self):
        """FusedAttention without bias."""
        from hypatia_core.fused_modules import FusedAttention

        hidden = 32
        n_heads = 4

        attn = FusedAttention(hidden, n_heads, bias=False)
        attn.eval()

        x = torch.randn(4, hidden)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (4, hidden)

    def test_module_parameters(self):
        """FusedAttention should have Q/K/V/O projection parameters."""
        from hypatia_core.fused_modules import FusedAttention

        attn = FusedAttention(64, 4)
        param_names = [name for name, _ in attn.named_parameters()]
        for expected in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                         "q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias"]:
            assert expected in param_names, f"Missing parameter: {expected}"

    def test_module_gradient_flow(self):
        """Gradients should flow through FusedAttention."""
        from hypatia_core.fused_modules import FusedAttention

        hidden = 32
        n_heads = 4

        attn = FusedAttention(hidden, n_heads)
        x = torch.randn(4, hidden, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, hidden)
        assert torch.all(torch.isfinite(x.grad))


# ============================================================================
# Test 6: FusedTransformerBlock
# ============================================================================

class TestFusedTransformerBlock:
    def test_block_forward(self):
        """Full transformer block: LN + Attention + MLP + residuals."""
        from hypatia_core.fused_modules import FusedTransformerBlock

        hidden = 64
        n_heads = 4
        mlp_hidden = 128

        block = FusedTransformerBlock(hidden, n_heads, mlp_hidden)
        block.eval()

        x = torch.randn(8, hidden)
        with torch.no_grad():
            out = block(x)

        assert out.shape == (8, hidden)
        assert torch.all(torch.isfinite(out))

    def test_block_residual(self):
        """Output should be different from input (non-trivial transform)."""
        from hypatia_core.fused_modules import FusedTransformerBlock

        block = FusedTransformerBlock(32, 4, 64)
        block.eval()

        x = torch.randn(4, 32)
        with torch.no_grad():
            out = block(x)

        assert not torch.allclose(x, out), "Block output should differ from input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
