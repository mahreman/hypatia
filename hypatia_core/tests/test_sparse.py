"""
Sparse Tensor IR Tests for Hypatia

Tests:
  1. Rust CSR conversion (to_sparse_csr)
  2. Rust sparse GEMM (sparse_linear_forward)
  3. Python SparseLinear module
  4. sparsify_model utility
  5. E-graph sparse rewrite rules
  6. Numerical correctness vs dense PyTorch
  7. Edge cases and error handling
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
def simple_mlp():
    """Simple MLP: 64→128→ReLU→32"""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    model.eval()
    return model


@pytest.fixture
def sparse_weights():
    """Weight matrix with known 50% sparsity."""
    torch.manual_seed(42)
    w = torch.randn(32, 64) * 0.1
    # Zero out ~50% of weights
    mask = torch.rand(32, 64) > 0.5
    w = w * mask.float()
    return w.numpy()


# ============================================================================
# Test 1: Rust CSR Conversion
# ============================================================================

class TestCSRConversion:
    def test_basic_conversion(self):
        """Dense matrix should convert to CSR correctly."""
        from _hypatia_core import to_sparse_csr

        w = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)

        assert csr["rows"] == 2
        assert csr["cols"] == 3
        assert csr["nnz"] == 3
        assert list(csr["row_ptrs"]) == [0, 2, 3]
        assert list(csr["col_indices"]) == [0, 2, 1]
        np.testing.assert_array_equal(np.array(csr["values"]), [1.0, 2.0, 3.0])

    def test_sparsity_ratio(self):
        """Sparsity ratio should be computed correctly."""
        from _hypatia_core import to_sparse_csr

        w = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)
        assert abs(csr["sparsity"] - 0.75) < 0.01

    def test_fully_dense(self):
        """Fully dense matrix should have sparsity ~0."""
        from _hypatia_core import to_sparse_csr

        w = np.ones((4, 4), dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)
        assert csr["nnz"] == 16
        assert csr["sparsity"] < 0.01

    def test_fully_sparse(self):
        """All-zero matrix should have sparsity ~1."""
        from _hypatia_core import to_sparse_csr

        w = np.zeros((4, 4), dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)
        assert csr["nnz"] == 0
        assert csr["sparsity"] > 0.99

    def test_threshold_pruning(self):
        """Threshold should prune small values."""
        from _hypatia_core import to_sparse_csr

        w = np.array([[0.001, 0.5, 0.002], [0.8, 0.003, 0.9]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)
        assert csr["nnz"] == 3  # only 0.5, 0.8, 0.9 survive
        np.testing.assert_array_almost_equal(np.array(csr["values"]), [0.5, 0.8, 0.9])

    def test_large_matrix(self):
        """Large matrix conversion should not crash."""
        from _hypatia_core import to_sparse_csr

        np.random.seed(42)
        w = np.random.randn(256, 512).astype(np.float32)
        # Prune 50%
        mask = np.random.rand(256, 512) > 0.5
        w *= mask

        csr = to_sparse_csr(w, 0.001)
        assert csr["rows"] == 256
        assert csr["cols"] == 512
        # ~50% should be non-zero
        assert 0.3 < (1.0 - csr["sparsity"]) < 0.7


# ============================================================================
# Test 2: Rust Sparse GEMM
# ============================================================================

class TestSparseGEMM:
    def test_identity_sparse(self):
        """Sparse identity matrix should pass through input."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = np.eye(4, dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)

        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), None, 4, 4, False,
        )
        np.testing.assert_array_almost_equal(np.array(out), [[1.0, 2.0, 3.0, 4.0]])

    def test_with_bias(self):
        """Sparse linear with bias should add bias correctly."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = np.eye(3, dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)
        bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), bias, 3, 3, False,
        )
        np.testing.assert_array_almost_equal(np.array(out), [[11.0, 22.0, 33.0]])

    def test_with_relu(self):
        """ReLU should clamp negative outputs to zero."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)

        x = np.array([[3.0, 5.0]], dtype=np.float32)
        out = sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), None, 2, 2, True,
        )
        np.testing.assert_array_almost_equal(np.array(out), [[3.0, 0.0]])

    def test_batch_forward(self):
        """Multiple batch rows should all be computed correctly."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        out = sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), None, 2, 2, False,
        )
        expected = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
        np.testing.assert_array_almost_equal(np.array(out), expected)

    def test_sparse_vs_dense(self, sparse_weights):
        """Sparse GEMM should produce same result as dense GEMM."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = sparse_weights
        rows, cols = w.shape
        # Use very small threshold so no extra values are pruned
        csr = to_sparse_csr(w, 1e-7)

        np.random.seed(99)
        x = np.random.randn(4, cols).astype(np.float32)
        bias = np.random.randn(rows).astype(np.float32)

        # Sparse forward
        out_sparse = np.array(sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), bias, rows, cols, False,
        ))

        # Dense forward
        out_dense = x @ w.T + bias[np.newaxis, :]

        np.testing.assert_array_almost_equal(out_sparse, out_dense, decimal=4)


# ============================================================================
# Test 3: Pruning Utilities
# ============================================================================

class TestPruning:
    def test_threshold_computation(self):
        """Threshold should produce target sparsity."""
        from _hypatia_core import compute_sparsity_threshold

        np.random.seed(42)
        w = np.random.randn(100, 100).astype(np.float32)

        threshold = compute_sparsity_threshold(w, 0.5)
        pruned = np.abs(w) >= threshold
        actual_density = pruned.sum() / pruned.size
        assert 0.4 < actual_density < 0.6, f"Expected ~50% dense, got {actual_density:.2f}"

    def test_threshold_zero_sparsity(self):
        """0% sparsity should return threshold 0."""
        from _hypatia_core import compute_sparsity_threshold

        w = np.random.randn(10, 10).astype(np.float32)
        threshold = compute_sparsity_threshold(w, 0.0)
        assert threshold == 0.0

    def test_sparsity_stats(self):
        """Stats should report correct counts."""
        from _hypatia_core import sparsity_stats

        w = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        stats = sparsity_stats(w)
        assert stats["total_elements"] == 6
        assert stats["nonzero_elements"] == 2
        assert stats["zero_elements"] == 4
        assert abs(stats["sparsity_ratio"] - 2 / 3) < 0.01


# ============================================================================
# Test 4: SparseLinear Module
# ============================================================================

class TestSparseLinearModule:
    def test_construction(self):
        """SparseLinear should be constructable."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(64, 32, sparsity=0.5)
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.sparsity == 0.5

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(64, 32, sparsity=0.5)
        layer.eval()

        x = torch.randn(8, 64)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (8, 32)

    def test_forward_1d_input(self):
        """1D input (no batch) should work."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(16, 8, sparsity=0.5)
        layer.eval()

        x = torch.randn(16)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (8,)

    def test_forward_3d_input(self):
        """3D input (batch, seq_len, features) should work."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(32, 16, sparsity=0.5)
        layer.eval()

        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (2, 4, 16)

    def test_training_uses_dense(self):
        """Training mode should use dense forward (for gradients)."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(16, 8, sparsity=0.5)
        layer.train()

        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 16)

    def test_no_bias(self):
        """SparseLinear without bias."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(16, 8, sparsity=0.5, bias=False)
        layer.eval()
        assert layer.bias is None

        x = torch.randn(4, 16)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (4, 8)

    def test_get_stats(self):
        """Stats should report sparsity info."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(64, 32, sparsity=0.5)
        layer.eval()
        stats = layer.get_stats()
        assert stats["in_features"] == 64
        assert stats["out_features"] == 32
        assert 0.3 < stats["actual_sparsity"] < 0.7

    def test_extra_repr(self):
        """repr should include sparsity info."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(64, 32, sparsity=0.7)
        repr_str = repr(layer)
        assert "sparsity=0.7" in repr_str


# ============================================================================
# Test 5: sparsify_model
# ============================================================================

class TestSparsifyModel:
    def test_basic_sparsify(self, simple_mlp):
        """sparsify_model should replace Linear with SparseLinear."""
        from hypatia_core import sparsify_model, SparseLinear

        sparse_model = sparsify_model(simple_mlp, sparsity=0.5)

        # Check that Linear layers were replaced
        has_sparse = False
        for module in sparse_model.modules():
            if isinstance(module, SparseLinear):
                has_sparse = True
        assert has_sparse, "No SparseLinear found in sparsified model"

    def test_sparsified_forward(self, simple_mlp):
        """Sparsified model should still produce valid output."""
        from hypatia_core import sparsify_model

        sparse_model = sparsify_model(simple_mlp, sparsity=0.5)

        x = torch.randn(4, 64)
        with torch.no_grad():
            out = sparse_model(x)

        assert out.shape == (4, 32)
        assert torch.all(torch.isfinite(out))

    def test_output_similarity(self, simple_mlp):
        """Sparse model output should be somewhat close to dense (low sparsity)."""
        from hypatia_core import sparsify_model

        torch.manual_seed(42)
        x = torch.randn(4, 64)

        with torch.no_grad():
            dense_out = simple_mlp(x)

        sparse_model = sparsify_model(simple_mlp, sparsity=0.1)  # Very low sparsity
        with torch.no_grad():
            sparse_out = sparse_model(x)

        # With 10% sparsity, output should be close to dense
        max_diff = (dense_out - sparse_out).abs().max().item()
        assert max_diff < 5.0, f"Max diff too large: {max_diff}"

    def test_min_size_filter(self):
        """Small layers should not be sparsified when min_size is set."""
        from hypatia_core import sparsify_model, SparseLinear

        model = nn.Sequential(
            nn.Linear(4, 4),  # Small layer
            nn.ReLU(),
            nn.Linear(4, 128),  # Large layer
        )
        model.eval()

        sparse_model = sparsify_model(model, sparsity=0.5, min_size=64)

        # First layer (4→4) should NOT be replaced
        assert isinstance(sparse_model[0], nn.Linear)
        # Second layer (4→128) should be replaced
        assert isinstance(sparse_model[2], SparseLinear)

    def test_sparsity_report(self, simple_mlp):
        """Model sparsity report should have correct structure."""
        from hypatia_core import sparsify_model, model_sparsity_report

        sparse_model = sparsify_model(simple_mlp, sparsity=0.5)
        report = model_sparsity_report(sparse_model)

        assert "layers" in report
        assert "total_parameters" in report
        assert "overall_sparsity" in report
        assert report["layer_count"] >= 1


# ============================================================================
# Test 6: E-graph Sparse Rewrite Rules
# ============================================================================

class TestEGraphSparse:
    def test_linear_to_sparse_rewrite(self):
        """linear(to_sparse(w, th), b, x) should rewrite to sparse_linear."""
        from _hypatia_core import optimize_ast
        import os

        # Enable sparse rules
        os.environ["HYPATIA_ENABLE_SPARSE"] = "1"
        try:
            expr = "(linear (to_sparse w 0.01) b x)"
            result = optimize_ast(expr)
            assert "sparse_linear" in result, \
                f"Expected sparse_linear in result, got: {result}"
        finally:
            os.environ.pop("HYPATIA_ENABLE_SPARSE", None)

    def test_sparse_linear_relu_fusion(self):
        """relu(sparse_linear(w,b,x,s)) should fuse to fused_sparse_linear_relu."""
        from _hypatia_core import optimize_ast
        import os

        os.environ["HYPATIA_ENABLE_SPARSE"] = "1"
        try:
            expr = "(relu (sparse_linear w b x s))"
            result = optimize_ast(expr)
            assert "fused_sparse_linear_relu" in result, \
                f"Expected fused_sparse_linear_relu, got: {result}"
        finally:
            os.environ.pop("HYPATIA_ENABLE_SPARSE", None)

    def test_sparse_ops_parse(self):
        """Sparse operators should parse without error."""
        from _hypatia_core import optimize_ast
        import os

        os.environ["HYPATIA_ENABLE_SPARSE"] = "1"
        try:
            exprs = [
                "(sparse_linear w b x s)",
                "(fused_sparse_linear_relu w b x s)",
                "(to_sparse w threshold)",
            ]
            for expr in exprs:
                result = optimize_ast(expr)
                assert "error" not in result.lower(), f"Parse failed for {expr}: {result}"
        finally:
            os.environ.pop("HYPATIA_ENABLE_SPARSE", None)

    def test_no_sparse_without_flag(self):
        """Without HYPATIA_ENABLE_SPARSE, sparse rules should not apply."""
        from _hypatia_core import optimize_ast
        import os

        os.environ.pop("HYPATIA_ENABLE_SPARSE", None)
        expr = "(linear (to_sparse w 0.01) b x)"
        result = optimize_ast(expr)
        # Without the flag, linear(to_sparse(w,th), b, x) stays as-is
        assert "to_sparse" in result, \
            f"Without sparse flag, should keep to_sparse: {result}"


# ============================================================================
# Test 7: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_zero_sparsity(self):
        """0% sparsity = fully dense, should still work."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(16, 8, sparsity=0.0)
        layer.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (4, 8)
        assert torch.all(torch.isfinite(out))

    def test_high_sparsity(self):
        """95% sparsity should still produce some output."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(64, 32, sparsity=0.95)
        layer.eval()

        x = torch.randn(4, 64)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (4, 32)
        assert torch.all(torch.isfinite(out))

    def test_single_element(self):
        """1x1 weight matrix should work."""
        from _hypatia_core import to_sparse_csr, sparse_linear_forward

        w = np.array([[5.0]], dtype=np.float32)
        csr = to_sparse_csr(w, 0.01)

        x = np.array([[3.0]], dtype=np.float32)
        out = sparse_linear_forward(
            x, list(csr["row_ptrs"]), list(csr["col_indices"]),
            np.array(csr["values"]), None, 1, 1, False,
        )
        np.testing.assert_array_almost_equal(np.array(out), [[15.0]])

    def test_cache_invalidation(self):
        """Modifying weights should invalidate sparse cache."""
        from hypatia_core import SparseLinear

        layer = SparseLinear(16, 8, sparsity=0.5)
        layer.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            out1 = layer(x)

        # Modify weights
        with torch.no_grad():
            layer.weight.fill_(1.0)
        layer.invalidate_cache()

        with torch.no_grad():
            out2 = layer(x)

        assert not torch.allclose(out1, out2), "Output should change after weight modification"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
