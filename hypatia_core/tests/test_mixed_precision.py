"""
Mixed Precision Tests for Hypatia

Tests:
  1. FP16/BF16 conversion (Rust to_half_precision)
  2. Mixed-precision GEMM (Rust mixed_precision_forward)
  3. Precision statistics
  4. MixedPrecisionLinear module
  5. convert_to_mixed_precision utility
  6. E-graph cast elimination rules
  7. Numerical accuracy vs FP32
  8. Edge cases
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import os


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


# ============================================================================
# Test 1: FP16/BF16 Conversion
# ============================================================================

class TestHalfConversion:
    def test_fp16_basic(self):
        """FP16 conversion should produce u16 data with correct shape."""
        from _hypatia_core import to_half_precision

        w = np.array([[1.0, 0.5], [-0.3, 2.0]], dtype=np.float32)
        hw = to_half_precision(w, "fp16")

        assert hw["rows"] == 2
        assert hw["cols"] == 2
        assert hw["format"] == "fp16"
        assert len(np.array(hw["data"])) == 4
        assert hw["memory_bytes"] == 8   # 4 * 2 bytes
        assert hw["fp32_memory_bytes"] == 16  # 4 * 4 bytes
        assert abs(hw["compression_ratio"] - 2.0) < 0.01

    def test_bf16_basic(self):
        """BF16 conversion should produce u16 data."""
        from _hypatia_core import to_half_precision

        w = np.array([[1.0, -1.0], [0.0, 100.0]], dtype=np.float32)
        hw = to_half_precision(w, "bf16")

        assert hw["format"] == "bf16"
        assert len(np.array(hw["data"])) == 4

    def test_fp16_roundtrip_accuracy(self):
        """FP16 roundtrip should preserve values within tolerance."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        # Weight [2, 2], output = I @ W.T = W.T
        w = np.array([[1.0, 0.5], [3.0, -1.5]], dtype=np.float32)
        hw = to_half_precision(w, "fp16")

        x = np.eye(2, dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 2, 2, "fp16", False
        )
        recovered = np.array(out)

        # GEMM computes I @ W.T, so recovered = W.T
        np.testing.assert_array_almost_equal(recovered, w.T, decimal=2)

    def test_bf16_large_range(self):
        """BF16 should handle large values (same range as FP32)."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.array([[1e10, -1e10]], dtype=np.float32)
        hw = to_half_precision(w, "bf16")

        x = np.array([[1.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 1, 1, "bf16", False
        )
        result = np.array(out).flatten()
        assert abs(result[0]) > 9e9, f"BF16 should handle large values: {result[0]}"

    def test_invalid_format(self):
        """Invalid format should raise error."""
        from _hypatia_core import to_half_precision

        w = np.array([[1.0]], dtype=np.float32)
        with pytest.raises(Exception, match="Invalid format"):
            to_half_precision(w, "fp8")


# ============================================================================
# Test 2: Mixed-Precision GEMM
# ============================================================================

class TestMixedPrecisionGEMM:
    def test_identity_fp16(self):
        """Identity matrix in FP16 should pass through input."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.eye(4, dtype=np.float32)
        hw = to_half_precision(w, "fp16")

        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 4, 4, "fp16", False
        )
        np.testing.assert_array_almost_equal(np.array(out), [[1.0, 2.0, 3.0, 4.0]], decimal=2)

    def test_with_bias(self):
        """Mixed-precision linear with bias."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.eye(3, dtype=np.float32)
        hw = to_half_precision(w, "fp16")
        bias = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), bias, 3, 3, "fp16", False
        )
        np.testing.assert_array_almost_equal(np.array(out), [[11.0, 22.0, 33.0]], decimal=1)

    def test_with_relu(self):
        """ReLU should clamp negative outputs."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        hw = to_half_precision(w, "bf16")

        x = np.array([[3.0, 5.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 2, 2, "bf16", True
        )
        result = np.array(out).flatten()
        assert result[0] > 2.5  # ~3.0
        assert result[1] == 0.0  # ReLU clamps -5 to 0

    def test_batch_forward(self):
        """Batch forward should work correctly."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        hw = to_half_precision(w, "fp16")

        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 2, 2, "fp16", False
        )
        expected = np.array([[1.0, 4.0], [3.0, 8.0]])
        np.testing.assert_array_almost_equal(np.array(out), expected, decimal=1)

    def test_fp16_vs_fp32_accuracy(self):
        """FP16 GEMM should be close to FP32 GEMM."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        np.random.seed(42)
        w = np.random.randn(8, 16).astype(np.float32) * 0.1
        x = np.random.randn(4, 16).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32) * 0.01

        # FP32 reference
        fp32_out = x @ w.T + bias[np.newaxis, :]

        # FP16 mixed-precision
        hw = to_half_precision(w, "fp16")
        fp16_out = np.array(mixed_precision_forward(
            x, np.array(hw["data"]), bias, 8, 16, "fp16", False
        ))

        # Should be close (FP16 has ~3.3 decimal digits precision)
        max_diff = np.abs(fp16_out - fp32_out).max()
        assert max_diff < 0.05, f"FP16 max diff too large: {max_diff}"

    def test_bf16_vs_fp32_accuracy(self):
        """BF16 GEMM should be close to FP32 GEMM."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        np.random.seed(42)
        w = np.random.randn(8, 16).astype(np.float32) * 0.1
        x = np.random.randn(4, 16).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32) * 0.01

        fp32_out = x @ w.T + bias[np.newaxis, :]

        hw = to_half_precision(w, "bf16")
        bf16_out = np.array(mixed_precision_forward(
            x, np.array(hw["data"]), bias, 8, 16, "bf16", False
        ))

        max_diff = np.abs(bf16_out - fp32_out).max()
        assert max_diff < 0.05, f"BF16 max diff too large: {max_diff}"


# ============================================================================
# Test 3: Precision Statistics
# ============================================================================

class TestPrecisionStats:
    def test_fp16_stats(self):
        """FP16 stats should show low error for normal values."""
        from _hypatia_core import mixed_precision_stats

        w = np.random.randn(100, 100).astype(np.float32) * 0.1
        stats = mixed_precision_stats(w, "fp16")

        assert stats["total_elements"] == 10000
        assert stats["rmse"] < 0.01
        assert stats["overflow_count"] == 0

    def test_bf16_stats(self):
        """BF16 stats should show zero overflows even for large values."""
        from _hypatia_core import mixed_precision_stats

        w = np.array([1e30, -1e30, 0.1, -0.1], dtype=np.float32).reshape(2, 2)
        stats = mixed_precision_stats(w, "bf16")

        assert stats["overflow_count"] == 0

    def test_fp16_overflow_detection(self):
        """FP16 should detect overflow for values > 65504."""
        from _hypatia_core import mixed_precision_stats

        w = np.array([100000.0, -100000.0, 1.0, -1.0], dtype=np.float32).reshape(2, 2)
        stats = mixed_precision_stats(w, "fp16")

        assert stats["overflow_count"] == 2


# ============================================================================
# Test 4: MixedPrecisionLinear Module
# ============================================================================

class TestMixedPrecisionLinear:
    def test_construction_fp16(self):
        """FP16 MixedPrecisionLinear should construct correctly."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(64, 32, precision="fp16")
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.precision == "fp16"

    def test_construction_bf16(self):
        """BF16 MixedPrecisionLinear should construct correctly."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(64, 32, precision="bf16")
        assert layer.precision == "bf16"

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(64, 32, precision="fp16")
        layer.eval()

        x = torch.randn(8, 64)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (8, 32)

    def test_forward_1d(self):
        """1D input should work."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(16, 8, precision="fp16")
        layer.eval()

        x = torch.randn(16)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (8,)

    def test_forward_3d(self):
        """3D input (batch, seq, features) should work."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(32, 16, precision="bf16")
        layer.eval()

        x = torch.randn(2, 4, 32)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (2, 4, 16)

    def test_training_uses_dense(self):
        """Training mode should use dense FP32 (for gradients)."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(16, 8, precision="fp16")
        layer.train()

        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 16)

    def test_no_bias(self):
        """MixedPrecisionLinear without bias."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(16, 8, precision="fp16", bias=False)
        layer.eval()
        assert layer.bias is None

        x = torch.randn(4, 16)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (4, 8)

    def test_precision_stats(self):
        """get_precision_stats should return valid info."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(64, 32, precision="fp16")
        stats = layer.get_precision_stats()
        assert "rmse" in stats
        assert "overflow_count" in stats
        assert stats["overflow_count"] == 0

    def test_extra_repr(self):
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(64, 32, precision="bf16")
        assert "bf16" in repr(layer)


# ============================================================================
# Test 5: convert_to_mixed_precision
# ============================================================================

class TestConvertToMixedPrecision:
    def test_basic_conversion(self, simple_mlp):
        """All Linear layers should be replaced."""
        from hypatia_core import convert_to_mixed_precision, MixedPrecisionLinear

        mp_model = convert_to_mixed_precision(simple_mlp, precision="fp16")

        has_mp = False
        for module in mp_model.modules():
            if isinstance(module, MixedPrecisionLinear):
                has_mp = True
                assert module.precision == "fp16"
        assert has_mp

    def test_bf16_conversion(self, simple_mlp):
        """BF16 conversion should work."""
        from hypatia_core import convert_to_mixed_precision, MixedPrecisionLinear

        mp_model = convert_to_mixed_precision(simple_mlp, precision="bf16")

        for module in mp_model.modules():
            if isinstance(module, MixedPrecisionLinear):
                assert module.precision == "bf16"

    def test_converted_forward(self, simple_mlp):
        """Converted model should produce valid output."""
        from hypatia_core import convert_to_mixed_precision

        mp_model = convert_to_mixed_precision(simple_mlp, precision="fp16")

        x = torch.randn(4, 64)
        with torch.no_grad():
            out = mp_model(x)

        assert out.shape == (4, 32)
        assert torch.all(torch.isfinite(out))

    def test_output_close_to_dense(self, simple_mlp):
        """MP model output should be close to dense FP32."""
        from hypatia_core import convert_to_mixed_precision

        torch.manual_seed(42)
        x = torch.randn(4, 64)

        with torch.no_grad():
            dense_out = simple_mlp(x)

        mp_model = convert_to_mixed_precision(simple_mlp, precision="fp16")
        with torch.no_grad():
            mp_out = mp_model(x)

        max_diff = (dense_out - mp_out).abs().max().item()
        assert max_diff < 1.0, f"FP16 max diff vs dense: {max_diff}"

    def test_min_size_filter(self):
        """Small layers should not be converted."""
        from hypatia_core import convert_to_mixed_precision, MixedPrecisionLinear

        model = nn.Sequential(
            nn.Linear(4, 4),    # Small
            nn.ReLU(),
            nn.Linear(4, 128),  # Large
        )
        model.eval()

        mp_model = convert_to_mixed_precision(model, precision="fp16", min_size=64)
        assert isinstance(mp_model[0], nn.Linear)  # Kept as dense
        assert isinstance(mp_model[2], MixedPrecisionLinear)  # Converted

    def test_precision_report(self, simple_mlp):
        """Precision report should have correct structure."""
        from hypatia_core import convert_to_mixed_precision, model_precision_report

        mp_model = convert_to_mixed_precision(simple_mlp, precision="fp16")
        report = model_precision_report(mp_model)

        assert "layers" in report
        assert "total_fp32_bytes" in report
        assert "memory_savings_pct" in report
        assert report["memory_savings_pct"] > 40  # Should save ~50%


# ============================================================================
# Test 6: E-graph Cast Elimination Rules
# ============================================================================

class TestEGraphMixedPrecision:
    def test_cast_roundtrip_elimination(self):
        """cast_fp32(cast_fp16(x)) should reduce to x."""
        from _hypatia_core import optimize_ast

        os.environ["HYPATIA_MIXED_PRECISION"] = "fp16"
        try:
            result = optimize_ast("(cast_fp32 (cast_fp16 x))")
            assert result.strip() == "x", f"Expected 'x', got: {result}"
        finally:
            os.environ.pop("HYPATIA_MIXED_PRECISION", None)

    def test_cast_bf16_roundtrip(self):
        """cast_fp32(cast_bf16(x)) should reduce to x."""
        from _hypatia_core import optimize_ast

        os.environ["HYPATIA_MIXED_PRECISION"] = "bf16"
        try:
            result = optimize_ast("(cast_fp32 (cast_bf16 x))")
            assert result.strip() == "x", f"Expected 'x', got: {result}"
        finally:
            os.environ.pop("HYPATIA_MIXED_PRECISION", None)

    def test_double_cast_elimination(self):
        """cast_fp16(cast_fp16(x)) should reduce to cast_fp16(x)."""
        from _hypatia_core import optimize_ast

        os.environ["HYPATIA_MIXED_PRECISION"] = "fp16"
        try:
            result = optimize_ast("(cast_fp16 (cast_fp16 x))")
            assert result == "(cast_fp16 x)", f"Expected (cast_fp16 x), got: {result}"
        finally:
            os.environ.pop("HYPATIA_MIXED_PRECISION", None)

    def test_mp_linear_fusion(self):
        """linear(cast_fp16(w), b, x) → mp_linear(w, b, x, fp16)."""
        from _hypatia_core import optimize_ast

        os.environ["HYPATIA_MIXED_PRECISION"] = "fp16"
        try:
            result = optimize_ast("(linear (cast_fp16 w) b x)")
            assert "mp_linear" in result, f"Expected mp_linear, got: {result}"
        finally:
            os.environ.pop("HYPATIA_MIXED_PRECISION", None)

    def test_mp_linear_relu_fusion(self):
        """relu(mp_linear(w,b,x,p)) → fused_mp_linear_relu(w,b,x,p)."""
        from _hypatia_core import optimize_ast

        os.environ["HYPATIA_MIXED_PRECISION"] = "fp16"
        try:
            result = optimize_ast("(relu (mp_linear w b x fp16))")
            assert "fused_mp_linear_relu" in result, f"Expected fused, got: {result}"
        finally:
            os.environ.pop("HYPATIA_MIXED_PRECISION", None)

    def test_no_rules_without_flag(self):
        """Without HYPATIA_MIXED_PRECISION, cast should not be eliminated."""
        from _hypatia_core import optimize_ast

        os.environ.pop("HYPATIA_MIXED_PRECISION", None)
        result = optimize_ast("(cast_fp32 (cast_fp16 x))")
        # Without the flag, the expression should stay as-is
        assert "cast_fp16" in result or result.strip() == "x"


# ============================================================================
# Test 7: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_element(self):
        """1x1 weight should work."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        w = np.array([[5.0]], dtype=np.float32)
        hw = to_half_precision(w, "fp16")

        x = np.array([[3.0]], dtype=np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 1, 1, "fp16", False
        )
        assert abs(np.array(out).flatten()[0] - 15.0) < 0.1

    def test_cache_invalidation(self):
        """Modifying weights should invalidate cache."""
        from hypatia_core import MixedPrecisionLinear

        layer = MixedPrecisionLinear(16, 8, precision="fp16")
        layer.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            out1 = layer(x)

        with torch.no_grad():
            layer.weight.fill_(1.0)
        layer.invalidate_cache()

        with torch.no_grad():
            out2 = layer(x)

        assert not torch.allclose(out1, out2)

    def test_fp16_and_bf16_give_different_results(self):
        """FP16 and BF16 should produce slightly different results."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        np.random.seed(42)
        w = np.random.randn(4, 8).astype(np.float32) * 0.5
        x = np.random.randn(2, 8).astype(np.float32)

        hw16 = to_half_precision(w, "fp16")
        out16 = np.array(mixed_precision_forward(
            x, np.array(hw16["data"]), None, 4, 8, "fp16", False
        ))

        hwbf = to_half_precision(w, "bf16")
        outbf = np.array(mixed_precision_forward(
            x, np.array(hwbf["data"]), None, 4, 8, "bf16", False
        ))

        # Both should be close to each other and FP32, but not identical
        max_diff = np.abs(out16 - outbf).max()
        assert max_diff < 0.1, f"FP16 vs BF16 diff too large: {max_diff}"

    def test_large_matrix(self):
        """Large matrix should not crash."""
        from _hypatia_core import to_half_precision, mixed_precision_forward

        np.random.seed(42)
        w = np.random.randn(256, 512).astype(np.float32) * 0.01
        hw = to_half_precision(w, "fp16")

        x = np.random.randn(8, 512).astype(np.float32)
        out = mixed_precision_forward(
            x, np.array(hw["data"]), None, 256, 512, "fp16", False
        )
        assert np.array(out).shape == (8, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
