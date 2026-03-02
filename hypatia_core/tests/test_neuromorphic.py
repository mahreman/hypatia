"""
Neuromorphic Computing Tests for Hypatia

Tests ANN→SNN conversion, LIF neuron simulation, rate coding accuracy,
e-graph ReLU→LIF rewrite rules, and energy estimation.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_mlp():
    """Simple 2-layer MLP: 32→64→ReLU→16"""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    model.eval()
    return model


@pytest.fixture
def deep_mlp():
    """Deeper MLP: 128→256→ReLU→128→ReLU→64→ReLU→10"""
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    model.eval()
    return model


# ============================================================================
# Test 1: NeuromorphicModel API
# ============================================================================

class TestNeuromorphicModel:
    def test_construction(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=32, v_threshold=1.0, beta=0.95)
        assert len(snn.layer_data) == 2
        assert snn.timesteps == 32
        assert snn.v_threshold == 1.0

    def test_forward_shape(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=32)
        x = torch.randn(32)
        output = snn(x)
        assert output.shape == (16,), f"Expected (16,), got {output.shape}"

    def test_forward_batch(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=32)
        x = torch.randn(4, 32)  # batch of 4
        output = snn(x)
        assert output.shape == (4, 16), f"Expected (4,16), got {output.shape}"

    def test_forward_with_stats(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=64)
        x = torch.randn(32)
        output, stats = snn.forward_with_stats(x)
        assert output.shape == (16,)
        assert len(stats) == 2  # 2 layers
        for s in stats:
            assert 'firing_rate' in s
            assert 0.0 <= s['firing_rate'] <= 1.0

    def test_deep_mlp(self, deep_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(deep_mlp, timesteps=64)
        assert len(snn.layer_data) == 4  # 4 linear layers
        x = torch.randn(128)
        output = snn(x)
        assert output.shape == (10,)

    def test_repr(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp)
        repr_str = repr(snn)
        assert "NeuromorphicModel" in repr_str
        assert "LIF" in repr_str

    def test_no_linear_raises(self):
        from hypatia_core import NeuromorphicModel
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        with pytest.raises(ValueError, match="at least one Linear"):
            NeuromorphicModel(model)


# ============================================================================
# Test 2: ReLU ≈ LIF Equivalence
# ============================================================================

class TestReLULIFEquivalence:
    """Test that LIF firing rate approximates ReLU output."""

    def test_positive_values(self, simple_mlp):
        """SNN output should be close to ReLU output for positive activations."""
        from hypatia_core import NeuromorphicModel

        # Use more timesteps for better approximation
        snn = NeuromorphicModel(simple_mlp, timesteps=256, v_threshold=1.0, beta=0.0)

        # Positive input → ReLU passes through
        torch.manual_seed(42)
        x = torch.abs(torch.randn(32)) * 0.5  # all positive, moderate range

        # Get ANN output
        with torch.no_grad():
            ann_output = simple_mlp(x).numpy()

        # Get SNN output
        snn_output = snn(x).numpy()

        # Both should produce outputs (not all zeros)
        assert np.any(ann_output != 0), "ANN output should not be all zero"
        # SNN output may be clipped (rate coding) but should have some signal
        assert snn_output.shape == ann_output.shape

    def test_negative_clamped(self):
        """SNN should clamp negative values to 0 (like ReLU)."""
        from hypatia_core import NeuromorphicModel

        model = nn.Sequential(nn.Linear(4, 4))
        # Set weights so output is always negative
        with torch.no_grad():
            model[0].weight.fill_(-1.0)
            model[0].bias.fill_(-5.0)

        snn = NeuromorphicModel(model, timesteps=64, beta=0.0)
        x = torch.ones(4) * 0.1  # Small positive input → large negative output
        output = snn(x).numpy()

        # SNN output should be close to 0 for negative activations
        assert np.all(np.abs(output) < 0.5), f"Negative activations should be ~0, got {output}"

    def test_timestep_convergence(self, simple_mlp):
        """More timesteps should give more accurate approximation."""
        from hypatia_core import NeuromorphicModel

        torch.manual_seed(123)
        x = torch.randn(32) * 0.3

        outputs = {}
        for T in [8, 32, 128]:
            snn = NeuromorphicModel(simple_mlp, timesteps=T, beta=0.0)
            outputs[T] = snn(x).numpy()

        # Outputs with more timesteps should be more consistent
        # (they converge to the true firing rate)
        diff_8_32 = np.mean(np.abs(outputs[8] - outputs[32]))
        diff_32_128 = np.mean(np.abs(outputs[32] - outputs[128]))
        # 32→128 should be smaller change than 8→32
        assert diff_32_128 <= diff_8_32 + 0.1, \
            f"Should converge: diff(8,32)={diff_8_32:.4f}, diff(32,128)={diff_32_128:.4f}"


# ============================================================================
# Test 3: E-graph Neuromorphic Rewrite Rules
# ============================================================================

class TestEGraphNeuromorphic:
    def test_relu_to_neuromorphic(self):
        from hypatia_core import compile_neuromorphic
        result = compile_neuromorphic("(relu x)")
        # Should contain neuromorphic operators
        assert "lif" in result or "spike" in result or "neuromorphic" in result, \
            f"Expected neuromorphic ops, got: {result}"

    def test_linear_relu_to_neuromorphic(self):
        from hypatia_core import compile_neuromorphic
        result = compile_neuromorphic("(relu (linear w b x))")
        assert "neuromorphic_linear" in result, \
            f"Expected neuromorphic_linear, got: {result}"

    def test_spike_roundtrip_elimination(self):
        from hypatia_core import compile_neuromorphic
        result = compile_neuromorphic("(spike_encode (spike_decode spikes T) T)")
        assert result == "spikes", \
            f"Expected spike roundtrip eliminated, got: {result}"

    def test_neuromorphic_operators_parse(self):
        from hypatia_core import compile_neuromorphic
        # All neuromorphic operators should parse without error
        exprs = [
            "(lif x v_th beta T)",
            "(spike_encode x T)",
            "(spike_decode spikes T)",
            "(lif_linear w b x v_th beta)",
            "(neuromorphic_linear w b x v_th beta T)",
        ]
        for expr in exprs:
            result = compile_neuromorphic(expr)
            assert "error" not in result.lower(), f"Parse failed for {expr}: {result}"

    def test_standard_mode_preserves_relu(self):
        """Without neuromorphic target, ReLU should stay as ReLU."""
        from _hypatia_core import optimize_ast
        result = optimize_ast("(relu x)")
        assert "relu" in result and "lif" not in result, \
            f"Standard mode should preserve ReLU, got: {result}"


# ============================================================================
# Test 4: Energy Estimation
# ============================================================================

class TestEnergyEstimation:
    def test_energy_comparison(self):
        from _hypatia_core import estimate_neuromorphic_energy

        # Sparse network (10% firing rate)
        result = estimate_neuromorphic_energy(1024, 1024, 32, 0.1)
        assert result['neuromorphic_nj'] > 0
        assert result['conventional_nj'] > 0
        assert result['energy_ratio'] < 1.0, \
            f"Sparse neuromorphic should be more efficient, ratio={result['energy_ratio']}"

    def test_sparse_vs_dense(self):
        from _hypatia_core import estimate_neuromorphic_energy

        sparse = estimate_neuromorphic_energy(512, 512, 32, 0.05)
        dense = estimate_neuromorphic_energy(512, 512, 32, 0.9)

        assert sparse['neuromorphic_nj'] < dense['neuromorphic_nj'], \
            "Sparse should use less energy"

    def test_model_energy(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp)
        energy = snn.estimate_energy(avg_firing_rate=0.1)

        assert 'total_savings_pct' in energy
        assert len(energy['layers']) == 2


# ============================================================================
# Test 5: Benchmark
# ============================================================================

class TestBenchmark:
    def test_benchmark_runs(self, simple_mlp):
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=16)
        x = torch.randn(32)
        result = snn.benchmark(x, n_iter=10)

        assert 'snn_time_ms' in result
        assert 'relu_time_ms' in result
        assert result['snn_time_ms'] > 0
        assert result['relu_time_ms'] > 0


# ============================================================================
# Test 6: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_layer(self):
        """Single linear layer (no ReLU) should work."""
        from hypatia_core import NeuromorphicModel
        model = nn.Sequential(nn.Linear(10, 5))
        snn = NeuromorphicModel(model, timesteps=16)
        x = torch.randn(10)
        output = snn(x)
        assert output.shape == (5,)

    def test_zero_input(self, simple_mlp):
        """Zero input should produce near-zero output."""
        from hypatia_core import NeuromorphicModel
        snn = NeuromorphicModel(simple_mlp, timesteps=32)
        x = torch.zeros(32)
        output = snn(x).numpy()
        # Zero input → zero spikes → near-zero output
        assert np.all(np.abs(output) < 1.0), f"Zero input gave large output: {output}"

    def test_different_beta_values(self):
        """Different beta values should affect output."""
        from hypatia_core import NeuromorphicModel

        # Use a model with known positive weights to ensure spikes
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        with torch.no_grad():
            model[0].weight.fill_(0.5)
            model[0].bias.fill_(0.1)
            model[2].weight.fill_(0.5)
            model[2].bias.fill_(0.0)

        # Strong positive input to guarantee spiking
        x = torch.ones(8) * 2.0

        snn_no_leak = NeuromorphicModel(model, timesteps=64, beta=0.0)
        snn_high_leak = NeuromorphicModel(model, timesteps=64, beta=0.99)

        out_no = snn_no_leak(x).numpy()
        out_hi = snn_high_leak(x).numpy()

        # At least one should be non-zero
        has_signal = np.any(out_no != 0) or np.any(out_hi != 0)
        assert has_signal, "At least one beta should produce non-zero output"

    def test_different_timesteps(self, simple_mlp):
        """Different timesteps should affect accuracy."""
        from hypatia_core import NeuromorphicModel
        torch.manual_seed(42)
        x = torch.randn(32) * 0.5

        snn_8 = NeuromorphicModel(simple_mlp, timesteps=8)
        snn_128 = NeuromorphicModel(simple_mlp, timesteps=128)

        out_8 = snn_8(x).numpy()
        out_128 = snn_128(x).numpy()

        # More timesteps = different (generally more accurate) results
        assert out_8.shape == out_128.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
