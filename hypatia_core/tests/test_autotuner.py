"""Tests for Hypatia Auto-tuner: quick_tune, benchmark_tune, TuneConfig."""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hypatia_core.autotuner import (
    auto_tune, quick_tune, benchmark_tune, TuneConfig, _detect_transformer,
)


# ============================================================================
# TEST MODELS
# ============================================================================

class TinyMLP(nn.Module):
    """< 1M params — should pick fusion mode."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MediumMLP(nn.Module):
    """~2M params — should pick native mode."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class TinyTransformer(nn.Module):
    """Transformer architecture — should be detected."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return self.fc(x)


# ============================================================================
# TUNE CONFIG
# ============================================================================

class TestTuneConfig:
    def test_default_config(self):
        config = TuneConfig()
        assert config.mode == "auto"
        assert config.quantize is None
        assert config.chain_compile is True
        assert config.checksum_mode == "off"

    def test_summary(self):
        config = TuneConfig(mode="native", strategy_name="Test")
        summary = config.summary()
        assert "native" in summary
        assert "Test" in summary

    def test_env_dict(self):
        config = TuneConfig(
            chain_compile=False,
            enable_fusion=True,
            mixed_precision="bf16",
        )
        env = config.env_dict()
        assert env["HYPATIA_CHAIN_COMPILE"] == "0"
        assert env["HYPATIA_ENABLE_LINRELU_FUSION"] == "1"
        assert env["HYPATIA_MIXED_PRECISION"] == "bf16"


# ============================================================================
# QUICK TUNE
# ============================================================================

class TestQuickTune:
    def test_tiny_model_picks_fusion(self):
        model = TinyMLP()
        config = quick_tune(model, (1, 64))
        assert config.mode == "fusion"
        assert len(config.search_log) > 0

    def test_medium_model_picks_native(self):
        model = MediumMLP()
        config = quick_tune(model, (1, 512))
        assert config.mode == "native"
        assert config.enable_fusion is True

    def test_transformer_detected(self):
        model = TinyTransformer()
        config = quick_tune(model, (1, 16, 64))
        # Transformer has < 1M params so it may pick transformer or fusion
        assert config.mode in ("transformer", "fusion")

    def test_search_log_populated(self):
        model = TinyMLP()
        config = quick_tune(model, (1, 64))
        assert len(config.search_log) >= 4  # Hardware, Model, Architecture, Mode
        assert any("Hardware" in line for line in config.search_log)
        assert any("Model" in line for line in config.search_log)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_enables_precision(self):
        model = TinyMLP().cuda()
        config = quick_tune(model, (1, 64))
        # Should suggest FP16 or BF16 on GPU
        assert config.mixed_precision in ("fp16", "bf16")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_enables_chain_compile(self):
        model = TinyMLP().cuda()
        config = quick_tune(model, (1, 64))
        assert config.chain_compile is True

    def test_cpu_disables_chain_compile(self):
        model = TinyMLP()
        config = quick_tune(model, (1, 64))
        assert config.chain_compile is False


# ============================================================================
# BENCHMARK TUNE
# ============================================================================

class TestBenchmarkTune:
    def test_benchmark_basic(self):
        """Benchmark with specific candidates (fast test)."""
        model = TinyMLP()
        config = benchmark_tune(
            model, (1, 64),
            warmup=1, runs=3,
            candidates=["fusion"],
        )
        assert config.inference_ms > 0
        assert "Winner" in config.strategy_name
        assert len(config.search_log) > 0

    def test_benchmark_multiple_candidates(self):
        model = TinyMLP()
        config = benchmark_tune(
            model, (1, 64),
            warmup=1, runs=3,
            candidates=["fusion", "native"],
        )
        assert config.inference_ms > 0
        # Should have tried both and picked the best
        log_text = "\n".join(config.search_log)
        assert "Fusion" in log_text or "fusion" in log_text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_gpu(self):
        model = TinyMLP().cuda()
        config = benchmark_tune(
            model, (1, 64),
            warmup=2, runs=5,
            candidates=["fusion", "fp16"],
        )
        assert config.inference_ms > 0


# ============================================================================
# AUTO TUNE
# ============================================================================

class TestAutoTune:
    def test_auto_tune_quick(self):
        model = TinyMLP()
        config = auto_tune(model, (1, 64), mode="quick")
        assert isinstance(config, TuneConfig)
        assert config.mode in ("fusion", "native", "quantized", "transformer")

    def test_auto_tune_benchmark(self):
        model = TinyMLP()
        config = auto_tune(
            model, (1, 64),
            mode="benchmark",
            warmup=1, runs=2,
            candidates=["fusion"],
        )
        assert isinstance(config, TuneConfig)
        assert config.inference_ms > 0

    def test_auto_tune_invalid_mode(self):
        model = TinyMLP()
        with pytest.raises(ValueError, match="Unknown auto_tune mode"):
            auto_tune(model, (1, 64), mode="invalid")


# ============================================================================
# TRANSFORMER DETECTION
# ============================================================================

class TestTransformerDetection:
    def test_mlp_not_transformer(self):
        model = TinyMLP()
        assert _detect_transformer(model) is False

    def test_transformer_detected(self):
        model = TinyTransformer()
        assert _detect_transformer(model) is True
