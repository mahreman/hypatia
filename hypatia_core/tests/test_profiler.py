"""Tests for Hypatia Profiler: FLOPs estimation, hardware detection, benchmarking."""

import pytest
import torch
import torch.nn as nn
import sys
import os
# Ensure hypatia_core Python package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hypatia_core.profiler import (
    estimate_flops, detect_hardware, profile_model, compare_flops,
    benchmark_inference, roofline_analysis,
    ModelProfile, LayerProfile, HardwareInfo, FlopsComparison,
)


# ============================================================================
# TEST MODELS
# ============================================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return self.fc(x)


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

class TestHardwareDetection:
    def test_detect_hardware_returns_info(self):
        hw = detect_hardware()
        assert isinstance(hw, HardwareInfo)
        assert hw.cpu_cores >= 1

    def test_hardware_summary(self):
        hw = detect_hardware()
        summary = hw.summary()
        assert "CPU:" in summary
        assert len(summary) > 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_detected(self):
        hw = detect_hardware()
        assert hw.has_cuda
        assert len(hw.gpu_name) > 0
        assert hw.gpu_memory_gb > 0
        assert hw.gpu_sm_count > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_precision_support(self):
        hw = detect_hardware()
        # Any modern GPU should support FP16
        assert hw.supports_fp16


# ============================================================================
# FLOPs ESTIMATION
# ============================================================================

class TestFlopsEstimation:
    def test_linear_flops(self):
        """Linear(256, 512) with batch=1: FLOPs = 2*256*512 + 512 = 262,656"""
        model = nn.Linear(256, 512)
        profile = estimate_flops(model, (1, 256))
        # 2 * 256 * 512 = 262,144 (MAC * 2) + 512 (bias) = 262,656
        assert profile.total_flops == 262_656
        assert profile.total_mac == 256 * 512

    def test_linear_no_bias_flops(self):
        model = nn.Linear(256, 512, bias=False)
        profile = estimate_flops(model, (1, 256))
        assert profile.total_flops == 2 * 256 * 512  # No bias addition
        assert profile.total_flops == 262_144

    def test_linear_batch_flops(self):
        """Batch=32 should scale FLOPs linearly."""
        model = nn.Linear(256, 512)
        p1 = estimate_flops(model, (1, 256))
        p32 = estimate_flops(model, (32, 256))
        assert p32.total_flops == p1.total_flops * 32

    def test_mlp_flops(self):
        model = SimpleMLP()
        profile = estimate_flops(model, (1, 256))
        assert profile.total_flops > 0
        # Should have layers for fc1, relu, fc2
        assert len(profile.layers) >= 3

    def test_conv2d_flops(self):
        model = nn.Conv2d(3, 16, 3, padding=1)
        profile = estimate_flops(model, (1, 3, 32, 32))
        # FLOPs = 2 * 1 * 16 * 32 * 32 * 3 * 3 * 3 + 16 * 32 * 32 (bias)
        expected_mac = 1 * 16 * 32 * 32 * 3 * 3 * 3
        expected_flops = 2 * expected_mac + 16 * 32 * 32
        assert profile.total_flops == expected_flops

    def test_cnn_flops(self):
        model = SmallCNN()
        profile = estimate_flops(model, (1, 3, 32, 32))
        assert profile.total_flops > 0
        assert profile.total_params > 0
        assert len(profile.layers) >= 4

    def test_layernorm_flops(self):
        model = nn.LayerNorm(128)
        profile = estimate_flops(model, (1, 32, 128))
        # 5 ops per element * 1 * 32 * 128 = 20,480
        assert profile.total_flops == 5 * 1 * 32 * 128

    def test_flops_str_formatting(self):
        """Test human-readable FLOPs formatting."""
        model = SimpleMLP()
        profile = estimate_flops(model, (1, 256))
        assert "FLOPs" in profile.total_flops_str or "MFLOPs" in profile.total_flops_str

    def test_memory_estimation(self):
        model = SimpleMLP()
        profile = estimate_flops(model, (1, 256))
        assert profile.total_weight_memory > 0
        assert profile.total_activation_memory > 0

    def test_arithmetic_intensity(self):
        model = SimpleMLP()
        profile = estimate_flops(model, (1, 256))
        ai = profile.arithmetic_intensity
        assert ai > 0  # Should be positive


# ============================================================================
# PROFILING
# ============================================================================

class TestProfiling:
    def test_profile_model_with_benchmark(self):
        model = SimpleMLP()
        profile = profile_model(model, (1, 256), benchmark=True, warmup=2, runs=5)
        assert profile.total_flops > 0
        assert profile.inference_ms > 0

    def test_profile_model_no_benchmark(self):
        model = SimpleMLP()
        profile = profile_model(model, (1, 256), benchmark=False)
        assert profile.total_flops > 0
        assert profile.inference_ms == 0.0

    def test_profile_summary(self):
        model = SimpleMLP()
        profile = profile_model(model, (1, 256), benchmark=True, warmup=2, runs=5)
        summary = profile.summary()
        assert "Total FLOPs" in summary
        assert "Parameters" in summary
        assert "Inference" in summary

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_profile_gpu(self):
        model = SimpleMLP().cuda()
        profile = profile_model(model, (1, 256), benchmark=True, warmup=3, runs=10)
        assert profile.total_flops > 0
        assert profile.inference_ms > 0


# ============================================================================
# BENCHMARK
# ============================================================================

class TestBenchmark:
    def test_benchmark_inference_cpu(self):
        model = SimpleMLP()
        ms = benchmark_inference(model, (1, 256), warmup=2, runs=5)
        assert ms > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_inference_gpu(self):
        model = SimpleMLP().cuda()
        ms = benchmark_inference(model, (1, 256), warmup=3, runs=10)
        assert ms > 0


# ============================================================================
# COMPARISON
# ============================================================================

class TestComparison:
    def test_compare_flops_basic(self):
        original = SimpleMLP()
        optimized = SimpleMLP()  # Same model, just testing API
        cmp = compare_flops(original, optimized, (1, 256), benchmark=False)
        assert isinstance(cmp, FlopsComparison)
        assert cmp.original_flops > 0
        assert cmp.optimized_flops > 0

    def test_compare_summary(self):
        original = SimpleMLP()
        optimized = SimpleMLP()
        cmp = compare_flops(original, optimized, (1, 256), benchmark=False)
        summary = cmp.summary()
        assert "FLOPs" in summary
        assert "Parameters" in summary


# ============================================================================
# ROOFLINE
# ============================================================================

class TestRoofline:
    def test_roofline_analysis(self):
        model = SimpleMLP()
        profile = profile_model(model, (1, 256), benchmark=False)
        result = roofline_analysis(profile)
        assert result["bottleneck"] in ("compute", "memory")
        assert result["arithmetic_intensity"] > 0
        assert len(result["recommendation"]) > 0
