"""Tests for hypatia_core.dashboard — Benchmark Dashboard HTML generation."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypatia_core.dashboard import (
    BenchmarkDashboard,
    BenchmarkResult,
    generate_benchmark_dashboard,
    _categorize,
)


# ============================================================================
# BenchmarkResult tests
# ============================================================================

def test_benchmark_result_creation():
    r = BenchmarkResult(name="CPU FP32", latency_ms=100.0)
    assert r.name == "CPU FP32"
    assert r.latency_ms == 100.0
    assert r.category == "general"


def test_benchmark_result_with_all_fields():
    r = BenchmarkResult(
        name="GPU FP16", latency_ms=5.0,
        throughput_tps=1000.0, memory_mb=512.0,
        compression=2.0, category="gpu", notes="Tensor Cores"
    )
    assert r.throughput_tps == 1000.0
    assert r.category == "gpu"
    assert r.notes == "Tensor Cores"


# ============================================================================
# BenchmarkDashboard tests
# ============================================================================

def test_dashboard_creation():
    dash = BenchmarkDashboard("TestModel")
    assert dash.model_name == "TestModel"
    assert len(dash.results) == 0


def test_dashboard_add_result():
    dash = BenchmarkDashboard("TestModel")
    dash.add_result("CPU FP32", latency_ms=100.0, category="cpu")
    dash.add_result("GPU FP16", latency_ms=10.0, category="gpu")
    assert len(dash.results) == 2
    assert dash.results[0].name == "CPU FP32"
    assert dash.results[1].latency_ms == 10.0


def test_dashboard_chaining():
    dash = (BenchmarkDashboard("TestModel")
            .add_result("A", 100.0)
            .add_result("B", 50.0)
            .set_tuner_recommendation("Strategy X"))
    assert len(dash.results) == 2
    assert dash.tuner_recommendation == "Strategy X"


def test_dashboard_set_model_info():
    dash = BenchmarkDashboard("TestModel")
    dash.set_model_info(n_params=494e6, arch="qwen2", hidden=896, layers=24, heads=14)
    assert dash.n_params == 494e6
    assert dash.arch == "qwen2"
    assert dash.hidden_size == 896
    assert dash.model_memory_mb > 0  # Should be ~1885 MB


def test_dashboard_set_generation_test():
    dash = BenchmarkDashboard("TestModel")
    dash.set_generation_test("Hello", "Hello world!")
    assert dash.prompt == "Hello"
    assert dash.generated_text == "Hello world!"


def test_dashboard_set_roofline():
    dash = BenchmarkDashboard("TestModel")
    dash.set_roofline(ridge_point=56.0, arithmetic_intensity=64.0, status="compute-bound")
    assert dash.roofline_ridge == 56.0
    assert dash.roofline_ai == 64.0
    assert "compute" in dash.roofline_status


# ============================================================================
# HTML generation tests
# ============================================================================

def test_dashboard_save_creates_file():
    dash = BenchmarkDashboard("TestModel")
    dash.add_result("Baseline", 100.0, category="cpu")
    dash.add_result("Optimized", 10.0, category="gpu")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = dash.save(path)
        assert os.path.exists(path)
        assert len(html) > 1000
        assert "TestModel" in html
        assert "Baseline" in html
        assert "Optimized" in html
    finally:
        os.unlink(path)


def test_dashboard_html_contains_sections():
    dash = BenchmarkDashboard("Qwen2.5-0.5B")
    dash.set_model_info(n_params=494e6, arch="qwen2", hidden=896, layers=24)
    dash.add_result("CPU FP32", 1449.0, category="cpu")
    dash.add_result("GPU FP16+compile", 8.9, category="compiled")
    dash.set_tuner_recommendation("Transformer", {"Mode": "transformer"})
    dash.set_generation_test("Hello", "Hello world!")
    dash.set_roofline(56.0, 64.0, "memory-bound")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = dash.save(path)
        # Check all sections present
        assert "Hardware" in html
        assert "Model" in html
        assert "Auto-Tuner" in html
        assert "Benchmark Results" in html
        assert "Detailed Results" in html
        assert "Roofline" in html
        assert "Token Generation" in html
        assert "Qwen2.5-0.5B" in html
        assert "494M" in html
        assert "transformer" in html.lower()
    finally:
        os.unlink(path)


def test_dashboard_html_escapes_special_chars():
    dash = BenchmarkDashboard("Model <b>test</b>")
    dash.add_result("Test & <b>bold</b>", 100.0)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = dash.save(path)
        # Model name should be escaped in the title/badge
        assert "&lt;b&gt;test&lt;/b&gt;" in html
        # Result name should be escaped in table
        assert "&amp;" in html
    finally:
        os.unlink(path)


def test_dashboard_speedup_calculation():
    dash = BenchmarkDashboard("Test")
    dash.add_result("CPU FP32 baseline", 1000.0, category="cpu")
    dash.add_result("GPU FP16", 100.0, category="gpu")
    dash.add_result("GPU compile", 10.0, category="compiled")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = dash.save(path)
        assert "FASTEST" in html  # Best result badge
        assert "10.0x" in html    # speedup for GPU FP16
        assert "100.0x" in html   # speedup for GPU compile
    finally:
        os.unlink(path)


def test_dashboard_log_scale_chart():
    """Chart should handle wide latency ranges (0.6ms to 1449ms)."""
    dash = BenchmarkDashboard("Test")
    dash.add_result("MLP INT4", 0.63, category="quantized")
    dash.add_result("CPU FP32", 1449.0, category="cpu")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = dash.save(path)
        assert "Math.log10" in html  # Log scale
        assert "0.63" in html
        assert "1449" in html or "1.45" in html
    finally:
        os.unlink(path)


# ============================================================================
# Convenience function tests
# ============================================================================

def test_generate_benchmark_dashboard():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = generate_benchmark_dashboard(
            model_name="TestModel",
            results={"CPU FP32": 100.0, "GPU FP16": 10.0},
            model_info={"n_params": 1e6, "arch": "mlp"},
            tuner_name="native",
            output_path=path,
        )
        assert os.path.exists(path)
        assert "TestModel" in html
        assert "native" in html
    finally:
        os.unlink(path)


def test_generate_dashboard_with_all_options():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        html = generate_benchmark_dashboard(
            model_name="FullTest",
            results={
                "CPU FP32 (vanilla)": 500.0,
                "CPU INT8 Dynamic": 250.0,
                "GPU FP32": 20.0,
                "GPU FP16+compile": 5.0,
                "MLP Block INT4": 0.5,
            },
            model_info={
                "n_params": 494e6, "arch": "qwen2",
                "hidden": 896, "layers": 24, "heads": 14,
                "input_shape": "batch=1, seq=128", "gflops": 126.5,
            },
            tuner_name="Transformer (Rust-native block)",
            tuner_details={"Mode": "transformer", "Precision": "bf16"},
            roofline={"ridge": 56.0, "ai": 64.0, "status": "memory-bound"},
            generation={"prompt": "Hello", "output": "Hello world!"},
            output_path=path,
        )
        assert len(html) > 5000
        assert "FullTest" in html
        assert "memory-bound" in html.lower() or "MEMORY-BOUND" in html
    finally:
        os.unlink(path)


# ============================================================================
# Categorize helper tests
# ============================================================================

def test_categorize_cpu():
    assert _categorize("CPU FP32 (vanilla)") == "cpu"
    assert _categorize("cpu baseline") == "cpu"


def test_categorize_gpu():
    assert _categorize("GPU FP32") == "gpu"
    assert _categorize("GPU FP16") == "gpu"


def test_categorize_quantized():
    assert _categorize("INT4 Block") == "quantized"
    assert _categorize("MLP Block INT8") == "quantized"


def test_categorize_compiled():
    assert _categorize("GPU FP16+compile") == "compiled"
    assert _categorize("Triton kernel") == "compiled"


def test_categorize_general():
    assert _categorize("MLP Block FP32") == "general"
    assert _categorize("Some random strategy") == "general"
