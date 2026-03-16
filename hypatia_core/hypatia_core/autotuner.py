# hypatia_core/autotuner.py
#
# Auto-tuning system for Hypatia compiler pipeline.
#
# Two modes:
#   1. quick_tune(model, input_shape) — Heuristic config based on model + hardware (< 100ms)
#   2. benchmark_tune(model, input_shape) — Benchmark multiple configs, pick fastest (seconds)
#
# Usage:
#     import hypatia_core
#     from hypatia_core.autotuner import auto_tune, quick_tune
#
#     # Fast heuristic selection
#     config = quick_tune(model, (1, 768))
#     fast_model = config.apply(model)
#
#     # Full benchmark search
#     config = auto_tune(model, (1, 768))
#     fast_model = config.apply(model)

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class TuneConfig:
    """Optimal configuration found by auto-tuner.

    Contains all pipeline knobs needed to reproduce the optimal setup.
    Call apply() to optimize a model with this config.
    """
    # Optimization mode
    mode: str = "auto"              # auto | native | quantized | fusion | transformer
    quantize: Optional[str] = None  # None | int4 | int8 | False

    # E-graph controls
    chain_compile: bool = True      # Enable torch.compile after e-graph
    enable_fusion: bool = False     # HYPATIA_ENABLE_LINRELU_FUSION
    enable_sparse: bool = False     # HYPATIA_ENABLE_SPARSE
    mixed_precision: str = ""       # "" | fp16 | bf16
    checksum_mode: str = "off"      # off | soft | strict

    # torch.compile settings
    compile_mode: str = "max-autotune"  # default | reduce-overhead | max-autotune

    # Profiling results (filled by benchmark_tune)
    inference_ms: float = 0.0
    memory_mb: float = 0.0
    strategy_name: str = ""
    search_log: List[str] = field(default_factory=list)

    def apply(self, model: nn.Module) -> nn.Module:
        """Apply this tuned configuration to optimize a model.

        Args:
            model: PyTorch model to optimize.

        Returns:
            Optimized model using the configuration found by auto-tuner.

        Examples:
            config = quick_tune(model, (1, 768))
            fast_model = config.apply(model)
            output = fast_model(input_tensor)
        """
        # Set environment variables for Rust e-graph optimizer
        os.environ["HYPATIA_CHAIN_COMPILE"] = "1" if self.chain_compile else "0"
        os.environ["HYPATIA_ENABLE_LINRELU_FUSION"] = "1" if self.enable_fusion else "0"
        os.environ["HYPATIA_ENABLE_SPARSE"] = "1" if self.enable_sparse else "0"
        os.environ["HYPATIA_CHECKSUM_MODE"] = self.checksum_mode

        if self.mixed_precision:
            os.environ["HYPATIA_MIXED_PRECISION"] = self.mixed_precision
        elif "HYPATIA_MIXED_PRECISION" in os.environ:
            del os.environ["HYPATIA_MIXED_PRECISION"]

        # Apply model-level optimization
        from .optimizer import optimize
        quantize_arg = self.quantize if self.quantize != "False" else False
        optimized = optimize(model, mode=self.mode, quantize=quantize_arg)

        return optimized

    def env_dict(self) -> Dict[str, str]:
        """Return environment variables as dict (for subprocess or logging)."""
        env = {
            "HYPATIA_CHAIN_COMPILE": "1" if self.chain_compile else "0",
            "HYPATIA_ENABLE_LINRELU_FUSION": "1" if self.enable_fusion else "0",
            "HYPATIA_ENABLE_SPARSE": "1" if self.enable_sparse else "0",
            "HYPATIA_CHECKSUM_MODE": self.checksum_mode,
        }
        if self.mixed_precision:
            env["HYPATIA_MIXED_PRECISION"] = self.mixed_precision
        return env

    def summary(self) -> str:
        """Human-readable summary of the tuned configuration."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  Hypatia Auto-Tune Result: {self.strategy_name}")
        lines.append(f"{'='*60}")
        lines.append(f"  Mode:            {self.mode}")
        lines.append(f"  Quantization:    {self.quantize or 'None'}")
        lines.append(f"  Chain Compile:   {self.chain_compile}")
        lines.append(f"  Fusion Rules:    {self.enable_fusion}")
        lines.append(f"  Sparse:          {self.enable_sparse}")
        lines.append(f"  Mixed Precision: {self.mixed_precision or 'FP32'}")
        lines.append(f"  Checksum:        {self.checksum_mode}")
        if self.inference_ms > 0:
            lines.append(f"  Inference:       {self.inference_ms:.2f} ms")
        if self.memory_mb > 0:
            lines.append(f"  Memory:          {self.memory_mb:.1f} MB")
        if self.search_log:
            lines.append(f"  {'-'*56}")
            lines.append(f"  Search Log:")
            for entry in self.search_log:
                lines.append(f"    {entry}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================================
# QUICK TUNE — Heuristic-based (< 100ms)
# ============================================================================

def quick_tune(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
) -> TuneConfig:
    """Select optimal configuration based on model analysis and hardware detection.

    This is a fast heuristic approach (< 100ms) that analyzes the model
    architecture and available hardware to select the best optimization strategy
    without actually benchmarking.

    Decision tree:
      1. Detect hardware (GPU? Tensor cores? Memory?)
      2. Analyze model (size, architecture, layer types)
      3. Select mode, precision, and feature flags

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224))

    Returns:
        TuneConfig with recommended settings.

    Examples:
        config = quick_tune(model, (1, 768))
        print(config.summary())
        fast_model = config.apply(model)
    """
    from .profiler import detect_hardware, estimate_flops

    config = TuneConfig()
    log = []

    # 1. Hardware detection
    hw = detect_hardware()
    has_gpu = hw.has_cuda
    has_tc = hw.has_tensor_cores
    gpu_mem_gb = hw.gpu_memory_gb

    log.append(f"Hardware: {'GPU ' + hw.gpu_name if has_gpu else 'CPU only'}")

    # 2. Model analysis
    n_params = sum(p.numel() for p in model.parameters())
    model_mb = n_params * 4 / 1024 / 1024  # FP32 size
    is_on_gpu = any(p.is_cuda for p in model.parameters())

    log.append(f"Model: {n_params/1e6:.1f}M params ({model_mb:.1f} MB FP32)")

    # Detect architecture type
    has_attention = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
    has_transformer_blocks = _detect_transformer(model)
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    arch = "transformer" if has_transformer_blocks else "cnn" if has_conv else "mlp"
    log.append(f"Architecture: {arch} ({n_linear} linear layers)")

    # 3. FLOPs analysis (fast, no benchmark)
    try:
        profile = estimate_flops(model, input_shape)
        total_gflops = profile.total_flops / 1e9
        log.append(f"FLOPs: {total_gflops:.2f} GFLOPs")
    except Exception:
        total_gflops = 0
        log.append("FLOPs: estimation failed")

    # ================================================================
    # DECISION LOGIC
    # ================================================================

    # --- Mode selection ---
    if has_transformer_blocks and n_params > 1_000_000:
        config.mode = "transformer"
        config.strategy_name = "Transformer (Rust-native block)"
        log.append("Mode: transformer (detected transformer blocks)")
    elif n_params >= 50_000_000:
        config.mode = "quantized"
        config.quantize = "int4"
        config.strategy_name = "INT4 Quantized (large model)"
        log.append("Mode: quantized INT4 (>50M params, memory-bandwidth bound)")
    elif n_params >= 1_000_000:
        config.mode = "native"
        config.strategy_name = "Native f32 (medium model)"
        log.append("Mode: native f32 (1-50M params, L3 cache fits)")
    else:
        config.mode = "fusion"
        config.strategy_name = "Layer Fusion (small model)"
        log.append("Mode: fusion (< 1M params, dispatch overhead negligible)")

    # --- Precision selection ---
    if has_gpu and has_tc:
        if hw.supports_bf16:
            config.mixed_precision = "bf16"
            log.append("Precision: BF16 (tensor cores + better dynamic range)")
        elif hw.supports_fp16:
            config.mixed_precision = "fp16"
            log.append("Precision: FP16 (tensor cores enabled)")
    elif has_gpu and hw.supports_fp16:
        config.mixed_precision = "fp16"
        log.append("Precision: FP16 (GPU without tensor cores)")

    # --- Feature flags ---
    # Enable fusion rules for MLP-heavy models
    if n_linear >= 2:
        config.enable_fusion = True
        log.append("Fusion rules: enabled (multiple linear layers)")

    # Enable sparse for large models with many linear layers
    if n_params >= 10_000_000 and n_linear >= 4:
        config.enable_sparse = True
        log.append("Sparse: enabled (large model, many linear layers)")

    # --- Chain compile ---
    if has_gpu and is_on_gpu:
        config.chain_compile = True
        log.append("Chain compile: enabled (GPU model, Triton kernels)")
    else:
        config.chain_compile = False
        log.append("Chain compile: disabled (CPU model)")

    # --- Checksum mode ---
    # Soft for medium+ models, off for small/perf-critical
    if n_params >= 10_000_000:
        config.checksum_mode = "soft"
        log.append("Checksum: soft (large model, safety validation)")
    else:
        config.checksum_mode = "off"
        log.append("Checksum: off (small model, maximum speed)")

    # --- Memory check ---
    if has_gpu and model_mb > gpu_mem_gb * 1024 * 0.8:
        # Model won't fit in GPU memory, force INT4
        config.quantize = "int4"
        config.mode = "quantized"
        config.strategy_name = "INT4 Quantized (GPU memory constraint)"
        log.append(f"WARNING: Model ({model_mb:.0f}MB) exceeds 80% GPU VRAM ({gpu_mem_gb:.1f}GB)")
        log.append("Switched to INT4 quantization for memory savings")

    config.search_log = log
    config.memory_mb = model_mb

    return config


# ============================================================================
# BENCHMARK TUNE — Measurement-based search
# ============================================================================

@dataclass
class _Candidate:
    """A single configuration candidate for benchmark evaluation."""
    name: str
    config: TuneConfig
    ms: float = float('inf')
    error: str = ""


def benchmark_tune(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    warmup: int = 3,
    runs: int = 10,
    candidates: Optional[List[str]] = None,
) -> TuneConfig:
    """Benchmark multiple optimization strategies and pick the fastest.

    Tries different combinations of mode, precision, and feature flags,
    measures actual inference time for each, and returns the winner.

    Args:
        model: PyTorch model (will be copied for each candidate)
        input_shape: Input tensor shape
        warmup: Warmup iterations per candidate
        runs: Timed iterations per candidate
        candidates: Optional list of strategy names to try.
                    Default: auto-selected based on model/hardware.

    Returns:
        TuneConfig for the fastest strategy, with benchmark results in search_log.

    Examples:
        config = benchmark_tune(model, (1, 768))
        print(config.summary())
        fast_model = config.apply(model)
    """
    import copy
    from .profiler import benchmark_inference, detect_hardware

    hw = detect_hardware()
    has_gpu = hw.has_cuda
    n_params = sum(p.numel() for p in model.parameters())
    is_on_gpu = any(p.is_cuda for p in model.parameters())

    # Build candidate list
    if candidates is None:
        candidates = _build_candidates(model, has_gpu, is_on_gpu, n_params, hw)
    else:
        candidates = [_make_candidate(name, has_gpu) for name in candidates]

    log = []
    log.append(f"Benchmarking {len(candidates)} strategies on {hw.gpu_name if has_gpu else 'CPU'}...")
    log.append(f"Model: {n_params/1e6:.1f}M params, input={input_shape}")
    log.append(f"{'-'*56}")

    best = None
    for cand in candidates:
        try:
            # Copy model for this candidate
            test_model = copy.deepcopy(model)
            test_model.eval()

            # Apply optimization
            optimized = cand.config.apply(test_model)

            # Benchmark
            ms = benchmark_inference(optimized, input_shape, warmup=warmup, runs=runs)
            cand.ms = ms

            marker = ""
            if best is None or ms < best.ms:
                best = cand
                marker = " <-- best"

            log.append(f"  {cand.name:<35} {ms:>8.2f} ms{marker}")

        except Exception as e:
            cand.error = str(e)
            log.append(f"  {cand.name:<35} {'FAILED':>8} ({str(e)[:40]})")

    log.append(f"{'-'*56}")

    if best is None:
        # All candidates failed, return default
        fallback = TuneConfig(mode="fusion", strategy_name="Fallback (all candidates failed)")
        fallback.search_log = log
        return fallback

    # Return winner config with benchmark data
    result = best.config
    result.inference_ms = best.ms
    result.strategy_name = f"Winner: {best.name}"
    result.search_log = log
    result.memory_mb = n_params * 4 / 1024 / 1024

    speedups = [c for c in candidates if c.ms < float('inf') and c != best]
    if speedups:
        slowest = max(c.ms for c in candidates if c.ms < float('inf'))
        log.append(f"Best: {best.name} ({best.ms:.2f}ms)")
        log.append(f"Speedup vs slowest: {slowest / best.ms:.2f}x")

    return result


def _build_candidates(
    model: nn.Module,
    has_gpu: bool,
    is_on_gpu: bool,
    n_params: int,
    hw,
) -> List[_Candidate]:
    """Build a smart list of candidates based on model and hardware."""
    candidates = []

    # Always try: baseline fusion (always works)
    candidates.append(_Candidate(
        name="Fusion Only",
        config=TuneConfig(
            mode="fusion", chain_compile=False,
            enable_fusion=True, checksum_mode="off",
        ),
    ))

    # Native f32 (for small/medium models)
    if n_params < 100_000_000:
        candidates.append(_Candidate(
            name="Native f32 (Rust GEMM)",
            config=TuneConfig(
                mode="native", chain_compile=False,
                enable_fusion=True, checksum_mode="off",
            ),
        ))

    # INT4 quantized (for large models)
    if n_params >= 5_000_000:
        candidates.append(_Candidate(
            name="INT4 Quantized",
            config=TuneConfig(
                mode="quantized", quantize="int4",
                chain_compile=False, enable_fusion=True,
                checksum_mode="off",
            ),
        ))

    # INT8 quantized
    if n_params >= 1_000_000:
        candidates.append(_Candidate(
            name="INT8 Dynamic Quantized",
            config=TuneConfig(
                mode="fusion", quantize="int8",
                chain_compile=False, enable_fusion=True,
                checksum_mode="off",
            ),
        ))

    # Transformer path
    if _detect_transformer(model):
        candidates.append(_Candidate(
            name="Transformer (Rust-native)",
            config=TuneConfig(
                mode="transformer", chain_compile=False,
                enable_fusion=True, checksum_mode="off",
            ),
        ))

    # GPU-specific candidates
    if has_gpu and is_on_gpu:
        # Fusion + torch.compile chain
        candidates.append(_Candidate(
            name="Fusion + torch.compile (Triton)",
            config=TuneConfig(
                mode="fusion", chain_compile=True,
                enable_fusion=True, checksum_mode="off",
            ),
        ))

        # FP16 variants
        if hw.supports_fp16:
            candidates.append(_Candidate(
                name="Fusion + FP16",
                config=TuneConfig(
                    mode="fusion", chain_compile=False,
                    enable_fusion=True, mixed_precision="fp16",
                    checksum_mode="off",
                ),
            ))

        # BF16 variants (better numerical stability)
        if hw.supports_bf16:
            candidates.append(_Candidate(
                name="Fusion + BF16",
                config=TuneConfig(
                    mode="fusion", chain_compile=False,
                    enable_fusion=True, mixed_precision="bf16",
                    checksum_mode="off",
                ),
            ))

        # Full pipeline: chain compile + mixed precision
        if hw.supports_bf16:
            candidates.append(_Candidate(
                name="Full Pipeline (chain + BF16)",
                config=TuneConfig(
                    mode="fusion", chain_compile=True,
                    enable_fusion=True, mixed_precision="bf16",
                    checksum_mode="off",
                ),
            ))

    return candidates


def _make_candidate(name: str, has_gpu: bool) -> _Candidate:
    """Create a candidate from a strategy name string."""
    presets = {
        "fusion": TuneConfig(mode="fusion", enable_fusion=True),
        "native": TuneConfig(mode="native", enable_fusion=True),
        "quantized": TuneConfig(mode="quantized", quantize="int4", enable_fusion=True),
        "int8": TuneConfig(mode="fusion", quantize="int8", enable_fusion=True),
        "transformer": TuneConfig(mode="transformer", enable_fusion=True),
        "fp16": TuneConfig(mode="fusion", enable_fusion=True, mixed_precision="fp16"),
        "bf16": TuneConfig(mode="fusion", enable_fusion=True, mixed_precision="bf16"),
        "chain": TuneConfig(mode="fusion", chain_compile=True, enable_fusion=True),
    }
    config = presets.get(name.lower(), TuneConfig(mode="fusion"))
    return _Candidate(name=name, config=config)


# ============================================================================
# CONVENIENCE API
# ============================================================================

def auto_tune(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    mode: str = "quick",
    **kwargs,
) -> TuneConfig:
    """Auto-tune model optimization — single entry point.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        mode: Tuning mode:
            - "quick": Heuristic selection (< 100ms, no benchmarking)
            - "benchmark": Full benchmark search (seconds, most accurate)
        **kwargs: Additional arguments passed to quick_tune or benchmark_tune

    Returns:
        TuneConfig with optimal settings.

    Examples:
        # Quick heuristic (recommended for most cases)
        config = auto_tune(model, (1, 768))
        fast_model = config.apply(model)

        # Full benchmark search (when you need maximum performance)
        config = auto_tune(model, (1, 768), mode="benchmark")
        fast_model = config.apply(model)
    """
    if mode == "quick":
        return quick_tune(model, input_shape, **kwargs)
    elif mode == "benchmark":
        return benchmark_tune(model, input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown auto_tune mode: {mode}. Use 'quick' or 'benchmark'.")


# ============================================================================
# HELPERS
# ============================================================================

def _detect_transformer(model: nn.Module) -> bool:
    """Check if model has transformer architecture (attention + layernorm blocks)."""
    has_attn = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())

    # Also check for common transformer block patterns
    for attr in ['h', 'layers', 'blocks', 'block', 'encoder', 'decoder']:
        child = getattr(model, attr, None)
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            return True
        # Check one level deeper
        for sub in model.children():
            sub_child = getattr(sub, attr, None)
            if isinstance(sub_child, nn.ModuleList) and len(sub_child) > 0:
                return True

    return has_attn and has_ln
