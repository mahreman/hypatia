# Hypatia User Guide

Hardware-Aware Symbolic Compiler for PyTorch.

## Quick Start

```python
import torch
import hypatia_core

# 1. Register backend (auto on import)
model = MyModel()

# 2. Option A: torch.compile integration
compiled = torch.compile(model, backend="hypatia")
output = compiled(input_tensor)

# 2. Option B: Direct optimization
fast_model = hypatia_core.optimize(model)
output = fast_model(input_tensor)

# 2. Option C: Auto-tuned optimization
from hypatia_core.autotuner import auto_tune
config = auto_tune(model, input_shape=(1, 768))
fast_model = config.apply(model)
```

## Installation

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Build Hypatia (Rust + Python)
cd hypatia_core
pip install maturin
maturin develop --release
```

## Optimization Modes

### `optimize(model)` — Direct Model Optimization

| Mode | Best For | Speed | Memory |
|------|----------|-------|--------|
| `auto` | Auto-select | Optimal | Optimal |
| `native` | Small models (<50M params) | 2-6x | Same |
| `quantized` | Large models (>50M params) | 15x+ | 7x less |
| `transformer` | Transformer architectures | 3-8x | Same |
| `fusion` | Any model (fallback) | 1.2-2x | Less |

```python
# Auto-select (recommended)
fast = hypatia_core.optimize(model)

# Force specific mode
fast = hypatia_core.optimize(model, mode='native')
fast = hypatia_core.optimize(model, quantize='int4')
fast = hypatia_core.optimize(model, mode='fusion')
```

### `torch.compile(backend='hypatia')` — E-graph + Triton Pipeline

Two-phase optimization:
1. **Phase 1:** E-graph equality saturation (Rust) — structural fusion rewrites
2. **Phase 2:** torch.compile/Triton (GPU) — kernel-level fusion

```python
compiled = torch.compile(model, backend="hypatia")
# First call triggers compilation, subsequent calls are fast
output = compiled(input_tensor)
```

### Auto-Tuner

```python
from hypatia_core.autotuner import auto_tune, quick_tune, benchmark_tune

# Quick heuristic (< 100ms, no benchmarking)
config = quick_tune(model, (batch, seq_len, hidden))
print(config.summary())
fast_model = config.apply(model)

# Full benchmark search (tries multiple strategies)
config = benchmark_tune(model, (1, 768), warmup=5, runs=20)
print(config.summary())
fast_model = config.apply(model)
```

## Profiling

```python
from hypatia_core.profiler import profile_model, detect_hardware, roofline_analysis

# Hardware info
hw = detect_hardware()
print(hw.summary())  # GPU name, VRAM, Tensor Cores, peak TFLOPS

# Model profiling (FLOPs + timing)
profile = profile_model(model, (1, 3, 224, 224))
print(profile.summary())  # Per-layer FLOPs, throughput, GPU utilization

# Roofline analysis (compute-bound vs memory-bound)
analysis = roofline_analysis(profile)
print(analysis["bottleneck"])       # "compute" or "memory"
print(analysis["recommendation"])   # Actionable optimization advice
```

## GPU Features

### Mixed Precision (FP16/BF16)

```python
from hypatia_core.mixed_precision import convert_to_mixed_precision

# Auto FP16 for GPU models
model_fp16 = convert_to_mixed_precision(model, precision='fp16')

# BF16 (better numerical stability on Ampere+)
model_bf16 = convert_to_mixed_precision(model, precision='bf16')
```

### Fused GPU Modules

```python
from hypatia_core.fused_modules import FusedGeluMLP, FusedAttention, FusedTransformerBlock

# Fused GELU MLP (Linear -> GELU -> Linear in one kernel)
mlp = FusedGeluMLP(hidden=768, mlp_hidden=3072)

# Fused Attention (Q/K/V projection + SDPA in one call)
attn = FusedAttention(hidden=768, n_heads=12)

# Full transformer block
block = FusedTransformerBlock(hidden=768, n_heads=12, mlp_hidden=3072)
```

### Dispatch Chain (4 tiers)

Hypatia automatically selects the fastest available backend:

1. **Custom CUDA extensions** (if built with nvcc)
2. **torch.compile + Triton** (no nvcc needed, GPU only)
3. **Rust native kernels** (CPU, OpenBLAS GEMM)
4. **Eager PyTorch** (universal fallback)

## Sparse Models

```python
from hypatia_core.sparse import sparsify_model, model_sparsity_report

# Prune and convert to sparse representation
sparse_model = sparsify_model(model, sparsity=0.5)
print(model_sparsity_report(sparse_model))
```

## Numerical Stability

**Important:** Hypatia guarantees numerical equivalence **within configurable
tolerance**, not bit-level identity. Floating-point arithmetic is non-associative:
`(a + b) + c` and `a + (b + c)` can differ at the ULP (Unit in the Last Place)
level on real hardware. Since kernel fusion inherently reorders operations,
bit-exact reproduction is not possible.

Default validation thresholds:
- **Max absolute difference**: < 1e-5 (FP32), < 1e-3 (FP16/BF16)
- **Cosine similarity**: > 0.9999
- **INT4 quantization**: cosine similarity > 0.995 (expected quality loss)

## Semantic Validation

Verify that optimized models produce numerically equivalent outputs:

```python
from hypatia_core.semantic_validation import SemanticValidator

validator = SemanticValidator(tolerance=1e-5)
result = validator.validate_models(original_model, optimized_model, input_shape=(1, 64))
print(f"Valid: {result['is_valid']}")
print(f"Max diff: {result['max_diff']:.2e}")
print(f"Cosine similarity: {result['cosine_similarity']:.6f}")
```

Checksum modes:
- `HYPATIA_CHECKSUM_MODE=strict` — Tight numerical validation (atol=1e-5)
- `HYPATIA_CHECKSUM_MODE=soft` — Structural + lenient forward pass (atol=1e-3)
- `HYPATIA_CHECKSUM_MODE=off` — No validation (fastest, production use)

## GPU Dispatch Chain

Hypatia uses a 4-tier fallback chain for maximum performance:

```
1. Custom CUDA Extension (fused kernels, if Ninja + CUDA available)
   |-- fused_linear_relu, fused_gelu_mlp, fused_attention, fused_layernorm
   |
2. torch.compile + Triton (GPU auto-tuned kernel generation)
   |-- max-autotune mode, epilogue fusion, memory coalescing
   |
3. Rust Native (CPU optimized GEMM, activations, quantization)
   |-- no Python overhead, SIMD-friendly memory layout
   |
4. Eager PyTorch (fallback, always works)
```

Each tier is tried in order; the first available one is used.

## Visualization

```python
from hypatia_core.visualization import model_summary, generate_html_report

# Model summary with FLOPs breakdown
model_summary(model, input_shape=(1, 3, 224, 224))

# HTML optimization report
generate_html_report("(linear w2 b2 (relu (linear w1 b1 x)))")
```

## Benchmark Dashboard

Generate interactive HTML benchmark reports:

```python
from hypatia_core.dashboard import generate_benchmark_dashboard

generate_benchmark_dashboard(
    model_name="MyModel",
    results={
        "CPU FP32": 100.0,
        "GPU FP16": 10.0,
        "GPU FP16+compile": 5.0,
    },
    hw=detect_hardware(),
    model_info={"n_params": 1e6, "arch": "transformer"},
    tuner_name="Transformer (Rust-native block)",
    output_path="benchmark_report.html",
)
```

The dashboard includes: hardware/model cards, animated bar charts (log scale),
auto-tuner recommendation, roofline analysis, and token generation test results.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPATIA_DEBUG_FX` | `0` | Enable FX graph debug logging |
| `HYPATIA_CHAIN_COMPILE` | `1` | Chain torch.compile after e-graph |
| `HYPATIA_CHECKSUM_MODE` | `off` | Validation mode: off/soft/strict |
| `HYPATIA_VALIDATION_TOLERANCE` | `1e-4` | Numerical tolerance for validation |
| `HYPATIA_ENABLE_LINRELU_FUSION` | `0` | Enable Linear+ReLU fusion rules |
| `HYPATIA_ENABLE_SPARSE` | `0` | Enable sparse optimization rules |
| `HYPATIA_MIXED_PRECISION` | (unset) | Mixed precision: fp16/bf16 |
| `HYPATIA_TARGET_NEUROMORPHIC` | `0` | Prefer neuromorphic operators |

## Supported Hardware

| GPU | Compute | Tensor Cores | BF16 | Tested |
|-----|---------|-------------|------|--------|
| RTX 4070 Laptop | SM 8.9 | Ada Lovelace | Yes | Yes |
| RTX 4090 | SM 8.9 | Ada Lovelace | Yes | - |
| A100 | SM 8.0 | Ampere | Yes | - |
| H100 | SM 9.0 | Hopper | Yes | - |
| RTX 3090 | SM 8.6 | Ampere | Yes | - |
| CPU (any) | - | - | - | Yes |
