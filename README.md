# Hypatia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahreman/hypatia/blob/main/notebooks/hypatia_colab_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A hardware-aware symbolic compiler for PyTorch, built on **Geometric Algebra** and **E-graph equality saturation**. Hypatia optimizes deep learning models at the graph level — fusing operations, quantizing weights, and dispatching to native kernels — all through a unified mathematical framework.

## Vision

To overcome the limitations of the von Neumann architecture and the unsustainable costs of modern AI training by creating a more elegant and efficient paradigm for computation.

## Key Features

- **E-graph Optimizer**: Equality saturation discovers optimal fusion patterns (Linear+ReLU, GELU+MLP, Attention)
- **torch.compile Backend**: Drop-in `torch.compile(model, backend='hypatia')` integration
- **Native Rust Kernels**: SIMD-vectorized forward/backward pass bypassing PyTorch dispatch
- **INT4/INT8 Quantization**: Block quantization with Rayon parallelism (11-16x speedup)
- **Sparse Tensor IR**: CSR format with automatic magnitude pruning
- **Mixed Precision**: FP16/BF16 storage with FP32 accumulation
- **Neuromorphic Computing**: ReLU-to-LIF neuron conversion via equality saturation
- **CUDA Fused Kernels**: cuBLAS-backed Linear+ReLU, GELU+MLP, multi-head Attention
- **Geometric Algebra Core**: 2D/3D MultiVector operations with symbolic differentiation
- **Visualization**: DOT graph export, ASCII trees, HTML optimization reports

## Quick Start

### Installation

```bash
# Prerequisites: Rust (1.70+), Python (3.8+), PyTorch (2.0+)
pip install -r requirements.txt

# Build the Rust extension
cd hypatia_core
pip install maturin
maturin develop --release
```

### Usage

#### As a torch.compile backend

```python
import torch
import hypatia_core

model = torch.nn.Sequential(
    torch.nn.Linear(768, 3072),
    torch.nn.ReLU(),
    torch.nn.Linear(3072, 768),
)

# Hypatia backend is auto-registered on import
compiled = torch.compile(model, backend="hypatia")
output = compiled(torch.randn(32, 768))
```

#### Direct model optimization

```python
import hypatia_core

# In-place fusion (no torch.compile overhead)
optimized_model = hypatia_core.optimize(model)
```

#### Native Rust forward pass

```python
from hypatia_core import NativeModel

native = NativeModel(model)
output = native.forward(input_tensor)  # Bypasses PyTorch dispatch
```

#### Quantized inference

```python
from hypatia_core import QuantizedModel

quantized = QuantizedModel(model, bits=4)
output = quantized.forward(input_tensor)  # INT4 quantized inference
```

#### Symbolic algebra

```python
from hypatia_core import parse_expr, optimize_ast

expr = parse_expr("(+ (* x 0) (* 1 y))")
optimized = optimize_ast(expr)  # Simplifies to "y"
```

### Running Tests

```bash
cd hypatia_core
python -m pytest tests/ -v
```

### Debug Mode

```bash
export HYPATIA_DEBUG_FX=1           # Verbose FX graph logging
export HYPATIA_CHECKSUM_MODE=strict # Parameter validation
python your_script.py
```

## Architecture

```
                    PyTorch Model
                         |
                    torch.compile
                         |
                  ┌──────┴──────┐
                  │ FX Graph    │
                  │ (Python)    │
                  └──────┬──────┘
                         |
              ┌──────────┴──────────┐
              │  Rust Core Engine   │
              │                     │
              │  S-expr → E-graph   │
              │  Equality Saturation│
              │  Fusion Rules       │
              │  Cost Extraction    │
              └──────────┬──────────┘
                         |
              ┌──────────┴──────────┐
              │  Backend Dispatch   │
              ├─────────────────────┤
              │ CPU: AVX2/AVX-512   │
              │ GPU: CUDA/cuBLAS    │
              │ Quantized: INT4/INT8│
              │ Sparse: CSR GEMM    │
              │ Neuromorphic: LIF   │
              └─────────────────────┘
```

## Project Structure

```
hypatia/
├── hypatia_core/
│   ├── src/                 # Rust source (core engine)
│   │   ├── egraph_optimizer.rs  # E-graph equality saturation
│   │   ├── fx_bridge.rs         # PyTorch FX ↔ S-expression bridge
│   │   ├── python_bindings.rs   # PyO3 Python API
│   │   ├── symbolic.rs          # Symbolic algebra & differentiation
│   │   ├── quantize.rs          # INT4/INT8 quantization
│   │   ├── native_ops.rs        # Fused SIMD kernels
│   │   ├── neuromorphic.rs      # ANN→SNN conversion
│   │   ├── sparse_ops.rs        # Sparse tensor operations
│   │   ├── mixed_precision.rs   # FP16/BF16 support
│   │   └── visualization.rs     # Graph visualization
│   ├── csrc/                # CUDA fused kernels
│   ├── hypatia_core/        # Python modules
│   ├── tests/               # Test suite (200+ tests)
│   └── examples/            # Benchmarks and demos
├── examples/                # Real model demos (GPT-2, DistilBERT)
├── docs/                    # Academic reports
├── MANIFESTO.md             # Project philosophy
├── ROADMAP                  # Development phases
└── CONTRIBUTING.md          # Contribution guidelines
```

## Benchmarks

### Hypatia vs TorchInductor (Fair GPU-to-GPU, Output-Verified)

All benchmarks use `torch.cuda.synchronize()`, 5 warmup + 100 measurement iterations, and **output correctness verification** (cosine similarity between vanilla and compiled outputs).

**RTX 4070 Laptop GPU:**

| Model | Inductor (max-autotune) | Hypatia | Hyp/Ind | Verified |
|-------|------------------------|---------|---------|----------|
| Small MLP (235K params) | 0.281 ms | 1.176 ms | 4.19x | ✅ |
| Medium MLP (4.9M) | 0.468 ms | 2.766 ms | 5.91x | ✅ |
| Large MLP (19.4M) | 0.901 ms | 2.010 ms | 2.23x | ✅ |
| **Transformer Block (3.2M)** | **3.004 ms** | **2.570 ms** | **0.86x** | ✅ |

**Google Colab T4 GPU:**

| Model | Inductor (max-autotune) | Hypatia | Hyp/Ind | Cosine Sim | Verified |
|-------|------------------------|---------|---------|------------|----------|
| Deep MLP (31.5M) | 1.42 ms | 1.33 ms | **0.94x** | 1.000000 | ✅ |
| GPT-2 Small (28.4M) | — | — | — | 0.176 | ❌ Graph breaks |
| Wide Transformer (75.6M) | — | — | — | 0.136 | ❌ Graph breaks |
| BERT-Base Encoder (85.1M) | — | — | — | 0.018 | ❌ Graph breaks |

> **Hyp/Ind < 1.0 = Hypatia faster.** Hypatia wins on Transformer blocks and deep MLPs where E-graph discovers fusion patterns that Inductor's greedy matcher misses. Multi-layer Transformer models currently suffer from graph breaks (see [Known Limitations](#known-limitations)).

### Qwen2.5-0.5B (494M params) on RTX 4070 Laptop

| Strategy | Latency | vs CPU FP32 |
|----------|---------|-------------|
| MLP Block INT4 (Hypatia) | **0.6 ms** | **2289x** |
| GPU FP16 + torch.compile | 8.9 ms | 164x |
| GPU FP16 (Tensor Cores) | 31.4 ms | 46x |
| GPU FP32 | 33.8 ms | 43x |
| CPU INT8 Dynamic | 793 ms | 1.8x |
| CPU FP32 (baseline) | 1449 ms | 1.0x |

### Known Limitations

- **Graph breaks on multi-layer Transformers**: Custom pybind11 ops (`hypatia_fused_gelu_mlp`) can't be traced by `torch._dynamo`, causing partial computation. Fix in progress: `torch.library` registration.
- **E-graph rewrite correctness**: Fused ops work for single blocks but produce incorrect results on deep pre-norm Transformer stacks (12+ layers). Fix in progress: LayerNorm-aware rewrite guards.
- **Inductor faster on standard MLPs**: For simple Linear+ReLU chains, Inductor's native Triton autotune is 2-6x faster. Hypatia's value is in complex fusion patterns.

## Numerical Stability

Hypatia guarantees numerical equivalence **within configurable tolerance**, not
bit-level identity. Floating-point arithmetic is non-associative (`(a+b)+c` !=
`a+(b+c)` at ULP level), and kernel fusion inherently reorders operations.

Default thresholds:
- **Max absolute diff**: < 1e-5 (FP32), < 1e-3 (FP16/BF16)
- **Cosine similarity**: > 0.9999
- **INT4 quantization**: cosine > 0.995 (expected quality loss)

```python
from hypatia_core import SemanticValidator
validator = SemanticValidator(tolerance=1e-5)
result = validator.validate_models(original, optimized, input_shape=(1, 64))
print(f"Max diff: {result['max_diff']:.2e}, Cosine: {result['cosine_similarity']:.6f}")
```

## Benchmark Dashboard

Generate interactive HTML reports with hardware info, animated charts, and strategy comparison:

```python
from hypatia_core.dashboard import generate_benchmark_dashboard
generate_benchmark_dashboard(
    model_name="MyModel",
    results={"CPU FP32": 100.0, "GPU FP16+compile": 5.0},
    hw=detect_hardware(),
    output_path="report.html",
)
```

## Documentation

- **[Colab Demo](https://colab.research.google.com/github/mahreman/hypatia/blob/main/notebooks/hypatia_colab_demo.ipynb)** — Interactive notebook, run in browser
- [Academic Paper (EN)](docs/hypatia_paper.md) — Full technical paper with benchmarks
- [Academic Paper (TR)](docs/hypatia_paper_tr.md) — Turkce akademik makale
- [User Guide](hypatia_core/docs/USER_GUIDE.md) — Installation, usage, configuration
- [Manifesto](MANIFESTO.md) — Project philosophy and vision
- [Roadmap](ROADMAP) — Development phases and milestones
- [Contributing](CONTRIBUTING.md) — How to contribute

## License

MIT License — see [LICENSE](LICENSE) for details.
