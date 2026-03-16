# Hypatia

A hardware-agnostic symbolic compiler for PyTorch, built on **Geometric Algebra** and **E-graph equality saturation**. Hypatia optimizes deep learning models at the graph level — fusing operations, quantizing weights, and dispatching to native kernels — all through a unified mathematical framework.

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

| Model | Optimization | Speedup vs PyTorch |
|-------|-------------|-------------------|
| MLP (768->3072->768) | Fused Linear+ReLU | 1.3-2.1x |
| GPT-2 XL | INT4 Quantization | 1.2x (+ 75% memory reduction) |
| Transformer | Native Rust Forward | 1.5-3.0x |
| Sparse Model (90%) | CSR GEMM | 2-5x |

### Qwen2.5-0.5B (494M params) on RTX 4070 Laptop

| Strategy | Latency | vs CPU FP32 |
|----------|---------|-------------|
| MLP Block INT4 (Hypatia) | **0.6 ms** | **2289x** |
| GPU FP16 + torch.compile | 8.9 ms | 164x |
| GPU FP16 (Tensor Cores) | 31.4 ms | 46x |
| GPU FP32 | 33.8 ms | 43x |
| CPU INT8 Dynamic | 793 ms | 1.8x |
| CPU FP32 (baseline) | 1449 ms | 1.0x |

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

- [Manifesto](MANIFESTO.md) — Project philosophy and vision
- [Roadmap](ROADMAP) — Development phases and milestones
- [Future Work](FUTURE_WORK.md) — Planned features and TODOs
- [Contributing](CONTRIBUTING.md) — How to contribute
- [Academic Report](docs/academic_report.md) — Technical details and benchmarks

## License

MIT License — see [LICENSE](LICENSE) for details.
