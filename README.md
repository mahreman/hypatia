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
| MLP (768→3072→768) | Fused Linear+ReLU | 1.3-2.1x |
| GPT-2 XL | INT4 Quantization | 1.2x (+ 75% memory reduction) |
| Transformer | Native Rust Forward | 1.5-3.0x |
| Sparse Model (90%) | CSR GEMM | 2-5x |

## Documentation

- [Manifesto](MANIFESTO.md) — Project philosophy and vision
- [Roadmap](ROADMAP) — Development phases and milestones
- [Future Work](FUTURE_WORK.md) — Planned features and TODOs
- [Contributing](CONTRIBUTING.md) — How to contribute
- [Academic Report](docs/academic_report.md) — Technical details and benchmarks

## License

MIT License — see [LICENSE](LICENSE) for details.
