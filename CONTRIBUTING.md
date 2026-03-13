# Contributing to Hypatia

Thank you for your interest in contributing to Hypatia! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- **Rust** (stable, 1.70+) — for building the core engine
- **Python** (3.8+) — for the Python bindings and test suite
- **PyTorch** (2.0+) — required runtime dependency
- **maturin** — for building the Rust-Python bridge

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/mahreman/hypatia.git
cd hypatia

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Build the Rust extension
cd hypatia_core
maturin develop --release
```

### Running Tests

```bash
cd hypatia_core
python -m pytest tests/ -v
```

## How to Contribute

### Reporting Bugs

- Open a GitHub Issue with a clear description
- Include steps to reproduce the problem
- Attach error logs and environment info (OS, Python version, PyTorch version)

### Suggesting Features

- Open a GitHub Issue tagged as "enhancement"
- Describe the use case and expected behavior
- Reference relevant sections of the ROADMAP if applicable

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `master` for your changes
3. **Write tests** for any new functionality
4. **Ensure all tests pass** before submitting
5. **Submit a Pull Request** with a clear description

### Code Style

- **Rust**: Follow standard `rustfmt` conventions
- **Python**: Follow PEP 8 guidelines
- **Commit messages**: Use clear, descriptive messages (e.g., "Fix parameter binding for multi-layer models")

## Project Structure

```
hypatia/
├── hypatia_core/
│   ├── src/            # Rust source (core engine)
│   ├── csrc/           # CUDA kernels
│   ├── hypatia_core/   # Python modules
│   ├── tests/          # Test suite
│   ├── examples/       # Example scripts and benchmarks
│   └── benchmarks/     # Performance benchmarks
├── examples/           # Top-level demos (GPT-2, DistilBERT)
├── docs/               # Academic reports and documentation
└── README.md
```

## Areas Where Help is Needed

- **E-graph fusion rules**: Adding new optimization patterns
- **Test coverage**: Expanding tests for edge cases
- **Documentation**: Improving inline code docs and tutorials
- **Benchmarks**: Testing on different hardware (GPU, ARM, etc.)
- **New backends**: Metal, Vulkan, neuromorphic hardware support

## License

By contributing to Hypatia, you agree that your contributions will be licensed under the MIT License.
