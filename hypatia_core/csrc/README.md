# Hypatia CUDA Kernels (csrc/)

This directory contains C++/CUDA implementations of fused operations for Hypatia.

## Files

### fused_linear_relu_kernel.cu
CUDA kernel implementation for fused Linear+ReLU operation.

**Features:**
- Single kernel computes: `y = ReLU(x @ W^T + b)`
- Naive GEMM implementation (research prototype)
- Template kernel for FP32, FP16, BF16 support

**Performance:**
- **Current:** Naive 2D grid kernel (simple but not optimal)
- **Future:** Can be replaced with cuBLAS + epilogue fusion or optimized tile-based GEMM

### fused_linear_relu.cpp
C++ PyTorch binding for the CUDA kernel.

**Features:**
- PyBind11 module registration
- Input validation
- Error checking
- c10::optional bias handling

## Building

The extension is built automatically via JIT compilation when first used:

```python
from hypatia_core.fused_modules import FusedLinearReLU

# First call triggers JIT compilation
layer = FusedLinearReLU(512, 256, device='cuda')
```

**Requirements:**
- CUDA toolkit (nvcc)
- PyTorch with CUDA support
- C++14 compatible compiler

**Build flags:**
- Automatically detects CUDA architecture
- Uses `torch.utils.cpp_extension.load()`
- Cached in `~/.cache/torch_extensions/`

## Usage

### Python API

```python
from hypatia_core.fused_modules import FusedLinearReLU

# Create fused layer
layer = FusedLinearReLU(in_features=512, out_features=256, bias=True, device='cuda')

# Forward pass (uses CUDA kernel)
x = torch.randn(32, 512, device='cuda')
y = layer(x)  # Single kernel: y = ReLU(x @ W^T + b)

# Backward pass (PyTorch ops for gradients)
loss = y.sum()
loss.backward()
```

### Automatic Fallback

If CUDA is unavailable or compilation fails, automatically falls back to:
```python
y = F.relu(F.linear(x, weight, bias))
```

No code changes needed - fallback is transparent.

## Implementation Details

### Forward Pass

**CUDA Kernel:**
```cuda
// 2D grid: (out_features, batch)
// Each thread computes one output element
acc = bias[out_idx]
for k in range(in_features):
    acc += input[batch_idx, k] * weight[out_idx, k]
output[batch_idx, out_idx] = max(acc, 0)  // ReLU
```

**Complexity:**
- FLOPs: 2 * batch * in_features * out_features + batch * out_features
- Memory: (batch * in_features + out_features * in_features + out_features) reads
- Kernel launches: 1 (fused)

### Backward Pass

**Implementation:** PyTorch operations (not fused)

```python
# ReLU gradient mask
grad_z = grad_output * (output > 0)

# Gradients
grad_input = grad_z @ weight
grad_weight = grad_z.T @ input
grad_bias = grad_z.sum(dim=0)
```

**Why not fused?**
- Correctness first - PyTorch ops are well-tested
- Training is less performance-critical than inference
- Can be optimized later if needed

## Performance Characteristics

### Current Implementation (Naive Kernel)

**Speedup:**
- Small models (256x2): ~1.0x (overhead)
- Medium models (1024x4): ~1.1x
- Large models (2048x4+): ~1.2-1.3x (TARGET)

**Bottleneck:**
- Naive GEMM is slower than cuBLAS
- But: Single kernel vs 2-3 kernels reduces overhead
- Memory bandwidth still benefits from fusion

### Future Optimizations

**Priority 1: cuBLAS + Epilogue**
Replace naive GEMM with cuBLAS, fuse ReLU via epilogue:
- Expected: 1.5-2.0x speedup over current
- Complexity: Medium (cuBLAS API)

**Priority 2: Tiled GEMM Kernel**
Hand-optimized tile-based GEMM with ReLU:
- Expected: 1.3-1.8x speedup (architecture-specific)
- Complexity: High (requires tuning per GPU)

**Priority 3: Multi-Precision**
FP16/BF16 support with Tensor Cores:
- Expected: 2-4x speedup on Ampere/Ada
- Complexity: Medium (template specialization)

## Debugging

### Check if kernel is being used

```python
from hypatia_core.fused_modules import _HAS_CUDA_KERNEL, _FUSED_LINEAR_RELU_EXT

print(f"CUDA kernel available: {_HAS_CUDA_KERNEL}")
if _HAS_CUDA_KERNEL:
    print(f"Extension module: {_FUSED_LINEAR_RELU_EXT}")
```

### Enable verbose build

Edit `fused_modules.py`:
```python
_FUSED_LINEAR_RELU_EXT = _load_ext(
    name="hypatia_fused_linear_relu",
    sources=[src_cpp, src_cu],
    verbose=True,  # Enable build output
)
```

### Common Issues

**"failed to build fused_linear_relu CUDA extension"**
→ Check: CUDA toolkit installed? `nvcc --version`
→ Check: PyTorch has CUDA? `torch.cuda.is_available()`

**Slower than baseline**
→ Expected for small models (overhead-bound)
→ Try larger batch/hidden dimensions
→ Naive kernel is not optimal - future versions will improve

**Numerical differences**
→ Should be < 1e-5 due to FP32 precision
→ If larger: check input data quality
→ Compare with test_fused_linear_relu.py

## Testing

```bash
# Unit test (correctness)
cd hypatia_core/examples
python test_fused_linear_relu.py

# CUDA extension test
cd ../tests
python test_cuda_extension.py

# Performance test
cd ../examples
python mlp_multiconfig_benchmark.py
```

## References

- PyTorch C++ Extension: https://pytorch.org/tutorials/advanced/cpp_extension.html
- CUDA C++ Programming: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- cuBLAS Epilogue Fusion: https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-api
