"""
Hypatia CUDA fused kernels module
"""

# Try to import CUDA extension if available
try:
    import hypatia_core._linear_relu_cuda as _C
    CUDA_AVAILABLE = True
except ImportError:
    _C = None
    CUDA_AVAILABLE = False

__all__ = ['CUDA_AVAILABLE']
