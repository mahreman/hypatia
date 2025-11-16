# hypatia_core/fused_modules.py

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.cpp_extension import load as _load_ext
except ImportError:
    _load_ext = None

# -----------------------------------------------------------------------------
# C++/CUDA extension yükleme (opsiyonel)
# -----------------------------------------------------------------------------

_FUSED_LINEAR_RELU_EXT = None
_HAS_CUDA_KERNEL = False

def _load_fused_linear_relu_ext() -> None:
    global _FUSED_LINEAR_RELU_EXT, _HAS_CUDA_KERNEL

    if _FUSED_LINEAR_RELU_EXT is not None:
        return

    if _load_ext is None:
        _FUSED_LINEAR_RELU_EXT = None
        _HAS_CUDA_KERNEL = False
        return

    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to hypatia_core root, then into csrc
    parent_dir = os.path.dirname(this_dir)
    src_cpp = os.path.join(parent_dir, "csrc", "fused_linear_relu.cpp")
    src_cu  = os.path.join(parent_dir, "csrc", "fused_linear_relu_kernel.cu")

    if not (os.path.exists(src_cpp) and os.path.exists(src_cu)):
        _FUSED_LINEAR_RELU_EXT = None
        _HAS_CUDA_KERNEL = False
        return

    try:
        _FUSED_LINEAR_RELU_EXT = _load_ext(
            name="hypatia_fused_linear_relu",
            sources=[src_cpp, src_cu],
            verbose=False,
        )
        _HAS_CUDA_KERNEL = True
    except Exception as exc:
        # Build sırasında hata olursa, sessizce fallback'e geç
        print(f"[Hypatia] Warning: failed to build fused_linear_relu CUDA extension: {exc}")
        _FUSED_LINEAR_RELU_EXT = None
        _HAS_CUDA_KERNEL = False


# -----------------------------------------------------------------------------
# Autograd.Function
# -----------------------------------------------------------------------------

class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Eğer CUDA kernel mevcutsa:
          y = ReLU( input @ weight.T + bias ) tek kernel ile hesaplanır.
        Aksi halde:
          F.linear + F.relu fallback'ine düşer.
        """
        use_cuda_kernel = (
            input.is_cuda
            and weight.is_cuda
            and (bias is None or bias.is_cuda)
            and _HAS_CUDA_KERNEL
        )

        if use_cuda_kernel:
            # C++/CUDA kernel çağrısı
            y = _FUSED_LINEAR_RELU_EXT.forward(input, weight, bias)
        else:
            # Graph-level fusion fallback (kernel fusion yok)
            y = F.relu(F.linear(input, weight, bias))

        ctx.save_for_backward(input, weight, bias if bias is not None else None, y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward'ı PyTorch op'larıyla yapıyoruz.
        Bu, kernel-level fused backward değil ama:
          - correctness için yeterli
          - training'de güvenli
        """
        input, weight, bias, output = ctx.saved_tensors

        # grad_output shape: [N, out_features]
        # output: [N, out_features]  (ReLU sonrası)
        grad_input = grad_weight = grad_bias = None

        # dL/dz = dL/dy * 1_{y>0}, z = x @ W^T + b, y = ReLU(z)
        grad_z = grad_output
        if output is not None:
            grad_z = grad_output * (output > 0).to(grad_output.dtype)

        if ctx.needs_input_grad[0]:
            # grad_input = grad_z @ W
            grad_input = grad_z.matmul(weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_z^T @ input
            grad_weight = grad_z.t().matmul(input)
        if ctx.needs_input_grad[2] and bias is not None:
            # grad_bias = sum_n grad_z[n, :]
            grad_bias = grad_z.sum(dim=0)

        return grad_input, grad_weight, grad_bias


# -----------------------------------------------------------------------------
# nn.Module wrapper (Hypatia FX e-graph'in ürettiği modül)
# -----------------------------------------------------------------------------

class FusedLinearReLU(nn.Module):
    """
    Hypatia'nın e-graph'ı tarafından kullanılan fused Linear+ReLU modülü.

    - State dict uyumluluğu için içerde bir nn.Linear tutuyoruz (self.fc).
    - CUDA'da mümkünse C++/CUDA kernel'i, aksi halde F.linear+ReLU kullanıyoruz.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)

        # Extension'ı lazy-load
        _load_fused_linear_relu_ext()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and _HAS_CUDA_KERNEL:
            return FusedLinearReLUFunction.apply(x, self.fc.weight, self.fc.bias)
        else:
            # CPU veya extension yoksa: normal path
            return F.relu(self.fc(x))


# Backward compatibility alias
HypatiaFusedLinearReLU = FusedLinearReLU


# -----------------------------------------------------------------------------
# Helper function for FX graph reconstruction
# -----------------------------------------------------------------------------

def create_fused_linear_relu_from_tensors(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> FusedLinearReLU:
    """
    Create a FusedLinearReLU module from existing weight and bias tensors.

    This helper is called from Rust (fx_bridge::reconstruct_fused_linear_relu)
    during graph reconstruction.
    """
    if not torch.is_tensor(weight):
        raise TypeError(f"weight must be a Tensor, got {type(weight)}")

    device = weight.device
    dtype = weight.dtype

    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D (out_features, in_features), got shape={tuple(weight.shape)}"
        )

    out_features, in_features = weight.shape
    use_bias = bias is not None and torch.is_tensor(bias)

    # Create module on correct device & dtype
    fused = FusedLinearReLU(
        in_features=in_features,
        out_features=out_features,
        bias=use_bias,
        device=device,
        dtype=dtype,
    )

    # Copy parameters
    with torch.no_grad():
        fused.fc.weight.copy_(weight)
        if use_bias:
            fused.fc.bias.copy_(bias.to(device=device, dtype=dtype))

    return fused


__all__ = ["FusedLinearReLU", "HypatiaFusedLinearReLU", "FusedLinearReLUFunction",
           "create_fused_linear_relu_from_tensors"]
