"""
Fused PyTorch modules for Hypatia optimizations.

These modules combine multiple operations into single efficient implementations,
resulting from E-graph fusion optimizations.

Currently available:
  - HypatiaFusedLinearReLU: Linear + ReLU combination
"""

from typing import Optional
import math

import torch
import torch.nn as nn
from torch.autograd import Function

# Try to import CUDA extension
try:
    import hypatia_core._linear_relu_cuda as _C
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    _C = None
    CUDA_EXTENSION_AVAILABLE = False


class FusedLinearReLUFunction(Function):
    """
    Autograd Function for fused Linear+ReLU using CUDA kernels.

    Forward: y = relu(x @ W^T + b)
    Backward: Compute gradients w.r.t. input, weight, and bias
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        Args:
            input: Input tensor, shape (..., in_features)
            weight: Weight tensor, shape (out_features, in_features)
            bias: Bias tensor, shape (out_features,)

        Returns:
            output: Output tensor, shape (..., out_features)
        """
        # Ensure all tensors are contiguous and on same device/dtype
        input = input.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        # Call CUDA kernel
        output = _C.forward(input, weight, bias)

        # Save for backward
        ctx.save_for_backward(input, weight, bias, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: Gradient w.r.t. output, shape (..., out_features)

        Returns:
            grad_input: Gradient w.r.t. input, shape (..., in_features)
            grad_weight: Gradient w.r.t. weight, shape (out_features, in_features)
            grad_bias: Gradient w.r.t. bias, shape (out_features,)
        """
        input, weight, bias, output = ctx.saved_tensors

        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Call CUDA backward kernel
        grad_input, grad_weight, grad_bias = _C.backward(
            grad_output, input, weight, bias, output
        )

        return grad_input, grad_weight, grad_bias


class HypatiaFusedLinearReLU(nn.Module):
    """
    Fused Linear + ReLU module.

    Semantically equivalent to:
        y = torch.relu(nn.Linear(in_features, out_features)(x))

    Advantages:
      - Fewer kernel calls / reduced Python overhead
      - Fewer intermediate tensor allocations (cache/latency benefits)
      - Better instruction-level parallelism
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Create weight and bias parameters directly
        # This gives us more control for kernel fusion
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters using Kaiming initialization (good for ReLU)
        self.reset_parameters()

        # Packed weight buffer for future optimization
        # (prepacking for optimal memory layout)
        self.register_buffer("_packed_weight", None, persistent=False)

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear transformation followed by ReLU"""

        # Check if we can use CUDA kernel
        use_cuda_kernel = (
            CUDA_EXTENSION_AVAILABLE and
            x.is_cuda and
            self.weight.is_cuda and
            (self.bias is None or self.bias.is_cuda) and
            x.dtype == torch.float32 and
            self.weight.dtype == torch.float32 and
            (self.bias is None or self.bias.dtype == torch.float32)
        )

        if use_cuda_kernel:
            # Use fused CUDA kernel
            # TODO: In future, use packed weight for better performance
            # if self._packed_weight is None or self._packed_weight.shape != self.weight.shape:
            #     self._packed_weight = self.weight.contiguous()

            weight = self.weight.contiguous()
            bias = self.bias if self.bias is not None else torch.zeros(
                self.out_features, device=self.weight.device, dtype=self.weight.dtype
            )

            return FusedLinearReLUFunction.apply(x, weight, bias)
        else:
            # Fallback to PyTorch implementation
            # (for CPU, fp16/bf16, or when CUDA extension not available)
            output = torch.nn.functional.linear(x, self.weight, self.bias)
            return torch.relu(output)

    def __repr__(self):
        return (f"HypatiaFusedLinearReLU(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None})")


def create_fused_linear_relu_from_tensors(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> HypatiaFusedLinearReLU:
    """
    Create a HypatiaFusedLinearReLU module from existing weight and bias tensors.

    This helper is called from Rust (fx_bridge::reconstruct_fused_linear_relu)
    during graph reconstruction.

    Task:
      - Create HypatiaFusedLinearReLU from weight and bias tensors
      - Initialize module DIRECTLY on correct device & dtype
      - Copy parameters (now safe since same device/dtype)

    Args:
        weight: Linear layer weight tensor, shape [out_features, in_features]
        bias: Linear layer bias tensor, shape [out_features], or None

    Returns:
        HypatiaFusedLinearReLU module with copied parameters on same device as inputs

    Raises:
        TypeError: If weight is not a tensor
        ValueError: If weight shape is invalid

    Example:
        >>> weight = torch.randn(128, 256, device='cuda')  # out=128, in=256
        >>> bias = torch.randn(128, device='cuda')
        >>> module = create_fused_linear_relu_from_tensors(weight, bias)
        >>> module.fc.weight.device  # cuda:0
        >>> x = torch.randn(32, 256, device='cuda')
        >>> out = module(x)  # shape: [32, 128], device: cuda:0
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

    # 1) Create module DIRECTLY on correct device & dtype
    #    This is CRITICAL for CUDA compatibility - avoids device mismatch errors
    fused = HypatiaFusedLinearReLU(
        in_features=in_features,
        out_features=out_features,
        bias=use_bias,
        device=device,
        dtype=dtype,
    )

    # 2) Copy parameters (safe now - same device & dtype)
    with torch.no_grad():
        fused.weight.copy_(weight)

        if use_bias:
            # Align bias to module's device/dtype if needed
            fused.bias.copy_(bias.to(device=device, dtype=dtype))

    return fused
