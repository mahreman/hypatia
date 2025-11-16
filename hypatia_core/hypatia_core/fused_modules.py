"""
Fused PyTorch modules for Hypatia optimizations.

These modules combine multiple operations into single efficient implementations,
resulting from E-graph fusion optimizations.

Currently available:
  - HypatiaFusedLinearReLU: Linear + ReLU combination
"""

from typing import Optional

import torch
import torch.nn as nn


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
        # CRITICAL: Create Linear directly on the target device
        # This ensures no CPU→CUDA copies during forward pass
        self.fc = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear transformation followed by ReLU"""
        # Single fused operation: Linear + ReLU
        return torch.relu(self.fc(x))

    def __repr__(self):
        return (f"HypatiaFusedLinearReLU(in_features={self.fc.in_features}, "
                f"out_features={self.fc.out_features}, bias={self.fc.bias is not None})")


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
        fused.fc.weight.copy_(weight)

        if use_bias:
            # Align bias to module's device/dtype if needed
            fused.fc.bias.copy_(bias.to(device=device, dtype=dtype))

    return fused
