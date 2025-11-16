"""
Fused PyTorch modules for Hypatia optimizations

These modules combine multiple operations into single efficient implementations.
Used as reconstruction targets for E-graph fusion patterns.
"""

import torch
import torch.nn as nn
from typing import Optional


class HypatiaFusedLinearReLU(nn.Module):
    """
    Fused Linear + ReLU module.

    Combines nn.Linear followed by ReLU activation into a single module.
    This reduces memory allocations and improves cache locality.

    Semantically equivalent to: torch.relu(nn.Linear(x))

    Used as reconstruction target for (relu (linear W B X)) E-graph patterns.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear transformation followed by ReLU"""
        return torch.relu(self.fc(x))

    def __repr__(self):
        return (f"HypatiaFusedLinearReLU(in_features={self.fc.in_features}, "
                f"out_features={self.fc.out_features}, bias={self.fc.bias is not None})")


def create_fused_linear_relu_from_tensors(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> HypatiaFusedLinearReLU:
    """
    Create a HypatiaFusedLinearReLU module from existing weight and bias tensors.

    This helper is called from Rust (PyO3) during graph reconstruction.

    Args:
        weight: Linear layer weight tensor, shape [out_features, in_features]
        bias: Linear layer bias tensor, shape [out_features], or None

    Returns:
        HypatiaFusedLinearReLU module with copied parameters on same device as input tensors

    Example:
        >>> weight = torch.randn(128, 256)  # out=128, in=256
        >>> bias = torch.randn(128)
        >>> module = create_fused_linear_relu_from_tensors(weight, bias)
        >>> x = torch.randn(32, 256)  # batch=32, in=256
        >>> out = module(x)  # shape: [32, 128]
    """
    out_features, in_features = weight.shape
    has_bias = bias is not None

    # Create module structure
    module = HypatiaFusedLinearReLU(in_features, out_features, bias=has_bias)

    # Copy weights and bias
    # Note: Assumes weight/bias are already on correct device (handled by Rust caller)
    module.fc.weight.data.copy_(weight)
    if has_bias:
        module.fc.bias.data.copy_(bias)

    return module
