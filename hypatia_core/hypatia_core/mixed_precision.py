# hypatia_core/mixed_precision.py
#
# Mixed Precision: FP16/BF16 storage with FP32 accumulation.
# PyTorch-compatible modules that dispatch to Rust half-precision GEMM kernels.

from __future__ import annotations

from typing import Optional, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from _hypatia_core import (
    to_half_precision,
    mixed_precision_forward,
    mixed_precision_stats,
)


class MixedPrecisionLinear(nn.Module):
    """Linear layer with half-precision weight storage and FP32 accumulation.

    Weights are stored as FP16 or BF16 (2x memory reduction).
    Forward pass: dequantize weight → FP32 GEMM → FP32 output.

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: include bias parameter
        precision: "fp16" or "bf16"
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: Literal["fp16", "bf16"] = "fp16",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cached half-precision data
        self._half_cache: Optional[Dict] = None
        self._cache_valid = False

    def _build_half(self):
        """Convert current weights to half-precision format."""
        w_np = self.weight.detach().cpu().to(torch.float32).numpy()
        hw = to_half_precision(w_np, self.precision)
        self._half_cache = {
            "data": np.array(hw["data"]),
            "format": hw["format"],
        }
        self._cache_valid = True
        return self._half_cache

    def _get_half(self):
        if not self._cache_valid:
            self._build_half()
        return self._half_cache

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Training: use dense FP32 for gradient accuracy
            return F.linear(input, self.weight, self.bias)

        # Inference: use mixed-precision GEMM
        half = self._get_half()

        orig_shape = input.shape
        if input.dim() == 1:
            input_2d = input.unsqueeze(0)
        elif input.dim() > 2:
            input_2d = input.reshape(-1, input.shape[-1])
        else:
            input_2d = input

        x_np = input_2d.detach().cpu().to(torch.float32).numpy()
        b_np = self.bias.detach().cpu().to(torch.float32).numpy() if self.bias is not None else None

        out_np = mixed_precision_forward(
            x_np,
            half["data"],
            b_np,
            self.out_features,
            self.in_features,
            self.precision,
            False,
        )

        result = torch.from_numpy(np.array(out_np)).to(input.device, input.dtype)

        if input.dim() == 1:
            result = result.squeeze(0)
        elif input.dim() > 2:
            result = result.reshape(*orig_shape[:-1], self.out_features)

        return result

    def invalidate_cache(self):
        """Call after modifying weights."""
        self._cache_valid = False

    def get_precision_stats(self) -> Dict:
        """Analyze precision loss from half-precision conversion."""
        w_np = self.weight.detach().cpu().to(torch.float32).numpy()
        stats = mixed_precision_stats(w_np, self.precision)
        return dict(stats)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"precision={self.precision}, bias={self.bias is not None}"
        )


def convert_to_mixed_precision(
    model: nn.Module,
    precision: Literal["fp16", "bf16"] = "fp16",
    min_size: int = 64,
) -> nn.Module:
    """Convert all nn.Linear layers to MixedPrecisionLinear.

    Args:
        model: PyTorch model
        precision: "fp16" or "bf16"
        min_size: minimum layer size to convert (skip small layers)

    Returns:
        Model with Linear layers replaced by MixedPrecisionLinear
    """
    model = _replace_linear_mp(model, precision, min_size)
    model.eval()
    return model


def _replace_linear_mp(
    module: nn.Module,
    precision: str,
    min_size: int,
) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features >= min_size or child.out_features >= min_size:
                mp = MixedPrecisionLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    precision=precision,
                )
                with torch.no_grad():
                    mp.weight.copy_(child.weight)
                    if child.bias is not None:
                        mp.bias.copy_(child.bias)
                setattr(module, name, mp)
        else:
            _replace_linear_mp(child, precision, min_size)
    return module


def model_precision_report(model: nn.Module) -> Dict:
    """Generate a precision report for a model with MixedPrecisionLinear layers.

    Returns:
        dict with per-layer precision info and total memory savings
    """
    layers = []
    total_fp32_bytes = 0
    total_half_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLinear):
            n_params = module.in_features * module.out_features
            fp32_bytes = n_params * 4
            half_bytes = n_params * 2
            total_fp32_bytes += fp32_bytes
            total_half_bytes += half_bytes

            layers.append({
                "name": name,
                "type": "MixedPrecisionLinear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "precision": module.precision,
                "fp32_bytes": fp32_bytes,
                "half_bytes": half_bytes,
            })
        elif isinstance(module, nn.Linear):
            n_params = module.in_features * module.out_features
            fp32_bytes = n_params * 4
            total_fp32_bytes += fp32_bytes
            total_half_bytes += fp32_bytes  # Still FP32

            layers.append({
                "name": name,
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "precision": "fp32",
                "fp32_bytes": fp32_bytes,
                "half_bytes": fp32_bytes,
            })

    return {
        "layers": layers,
        "total_fp32_bytes": total_fp32_bytes,
        "total_half_bytes": total_half_bytes,
        "memory_savings_pct": (1.0 - total_half_bytes / max(total_fp32_bytes, 1)) * 100,
        "compression_ratio": total_fp32_bytes / max(total_half_bytes, 1),
    }


__all__ = [
    "MixedPrecisionLinear",
    "convert_to_mixed_precision",
    "model_precision_report",
]
