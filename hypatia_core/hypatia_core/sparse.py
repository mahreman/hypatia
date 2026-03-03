# hypatia_core/sparse.py
#
# Sparse Tensor IR: PyTorch-compatible sparse linear layers with Rust GEMM backend.
# Supports magnitude pruning, CSR conversion, and auto-dispatch to sparse kernels.

from __future__ import annotations

from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from _hypatia_core import (
    to_sparse_csr,
    sparse_linear_forward,
    compute_sparsity_threshold,
    sparsity_stats,
)


class SparseLinear(nn.Module):
    """Sparse linear layer using Hypatia's CSR sparse-dense GEMM.

    Replaces nn.Linear with magnitude-pruned sparse weights stored in CSR
    format. Forward pass dispatches to Rust sparse GEMM kernel, which only
    iterates over non-zero weight elements.

    Args:
        in_features: input dimension
        out_features: output dimension
        sparsity: target sparsity ratio (0.0-1.0), e.g. 0.5 = 50% zeros
        bias: include bias parameter
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cached CSR data (invalidated on weight update)
        self._csr_cache: Optional[Dict] = None
        self._cache_valid = False

    def _build_csr(self):
        """Convert current weights to CSR format with magnitude pruning."""
        w_np = self.weight.detach().cpu().to(torch.float32).numpy()
        threshold = compute_sparsity_threshold(w_np, self.sparsity)
        csr = to_sparse_csr(w_np, threshold)

        self._csr_cache = {
            "row_ptrs": list(csr["row_ptrs"]),
            "col_indices": list(csr["col_indices"]),
            "values": np.array(csr["values"]),
            "nnz": csr["nnz"],
            "sparsity": csr["sparsity"],
        }
        self._cache_valid = True
        return self._csr_cache

    def _get_csr(self):
        """Get cached CSR or rebuild."""
        if not self._cache_valid:
            self._build_csr()
        return self._csr_cache

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, use dense forward (gradients need dense weights)
            return F.linear(input, self.weight, self.bias)

        # Inference: use sparse GEMM
        csr = self._get_csr()

        # Handle batched input
        orig_shape = input.shape
        if input.dim() == 1:
            input_2d = input.unsqueeze(0)
        elif input.dim() > 2:
            input_2d = input.reshape(-1, input.shape[-1])
        else:
            input_2d = input

        x_np = input_2d.detach().cpu().to(torch.float32).numpy()
        b_np = self.bias.detach().cpu().to(torch.float32).numpy() if self.bias is not None else None

        out_np = sparse_linear_forward(
            x_np,
            csr["row_ptrs"],
            csr["col_indices"],
            csr["values"],
            b_np,
            self.out_features,
            self.in_features,
            False,  # relu handled separately
        )

        result = torch.from_numpy(np.array(out_np)).to(input.device, input.dtype)

        # Restore original batch dimensions
        if input.dim() == 1:
            result = result.squeeze(0)
        elif input.dim() > 2:
            result = result.reshape(*orig_shape[:-1], self.out_features)

        return result

    def invalidate_cache(self):
        """Call after modifying weights (e.g., fine-tuning)."""
        self._cache_valid = False

    def get_stats(self) -> Dict:
        """Get sparsity statistics."""
        csr = self._get_csr()
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "target_sparsity": self.sparsity,
            "actual_sparsity": csr["sparsity"],
            "nnz": csr["nnz"],
            "total_elements": self.in_features * self.out_features,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sparsity={self.sparsity}, bias={self.bias is not None}"
        )


def sparsify_model(
    model: nn.Module,
    sparsity: float = 0.5,
    min_size: int = 64,
) -> nn.Module:
    """Convert all nn.Linear layers in a model to SparseLinear.

    Replaces each nn.Linear with a SparseLinear that uses magnitude pruning
    and CSR sparse-dense GEMM for inference.

    Args:
        model: PyTorch model to sparsify
        sparsity: target sparsity ratio (0.0-1.0)
        min_size: minimum layer size to sparsify (skip small layers)

    Returns:
        Model with Linear layers replaced by SparseLinear
    """
    model = _replace_linear_recursive(model, sparsity, min_size)
    model.eval()
    return model


def _replace_linear_recursive(
    module: nn.Module,
    sparsity: float,
    min_size: int,
) -> nn.Module:
    """Recursively replace nn.Linear with SparseLinear."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features >= min_size or child.out_features >= min_size:
                sparse = SparseLinear(
                    child.in_features,
                    child.out_features,
                    sparsity=sparsity,
                    bias=child.bias is not None,
                )
                with torch.no_grad():
                    sparse.weight.copy_(child.weight)
                    if child.bias is not None:
                        sparse.bias.copy_(child.bias)
                setattr(module, name, sparse)
        else:
            _replace_linear_recursive(child, sparsity, min_size)
    return module


def model_sparsity_report(model: nn.Module) -> Dict:
    """Generate a sparsity report for a model.

    Args:
        model: PyTorch model

    Returns:
        dict with per-layer and total sparsity info
    """
    layers = []
    total_params = 0
    total_nonzero = 0
    total_dense_bytes = 0
    total_sparse_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            stats = module.get_stats()
            layers.append({
                "name": name,
                "type": "SparseLinear",
                **stats,
            })
            total_params += stats["total_elements"]
            total_nonzero += stats["nnz"]
        elif isinstance(module, nn.Linear):
            w_np = module.weight.detach().cpu().numpy()
            stats = sparsity_stats(w_np)
            total_elements = stats["total_elements"]
            nonzero = stats["nonzero_elements"]
            layers.append({
                "name": name,
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "actual_sparsity": stats["sparsity_ratio"],
                "nnz": nonzero,
                "total_elements": total_elements,
            })
            total_params += total_elements
            total_nonzero += nonzero
            total_dense_bytes += stats["dense_bytes"]
            total_sparse_bytes += stats["sparse_bytes_estimate"]

    return {
        "layers": layers,
        "total_parameters": total_params,
        "total_nonzero": total_nonzero,
        "overall_sparsity": 1.0 - (total_nonzero / max(total_params, 1)),
        "layer_count": len(layers),
    }


__all__ = [
    "SparseLinear",
    "sparsify_model",
    "model_sparsity_report",
]
