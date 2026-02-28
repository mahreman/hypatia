"""
Hypatia NativeModel - Rust-native forward pass that bypasses PyTorch dispatch.

For small-to-medium models, PyTorch's per-operator dispatch overhead
(Python -> C++ -> ATen -> kernel) dominates actual compute. Each op
(matmul, bias add, relu) crosses this boundary separately.

NativeModel eliminates this by doing the ENTIRE forward pass in a single
Python -> Rust call, using matrixmultiply's AVX2-optimized GEMM with
fused bias + activation.

Usage:
    import hypatia_core

    model = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Inference (2-5x faster for small models)
    fast_model = hypatia_core.NativeModel(model)
    output = fast_model(input_tensor)

    # Training (single Rust call per step)
    trainer = hypatia_core.NativeTrainer(model, lr=0.01)
    loss = trainer.step(inputs, targets)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

try:
    from _hypatia_core import native_forward, native_train_step
except ImportError:
    native_forward = None
    native_train_step = None


class NativeModel(nn.Module):
    """Rust-native inference model. Drop-in replacement for Sequential MLPs."""

    def __init__(self, model: nn.Module):
        super().__init__()

        if native_forward is None:
            raise RuntimeError("native_forward not available. Rebuild with: maturin develop --release")

        # Extract layer structure
        self._layer_info = _extract_layers(model)
        if not self._layer_info:
            raise ValueError(
                "Could not extract MLP layers. Model must be Sequential "
                "with Linear + optional ReLU layers."
            )

        # Pre-extract numpy arrays (zero-copy from torch contiguous tensors)
        self._np_layers = []
        for weight, bias, activation in self._layer_info:
            w_np = weight.detach().float().contiguous().numpy()
            b_np = bias.detach().float().contiguous().numpy() if bias is not None else None
            self._np_layers.append((w_np, b_np, activation))

        # Store dims for repr
        self._dims = [(w.shape[1], w.shape[0]) for w, _, _ in self._layer_info]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().float().contiguous().numpy()
        out_np = native_forward(x_np, self._np_layers)
        return torch.from_numpy(out_np)

    def __repr__(self):
        dims_str = " -> ".join(
            f"{d[0]}" + (f" -> {d[1]}" if i == len(self._dims) - 1 else f" -> [{d[1]}]")
            for i, d in enumerate(self._dims)
        )
        return f"NativeModel({dims_str})"


class NativeTrainer:
    """Rust-native training loop. Single Python->Rust call per training step.

    Implements: forward -> MSE loss -> backward -> SGD update, all in Rust.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001):
        if native_train_step is None:
            raise RuntimeError("native_train_step not available. Rebuild with: maturin develop --release")

        self.lr = lr
        layer_info = _extract_layers(model)
        if not layer_info:
            raise ValueError("Could not extract MLP layers from model")

        # Create mutable weight copies
        self.weights = []
        self.biases = []
        self.activations = []

        for weight, bias, activation in layer_info:
            self.weights.append(weight.detach().float().contiguous().numpy().copy())
            self.biases.append(
                bias.detach().float().contiguous().numpy().copy() if bias is not None else None
            )
            self.activations.append(activation)

        # Store dims
        self._in_features = layer_info[0][0].shape[1]
        self._out_features = layer_info[-1][0].shape[0]

    def step(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Single training step. Returns loss value."""
        x_np = inputs.detach().float().contiguous().numpy()
        y_np = targets.detach().float().contiguous().numpy()

        loss = native_train_step(
            x_np,
            y_np,
            self.weights,
            self.biases,
            self.activations,
            self.lr,
        )
        return loss

    def get_model_state(self):
        """Return current weights as torch tensors (for evaluation)."""
        state = []
        for w, b in zip(self.weights, self.biases):
            state.append(torch.from_numpy(w.copy()))
            if b is not None:
                state.append(torch.from_numpy(b.copy()))
        return state


def _extract_layers(
    model: nn.Module,
) -> Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor], str]]]:
    """Extract (weight, bias, activation_name) tuples from a model.

    Supports:
    - nn.Sequential(Linear, ReLU, Linear, ...)
    - Models with named children that follow Linear+ReLU pattern
    """
    children = list(model.children()) if hasattr(model, "children") else []

    # Handle nested Sequential
    if len(children) == 1 and isinstance(children[0], nn.Sequential):
        children = list(children[0].children())
    elif isinstance(model, nn.Sequential):
        children = list(model.children())

    if not children:
        return None

    layers = []
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, nn.Linear):
            weight = child.weight.data
            bias = child.bias.data if child.bias is not None else None
            activation = "none"

            # Check if next child is an activation
            if i + 1 < len(children):
                next_child = children[i + 1]
                if isinstance(next_child, nn.ReLU):
                    activation = "relu"
                    i += 1

            layers.append((weight, bias, activation))
        elif isinstance(child, nn.Sequential):
            sub_layers = _extract_layers(child)
            if sub_layers:
                layers.extend(sub_layers)
        i += 1

    return layers if layers else None
