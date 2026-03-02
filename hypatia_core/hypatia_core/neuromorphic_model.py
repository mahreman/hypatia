"""
Neuromorphic Model API for Hypatia

Provides ANN→SNN (Artificial → Spiking Neural Network) conversion
using Leaky Integrate-and-Fire (LIF) neurons.

Theory:
    ReLU(x) = max(0, x) is approximated by LIF neuron firing rate:
    - Membrane: V[t+1] = beta * V[t] + I[t]
    - Spike:    s[t] = 1 if V[t] >= v_threshold
    - Reset:    V[t] -= v_threshold * s[t]
    - Rate:     firing_rate ≈ ReLU(x) / v_threshold

Usage:
    import torch
    from hypatia_core import NeuromorphicModel

    # Create a standard ANN
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

    # Convert to SNN
    snn = NeuromorphicModel(model, timesteps=32, v_threshold=1.0, beta=0.95)

    # Run neuromorphic inference
    x = torch.randn(784)
    output = snn(x)

    # Get spike statistics
    output, stats = snn.forward_with_stats(x)
"""

import torch
import torch.nn as nn
import numpy as np
import time

try:
    from _hypatia_core import (
        neuromorphic_forward,
        neuromorphic_forward_with_stats,
        optimize_for_neuromorphic,
        estimate_neuromorphic_energy,
    )
except ImportError:
    raise ImportError(
        "Neuromorphic bindings not available. "
        "Rebuild with: cd hypatia_core && maturin develop --release"
    )


class NeuromorphicModel:
    """ANN-to-SNN converted model using LIF neurons.

    Converts a PyTorch model with Linear+ReLU layers into an equivalent
    Spiking Neural Network using Leaky Integrate-and-Fire neurons.

    The conversion preserves functional equivalence:
    - Over T timesteps, LIF firing rate ≈ ReLU(x) / v_threshold
    - Weights are normalized for proper rate coding
    - Last layer outputs membrane potential (no spiking)

    Args:
        model: PyTorch model (Sequential of Linear+ReLU layers)
        timesteps: Number of simulation timesteps (more = better accuracy)
        v_threshold: Spike threshold voltage
        beta: Membrane leak factor (0 = no memory, 1 = perfect memory)
    """

    def __init__(self, model, timesteps=32, v_threshold=1.0, beta=0.95):
        self.timesteps = timesteps
        self.v_threshold = v_threshold
        self.beta = beta

        # Extract Linear layers from model
        self.layer_data = []
        self._extract_layers(model)

        if not self.layer_data:
            raise ValueError("Model must contain at least one Linear layer")

    def _extract_layers(self, model):
        """Extract weight/bias from Linear layers, skipping activations."""
        if isinstance(model, nn.Sequential):
            for module in model:
                self._extract_layers(module)
        elif isinstance(model, nn.Linear):
            weight = model.weight.detach().cpu().numpy().astype(np.float32)
            bias = None
            if model.bias is not None:
                bias = model.bias.detach().cpu().numpy().astype(np.float32)
            self.layer_data.append((weight, bias))
        # Skip ReLU, BatchNorm, Dropout etc. (ReLU → LIF conversion is implicit)

    def __call__(self, x):
        """Run neuromorphic inference.

        Args:
            x: Input tensor (1D or 2D batch)

        Returns:
            Output tensor
        """
        return self.forward(x)

    def forward(self, x):
        """Run neuromorphic forward pass through Rust SNN simulation.

        Args:
            x: Input tensor [features] or [batch, features]

        Returns:
            Output tensor matching input batch dimensions
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        # Handle batched input
        if x_np.ndim == 2:
            outputs = []
            for i in range(x_np.shape[0]):
                out = self._forward_single(x_np[i])
                outputs.append(out)
            return torch.from_numpy(np.stack(outputs))
        else:
            out = self._forward_single(x_np)
            return torch.from_numpy(out)

    def _forward_single(self, x_np):
        """Single-sample forward through Rust."""
        layer_weights = [(w, b) for w, b in self.layer_data]
        result = neuromorphic_forward(
            layer_weights,
            x_np,
            v_threshold=self.v_threshold,
            beta=self.beta,
            timesteps=self.timesteps,
        )
        return np.array(result)

    def forward_with_stats(self, x):
        """Run forward pass and return spike statistics per layer.

        Args:
            x: Input tensor [features]

        Returns:
            (output_tensor, stats_list)
            stats_list: List of dicts with keys:
                layer_idx, total_spikes, total_possible, firing_rate
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        layer_weights = [(w, b) for w, b in self.layer_data]
        output, stats = neuromorphic_forward_with_stats(
            layer_weights,
            x_np,
            v_threshold=self.v_threshold,
            beta=self.beta,
            timesteps=self.timesteps,
        )

        return torch.from_numpy(np.array(output)), stats

    def estimate_energy(self, avg_firing_rate=0.1):
        """Estimate energy savings vs conventional (GPU/CPU) execution.

        Args:
            avg_firing_rate: Expected average firing rate (0.0-1.0)
                Sparse models (rate ~0.1) save most energy.

        Returns:
            Dict with energy comparison per layer and total
        """
        results = []
        total_neuro = 0.0
        total_conv = 0.0

        for i, (w, _) in enumerate(self.layer_data):
            out_feat, in_feat = w.shape
            energy = estimate_neuromorphic_energy(
                in_feat, out_feat, self.timesteps, avg_firing_rate,
            )
            energy['layer_idx'] = i
            energy['in_features'] = in_feat
            energy['out_features'] = out_feat
            total_neuro += energy['neuromorphic_nj']
            total_conv += energy['conventional_nj']
            results.append(energy)

        return {
            'layers': results,
            'total_neuromorphic_nj': total_neuro,
            'total_conventional_nj': total_conv,
            'total_energy_ratio': total_neuro / total_conv if total_conv > 0 else float('inf'),
            'total_savings_pct': (1.0 - total_neuro / total_conv) * 100 if total_conv > 0 else 0,
        }

    def benchmark(self, x, n_iter=100, compare_relu=True):
        """Benchmark neuromorphic vs ReLU inference.

        Args:
            x: Input tensor
            n_iter: Number of iterations
            compare_relu: Also benchmark original ReLU model

        Returns:
            Dict with timing results
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        # Warm up
        self._forward_single(x_np)

        # Benchmark SNN
        start = time.perf_counter()
        for _ in range(n_iter):
            self._forward_single(x_np)
        snn_time = (time.perf_counter() - start) / n_iter

        result = {
            'snn_time_ms': snn_time * 1000,
            'timesteps': self.timesteps,
            'n_layers': len(self.layer_data),
        }

        if compare_relu:
            # Benchmark equivalent ReLU computation with numpy
            def relu_forward(x):
                for w, b in self.layer_data[:-1]:
                    x = x @ w.T
                    if b is not None:
                        x = x + b
                    x = np.maximum(x, 0)  # ReLU
                w, b = self.layer_data[-1]
                x = x @ w.T
                if b is not None:
                    x = x + b
                return x

            relu_forward(x_np)  # warmup
            start = time.perf_counter()
            for _ in range(n_iter):
                relu_forward(x_np)
            relu_time = (time.perf_counter() - start) / n_iter

            result['relu_time_ms'] = relu_time * 1000
            result['slowdown'] = snn_time / relu_time if relu_time > 0 else float('inf')

        return result

    def __repr__(self):
        layers_str = ", ".join(
            f"LIF({w.shape[1]}→{w.shape[0]})" for w, _ in self.layer_data
        )
        return (
            f"NeuromorphicModel(\n"
            f"  layers=[{layers_str}],\n"
            f"  timesteps={self.timesteps}, v_th={self.v_threshold}, "
            f"beta={self.beta}\n"
            f")"
        )


def compile_neuromorphic(expr_str):
    """Compile an S-expression for neuromorphic hardware using e-graph optimization.

    Applies ReLU→LIF rewrite rules via equality saturation to find
    the optimal neuromorphic representation.

    Args:
        expr_str: S-expression string, e.g. "(relu (linear w b x))"

    Returns:
        Optimized S-expression with neuromorphic operators

    Example:
        >>> compile_neuromorphic("(relu (linear w b x))")
        '(neuromorphic_linear w b x v_th beta T)'
    """
    return optimize_for_neuromorphic(expr_str)
