#!/usr/bin/env python3
"""
Benchmark: Hypatia NativeModel vs PyTorch Eager

Measures the actual speedup of Rust-native fused forward pass
over PyTorch's standard operator dispatch.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import Hypatia native ops
from _hypatia_core import native_forward, native_train_step


def benchmark_inference(name, model_fn, configs, warmup=50, iterations=500):
    """Benchmark inference: PyTorch eager vs Hypatia native."""
    print(f"\n{'='*70}")
    print(f" INFERENCE BENCHMARK: {name}")
    print(f" Warmup={warmup}, Iterations={iterations}")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'PyTorch':>10} {'Hypatia':>10} {'Speedup':>10}")
    print(f"{'-'*70}")

    for config_name, (in_feat, layers_spec, batch) in configs.items():
        # Build model
        model = model_fn(in_feat, layers_spec)
        model.eval()

        # Extract layers for native forward
        np_layers = extract_np_layers(model)

        # Create input
        x = torch.randn(batch, in_feat)
        x_np = x.float().contiguous().numpy()

        # === PyTorch Eager ===
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)

            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = model(x)
            torch_time = (time.perf_counter() - t0) / iterations * 1e6  # microseconds

        # === Hypatia Native ===
        for _ in range(warmup):
            _ = native_forward(x_np, np_layers)

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = native_forward(x_np, np_layers)
        native_time = (time.perf_counter() - t0) / iterations * 1e6

        speedup = torch_time / native_time
        marker = " <--" if speedup >= 1.5 else ""
        print(f"{config_name:<30} {torch_time:>8.0f}us {native_time:>8.0f}us {speedup:>8.2f}x{marker}")

        # Verify correctness
        with torch.no_grad():
            torch_out = model(x).numpy()
        native_out = native_forward(x_np, np_layers)
        max_diff = np.abs(torch_out - native_out).max()
        if max_diff > 0.01:
            print(f"  WARNING: max diff = {max_diff:.6f}")


def benchmark_training(name, model_fn, configs, warmup=20, iterations=200):
    """Benchmark training: PyTorch vs Hypatia native train step."""
    print(f"\n{'='*70}")
    print(f" TRAINING BENCHMARK: {name}")
    print(f" Warmup={warmup}, Iterations={iterations}")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'PyTorch':>10} {'Hypatia':>10} {'Speedup':>10}")
    print(f"{'-'*70}")

    lr = 0.001

    for config_name, (in_feat, layers_spec, batch) in configs.items():
        out_feat = layers_spec[-1][0]

        # === PyTorch Training ===
        model_pt = model_fn(in_feat, layers_spec)
        model_pt.train()
        optimizer = torch.optim.SGD(model_pt.parameters(), lr=lr)

        x = torch.randn(batch, in_feat)
        y = torch.randn(batch, out_feat)

        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            out = model_pt(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

        t0 = time.perf_counter()
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            out = model_pt(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
        torch_time = (time.perf_counter() - t0) / iterations * 1e6

        # === Hypatia Native Training ===
        model_hy = model_fn(in_feat, layers_spec)
        np_weights, np_biases, np_activations = extract_train_data(model_hy)

        x_np = x.float().contiguous().numpy()
        y_np = y.float().contiguous().numpy()

        for _ in range(warmup):
            _ = native_train_step(x_np, y_np, np_weights, np_biases, np_activations, lr)

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = native_train_step(x_np, y_np, np_weights, np_biases, np_activations, lr)
        native_time = (time.perf_counter() - t0) / iterations * 1e6

        speedup = torch_time / native_time
        marker = " <--" if speedup >= 1.5 else ""
        print(f"{config_name:<30} {torch_time:>8.0f}us {native_time:>8.0f}us {speedup:>8.2f}x{marker}")


def extract_np_layers(model):
    """Extract numpy layer data from a Sequential model."""
    layers = []
    children = list(model.children())
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, nn.Linear):
            w = child.weight.data.detach().float().contiguous().numpy()
            b = child.bias.data.detach().float().contiguous().numpy() if child.bias is not None else None
            act = "none"
            if i + 1 < len(children) and isinstance(children[i + 1], nn.ReLU):
                act = "relu"
                i += 1
            layers.append((w, b, act))
        i += 1
    return layers


def extract_train_data(model):
    """Extract mutable numpy arrays for native training."""
    weights = []
    biases = []
    activations = []

    children = list(model.children())
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, nn.Linear):
            w = child.weight.data.detach().float().contiguous().numpy().copy()
            b = child.bias.data.detach().float().contiguous().numpy().copy() if child.bias is not None else None
            act = "none"
            if i + 1 < len(children) and isinstance(children[i + 1], nn.ReLU):
                act = "relu"
                i += 1
            weights.append(w)
            biases.append(b)
            activations.append(act)
        i += 1
    return weights, biases, activations


def build_mlp(in_feat, layers_spec):
    """Build Sequential MLP from spec: [(out_features, has_relu), ...]"""
    modules = []
    prev = in_feat
    for out_f, has_relu in layers_spec:
        modules.append(nn.Linear(prev, out_f))
        if has_relu:
            modules.append(nn.ReLU())
        prev = out_f
    return nn.Sequential(*modules)


if __name__ == "__main__":
    torch.set_num_threads(1)  # Single-threaded for fair comparison

    print("Hypatia Native Ops Benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")

    # Test correctness first
    print("\n--- Correctness Check ---")
    model = build_mlp(64, [(256, True), (10, False)])
    model.eval()
    x = torch.randn(32, 64)
    np_layers = extract_np_layers(model)

    with torch.no_grad():
        torch_out = model(x).numpy()
    native_out = native_forward(x.float().contiguous().numpy(), np_layers)
    max_diff = np.abs(torch_out - native_out).max()
    print(f"Max abs diff: {max_diff:.2e} {'PASS' if max_diff < 1e-4 else 'FAIL'}")

    # === INFERENCE BENCHMARKS ===
    inference_configs = {
        "Tiny   16->32->8   B=256":  (16,  [(32, True), (8, False)], 256),
        "Small  64->256->10 B=128":  (64,  [(256, True), (10, False)], 128),
        "Med   256->512->10 B=64":   (256, [(512, True), (10, False)], 64),
        "Large 512->1024->10 B=32":  (512, [(1024, True), (10, False)], 32),
        "XL   1024->2048->10 B=16":  (1024, [(2048, True), (10, False)], 16),
        "XXL  2048->4096->10 B=8":   (2048, [(4096, True), (10, False)], 8),
    }
    benchmark_inference("2-Layer MLP", build_mlp, inference_configs)

    # Deep MLP
    deep_configs = {
        "3L  64->128->64->10 B=128":  (64, [(128, True), (64, True), (10, False)], 128),
        "4L  64->128->128->64->10 B=64": (64, [(128, True), (128, True), (64, True), (10, False)], 64),
        "5L  64->256->256->128->64->10 B=32": (64, [(256, True), (256, True), (128, True), (64, True), (10, False)], 32),
    }
    benchmark_inference("Deep MLP", build_mlp, deep_configs)

    # === TRAINING BENCHMARKS ===
    training_configs = {
        "Tiny   16->32->8   B=256":  (16,  [(32, True), (8, False)], 256),
        "Small  64->256->10 B=128":  (64,  [(256, True), (10, False)], 128),
        "Med   256->512->10 B=64":   (256, [(512, True), (10, False)], 64),
        "Large 512->1024->10 B=32":  (512, [(1024, True), (10, False)], 32),
    }
    benchmark_training("2-Layer MLP Training", build_mlp, training_configs)

    print(f"\n{'='*70}")
    print(" DONE")
    print(f"{'='*70}")
