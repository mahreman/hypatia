#!/usr/bin/env python3
"""
Hypatia Quick Start - End-to-End Demo

Shows how to use Hypatia to accelerate PyTorch model inference.
One function call: hypatia.optimize(model) - that's it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import time

# ====================================================================
# Step 1: Import Hypatia
# ====================================================================
import hypatia_core as hypatia
print(f"Hypatia v{hypatia.__version__} loaded")
print()

# ====================================================================
# Step 2: Create a model (standard PyTorch)
# ====================================================================
print("=" * 60)
print(" DEMO 1: Small Model (auto -> NativeModel)")
print("=" * 60)

small_model = nn.Sequential(
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 64),
)
small_model.eval()
n_params = sum(p.numel() for p in small_model.parameters())
print(f"Model: 256 -> 1024 -> 512 -> 64  ({n_params/1e6:.1f}M params)")

# Step 3: Optimize with one call
fast_small = hypatia.optimize(small_model)

# Step 4: Use it like normal
x = torch.randn(128, 256)
with torch.no_grad():
    out_orig = small_model(x)
    out_fast = fast_small(x)

rmse = (out_orig - out_fast).pow(2).mean().sqrt().item()
print(f"Correctness: RMSE = {rmse:.6f} (should be 0.0 for f32 native)")

# Benchmark
def bench(fn, iters=200):
    for _ in range(10): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1000

torch.set_num_threads(1)
with torch.no_grad():
    pt_ms = bench(lambda: small_model(x))
    hy_ms = bench(lambda: fast_small(x))

print(f"PyTorch:  {pt_ms:.3f} ms")
print(f"Hypatia:  {hy_ms:.3f} ms")
print(f"Speedup:  {pt_ms/hy_ms:.2f}x")

# ====================================================================
# DEMO 2: Large Model (auto -> QuantizedModel with INT4)
# ====================================================================
print()
print("=" * 60)
print(" DEMO 2: Large Model (auto -> QuantizedModel INT4)")
print("=" * 60)

# LLaMA-7B FFN layer dimensions
large_model = nn.Sequential(
    nn.Linear(4096, 11008),
    nn.ReLU(),
    nn.Linear(11008, 4096),
)
large_model.eval()
n_params = sum(p.numel() for p in large_model.parameters())
print(f"Model: 4096 -> 11008 -> 4096  ({n_params/1e6:.0f}M params, LLaMA-7B FFN)")

# Auto-selects INT4 quantization for large models
fast_large = hypatia.optimize(large_model)

# Use it
x = torch.randn(1, 4096)
with torch.no_grad():
    out_orig = large_model(x)
    out_fast = fast_large(x)

rmse = (out_orig - out_fast).pow(2).mean().sqrt().item()
print(f"Correctness: RMSE = {rmse:.4f} (expected ~0.02 for INT4)")

# Memory savings
if hasattr(fast_large, 'memory_saved_mb'):
    print(f"Memory saved: {fast_large.memory_saved_mb:.0f} MB ({fast_large.compression_ratio:.1f}x compression)")

# Benchmark
with torch.no_grad():
    pt_ms = bench(lambda: large_model(x), iters=50)
    hy_ms = bench(lambda: fast_large(x), iters=50)

print(f"PyTorch:  {pt_ms:.2f} ms")
print(f"Hypatia:  {hy_ms:.2f} ms")
print(f"Speedup:  {pt_ms/hy_ms:.2f}x")

# ====================================================================
# DEMO 3: Training (small model)
# ====================================================================
print()
print("=" * 60)
print(" DEMO 3: Training with NativeTrainer")
print("=" * 60)

train_model = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)

trainer = hypatia.NativeTrainer(train_model, lr=0.01)
print(f"NativeTrainer: 16 -> 64 -> 4")

# Training loop
for epoch in range(5):
    x = torch.randn(64, 16)
    y = torch.randn(64, 4)
    loss = trainer.step(x, y)
    if epoch == 0 or epoch == 4:
        print(f"  Epoch {epoch}: loss = {loss:.4f}")

print()
print("=" * 60)
print(" SUMMARY")
print("=" * 60)
print()
print("hypatia.optimize(model) automatically selects:")
print("  - Small models  -> NativeModel  (fused GEMM, 2-6x faster)")
print("  - Large models  -> QuantizedModel (INT4 SIMD, 11-16x faster)")
print()
print("That's it. One function call.")
