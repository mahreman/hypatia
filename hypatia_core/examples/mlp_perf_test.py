"""
MLP Performance Test

Benchmarks Hypatia compilation with aggressive optimization settings.
Uses OFF or SOFT checksum mode to allow maximum fusion opportunities.
Measures actual speedup from e-graph optimizations (e.g., Linear+ReLU fusion).
"""

import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time
import hypatia_core  # Auto-registers 'hypatia' backend

# Debug: Check registered backends
print(f"[DEBUG] Registered backends: {torch._dynamo.list_backends()}")

# Aggressive mode: disable checksum validation to maximize performance
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"  # or "soft" for warnings only


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def bench(fn, x, iters=100, device="cpu"):
    """Benchmark a function with warmup and synchronization"""
    # Warmup
    for _ in range(10):
        _ = fn(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    for _ in range(iters):
        _ = fn(x)

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    return (t1 - t0) * 1000.0 / iters  # ms/iter


def main():
    print("=" * 70)
    print("MLP Performance Test (Aggressive Mode)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checksum Mode: {os.environ.get('HYPATIA_CHECKSUM_MODE', 'not set')}")
    print()

    # Create model and input (larger batch for better GPU utilization)
    model = MLP().to(device)
    model.eval()  # Inference mode
    x = torch.randn(256, 784, device=device)

    print("Model Architecture:")
    print(f"  Input:  {784} features")
    print(f"  Hidden: {256} units (2 layers)")
    print(f"  Output: {10} classes")
    print(f"  Batch:  {x.shape[0]}")
    print()

    # Compile with Hypatia
    print("Compiling with Hypatia backend...")
    opt = torch.compile(model, backend="hypatia")
    print()

    # Benchmark eager model
    print("Benchmarking eager model...")
    with torch.no_grad():
        eager_ms = bench(model, x, iters=100, device=device)
    print(f"  Eager:   {eager_ms:.4f} ms/iter")

    # Benchmark optimized model
    print("\nBenchmarking Hypatia-compiled model...")
    with torch.no_grad():
        hypatia_ms = bench(opt, x, iters=100, device=device)
    print(f"  Hypatia: {hypatia_ms:.4f} ms/iter")

    # Calculate speedup
    print()
    print("=" * 70)
    speedup = eager_ms / hypatia_ms
    improvement = (speedup - 1) * 100

    if speedup > 1.0:
        print(f"✅ Speedup: {speedup:.4f}x ({improvement:+.2f}% faster)")
    elif speedup < 1.0:
        slowdown = (1 / speedup - 1) * 100
        print(f"⚠️  Slowdown: {1/speedup:.4f}x ({slowdown:+.2f}% slower)")
    else:
        print(f"➡️  No change: {speedup:.4f}x")

    print("=" * 70)
    print()
    print("Note: Speedup comes from e-graph optimizations like:")
    print("  - Linear + ReLU fusion")
    print("  - Algebraic simplifications")
    print("  - Dead code elimination")


if __name__ == "__main__":
    main()
