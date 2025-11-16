#!/usr/bin/env python3
"""
Isolated CUDA kernel performance test.

This isolates the kernel to check if the issue is:
1. Kernel not loading (using fallback)
2. Kernel overhead
3. Wrapper overhead
"""

import torch
import time
import sys
import os

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration
B, IN, OUT = 1024, 784, 2048
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("ISOLATED KERNEL PERFORMANCE TEST")
print("=" * 70)
print(f"Config: Batch={B}, In={IN}, Out={OUT}")
print(f"Device: {device}")
print()

x = torch.randn(B, IN, device=device, dtype=torch.float32)
w = torch.randn(OUT, IN, device=device, dtype=torch.float32)
b = torch.randn(OUT, device=device, dtype=torch.float32)

def bench(fn, n=500, warmup=10):
    """Benchmark function with proper CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        _ = fn()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        _ = fn()

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1000 / n  # ms/iter

# Baseline: PyTorch eager
def eager_fn():
    return torch.relu(torch.nn.functional.linear(x, w, b))

print("🔥 Testing PyTorch eager baseline...")
eager_ms = bench(eager_fn)
print(f"✅ Eager:  {eager_ms:.4f} ms/iter")
print()

# Fused kernel - attempt 1: Direct CUDA extension
print("🔍 Attempting to load CUDA extension...")
cuda_ext_loaded = False
fused_fn = None

try:
    # Try loading the JIT-compiled extension
    from hypatia_core.fused_modules import _FUSED_LINEAR_RELU_EXT, _HAS_CUDA_KERNEL

    if _HAS_CUDA_KERNEL and _FUSED_LINEAR_RELU_EXT is not None:
        print("✅ CUDA extension loaded via JIT compilation")
        cuda_ext_loaded = True

        def fused_fn():
            return _FUSED_LINEAR_RELU_EXT.forward(x, w, b)
    else:
        print("❌ CUDA extension available but kernel flag is False")
        print(f"   _HAS_CUDA_KERNEL = {_HAS_CUDA_KERNEL}")
        print(f"   _FUSED_LINEAR_RELU_EXT = {_FUSED_LINEAR_RELU_EXT}")

except Exception as e:
    print(f"❌ Failed to load CUDA extension: {e}")

# Attempt 2: Try direct import
if not cuda_ext_loaded:
    print("\n🔍 Attempting direct import...")
    try:
        import hypatia_fused_linear_relu as cuda_ext
        print("✅ Direct import successful")
        cuda_ext_loaded = True

        def fused_fn():
            return cuda_ext.forward(x, w, b)

    except ImportError as e:
        print(f"❌ Direct import failed: {e}")

# Attempt 3: Check for any built extension
if not cuda_ext_loaded:
    print("\n🔍 Searching for built extensions...")
    import glob
    so_files = glob.glob(os.path.join(os.path.dirname(__file__), "**/*.so"), recursive=True)
    print(f"Found {len(so_files)} .so files:")
    for so in so_files[:5]:
        print(f"  - {so}")

print()

# Run benchmark
if cuda_ext_loaded and fused_fn is not None:
    print("🔥 Testing fused CUDA kernel...")
    fused_ms = bench(fused_fn)
    print(f"✅ Fused:  {fused_ms:.4f} ms/iter")
    print()

    # Calculate speedup
    speedup = eager_ms / fused_ms
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Eager:     {eager_ms:.4f} ms/iter")
    print(f"Fused:     {fused_ms:.4f} ms/iter")
    print(f"Speedup:   {speedup:.2f}x")

    if speedup < 0.95:
        print(f"\n❌ SLOWDOWN: {1/speedup:.2f}x slower!")
        print("\nPossible causes:")
        print("  1. Reshape overhead (9 reshapes vs 0 in eager)")
        print("  2. Extra allocations in backward pass")
        print("  3. CUDA synchronization issues")
        print("  4. Wrapper overhead")
    elif speedup < 1.05:
        print("\n⚠️  NEUTRAL: No significant speedup")
        print("  This is expected for v1.0 with naive kernel")
    else:
        print("\n✅ SPEEDUP ACHIEVED!")

    print()

    # Correctness check
    print("🔍 Checking correctness...")
    with torch.no_grad():
        y_eager = eager_fn()
        y_fused = fused_fn()
        diff = (y_eager - y_fused).abs().max().item()
        mean_diff = (y_eager - y_fused).abs().mean().item()

        print(f"Max diff:  {diff:.2e}")
        print(f"Mean diff: {mean_diff:.2e}")

        if diff < 1e-5:
            print("✅ CORRECTNESS: PASS")
        else:
            print(f"❌ CORRECTNESS: FAIL (diff = {diff:.2e})")

else:
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print("❌ CUDA extension NOT loaded - using PyTorch fallback")
    print("\nThis explains the slowdown in benchmarks!")
    print("\nTo fix:")
    print("1. Check if CUDA is available:")
    print("   python -c 'import torch; print(torch.cuda.is_available())'")
    print()
    print("2. Try to trigger JIT compilation:")
    print("   python -c 'from hypatia_core.fused_modules import FusedLinearReLU; print(FusedLinearReLU)'")
    print()
    print("3. Check for compilation errors in /tmp/torch_extensions/")

print("=" * 70)
