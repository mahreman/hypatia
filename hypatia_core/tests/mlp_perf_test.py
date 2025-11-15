#!/usr/bin/env python3
"""
MLP Performance Test - Hypatia Backend Benchmarking
Bu test, Hypatia backend'inin performansını eager mode ile karşılaştırır.
"""

import os
# CRITICAL: Disable checksum validation to allow optimizations to apply
# See SETUP_GUIDE.md "Checksum Mode" section for details
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"

import torch
import torch.nn as nn
import time
import hypatia_core  # Auto-registers 'hypatia' backend

# Ensure backend is registered
hypatia_core.register_backend()


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for benchmarking"""
    def __init__(self, in_features=256, hidden_dim=512, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def benchmark_model(model, inputs, num_warmup=10, num_iterations=100, device='cpu'):
    """
    Model performansını ölç

    Args:
        model: Test edilecek model
        inputs: Tuple of input tensors
        num_warmup: Warmup iterasyon sayısı
        num_iterations: Benchmark iterasyon sayısı
        device: 'cpu' veya 'cuda'

    Returns:
        Ortalama inference süresi (milliseconds)
    """
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(*inputs)

    # Sync if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    with torch.inference_mode():
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(*inputs)

        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    return avg_time_ms


def test_performance_comparison():
    """Eager vs Hypatia performans karşılaştırması"""
    print("\n" + "="*80)
    print("HYPATIA MLP PERFORMANCE BENCHMARK")
    print("="*80)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Model ve input
    model = MLP().to(device).eval()
    batch_size = 128
    inputs = (torch.randn(batch_size, 256, device=device),)

    print(f"Model: MLP (256 -> 512 -> 512 -> 10)")
    print(f"Batch size: {batch_size}")

    # Benchmark parametreleri
    WARMUP_ITER = 10
    BENCH_ITER = 100

    print(f"Warmup iterations: {WARMUP_ITER}")
    print(f"Benchmark iterations: {BENCH_ITER}")

    # 1. Eager mode benchmark
    print("\n" + "-"*80)
    print("1. EAGER MODE (Original PyTorch)")
    print("-"*80)
    eager_time = benchmark_model(
        model, inputs,
        num_warmup=WARMUP_ITER,
        num_iterations=BENCH_ITER,
        device=device
    )
    print(f"Average time: {eager_time:.4f} ms")

    # 2. Hypatia compiled benchmark
    print("\n" + "-"*80)
    print("2. HYPATIA COMPILED")
    print("-"*80)

    try:
        compiled_model = torch.compile(model, backend="hypatia")
        hypatia_time = benchmark_model(
            compiled_model, inputs,
            num_warmup=WARMUP_ITER,
            num_iterations=BENCH_ITER,
            device=device
        )
        print(f"Average time: {hypatia_time:.4f} ms")

        # Speedup hesapla
        print("\n" + "-"*80)
        print("PERFORMANCE COMPARISON")
        print("-"*80)
        print(f"Eager time:   {eager_time:.4f} ms")
        print(f"Hypatia time: {hypatia_time:.4f} ms")

        if hypatia_time < eager_time:
            speedup = (eager_time / hypatia_time - 1) * 100
            print(f"Speedup:      {speedup:.2f}% faster 🚀")
        else:
            slowdown = (hypatia_time / eager_time - 1) * 100
            print(f"Slowdown:     {slowdown:.2f}% slower ⚠️")

    except Exception as e:
        print(f"❌ Hypatia compilation failed: {e}")
        import traceback
        traceback.print_exc()


def test_different_batch_sizes():
    """Farklı batch size'lar ile performans testi"""
    print("\n" + "="*80)
    print("BATCH SIZE SCALING TEST")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP().to(device).eval()
    compiled_model = torch.compile(model, backend="hypatia")

    batch_sizes = [1, 16, 64, 128, 256]
    WARMUP = 5
    BENCH = 50

    print(f"\n{'Batch Size':<12} {'Eager (ms)':<15} {'Hypatia (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for bs in batch_sizes:
        inputs = (torch.randn(bs, 256, device=device),)

        eager_time = benchmark_model(model, inputs, WARMUP, BENCH, device)
        hypatia_time = benchmark_model(compiled_model, inputs, WARMUP, BENCH, device)

        speedup = (eager_time / hypatia_time - 1) * 100 if hypatia_time > 0 else 0

        print(f"{bs:<12} {eager_time:<15.4f} {hypatia_time:<15.4f} {speedup:+.2f}%")


def test_memory_usage():
    """Memory kullanımı karşılaştırması (sadece CUDA için)"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping memory test")
        return

    print("\n" + "="*80)
    print("MEMORY USAGE TEST (CUDA)")
    print("="*80)

    device = 'cuda'
    model = MLP().to(device).eval()
    inputs = (torch.randn(128, 256, device=device),)

    # Eager mode memory
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _ = model(*inputs)
    eager_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Hypatia compiled memory
    torch.cuda.reset_peak_memory_stats()
    compiled_model = torch.compile(model, backend="hypatia")
    with torch.inference_mode():
        _ = compiled_model(*inputs)
    hypatia_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    print(f"Eager mode:   {eager_mem:.2f} MB")
    print(f"Hypatia mode: {hypatia_mem:.2f} MB")

    mem_diff = hypatia_mem - eager_mem
    if mem_diff > 0:
        print(f"Difference:   +{mem_diff:.2f} MB (Hypatia uses more)")
    else:
        print(f"Difference:   {mem_diff:.2f} MB (Hypatia uses less)")


def main():
    print("\n" + "="*80)
    print("HYPATIA MLP PERFORMANCE TEST SUITE")
    print("="*80)

    # Check backend registration
    backends = torch._dynamo.list_backends()
    if "hypatia" not in backends:
        print(f"\n❌ ERROR: 'hypatia' backend not registered!")
        print(f"   Available backends: {backends}")
        return

    print(f"✅ Hypatia backend registered")

    try:
        test_performance_comparison()
        test_different_batch_sizes()
        test_memory_usage()

        print("\n" + "="*80)
        print("🎉 PERFORMANCE TESTS COMPLETED!")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
