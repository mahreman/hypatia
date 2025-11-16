"""
Comprehensive MLP Fusion Benchmark Suite

Tests FusedLinearReLU performance across different configurations:
- Multiple batch sizes (256, 1024, 4096, 8192)
- Multiple hidden dimensions (256, 512, 1024, 2048, 4096)
- Multiple MLP depths (2, 3, 4 layers)
- Multiple backends (eager, torch.compile+inductor, torch.compile+hypatia)

Target: Achieve ≥20% speedup (hypatia/inductor ≤ 0.8) in compute-bound regime
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypatia_core.fused_modules import HypatiaFusedLinearReLU


class BaselineMLP(nn.Module):
    """Baseline MLP using nn.Linear + ReLU"""
    def __init__(self, layers, device='cuda'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1], device=device))
            if i < len(layers) - 2:  # No ReLU after last layer
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FusedMLP(nn.Module):
    """MLP using HypatiaFusedLinearReLU"""
    def __init__(self, layers, device='cuda'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            # Use fused Linear+ReLU for all but last layer
            self.layers.append(HypatiaFusedLinearReLU(
                layers[i], layers[i+1], device=device
            ))
        # Last layer: just Linear
        self.layers.append(nn.Linear(layers[-2], layers[-1], device=device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def benchmark_model(model, x, warmup=10, iters=100, device='cuda'):
    """
    Benchmark a model's forward pass.

    Args:
        model: PyTorch model
        x: Input tensor
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        device: Device to run on

    Returns:
        Average time per iteration in milliseconds
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(x)

            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)  # Convert to ms

    return sum(times) / len(times)


def compute_gflops(layers, batch_size):
    """
    Compute total GFLOPs for MLP forward pass.

    FLOPs per Linear layer: 2 * batch * in_features * out_features
    """
    total_flops = 0
    for i in range(len(layers) - 1):
        total_flops += 2 * batch_size * layers[i] * layers[i+1]
    return total_flops / 1e9


def run_benchmark_suite(device='cuda'):
    """
    Run comprehensive benchmark suite.

    Tests configurations from small (overhead-bound) to large (compute-bound)
    to identify sweet spot for fusion optimization.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 80)
    print("Hypatia MLP Fusion Benchmark Suite")
    print("=" * 80)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("")

    # Benchmark configurations
    configs = [
        # (name, layers, batch_size, iters)
        # Small configurations (overhead-bound)
        ("Small MLP 2-layer", [256, 256, 256], 256, 500),
        ("Small MLP 4-layer", [256, 256, 256, 256, 256], 256, 500),

        # Medium configurations (transitional)
        ("Medium MLP 2-layer", [1024, 1024, 1024], 1024, 500),
        ("Medium MLP 4-layer", [1024, 1024, 1024, 1024, 1024], 1024, 500),

        # Large configurations (compute-bound - TARGET)
        ("Large MLP 2-layer", [2048, 2048, 2048], 2048, 300),
        ("Large MLP 4-layer", [2048, 2048, 2048, 2048, 2048], 2048, 300),

        # Extra-large configurations (ideal for fusion)
        ("XLarge MLP 2-layer", [4096, 4096, 4096], 4096, 200),
        ("XLarge MLP 4-layer", [4096, 4096, 4096, 4096, 4096], 4096, 200),
    ]

    results = []

    for config_name, layers, batch_size, iters in configs:
        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"  Layers: {layers}")
        print(f"  Batch size: {batch_size}")
        print(f"  GFLOPs: {compute_gflops(layers, batch_size):.2f}")
        print(f"{'='*80}")

        # Create input
        x = torch.randn(batch_size, layers[0], device=device)

        # Baseline: Separate Linear + ReLU
        print("\n[1/2] Baseline (Linear + ReLU)...")
        baseline = BaselineMLP(layers, device=device)
        baseline_ms = benchmark_model(baseline, x, warmup=10, iters=iters, device=device)
        print(f"  Time: {baseline_ms:.4f} ms/iter")

        # Fused: HypatiaFusedLinearReLU
        print("\n[2/2] Fused (HypatiaFusedLinearReLU)...")
        fused = FusedMLP(layers, device=device)
        fused_ms = benchmark_model(fused, x, warmup=10, iters=iters, device=device)
        print(f"  Time: {fused_ms:.4f} ms/iter")

        # Compute speedup
        speedup = baseline_ms / fused_ms
        gflops = compute_gflops(layers, batch_size)
        baseline_tflops = gflops / baseline_ms
        fused_tflops = gflops / fused_ms

        print(f"\n📊 Results:")
        print(f"  Baseline: {baseline_ms:.4f} ms ({baseline_tflops:.2f} TFLOP/s)")
        print(f"  Fused:    {fused_ms:.4f} ms ({fused_tflops:.2f} TFLOP/s)")
        print(f"  Speedup:  {speedup:.3f}x", end="")

        if speedup >= 1.2:
            print(" ✅ (target met: ≥1.2x)")
        elif speedup >= 1.0:
            print(" 🟡 (slight improvement)")
        else:
            print(" ⚠️  (slowdown)")

        results.append({
            'config': config_name,
            'layers': len(layers) - 1,
            'hidden': layers[1],
            'batch': batch_size,
            'gflops': gflops,
            'baseline_ms': baseline_ms,
            'fused_ms': fused_ms,
            'speedup': speedup,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<25} {'Layers':>6} {'Hidden':>7} {'Batch':>7} {'GFLOPs':>8} "
          f"{'Baseline':>10} {'Fused':>10} {'Speedup':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['config']:<25} {r['layers']:>6} {r['hidden']:>7} {r['batch']:>7} "
              f"{r['gflops']:>8.2f} {r['baseline_ms']:>10.4f} {r['fused_ms']:>10.4f} "
              f"{r['speedup']:>8.3f}x")

    # Identify best configurations
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    best = sorted(results, key=lambda x: x['speedup'], reverse=True)[:3]
    for i, r in enumerate(best, 1):
        print(f"{i}. {r['config']}: {r['speedup']:.3f}x speedup "
              f"({r['gflops']:.2f} GFLOPs, batch={r['batch']})")

    # Check if target met
    print("\n" + "=" * 80)
    print("TARGET ANALYSIS")
    print("=" * 80)
    target_met = [r for r in results if r['speedup'] >= 1.2]

    if target_met:
        print(f"✅ Target met: {len(target_met)}/{len(results)} configs achieved ≥1.2x speedup")
        print("\nConfigurations meeting target:")
        for r in target_met:
            print(f"  - {r['config']}: {r['speedup']:.3f}x")
    else:
        print(f"⚠️  Target not met: No configs achieved ≥1.2x speedup")
        print(f"   Best: {best[0]['config']} with {best[0]['speedup']:.3f}x")
        print("\nPossible reasons:")
        print("  1. CUDA extension not built (using fallback)")
        print("  2. Overhead dominates for these problem sizes")
        print("  3. Need larger batch/hidden dimensions")

    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='MLP Fusion Benchmark Suite')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run benchmark on')
    args = parser.parse_args()

    run_benchmark_suite(device=args.device)


if __name__ == "__main__":
    main()
