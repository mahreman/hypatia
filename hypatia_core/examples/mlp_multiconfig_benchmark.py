"""
Comprehensive MLP Performance Benchmark with Multi-Config Sweep

Tests multiple model sizes to identify where Hypatia fusion provides benefits:
- Small models (overhead-bound)
- Medium models (transitional)
- Large models (compute-bound)

Also verifies if CUDA kernel fusion is actually active.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time
import hypatia_core

# Configure environment for testing
os.environ["HYPATIA_DEBUG_FX"] = "0"
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"
os.environ["HYPATIA_ENABLE_LINRELU_FUSION"] = "1"


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""

    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_hidden - 1)
        ])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc in self.fc_hidden:
            x = self.act(fc(x))
        x = self.fc_out(x)
        return x


def bench(fn, x, warmup=10, iters=100, device="cpu"):
    """Benchmark with proper synchronization"""
    # Warmup
    for _ in range(warmup):
        _ = fn(x)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(x)

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms/iter


def calculate_flops(batch, in_dim, hidden_dim, num_hidden, out_dim):
    """Calculate FLOPs for forward pass"""
    flops = 2 * batch * in_dim * hidden_dim
    flops += (num_hidden - 1) * 2 * batch * hidden_dim * hidden_dim
    flops += 2 * batch * hidden_dim * out_dim
    return flops


def check_cuda_extension_status():
    """Check if CUDA extension is built and available"""
    print("=" * 80)
    print("CUDA Extension Status Check")
    print("=" * 80)

    try:
        from hypatia_core.fused_modules import CUDA_EXTENSION_AVAILABLE
        print(f"CUDA_EXTENSION_AVAILABLE: {CUDA_EXTENSION_AVAILABLE}")

        if CUDA_EXTENSION_AVAILABLE:
            import hypatia_core._linear_relu_cuda as cuda_ext
            print("✅ CUDA extension successfully imported")
            print(f"   Available functions: {dir(cuda_ext)}")
        else:
            print("⚠️  CUDA extension not available (will use PyTorch fallback)")
            print("   To enable CUDA kernel fusion:")
            print("   cd hypatia_core/hypatia_core/fused_kernels")
            print("   ./build.sh")
    except ImportError as e:
        print(f"❌ Failed to import fused_modules: {e}")

    print()


def verify_fusion_active(model, x, device):
    """
    Verify if fusion is actually happening by inspecting the graph
    """
    print("=" * 80)
    print("Fusion Verification")
    print("=" * 80)

    # Check if FusedLinearReLU is being used
    has_fused = False
    for name, module in model.named_modules():
        if 'Fused' in type(module).__name__:
            has_fused = True
            print(f"✅ Found fused module: {name} ({type(module).__name__})")

    if not has_fused:
        print("ℹ️  No fused modules found in model (expected for standard nn.Linear + ReLU)")
        print("   Fusion happens during torch.compile optimization")

    # Compile and check graph
    print("\nCompiling model...")
    compiled = torch.compile(model, backend="hypatia")

    # Run once to trigger compilation
    with torch.no_grad():
        _ = compiled(x)

    print("✅ Compilation successful")
    print()


def run_config(config_name, in_dim, hidden_dim, num_hidden, out_dim, batch, iters, device):
    """Run benchmark for a single configuration"""

    # Create model and input
    model = MLP(in_dim, hidden_dim, num_hidden, out_dim).to(device)
    model.eval()
    x = torch.randn(batch, in_dim, device=device)

    # Calculate metrics
    flops = calculate_flops(batch, in_dim, hidden_dim, num_hidden, out_dim)
    gflops = flops / 1e9

    # Benchmark eager
    with torch.no_grad():
        eager_ms = bench(model, x, warmup=10, iters=iters, device=device)
    eager_tflops = gflops / eager_ms

    # Compile with Hypatia
    compiled = torch.compile(model, backend="hypatia")

    # Benchmark compiled
    with torch.no_grad():
        hypatia_ms = bench(compiled, x, warmup=10, iters=iters, device=device)
    hypatia_tflops = gflops / hypatia_ms

    # Calculate speedup
    speedup = eager_ms / hypatia_ms

    return {
        'config': config_name,
        'in_dim': in_dim,
        'hidden_dim': hidden_dim,
        'num_hidden': num_hidden,
        'out_dim': out_dim,
        'batch': batch,
        'gflops': gflops,
        'eager_ms': eager_ms,
        'hypatia_ms': hypatia_ms,
        'eager_tflops': eager_tflops,
        'hypatia_tflops': hypatia_tflops,
        'speedup': speedup,
    }


def main():
    print("=" * 80)
    print("Hypatia MLP Multi-Config Performance Sweep")
    print("=" * 80)
    print()

    # Check CUDA extension status
    check_cuda_extension_status()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Verify fusion on a sample model
    sample_model = MLP(784, 256, 2, 10).to(device)
    sample_x = torch.randn(64, 784, device=device)
    verify_fusion_active(sample_model, sample_x, device)

    # Define benchmark configurations
    # Format: (name, in_dim, hidden_dim, num_hidden, out_dim, batch, iters)
    configs = [
        # Small models (overhead-bound)
        ("Tiny MLP 2-layer", 784, 256, 2, 10, 64, 200),
        ("Tiny MLP 2-layer (larger batch)", 784, 256, 2, 10, 256, 200),
        ("Small MLP 3-layer", 512, 512, 3, 128, 128, 200),

        # Medium models (transitional)
        ("Medium MLP 3-layer", 784, 512, 3, 128, 256, 300),
        ("Medium MLP 4-layer", 1024, 512, 4, 256, 256, 300),

        # Large models (compute-bound - TARGET for kernel fusion)
        ("Large MLP 4-layer", 784, 2048, 4, 1000, 512, 300),
        ("Large MLP 4-layer (big batch)", 784, 2048, 4, 1000, 1024, 300),

        # Extra-large models (ideal for fusion)
        ("XLarge MLP 4-layer", 2048, 2048, 4, 1000, 1024, 200),
        ("XLarge MLP 6-layer", 1024, 4096, 6, 1000, 512, 200),
    ]

    print("=" * 80)
    print("Running Benchmarks")
    print("=" * 80)
    print()

    results = []
    for i, (name, in_d, hid_d, num_h, out_d, bs, iters) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name}")
        print(f"  Architecture: {in_d} → {hid_d}×{num_h} → {out_d}, batch={bs}")

        result = run_config(name, in_d, hid_d, num_h, out_d, bs, iters, device)
        results.append(result)

        print(f"  Eager:   {result['eager_ms']:.4f} ms ({result['eager_tflops']:.2f} TFLOP/s)")
        print(f"  Hypatia: {result['hypatia_ms']:.4f} ms ({result['hypatia_tflops']:.2f} TFLOP/s)")
        print(f"  Speedup: {result['speedup']:.3f}x", end="")

        if result['speedup'] >= 1.05:
            print(" ✅")
        elif result['speedup'] >= 0.95:
            print(" 🟡")
        else:
            print(" ⚠️")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<30} {'Batch':>6} {'GFLOPs':>8} {'Eager':>10} {'Hypatia':>10} {'Speedup':>9}")
    print("-" * 80)

    for r in results:
        print(f"{r['config']:<30} {r['batch']:>6} {r['gflops']:>8.2f} "
              f"{r['eager_ms']:>10.4f} {r['hypatia_ms']:>10.4f} {r['speedup']:>9.3f}x")

    # Analysis by category
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL SIZE")
    print("=" * 80)

    small = [r for r in results if 'Tiny' in r['config'] or 'Small' in r['config']]
    medium = [r for r in results if 'Medium' in r['config']]
    large = [r for r in results if 'Large' in r['config'] or 'XLarge' in r['config']]

    def analyze_category(name, configs):
        if not configs:
            return
        avg_speedup = sum(c['speedup'] for c in configs) / len(configs)
        best = max(configs, key=lambda x: x['speedup'])
        worst = min(configs, key=lambda x: x['speedup'])

        print(f"\n{name}:")
        print(f"  Configs: {len(configs)}")
        print(f"  Avg speedup: {avg_speedup:.3f}x")
        print(f"  Best: {best['config']} ({best['speedup']:.3f}x)")
        print(f"  Worst: {worst['config']} ({worst['speedup']:.3f}x)")

    analyze_category("Small Models (overhead-bound)", small)
    analyze_category("Medium Models (transitional)", medium)
    analyze_category("Large Models (compute-bound)", large)

    # Check if target met
    print("\n" + "=" * 80)
    print("TARGET ANALYSIS")
    print("=" * 80)

    target_configs = [r for r in results if r['speedup'] >= 1.05]

    if target_configs:
        print(f"✅ {len(target_configs)}/{len(results)} configs achieved ≥1.05x speedup")
        print("\nConfigs meeting target:")
        for r in sorted(target_configs, key=lambda x: x['speedup'], reverse=True):
            print(f"  • {r['config']}: {r['speedup']:.3f}x ({r['gflops']:.2f} GFLOPs)")
    else:
        best = max(results, key=lambda x: x['speedup'])
        print(f"⚠️  No configs achieved ≥1.05x speedup")
        print(f"   Best: {best['config']} with {best['speedup']:.3f}x")
        print("\nPossible reasons:")
        print("  1. CUDA extension not built (check status above)")
        print("  2. Kernel fusion not active (e-graph rewrite not triggering)")
        print("  3. Backend overhead dominates for these sizes")
        print("  4. Need larger compute-bound workloads")

    # Check if CUDA kernel is actually being used
    print("\n" + "=" * 80)
    print("KERNEL FUSION STATUS")
    print("=" * 80)

    from hypatia_core.fused_modules import CUDA_EXTENSION_AVAILABLE

    if not CUDA_EXTENSION_AVAILABLE:
        print("⚠️  CUDA extension not available")
        print("   Current implementation: PyTorch nn.Linear + torch.relu (2 kernels)")
        print("   Expected with CUDA extension: Single fused kernel")
        print("\n   To enable true kernel fusion:")
        print("   1. Install CUDA toolkit")
        print("   2. cd hypatia_core/hypatia_core/fused_kernels")
        print("   3. ./build.sh")
        print("   4. Re-run this benchmark")
    else:
        print("✅ CUDA extension available")
        print("   Kernel fusion should be active for CUDA tensors with FP32")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
