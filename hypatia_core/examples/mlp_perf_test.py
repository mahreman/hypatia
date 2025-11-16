"""
MLP Performance Test (Large-Scale Workload)

Benchmarks Hypatia compilation with aggressive optimization settings.
Uses OFF checksum mode to allow maximum fusion opportunities.
Measures actual speedup from e-graph optimizations (e.g., Linear+ReLU fusion).

Large-scale configuration to reduce overhead-to-compute ratio:
  - HIDDEN_SIZE: 2048 (vs 256 in small test)
  - NUM_HIDDEN_LAYERS: 4 (vs 2 in small test)
  - BATCH_SIZE: 1024 (vs 256 in small test)
  - ITERS: 500 (vs 100 in small test)
"""

import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time
import hypatia_core  # Auto-registers 'hypatia' backend

# ============================================================================
# Configuration: Large-scale MLP for realistic workload
# ============================================================================
INPUT_SIZE = 784
HIDDEN_SIZE = 2048        # 256 → 2048 (8x larger)
NUM_HIDDEN_LAYERS = 4     # 2 → 4 (2x more layers)
OUTPUT_SIZE = 1000        # 10 → 1000 (100x larger, more realistic)
BATCH_SIZE = 1024         # 256 → 1024 (4x larger)
ITERS = 500               # 100 → 500 (more stable measurements)

# ============================================================================
# Environment: Aggressive mode for maximum performance
# ============================================================================

# Disable debug logging for clean performance measurement
os.environ["HYPATIA_DEBUG_FX"] = "0"

# Disable checksum validation to maximize performance
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"

# Enable fusion for performance testing
os.environ["HYPATIA_ENABLE_LINRELU_FUSION"] = "1"

print(f"[INFO] Debug logs disabled: HYPATIA_DEBUG_FX=0")
print(f"[INFO] Checksum mode: OFF")
print(f"[INFO] Fusion enabled: HYPATIA_ENABLE_LINRELU_FUSION=1")
print(f"[DEBUG] Registered backends: {torch._dynamo.list_backends()}")
print()


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth and width"""

    def __init__(self, in_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE,
                 num_hidden=NUM_HIDDEN_LAYERS, out_dim=OUTPUT_SIZE):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim

        # Input layer
        self.fc_in = nn.Linear(in_dim, hidden_dim)

        # Hidden layers
        self.fc_hidden = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_hidden - 1)
        ])

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    def forward(self, x):
        # Input → Hidden
        x = self.act(self.fc_in(x))

        # Hidden → Hidden
        for fc in self.fc_hidden:
            x = self.act(fc(x))

        # Hidden → Output (no activation)
        x = self.fc_out(x)
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


def calculate_flops(batch, in_dim, hidden_dim, num_hidden, out_dim):
    """Calculate approximate FLOPs for one forward pass"""
    # Input → Hidden: 2 * batch * in_dim * hidden_dim
    flops = 2 * batch * in_dim * hidden_dim
    # Hidden → Hidden (num_hidden - 1 times): 2 * batch * hidden_dim * hidden_dim
    flops += (num_hidden - 1) * 2 * batch * hidden_dim * hidden_dim
    # Hidden → Output: 2 * batch * hidden_dim * out_dim
    flops += 2 * batch * hidden_dim * out_dim
    return flops


def count_params(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 80)
    print("MLP Performance Test (Large-Scale Workload)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Create model and input
    model = MLP().to(device)
    model.eval()  # Inference mode
    x = torch.randn(BATCH_SIZE, INPUT_SIZE, device=device)

    # Calculate model statistics
    num_params = count_params(model)
    flops_per_iter = calculate_flops(BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE,
                                      NUM_HIDDEN_LAYERS, OUTPUT_SIZE)

    print("Model Architecture:")
    print(f"  Input:       {INPUT_SIZE} features")
    print(f"  Hidden:      {HIDDEN_SIZE} units × {NUM_HIDDEN_LAYERS} layers")
    print(f"  Output:      {OUTPUT_SIZE} classes")
    print(f"  Parameters:  {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  FLOPs/iter:  {flops_per_iter / 1e9:.3f} GFLOPs")
    print()

    # Compile with Hypatia
    print("Compiling with Hypatia backend...")
    opt = torch.compile(model, backend="hypatia")
    print()

    # Benchmark eager model
    print(f"Benchmarking eager model ({ITERS} iterations)...")
    with torch.no_grad():
        eager_ms = bench(model, x, iters=ITERS, device=device)
    eager_gflops = (flops_per_iter / 1e9) / (eager_ms / 1000.0)
    print(f"  Eager:   {eager_ms:.4f} ms/iter  ({eager_gflops:.2f} GFLOP/s)")

    # Benchmark optimized model
    print(f"\nBenchmarking Hypatia-compiled model ({ITERS} iterations)...")
    with torch.no_grad():
        hypatia_ms = bench(opt, x, iters=ITERS, device=device)
    hypatia_gflops = (flops_per_iter / 1e9) / (hypatia_ms / 1000.0)
    print(f"  Hypatia: {hypatia_ms:.4f} ms/iter  ({hypatia_gflops:.2f} GFLOP/s)")

    # Calculate speedup
    print()
    print("=" * 80)
    speedup = eager_ms / hypatia_ms
    improvement = (speedup - 1) * 100

    if speedup > 1.05:
        print(f"✅ Speedup: {speedup:.4f}x ({improvement:+.2f}% faster)")
    elif speedup < 0.95:
        slowdown = (1 / speedup - 1) * 100
        print(f"⚠️  Slowdown: {1/speedup:.4f}x ({slowdown:+.2f}% slower)")
    else:
        print(f"➡️  Neutral: {speedup:.4f}x (within measurement noise)")

    print("=" * 80)
    print()
    print("Interpretation:")
    print("  • Speedup > 1.05x: E-graph fusion is working effectively")
    print("  • Speedup ≈ 1.0x:  Overhead balanced with optimizations")
    print("  • Speedup < 0.95x: Backend overhead dominates (needs profiling)")
    print()
    print("Expected optimizations (with HYPATIA_ENABLE_LINRELU_FUSION=1):")
    print(f"  • {NUM_HIDDEN_LAYERS} Linear+ReLU fusions")
    print("  • Graph structure simplifications")
    print("  • Algebraic rewrites")


if __name__ == "__main__":
    main()
