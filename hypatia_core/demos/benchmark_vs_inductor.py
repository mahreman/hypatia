"""
Hypatia vs TorchInductor: Fair GPU Benchmark
=============================================

Answers the question: On the SAME GPU hardware, how does
torch.compile(backend='hypatia') compare to torch.compile(backend='inductor')?

Benchmark matrix:
  - Models: Small MLP, Medium MLP, Large MLP, Transformer-like block
  - Backends: Vanilla PyTorch (GPU), TorchInductor (max-autotune), Hypatia, CPU (reference)
  - Warmup: 5 iterations, Measurement: 100 iterations
  - GPU timing: torch.cuda.synchronize() for accuracy

Usage:
    python benchmark_vs_inductor.py
"""

import sys
import os
import time
import gc
import statistics

# --- Path setup for import compatibility ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


# ============================================================================
# Configuration
# ============================================================================

WARMUP_ITERS = 5
MEASURE_ITERS = 100
DEVICE_GPU = "cuda"
DEVICE_CPU = "cpu"


# ============================================================================
# Model Definitions
# ============================================================================

class SimpleMLP(nn.Module):
    """Small MLP: 784 -> 256 -> 128 -> 10"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class MediumMLP(nn.Module):
    """Medium MLP: 1024 -> 2048 -> 1024 -> 512 -> 256 -> 10"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class LargeMLP(nn.Module):
    """Large MLP: 2048 -> 4096 -> 2048 -> 1024 -> 512 -> 10"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer-like block: multi-head attention + feed-forward."""
    def __init__(self, d_model=512, nhead=8, dim_ff=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# ============================================================================
# Benchmark Helpers
# ============================================================================

def count_params(model):
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_on_gpu(model, dummy_input, label, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Benchmark a model on GPU with proper synchronization.

    Returns:
        Mean time in milliseconds, or None if failed.
    """
    try:
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(dummy_input)
            torch.cuda.synchronize()

            # Measurement: collect per-iteration times
            times = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"    {label}: {mean_ms:.3f} ms  (std: {std_ms:.3f} ms)")
        return mean_ms

    except Exception as e:
        print(f"    {label}: FAILED - {e}")
        return None


def benchmark_on_cpu(model, dummy_input, label, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Benchmark a model on CPU (reference only).

    Returns:
        Mean time in milliseconds, or None if failed.
    """
    try:
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(dummy_input)

            # Measurement
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"    {label}: {mean_ms:.3f} ms  (std: {std_ms:.3f} ms)")
        return mean_ms

    except Exception as e:
        print(f"    {label}: FAILED - {e}")
        return None


def compile_with_backend(model, backend, mode=None):
    """Compile a model with a given backend. Returns compiled model or None."""
    try:
        kwargs = {"backend": backend}
        if mode is not None:
            kwargs["mode"] = mode
        compiled = torch.compile(model, **kwargs)
        return compiled
    except Exception as e:
        print(f"    [!] torch.compile(backend='{backend}') failed: {e}")
        return None


def reset_dynamo():
    """Reset torch._dynamo state between benchmarks to avoid cross-contamination."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark():
    print("=" * 74)
    print("  Hypatia vs TorchInductor: Fair GPU Benchmark")
    print("=" * 74)

    # ------------------------------------------------------------------
    # Hardware check
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\n  [ERROR] CUDA is not available. This benchmark requires a GPU.")
        print("  Exiting.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Measurement: {MEASURE_ITERS} iters")

    # ------------------------------------------------------------------
    # Register Hypatia backend
    # ------------------------------------------------------------------
    hypatia_available = False
    try:
        import hypatia_core
        hypatia_core.register_backend()
        hypatia_available = True
        print("  Hypatia backend: registered")
    except Exception as e:
        print(f"  Hypatia backend: NOT available ({e})")

    # ------------------------------------------------------------------
    # Define benchmark suite
    # ------------------------------------------------------------------
    batch_size = 32

    models_config = [
        {
            "name": "Small MLP",
            "model_fn": SimpleMLP,
            "input_shape": (batch_size, 784),
        },
        {
            "name": "Medium MLP",
            "model_fn": MediumMLP,
            "input_shape": (batch_size, 1024),
        },
        {
            "name": "Large MLP",
            "model_fn": LargeMLP,
            "input_shape": (batch_size, 2048),
        },
        {
            "name": "Transformer Block",
            "model_fn": TransformerBlock,
            "input_shape": (batch_size, 64, 512),  # (batch, seq_len, d_model)
        },
    ]

    # Collect all results for the summary table
    # results[model_name] = { "backend_name": mean_ms_or_None, ... }
    all_results = {}

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    for cfg in models_config:
        model_name = cfg["name"]
        model_fn = cfg["model_fn"]
        input_shape = cfg["input_shape"]

        print(f"\n{'-' * 74}")
        print(f"  Model: {model_name}")

        # Create model and dummy input
        base_model = model_fn()
        n_params = count_params(base_model)
        print(f"  Parameters: {n_params:,}")
        print(f"  Input shape: {input_shape}")
        print()

        results = {}

        # --- 1. CPU Reference ---
        cpu_model = model_fn().to(DEVICE_CPU).eval()
        cpu_input = torch.randn(input_shape, device=DEVICE_CPU)
        cpu_ms = benchmark_on_cpu(cpu_model, cpu_input, "CPU (reference)")
        results["CPU (ref)"] = cpu_ms
        del cpu_model, cpu_input
        gc.collect()

        # --- 2. Vanilla PyTorch on GPU ---
        reset_dynamo()
        gpu_model = model_fn().to(DEVICE_GPU).eval()
        gpu_input = torch.randn(input_shape, device=DEVICE_GPU)
        vanilla_ms = benchmark_on_gpu(gpu_model, gpu_input, "Vanilla PyTorch (GPU)")
        results["Vanilla GPU"] = vanilla_ms
        del gpu_model
        torch.cuda.empty_cache()
        gc.collect()

        # --- 3. TorchInductor (max-autotune) ---
        reset_dynamo()
        gpu_model = model_fn().to(DEVICE_GPU).eval()
        print("    Compiling with TorchInductor (max-autotune)...")
        inductor_model = compile_with_backend(gpu_model, "inductor", mode="max-autotune")
        if inductor_model is not None:
            inductor_ms = benchmark_on_gpu(inductor_model, gpu_input, "TorchInductor (max-autotune)")
            results["Inductor"] = inductor_ms
        else:
            results["Inductor"] = None
        del gpu_model, inductor_model
        torch.cuda.empty_cache()
        gc.collect()

        # --- 4. Hypatia backend ---
        reset_dynamo()
        if hypatia_available:
            gpu_model = model_fn().to(DEVICE_GPU).eval()
            gpu_input = torch.randn(input_shape, device=DEVICE_GPU)
            print("    Compiling with Hypatia backend...")
            hypatia_model = compile_with_backend(gpu_model, "hypatia")
            if hypatia_model is not None:
                hypatia_ms = benchmark_on_gpu(hypatia_model, gpu_input, "Hypatia")
                results["Hypatia"] = hypatia_ms
            else:
                results["Hypatia"] = None
            del gpu_model, hypatia_model
            torch.cuda.empty_cache()
            gc.collect()
        else:
            results["Hypatia"] = None
            print("    Hypatia: SKIPPED (backend not available)")

        all_results[model_name] = results

        # Per-model mini comparison
        baseline = results.get("Vanilla GPU")
        if baseline is not None and baseline > 0:
            print()
            print(f"    -- Speedup vs Vanilla GPU --")
            for bname, bms in results.items():
                if bname == "Vanilla GPU" or bms is None:
                    continue
                speedup = baseline / bms
                print(f"    {bname}: {speedup:.2f}x")

        # Clean up GPU memory between models
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("  SUMMARY TABLE  (all times in ms, lower is better)")
    print("=" * 74)

    backends = ["CPU (ref)", "Vanilla GPU", "Inductor", "Hypatia"]

    # Header
    header = f"  {'Model':<22}"
    for b in backends:
        header += f" {b:>12}"
    header += f" {'Hyp/Ind':>10}"
    print(header)
    print("  " + "-" * (22 + 12 * len(backends) + 10))

    for model_name, results in all_results.items():
        row = f"  {model_name:<22}"
        for b in backends:
            val = results.get(b)
            if val is not None:
                row += f" {val:>12.3f}"
            else:
                row += f" {'N/A':>12}"

        # Hypatia / Inductor ratio
        hyp = results.get("Hypatia")
        ind = results.get("Inductor")
        if hyp is not None and ind is not None and ind > 0:
            ratio = hyp / ind
            row += f" {ratio:>10.2f}x"
        else:
            row += f" {'N/A':>10}"

        print(row)

    print("  " + "-" * (22 + 12 * len(backends) + 10))

    # ------------------------------------------------------------------
    # Key Insights
    # ------------------------------------------------------------------
    print()
    print("  KEY:")
    print("  - Hyp/Ind < 1.00x means Hypatia is FASTER than Inductor")
    print("  - Hyp/Ind > 1.00x means Inductor is FASTER than Hypatia")
    print("  - CPU (ref) is included only as a baseline reference")
    print()
    print("  NOTES:")
    print("  - TorchInductor uses mode='max-autotune' (best Inductor config)")
    print("  - Hypatia backend applies e-graph optimization + optional chain compile")
    print("  - Set HYPATIA_CHAIN_COMPILE=0 to benchmark e-graph only (no Triton)")
    print("  - All GPU timings use torch.cuda.synchronize() for accuracy")
    print("  - First compile may be slow; only steady-state runtime is measured")
    print("=" * 74)


if __name__ == "__main__":
    run_benchmark()
