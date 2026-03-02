#!/usr/bin/env python3
"""
Hypatia Real Model Test - HuggingFace GPT-2 Integration

Demonstrates Hypatia's optimization on real pre-trained model weights
from HuggingFace's GPT-2 family. Tests both:
  - NativeModel (f32, small models < 10M params)
  - QuantizedModel (INT4, large models >= 10M params)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import time
import hypatia_core as hypatia

print(f"Hypatia v{hypatia.__version__}")

def bench(fn, iters=50):
    for _ in range(5): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1000


def test_gpt2_mlp(model_name="gpt2"):
    """Test Hypatia on GPT-2's MLP layers with real pre-trained weights."""
    from transformers import GPT2Model

    print(f"\n{'='*70}")
    print(f" Loading {model_name}...")
    print(f"{'='*70}")

    model = GPT2Model.from_pretrained(model_name)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name}: {n_params/1e6:.0f}M params total")

    # GPT-2 uses Conv1D (weight: [in, out]) - transpose to Linear format [out, in]
    block = model.h[0]
    hidden = block.mlp.c_fc.weight.shape[0]
    mlp_dim = block.mlp.c_fc.weight.shape[1]
    print(f"  MLP dimensions: {hidden} -> {mlp_dim} -> {hidden}")

    # Build Sequential MLP with real weights (ReLU approximation of GELU)
    mlp = nn.Sequential(
        nn.Linear(hidden, mlp_dim),
        nn.ReLU(),
        nn.Linear(mlp_dim, hidden),
    )
    with torch.no_grad():
        mlp[0].weight.copy_(block.mlp.c_fc.weight.data.t())
        mlp[0].bias.copy_(block.mlp.c_fc.bias.data)
        mlp[2].weight.copy_(block.mlp.c_proj.weight.data.t())
        mlp[2].bias.copy_(block.mlp.c_proj.bias.data)
    mlp.eval()

    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP params: {mlp_params/1e6:.1f}M")

    # Optimize
    fast = hypatia.optimize(mlp)

    # Correctness
    print(f"\n  Correctness (RMSE):")
    for bs in [1, 8, 32]:
        x = torch.randn(bs, hidden)
        with torch.no_grad():
            orig = mlp(x)
            opt = fast(x)
        rmse = (orig - opt).pow(2).mean().sqrt().item()
        print(f"    batch={bs:2d}: {rmse:.6f}")

    # Memory
    if hasattr(fast, 'memory_saved_mb'):
        print(f"  Memory: {fast.memory_saved_mb:.0f}MB saved ({fast.compression_ratio:.1f}x)")

    # Benchmark
    torch.set_num_threads(1)
    print(f"\n  Benchmark (single-thread):")
    results = []
    for bs in [1, 4, 8, 16, 32]:
        x = torch.randn(bs, hidden)
        with torch.no_grad():
            pt_ms = bench(lambda: mlp(x))
            hy_ms = bench(lambda: fast(x))
        speedup = pt_ms / hy_ms
        results.append((bs, pt_ms, hy_ms, speedup))
        print(f"    batch={bs:2d}: PyTorch={pt_ms:.2f}ms  Hypatia={hy_ms:.2f}ms  {speedup:.2f}x")

    return results


def test_synthetic_llama():
    """Test with LLaMA-7B dimensions using synthetic weights."""
    print(f"\n{'='*70}")
    print(f" Synthetic LLaMA-7B FFN Layer")
    print(f"{'='*70}")

    model = nn.Sequential(
        nn.Linear(4096, 11008),
        nn.ReLU(),
        nn.Linear(11008, 4096),
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Dimensions: 4096 -> 11008 -> 4096 ({n_params/1e6:.0f}M params)")

    fast = hypatia.optimize(model)

    print(f"\n  Correctness:")
    for bs in [1, 8, 32]:
        x = torch.randn(bs, 4096)
        with torch.no_grad():
            rmse = (model(x) - fast(x)).pow(2).mean().sqrt().item()
        print(f"    batch={bs:2d}: RMSE={rmse:.4f}")

    if hasattr(fast, 'memory_saved_mb'):
        print(f"  Memory: {fast.memory_saved_mb:.0f}MB saved ({fast.compression_ratio:.1f}x)")

    torch.set_num_threads(1)
    print(f"\n  Benchmark (single-thread):")
    for bs in [1, 4, 8, 16, 32]:
        x = torch.randn(bs, 4096)
        with torch.no_grad():
            pt_ms = bench(lambda: model(x))
            hy_ms = bench(lambda: fast(x))
        print(f"    batch={bs:2d}: PyTorch={pt_ms:.2f}ms  Hypatia={hy_ms:.2f}ms  {pt_ms/hy_ms:.2f}x")


if __name__ == "__main__":
    # GPT-2 Small (124M total, 4.7M per MLP block) -> NativeModel
    test_gpt2_mlp("gpt2")

    # GPT-2 Large (774M total, 13.1M per MLP block) -> QuantizedModel
    try:
        test_gpt2_mlp("gpt2-large")
    except Exception as e:
        print(f"\n  Skipping gpt2-large: {e}")

    # Synthetic LLaMA-7B FFN -> QuantizedModel
    test_synthetic_llama()

    print(f"\n{'='*70}")
    print(f" REAL MODEL TEST COMPLETE")
    print(f"{'='*70}")
