#!/usr/bin/env python3
"""
Benchmark: INT4 Quantized (AVX2 SIMD + Rayon) vs PyTorch f32
Tests on realistic transformer FFN sizes (GPT-2 through LLaMA-13B)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from _hypatia_core import native_forward, quantize_weights, quantized_forward


def extract_np_layers(model):
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


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000  # ms


if __name__ == "__main__":
    # Let PyTorch use single thread for fair comparison
    torch.set_num_threads(1)

    print("=" * 95)
    print(" INT4 QUANTIZED INFERENCE BENCHMARK (AVX2 SIMD + Rayon)")
    print(" Hypatia INT4 vs PyTorch f32 (single-thread) vs Hypatia f32")
    print("=" * 95)
    print(f"PyTorch: {torch.__version__}, Threads: {torch.get_num_threads()}")
    print(f"CPU cores available for Rayon: {os.cpu_count()}")

    configs = [
        # (name, hidden, ffn_dim, batch)
        ("GPT-2 Small",   768,  3072,   1),
        ("GPT-2 Medium", 1024,  4096,   1),
        ("GPT-2 Large",  1280,  5120,   1),
        ("GPT-2 XL",     1600,  6400,   1),
        ("LLaMA-1B",     2048,  5504,   1),
        ("LLaMA-7B",     4096, 11008,   1),
        ("LLaMA-13B",    5120, 13824,   1),
    ]

    print(f"\n{'Model':<18} {'Params':>8} {'f32 MB':>7} {'q4 MB':>6} {'Comp':>5}"
          f" {'PT f32':>8} {'HY q4':>8} {'Speedup':>8} {'RMSE':>8}")
    print("-" * 95)

    for name, hidden, ffn, batch in configs:
        # Build FFN
        model = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.ReLU(),
            nn.Linear(ffn, hidden),
        )
        model.eval()
        params = sum(p.numel() for p in model.parameters())

        # Extract layers
        np_layers = extract_np_layers(model)

        # Quantize
        q_layers = quantize_weights(np_layers, 128)

        # Memory stats
        f32_bytes = 0
        q4_bytes = 0
        for ql in q_layers:
            f32_bytes += ql[8]  # orig_bytes
            q4_bytes += ql[9]   # quant_bytes
        f32_mb = f32_bytes / 1024 / 1024
        q4_mb = q4_bytes / 1024 / 1024
        compression = f32_bytes / q4_bytes if q4_bytes > 0 else 0

        # Create input
        x = torch.randn(batch, hidden)
        x_np = x.float().contiguous().numpy()

        # Correctness: measure quantization error
        with torch.no_grad():
            ref = model(x).numpy()
        q4_out = quantized_forward(x_np, q_layers)
        rmse = np.sqrt(np.mean((ref - q4_out) ** 2))

        # Benchmark PyTorch f32 (single-threaded)
        with torch.no_grad():
            pt_ms = bench(lambda: model(x), warmup=3, iters=15)

        # Benchmark Hypatia INT4 (AVX2 SIMD + Rayon multi-threaded)
        hy_ms = bench(lambda: quantized_forward(x_np, q_layers), warmup=3, iters=15)

        speedup = pt_ms / hy_ms
        marker = " <--" if speedup >= 1.3 else ""

        params_str = f"{params/1e6:.0f}M"
        print(f"{name:<18} {params_str:>8} {f32_mb:>6.1f} {q4_mb:>6.1f} {compression:>4.1f}x"
              f" {pt_ms:>7.2f}ms {hy_ms:>7.2f}ms {speedup:>7.2f}x {rmse:>7.4f}{marker}")

    # === Also compare with Hypatia f32 native (multi-threaded BLAS) ===
    print(f"\n{'='*95}")
    print(" COMPARISON: Hypatia f32 (MT BLAS) vs INT4 (AVX2+Rayon) vs PyTorch f32 (1T)")
    print(f"{'='*95}")

    print(f"\n{'Model':<18} {'PT f32(1T)':>10} {'HY f32(MT)':>11} {'HY q4(MT)':>10}"
          f" {'q4 vs PT':>9} {'q4 vs HYf32':>12}")
    print("-" * 80)

    for name, hidden, ffn, batch in configs:
        model = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.ReLU(),
            nn.Linear(ffn, hidden),
        )
        model.eval()

        np_layers = extract_np_layers(model)
        q_layers = quantize_weights(np_layers, 128)

        x = torch.randn(batch, hidden)
        x_np = x.float().contiguous().numpy()

        # PyTorch f32 single-threaded
        with torch.no_grad():
            pt_ms = bench(lambda: model(x), warmup=3, iters=15)

        # Hypatia f32 multi-threaded BLAS
        hy_f32_ms = bench(lambda: native_forward(x_np, np_layers), warmup=3, iters=15)

        # Hypatia INT4 AVX2+Rayon
        hy_q4_ms = bench(lambda: quantized_forward(x_np, q_layers), warmup=3, iters=15)

        q4_vs_pt = pt_ms / hy_q4_ms
        q4_vs_f32 = hy_f32_ms / hy_q4_ms

        marker = " <--" if q4_vs_pt >= 1.3 else ""
        print(f"{name:<18} {pt_ms:>8.2f}ms {hy_f32_ms:>9.2f}ms {hy_q4_ms:>8.2f}ms"
              f" {q4_vs_pt:>8.2f}x {q4_vs_f32:>10.2f}x{marker}")

    # === Batched inference test ===
    print(f"\n{'='*95}")
    print(" BATCHED INFERENCE (batch=32)")
    print(f"{'='*95}")

    batch_configs = [
        ("GPT-2 Small",   768,  3072,  32),
        ("GPT-2 Medium", 1024,  4096,  32),
        ("LLaMA-1B",     2048,  5504,  32),
    ]

    print(f"\n{'Model':<18} {'PT f32':>10} {'HY q4':>10} {'Speedup':>8}")
    print("-" * 55)

    for name, hidden, ffn, batch in batch_configs:
        model = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.ReLU(),
            nn.Linear(ffn, hidden),
        )
        model.eval()

        np_layers = extract_np_layers(model)
        q_layers = quantize_weights(np_layers, 128)

        x = torch.randn(batch, hidden)
        x_np = x.float().contiguous().numpy()

        with torch.no_grad():
            pt_ms = bench(lambda: model(x), warmup=3, iters=10)

        hy_ms = bench(lambda: quantized_forward(x_np, q_layers), warmup=3, iters=10)

        speedup = pt_ms / hy_ms
        marker = " <--" if speedup >= 1.3 else ""
        print(f"{name:<18} {pt_ms:>8.2f}ms {hy_ms:>8.2f}ms {speedup:>7.2f}x{marker}")

    print(f"\n{'='*95}")
    print(" DONE")
    print(f"{'='*95}")
