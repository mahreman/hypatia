#!/usr/bin/env python3
"""
Benchmark: Large model simulation
Testing realistic sizes for 8B-class models (single layer)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from _hypatia_core import native_forward, native_train_step


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


def extract_train_data(model):
    weights, biases, activations = [], [], []
    children = list(model.children())
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, nn.Linear):
            w = child.weight.data.detach().float().contiguous().numpy().copy()
            b = child.bias.data.detach().float().contiguous().numpy().copy() if child.bias is not None else None
            act = "none"
            if i + 1 < len(children) and isinstance(children[i + 1], nn.ReLU):
                act = "relu"
                i += 1
            weights.append(w)
            biases.append(b)
            activations.append(act)
        i += 1
    return weights, biases, activations


def bench(name, fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000  # ms


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("=" * 70)
    print(" LARGE MODEL BENCHMARK - Realistic Transformer Layer Sizes")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}, Threads: {torch.get_num_threads()}")

    configs = [
        # (name, hidden, ffn_dim, batch, seq_len_equiv)
        # GPT-2 Small (117M)
        ("GPT-2 Small FFN",      768,  3072,  32, "B=32"),
        # GPT-2 Medium (345M)
        ("GPT-2 Medium FFN",    1024,  4096,  16, "B=16"),
        # GPT-2 Large (774M)
        ("GPT-2 Large FFN",     1280,  5120,   8, "B=8"),
        # LLaMA-1B style
        ("LLaMA-1B FFN",        2048,  5504,   4, "B=4"),
        # LLaMA-7B / 8B style (single FFN layer)
        ("LLaMA-7B FFN",        4096, 11008,   1, "B=1"),
        # LLaMA-13B style
        ("LLaMA-13B FFN",       5120, 13824,   1, "B=1"),
    ]

    print(f"\n{'Model':<25} {'Params':>10} {'PyTorch':>10} {'Hypatia':>10} {'Speedup':>8}")
    print("-" * 70)

    for name, hidden, ffn, batch, bs_label in configs:
        # Build FFN: Linear(hidden, ffn) -> ReLU -> Linear(ffn, hidden)
        model = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.ReLU(),
            nn.Linear(ffn, hidden),
        )
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        params_str = f"{params/1e6:.1f}M"

        x = torch.randn(batch, hidden)
        np_layers = extract_np_layers(model)
        x_np = x.float().contiguous().numpy()

        # Correctness check
        with torch.no_grad():
            ref = model(x).numpy()
        hyp = native_forward(x_np, np_layers)
        diff = np.abs(ref - hyp).max()
        if diff > 0.01:
            print(f"  {name}: CORRECTNESS FAIL (diff={diff:.4f})")
            continue

        # Benchmark inference
        with torch.no_grad():
            pt_ms = bench(f"pt-{name}", lambda: model(x))
        hy_ms = bench(f"hy-{name}", lambda: native_forward(x_np, np_layers))

        speedup = pt_ms / hy_ms
        marker = " <--" if speedup >= 1.3 else ""
        print(f"{name+' '+bs_label:<25} {params_str:>10} {pt_ms:>8.2f}ms {hy_ms:>8.2f}ms {speedup:>7.2f}x{marker}")

    # === TRAINING on larger sizes ===
    print(f"\n{'='*70}")
    print(" TRAINING BENCHMARK - Larger Models")
    print(f"{'='*70}")

    train_configs = [
        ("GPT-2 Small FFN",   768,  3072, 32),
        ("GPT-2 Medium FFN",  1024, 4096, 16),
        ("GPT-2 Large FFN",   1280, 5120,  8),
        ("LLaMA-1B FFN",      2048, 5504,  4),
    ]

    print(f"\n{'Model':<25} {'Params':>10} {'PyTorch':>10} {'Hypatia':>10} {'Speedup':>8}")
    print("-" * 70)

    lr = 0.001
    for name, hidden, ffn, batch in train_configs:
        model_pt = nn.Sequential(
            nn.Linear(hidden, ffn), nn.ReLU(), nn.Linear(ffn, hidden)
        )
        model_pt.train()
        opt = torch.optim.SGD(model_pt.parameters(), lr=lr)

        x = torch.randn(batch, hidden)
        y = torch.randn(batch, hidden)

        def torch_step():
            opt.zero_grad(set_to_none=True)
            out = model_pt(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()

        pt_ms = bench("pt-train", torch_step, warmup=3, iters=10)

        # Hypatia native train
        model_hy = nn.Sequential(
            nn.Linear(hidden, ffn), nn.ReLU(), nn.Linear(ffn, hidden)
        )
        w, b, a = extract_train_data(model_hy)
        x_np = x.float().contiguous().numpy()
        y_np = y.float().contiguous().numpy()

        def hypatia_step():
            native_train_step(x_np, y_np, w, b, a, lr)

        hy_ms = bench("hy-train", hypatia_step, warmup=3, iters=10)

        params = sum(p.numel() for p in model_pt.parameters())
        speedup = pt_ms / hy_ms
        marker = " <--" if speedup >= 1.3 else ""
        params_str = f"{params/1e6:.1f}M"
        print(f"{name+' B='+str(batch):<25} {params_str:>10} {pt_ms:>8.2f}ms {hy_ms:>8.2f}ms {speedup:>7.2f}x{marker}")

    # === Analysis ===
    print(f"\n{'='*70}")
    print(" ANALYSIS")
    print(f"{'='*70}")
    print("""
The bottleneck for large models is NOT dispatch overhead (~50us).
For a 7B model forward pass taking 10+ seconds, eliminating dispatch
saves 0.0005% - completely irrelevant.

For large models, the bottleneck is MEMORY BANDWIDTH:
- LLaMA-7B FFN weights: 4096*11008*4*2 = 344MB per layer
- 32 layers = 11GB just for FFN
- CPU memory bandwidth: ~40 GB/s
- Time to read weights: 11GB / 40GB/s = 275ms per forward pass

Strategies that ACTUALLY help for large models:
1. INT4/INT8 quantization: 2-4x less memory traffic
2. Weight-only quantization (like GPTQ/AWQ): keeps activations in fp16
3. KV-cache optimization: reduces recomputation
4. Model parallelism: split across cores/machines
5. Flash attention: O(n) memory instead of O(n^2)
""")
