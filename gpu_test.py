"""
Hypatia GPU/CUDA Benchmark Script
RTX 4070 ile test icin: python gpu_test.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

# Hypatia'yi bul
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hypatia_core"))

print("=" * 60)
print("  HYPATIA GPU BENCHMARK")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available! Running CPU-only tests.")
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Test 1: FusedGeluMLP
# ================================================================
print("-" * 60)
print("TEST 1: FusedGeluMLP (768 -> 3072 -> 768)")
print("-" * 60)

try:
    from hypatia_core.fused_modules import FusedGeluMLP

    mlp = FusedGeluMLP(768, 3072, 768).to(device).eval()
    x = torch.randn(32, 768, device=device)

    # Correctness check
    with torch.no_grad():
        out = mlp(x)
        ref_h = F.gelu(F.linear(x, mlp.fc1.weight, mlp.fc1.bias))
        ref = F.linear(ref_h, mlp.fc2.weight, mlp.fc2.bias)
        err = (ref - out).abs().max().item()
    print(f"  Correctness: max_err = {err:.2e}")

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            mlp(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark: Hypatia FusedGeluMLP
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(1000):
            mlp(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        hypatia_time = time.perf_counter() - t0
    print(f"  Hypatia FusedGeluMLP: {hypatia_time*1000:.1f}ms / 1000 iters ({hypatia_time:.4f}ms/iter)")

    # Benchmark: PyTorch baseline
    fc1 = nn.Linear(768, 3072).to(device).eval()
    fc2 = nn.Linear(3072, 768).to(device).eval()
    with torch.no_grad():
        fc1.weight.copy_(mlp.fc1.weight)
        fc1.bias.copy_(mlp.fc1.bias)
        fc2.weight.copy_(mlp.fc2.weight)
        fc2.bias.copy_(mlp.fc2.bias)

    with torch.no_grad():
        for _ in range(20):
            F.linear(F.gelu(F.linear(x, fc1.weight, fc1.bias)), fc2.weight, fc2.bias)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(1000):
            F.linear(F.gelu(F.linear(x, fc1.weight, fc1.bias)), fc2.weight, fc2.bias)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - t0
    print(f"  PyTorch baseline:    {pytorch_time*1000:.1f}ms / 1000 iters ({pytorch_time:.4f}ms/iter)")
    print(f"  Speedup: {pytorch_time/hypatia_time:.2f}x")
except Exception as e:
    print(f"  ERROR: {e}")

print()


# ================================================================
# Test 2: FusedLayerNorm
# ================================================================
print("-" * 60)
print("TEST 2: FusedLayerNorm (hidden=768)")
print("-" * 60)

try:
    from hypatia_core.fused_modules import FusedLayerNorm

    ln_torch = nn.LayerNorm(768).to(device).eval()
    ln_fused = FusedLayerNorm.from_torch_layernorm(ln_torch).to(device).eval()
    x = torch.randn(32, 768, device=device)

    with torch.no_grad():
        ref = ln_torch(x)
        out = ln_fused(x)
        err = (ref - out).abs().max().item()
    print(f"  Correctness: max_err = {err:.2e}")

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            ln_fused(x)
            ln_torch(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Hypatia
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(5000):
            ln_fused(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        hypatia_time = time.perf_counter() - t0
    print(f"  Hypatia FusedLayerNorm: {hypatia_time*1000:.1f}ms / 5000 iters")

    # PyTorch
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(5000):
            ln_torch(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - t0
    print(f"  PyTorch LayerNorm:     {pytorch_time*1000:.1f}ms / 5000 iters")
    print(f"  Speedup: {pytorch_time/hypatia_time:.2f}x")
except Exception as e:
    print(f"  ERROR: {e}")

print()


# ================================================================
# Test 3: FusedAttention
# ================================================================
print("-" * 60)
print("TEST 3: FusedAttention (hidden=768, heads=12)")
print("-" * 60)

try:
    from hypatia_core.fused_modules import FusedAttention

    attn = FusedAttention(768, 12).to(device).eval()
    x = torch.randn(32, 768, device=device)  # [seq_len, hidden]

    with torch.no_grad():
        out = attn(x)
    print(f"  Output shape: {out.shape} (expected [32, 768])")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            attn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Bench
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(500):
            attn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
    print(f"  FusedAttention: {t*1000:.1f}ms / 500 iters ({t/500*1000:.3f}ms/iter)")
except Exception as e:
    print(f"  ERROR: {e}")

print()


# ================================================================
# Test 4: GpuTransformerModel (GPT-2 style, 6 blocks)
# ================================================================
print("-" * 60)
print("TEST 4: GpuTransformerModel (GPT-2 style, 6 blocks, 768 hidden)")
print("-" * 60)

try:
    from hypatia_core.native_model import GpuTransformerModel, TransformerModel

    # Create mock GPT-2 model
    class Conv1D(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_f, out_f) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_f))
        def forward(self, x):
            return x @ self.weight + self.bias

    class FakeAttn(nn.Module):
        def __init__(self, h=768):
            super().__init__()
            self.c_attn = Conv1D(h, 3 * h)
            self.c_proj = Conv1D(h, h)

    class FakeMLP(nn.Module):
        def __init__(self, h=768):
            super().__init__()
            self.c_fc = Conv1D(h, 4 * h)
            self.c_proj = Conv1D(4 * h, h)

    class FakeBlock(nn.Module):
        def __init__(self, h=768):
            super().__init__()
            self.ln_1 = nn.LayerNorm(h)
            self.attn = FakeAttn(h)
            self.ln_2 = nn.LayerNorm(h)
            self.mlp = FakeMLP(h)

    class FakeGPT2(nn.Module):
        def __init__(self, h=768, n_blocks=6):
            super().__init__()
            self.h = nn.ModuleList([FakeBlock(h) for _ in range(n_blocks)])
            self.ln_f = nn.LayerNorm(h)

    model = FakeGPT2(768, 6)
    gpu_model = GpuTransformerModel(model, n_heads=12, device=device)
    print(f"  Model: {gpu_model}")

    x = torch.randn(1, 32, 768, device=device)

    # Correctness
    with torch.no_grad():
        out = gpu_model(x)
    print(f"  Output shape: {out.shape} (expected [1, 32, 768])")

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            gpu_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark: GpuTransformerModel
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200):
            gpu_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        hypatia_time = time.perf_counter() - t0
    ms_per_iter = hypatia_time / 200 * 1000
    print(f"  Hypatia GpuTransformerModel: {hypatia_time*1000:.0f}ms / 200 iters ({ms_per_iter:.2f}ms/iter)")

    # Benchmark: CPU TransformerModel (if available)
    try:
        cpu_model = TransformerModel(model, n_heads=12)
        x_cpu = torch.randn(1, 32, 768)
        with torch.no_grad():
            for _ in range(3):
                cpu_model(x_cpu)

            t0 = time.perf_counter()
            for _ in range(50):
                cpu_model(x_cpu)
            cpu_time = time.perf_counter() - t0
        cpu_ms = cpu_time / 50 * 1000
        print(f"  CPU TransformerModel:        {cpu_time*1000:.0f}ms / 50 iters ({cpu_ms:.2f}ms/iter)")
        print(f"  GPU vs CPU speedup: {cpu_ms/ms_per_iter:.2f}x")
    except Exception as e:
        print(f"  (CPU TransformerModel: {e})")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print()


# ================================================================
# Test 5: CUDA Extension JIT compilation status
# ================================================================
print("-" * 60)
print("TEST 5: CUDA Extension Status")
print("-" * 60)

try:
    from hypatia_core.fused_modules import _has_ext, _load_cuda_ext

    extensions = [
        ("fused_linear_relu", "fused_linear_relu.cpp", "fused_linear_relu_kernel.cu"),
        ("fused_gelu_mlp", "fused_gelu_mlp.cpp", "fused_gelu_mlp_kernel.cu"),
        ("fused_attention", "fused_attention.cpp", "fused_attention_kernel.cu"),
        ("fused_layernorm", "fused_layernorm.cpp", "fused_layernorm_kernel.cu"),
    ]

    for name, cpp, cu in extensions:
        _load_cuda_ext(name, cpp, cu)
        status = "LOADED" if _has_ext(name) else "fallback (CPU)"
        print(f"  {name}: {status}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("  BENCHMARK COMPLETE")
print("=" * 60)
