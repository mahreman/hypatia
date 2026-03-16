"""
Hypatia Demo: Qwen2.5-0.5B Inference + Auto-Tune Benchmark
============================================================

RTX 4070 Laptop GPU (8GB VRAM) uzerinde:
  - Qwen2.5-0.5B (494M params, ~1 GB FP32)
  - Auto-tuner: quick_tune
  - GPU FP32 vs FP16 vs BF16 karsilastirmasi
  - INT4 / INT8 quantization
  - FLOPs profiling + roofline analysis
  - Tum stratejilerin benchmark karsilastirmasi
"""

import sys
import os
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# STEP 1: Model yukleme
# ============================================================================

print("=" * 70)
print("  Hypatia Demo: Qwen2.5-0.5B + Auto-Tune Benchmark")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n[1/7] Qwen2.5-0.5B modeli yukleniyor...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
model_mb = n_params * 4 / 1024**2
print(f"  Parametreler: {n_params/1e6:.1f}M ({model_mb:.0f} MB FP32)")
print(f"  Yukleme suresi: {time.time()-t0:.1f}s")
print(f"  Mimari: {model.config.model_type}")
print(f"  Hidden: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")

# ============================================================================
# STEP 2: Hardware Detection
# ============================================================================

print("\n[2/7] Hardware tespit ediliyor...")
from hypatia_core.profiler import detect_hardware
hw = detect_hardware()
print(hw.summary())

# ============================================================================
# STEP 3: Quick Auto-Tune
# ============================================================================

print("\n[3/7] Quick auto-tune (heuristic)...")
from hypatia_core.autotuner import quick_tune

t0 = time.time()
config = quick_tune(model, (1, 128))
print(f"  Karar suresi: {(time.time()-t0)*1000:.0f}ms")
print(config.summary())

# ============================================================================
# STEP 4: Token Generation Test
# ============================================================================

print("\n[4/7] Token generation testi (CPU)...")
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"  Prompt: '{prompt}'")
print(f"  Token sayisi: {input_ids.shape[1]}")

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Output: '{generated}'")

# ============================================================================
# STEP 5: CPU Benchmark
# ============================================================================

print("\n[5/7] CPU benchmark (batch=1, seq=128)...")

# Pad input to 128 tokens
pad_ids = F.pad(input_ids, (0, 128 - input_ids.shape[1]), value=tokenizer.pad_token_id or 0)

results = {}

# --- Vanilla PyTorch CPU ---
print("  [A] Vanilla PyTorch FP32 (CPU)...")
model.eval()
with torch.no_grad():
    for _ in range(2):
        _ = model(pad_ids)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model(pad_ids)
    cpu_fp32_ms = (time.perf_counter() - t0) / 5 * 1000
results["CPU FP32 (vanilla)"] = cpu_fp32_ms
print(f"      -> {cpu_fp32_ms:.0f} ms")

# --- INT8 Dynamic Quantization ---
print("  [B] INT8 Dynamic Quantized (CPU)...")
try:
    int8_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    int8_model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = int8_model(pad_ids)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = int8_model(pad_ids)
        cpu_int8_ms = (time.perf_counter() - t0) / 5 * 1000
    results["CPU INT8 Dynamic"] = cpu_int8_ms
    print(f"      -> {cpu_int8_ms:.0f} ms ({cpu_fp32_ms/cpu_int8_ms:.2f}x speedup)")
    del int8_model; gc.collect()
except Exception as e:
    print(f"      -> Basarisiz: {e}")

# --- INT4 Block (on MLP sub-module) ---
print("  [C] Hypatia INT4 (tek MLP block)...")
try:
    from hypatia_core.native_model import QuantizedModel
    # Extract one MLP layer as simple Sequential
    layer0 = model.model.layers[0].mlp
    up = layer0.up_proj if hasattr(layer0, 'up_proj') else layer0.gate_proj
    down = layer0.down_proj

    simple_mlp = nn.Sequential(
        nn.Linear(up.in_features, up.out_features, bias=up.bias is not None),
        nn.SiLU(),
        nn.Linear(down.in_features, down.out_features, bias=down.bias is not None),
    )
    simple_mlp[0].weight.data = up.weight.data.clone()
    if up.bias is not None:
        simple_mlp[0].bias.data = up.bias.data.clone()
    simple_mlp[2].weight.data = down.weight.data.clone()
    if down.bias is not None:
        simple_mlp[2].bias.data = down.bias.data.clone()

    qmodel = QuantizedModel(simple_mlp)
    print(f"      Compression: {qmodel.compression_ratio:.1f}x, Saved: {qmodel.memory_saved_mb:.0f} MB")

    dummy = torch.randn(1, up.in_features)
    with torch.no_grad():
        for _ in range(3):
            qmodel(dummy)
        t0 = time.perf_counter()
        for _ in range(10):
            qmodel(dummy)
        q4_ms = (time.perf_counter() - t0) / 10 * 1000

    # Baseline for same block
    simple_mlp.eval()
    with torch.no_grad():
        for _ in range(3):
            simple_mlp(dummy)
        t0 = time.perf_counter()
        for _ in range(10):
            simple_mlp(dummy)
        block_ms = (time.perf_counter() - t0) / 10 * 1000

    results["MLP Block FP32"] = block_ms
    results["MLP Block INT4"] = q4_ms
    print(f"      FP32 block: {block_ms:.2f} ms")
    print(f"      INT4 block: {q4_ms:.2f} ms")
    print(f"      Speedup: {block_ms/q4_ms:.2f}x")
    del qmodel, simple_mlp; gc.collect()
except Exception as e:
    print(f"      -> Basarisiz: {e}")

# ============================================================================
# STEP 6: GPU Benchmark
# ============================================================================

if torch.cuda.is_available():
    print("\n[6/7] GPU benchmark...")
    torch.cuda.empty_cache()

    # --- GPU FP32 ---
    print("  [D] GPU FP32...")
    try:
        gpu_model = model.cuda()
        gpu_ids = pad_ids.cuda()
        with torch.no_grad():
            for _ in range(3):
                _ = gpu_model(gpu_ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = gpu_model(gpu_ids)
            torch.cuda.synchronize()
            gpu_fp32_ms = (time.perf_counter() - t0) / 10 * 1000
        results["GPU FP32"] = gpu_fp32_ms
        print(f"      -> {gpu_fp32_ms:.1f} ms ({cpu_fp32_ms/gpu_fp32_ms:.1f}x vs CPU)")
    except Exception as e:
        print(f"      -> Basarisiz: {e}")
        gpu_model = None

    # --- GPU FP16 ---
    if gpu_model is not None:
        print("  [E] GPU FP16 (Tensor Cores)...")
        try:
            gpu_fp16 = gpu_model.half()
            with torch.no_grad():
                for _ in range(3):
                    _ = gpu_fp16(gpu_ids)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = gpu_fp16(gpu_ids)
                torch.cuda.synchronize()
                gpu_fp16_ms = (time.perf_counter() - t0) / 10 * 1000
            results["GPU FP16"] = gpu_fp16_ms
            print(f"      -> {gpu_fp16_ms:.1f} ms ({gpu_fp32_ms/gpu_fp16_ms:.1f}x vs GPU FP32)")
            del gpu_fp16
        except Exception as e:
            print(f"      -> Basarisiz: {e}")

    # --- GPU BF16 ---
    if gpu_model is not None:
        print("  [F] GPU BF16 (better precision)...")
        try:
            gpu_bf16 = gpu_model.to(torch.bfloat16)
            with torch.no_grad():
                for _ in range(3):
                    _ = gpu_bf16(gpu_ids)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = gpu_bf16(gpu_ids)
                torch.cuda.synchronize()
                gpu_bf16_ms = (time.perf_counter() - t0) / 10 * 1000
            results["GPU BF16"] = gpu_bf16_ms
            print(f"      -> {gpu_bf16_ms:.1f} ms ({gpu_fp32_ms/gpu_bf16_ms:.1f}x vs GPU FP32)")
            del gpu_bf16
        except Exception as e:
            print(f"      -> Basarisiz: {e}")

    # --- GPU FP16 + torch.compile ---
    if gpu_model is not None:
        print("  [G] GPU FP16 + torch.compile (Triton)...")
        try:
            gpu_compiled = torch.compile(gpu_model.half(), mode="max-autotune")
            with torch.no_grad():
                for _ in range(3):
                    _ = gpu_compiled(gpu_ids)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = gpu_compiled(gpu_ids)
                torch.cuda.synchronize()
                gpu_compiled_ms = (time.perf_counter() - t0) / 10 * 1000
            results["GPU FP16+compile"] = gpu_compiled_ms
            print(f"      -> {gpu_compiled_ms:.1f} ms ({gpu_fp32_ms/gpu_compiled_ms:.1f}x vs GPU FP32)")
            del gpu_compiled
        except Exception as e:
            print(f"      -> Basarisiz: {e}")

    # Cleanup
    del gpu_model
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("\n[6/7] GPU benchmark atlandirildi (CUDA yok)")

# ============================================================================
# STEP 7: Final Results
# ============================================================================

print("\n[7/7] SONUC TABLOSU")
print("=" * 70)
print(f"  Model: Qwen2.5-0.5B ({n_params/1e6:.0f}M params)")
print(f"  Hardware: {hw.gpu_name if hw.has_cuda else 'CPU'}")
print(f"  Input: batch=1, seq_len=128")
print(f"  Tahmini FLOPs/inference: {2*n_params*128/1e9:.1f} GFLOPs")
print("-" * 70)
print(f"  {'Strateji':<30} {'Zaman':>12} {'vs CPU FP32':>12}")
print(f"  {'-'*54}")

baseline = results.get("CPU FP32 (vanilla)", 1)
for name, ms in sorted(results.items(), key=lambda x: x[1]):
    speedup = baseline / ms if ms > 0 else 0
    if ms >= 1000:
        time_str = f"{ms/1000:.2f} s"
    else:
        time_str = f"{ms:.1f} ms"
    print(f"  {name:<30} {time_str:>12} {speedup:>10.1f}x")

print("=" * 70)

# Best strategy
best_name = min(results, key=results.get)
best_ms = results[best_name]
print(f"\n  En hizli: {best_name} ({best_ms:.1f} ms)")
print(f"  Speedup vs baseline: {baseline/best_ms:.1f}x")
print(f"\n  Hypatia onerisi: {config.strategy_name}")

# ============================================================================
# STEP 8: Generate HTML Dashboard
# ============================================================================

print("\n[8/8] HTML Dashboard olusturuluyor...")
try:
    from hypatia_core.dashboard import generate_benchmark_dashboard

    dashboard_path = os.path.join(os.path.dirname(__file__), "benchmark_qwen.html")
    generate_benchmark_dashboard(
        model_name="Qwen2.5-0.5B",
        results=results,
        hw=hw,
        model_info={
            "n_params": n_params,
            "arch": model.config.model_type,
            "hidden": model.config.hidden_size,
            "layers": model.config.num_hidden_layers,
            "heads": model.config.num_attention_heads,
            "input_shape": "batch=1, seq_len=128",
            "gflops": 2 * n_params * 128 / 1e9,
        },
        tuner_name=config.strategy_name,
        tuner_details={
            "Mode": config.mode,
            "Quantization": config.quantize or "None",
            "Mixed Precision": config.mixed_precision or "None",
            "Chain Compile": str(config.chain_compile),
            "Fusion Rules": str(config.enable_fusion),
            "Sparse": str(config.enable_sparse),
        },
        generation={"prompt": prompt, "output": generated},
        output_path=dashboard_path,
    )
    print(f"  Dashboard: {dashboard_path}")
except Exception as e:
    print(f"  Dashboard olusturulamadi: {e}")

print("\nDemo tamamlandi!")
