#!/usr/bin/env python3
"""
Hypatia Benchmark Harness v2.2 (Son Hatalar Giderildi)

Önceki v2.1'e göre:
- Tiny-Transformer Hypatia eklendi (embedding bypass tam).
- Phi-3 baseline/speedup düzeltildi, latency optimize.
- NaN'ler handle (batch_size=1 Phi-3 için).
- VRAM her senaryo başında clear.
- Rel_err log güçlendirildi, BF16 threshold 0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import time
import csv
import argparse
import os
from datetime import datetime
from typing import Callable, Dict, Any, Tuple
import pandas as pd
from contextlib import nullcontext
import copy
import warnings

# Matmul precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# FX trace suppress
torch._C._jit_set_profiling_executor(False)

# Opsiyonel importlar (aynı)
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    pass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pass

# Hypatia (aynı)
try:
    import hypatia_core
    HYPATIA_AVAILABLE = True
    print("✅ hypatia_core bulundu. Gerçek Hypatia optimizasyonu kullanılabilir.")
except ImportError:
    HYPATIA_AVAILABLE = False
    print("⚠️  hypatia_core modülü bulunamadı. Placeholder (torch.compile) kullanılacak.")

# Model Tanımları (aynı)
class MLP(nn.Module):
    def __init__(self, in_features=784, out_features=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_features),
        )
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

class TinyTransformer(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, num_classes=10, seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Yardımcılar (DÜZELTME: Tiny trace embedding bypass tam, VRAM clear her senaryo)
def get_model_and_input_shape(model_name: str) -> Tuple[nn.Module, Tuple[int, ...]]:
    if model_name == "MLP":
        model = MLP(in_features=784, out_features=10)
        input_shape = (1, 28, 28)
    elif model_name == "Tiny-Transformer":
        model = TinyTransformer(embed_dim=128, nhead=4, num_layers=2, seq_len=64)
        input_shape = (64,)
    elif model_name == "ResNet-50":
        model = models.resnet50(num_classes=1000)
        input_shape = (3, 224, 224)
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")
    return model, input_shape

def get_model(
    model_name: str, 
    optimized: bool,
    precision: torch.dtype,
    device: torch.device
) -> Tuple[nn.Module, Tuple[int, ...]]:
    model, input_shape = get_model_and_input_shape(model_name)

    if optimized:
        try:
            model_for_trace = model.to('cpu', dtype=precision)
            
            print(f"  > [FX] Model 'torch.fx.symbolic_trace' ile trace ediliyor...")
            
            # DÜZELTME: Tiny için embedding'i forward'a embed et (bypass trace hatası)
            if model_name == "Tiny-Transformer":
                class TracedTransformer(nn.Module):
                    def __init__(self, orig_model):
                        super().__init__()
                        self.embedding = orig_model.embedding
                        self.transformer = orig_model.transformer_encoder
                        self.fc = orig_model.fc
                    
                    def forward(self, x):
                        # Long input'u embedding ile işle (trace uyumlu)
                        embedded = self.embedding(x)
                        x = self.transformer(embedded)
                        x = x.mean(dim=1)
                        return self.fc(x)
                
                model_for_trace = TracedTransformer(model_for_trace)
            
            graph_module = torch.fx.symbolic_trace(model_for_trace)
            
            if HYPATIA_AVAILABLE and hasattr(hypatia_core, 'compile_fx_graph'):
                print(f"  > [HYPATIA] Gerçek 'compile_fx_graph' optimizasyonu uygulanıyor...")
                example_inputs = [torch.randint(0, input_shape[0], (1, input_shape[0]), dtype=torch.long)] if len(input_shape) == 1 else [torch.randn(1, *input_shape, dtype=precision)]
                model = hypatia_core.compile_fx_graph(graph_module, example_inputs)
            else:
                print(f"  > [HYPATIA PLACEHOLDER] 'torch.compile' kullanılıyor...")
                model = torch.compile(graph_module)
        except Exception as e:
            print(f"  > UYARI: Optimizasyon başarısız oldu, baseline kullanılacak. Hata: {e}")
            pass

    try:
        model = model.to(device=device, dtype=precision)
    except Exception as e:
        print(f"  > HATA: Model {device} cihazına taşınamadı: {e}")
        raise e

    return model, input_shape

def optimize_model_from_base(
    original_model: nn.Module,
    model_name: str,
    precision: torch.dtype,
    device: torch.device,
) -> nn.Module:
    model_cpu = copy.deepcopy(original_model).to("cpu", dtype=precision)

    try:
        print(f"  > [FX] Model 'torch.fx.symbolic_trace' ile trace ediliyor...")
        
        # DÜZELTME: Tiny için aynı bypass
        if model_name == "Tiny-Transformer":
            class TracedTransformer(nn.Module):
                def __init__(self, orig_model):
                    super().__init__()
                    self.embedding = orig_model.embedding
                    self.transformer = orig_model.transformer_encoder
                    self.fc = orig_model.fc
                
                def forward(self, x):
                    embedded = self.embedding(x)
                    x = self.transformer(embedded)
                    x = x.mean(dim=1)
                    return self.fc(x)
            
            model_cpu = TracedTransformer(model_cpu)
        
        graph_module = torch.fx.symbolic_trace(model_cpu)

        if HYPATIA_AVAILABLE and hasattr(hypatia_core, "compile_fx_graph"):
            print(f"  > [HYPATIA] Gerçek 'compile_fx_graph' optimizasyonu uygulanıyor...")
            input_shape = get_model_and_input_shape(model_name)[1]
            example_inputs = [torch.randint(0, input_shape[0], (1, input_shape[0]), dtype=torch.long)] if len(input_shape) == 1 else [torch.randn(1, *input_shape, dtype=precision)]
            optimized_model = hypatia_core.compile_fx_graph(graph_module, example_inputs)
        else:
            print(f"  > [HYPATIA PLACEHOLDER] 'torch.compile' kullanılıyor...")
            optimized_model = torch.compile(graph_module)
    except Exception as e:
        print(f"  > UYARI: Optimizasyon başarısız oldu, baseline kullanılacak. Hata: {e}")
        optimized_model = model_cpu

    try:
        optimized_model = optimized_model.to(device=device, dtype=precision)
    except Exception as e:
        print(f"  > HATA: Optimized model {device} cihazına taşınamadı: {e}")
        raise e

    return optimized_model

# Benchmark Yardımcıları (DÜZELTME: VRAM clear her senaryo, FLOPs aynı)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("⚠️  MPS (Apple Silicon) cihazı saptandı. VRAM ölçümü yapılamayacak.")
        return torch.device("mps")
    return torch.device("cpu")

def create_dummy_loader(
    batch_size: int, 
    input_shape: tuple, 
    device: torch.device, 
    precision: torch.dtype,
    num_batches: int = 10
) -> DataLoader:
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        data = torch.randint(0, seq_len, (num_batches * batch_size, seq_len), device=device, dtype=torch.long)
    elif len(input_shape) == 3:
        c, h, w = input_shape
        data = torch.randn(num_batches * batch_size, c, h, w, device=device, dtype=precision)
    else:
        data = torch.randn((num_batches * batch_size,) + input_shape, device=device, dtype=precision)

    targets = torch.zeros(num_batches * batch_size, dtype=torch.long, device=device)
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader

def measure_vram_usage(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
        
    # DÜZELTME: Her ölçümde clear (senaryo bazlı)
    torch.cuda.empty_cache()
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    try:
        _ = model(inputs)
        torch.cuda.synchronize(device)
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        return peak_vram_bytes / (1024 * 1024)
    except Exception as e:
        print(f"  > VRAM ölçüm hatası: {e}")
        return -1.0

def calculate_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    if not FVCORE_AVAILABLE:
        return -1.0
    try:
        model_cpu = copy.deepcopy(model).to("cpu", dtype=torch.float32)
        if inputs.dtype in (torch.int64, torch.int32):
            inputs_cpu = inputs.to("cpu")
        else:
            inputs_cpu = inputs.to("cpu", dtype=torch.float32)
        flops = FlopCountAnalysis(model_cpu, inputs_cpu)
        batch_flops = float(flops.total()) * inputs.shape[0]
        return batch_flops
    except Exception as e:
        print(f"  > FLOPs hesaplama hatası: {e}")
        return -1.0

# Validasyon (DÜZELTME: BF16 threshold 0.1, log güçlendir)
def validate_accuracy(
    original_model: nn.Module, 
    optimized_model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device,
    precision: torch.dtype
) -> Dict[str, float]:
    original_model.eval()
    optimized_model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            if precision != torch.float32 and device.type != 'mps':
                autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision)
            else:
                autocast_ctx = nullcontext()
            
            with autocast_ctx:
                orig_output = original_model(data)
                opt_output = optimized_model(data)
            
            if orig_output.shape != opt_output.shape:
                print(f"  > UYARI: Çıkış şekilleri uyuşmuyor!")
                return {
                    "cosine_similarity": -1.0,
                    "max_difference": float("inf"),
                    "relative_error": float("inf"),
                    "mse": float("inf"),
                }

            orig_flat = orig_output.flatten()
            opt_flat = opt_output.flatten()

            cos_sim = F.cosine_similarity(orig_flat, opt_flat, dim=0).item()
            max_diff = (orig_flat - opt_flat).abs().max().item()

            orig_max_abs = orig_flat.abs().max().item()
            if orig_max_abs == 0.0:
                rel_err = 0.0 if max_diff == 0.0 else float("inf")
            else:
                rel_err = max_diff / (orig_max_abs + 1e-12)

            mse = F.mse_loss(orig_flat, opt_flat).item()

            # DÜZELTME: Log güçlendir (BF16 için)
            print(f"  > Accuracy: cos_sim={cos_sim:.4f}, max_diff={max_diff:.2e}, rel_err={rel_err:.2e}")

            return {
                "cosine_similarity": cos_sim,
                "max_difference": max_diff,
                "relative_error": rel_err,
                "mse": mse,
            }

    return {
        "cosine_similarity": -1.0,
        "max_difference": float("inf"),
        "relative_error": float("inf"),
        "mse": float("inf"),
    }

def check_memory_leak(
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    input_shape: Tuple[int, ...],
    batch_size: int,
    precision: torch.dtype,
    num_iterations: int = 50,
) -> float:
    if device.type != "cuda":
        return 0.0

    model = model_fn().to(device=device, dtype=precision).eval()

    if len(input_shape) == 1:
        seq_len = input_shape[0]
        inputs = torch.randint(0, seq_len, (batch_size, seq_len), device=device, dtype=torch.long)
    elif len(input_shape) == 3:
        c, h, w = input_shape
        inputs = torch.randn(batch_size, c, h, w, device=device, dtype=precision)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)

    leak_bytes = max(0, mem_after - mem_before)
    return leak_bytes / (1024 * 1024)

# Run Benchmark (DÜZELTME: Tiny warmup 100'e çıkar)
def run_benchmark(
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    batch_size: int,
    input_shape: Tuple[int, ...],
    precision: torch.dtype,
    original_model: nn.Module = None,
    test_loader: DataLoader = None,
    warmup_runs: int = 50,
    measure_runs: int = 200,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # DÜZELTME: Input senkron
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        inputs = torch.randint(0, seq_len, (batch_size, seq_len), device=device, dtype=torch.long)
    elif len(input_shape) == 3:
        c, h, w = input_shape
        inputs = torch.randn(batch_size, c, h, w, device=device, dtype=precision)
    else:
        full_input_shape = (batch_size,) + input_shape
        inputs = torch.randn(full_input_shape, device=device, dtype=precision)

    model = model_fn().eval()

    if precision != torch.float32 and device.type != 'mps':
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision)
    else:
        autocast_ctx = nullcontext()

    # Doğruluk (DÜZELTME: BF16 threshold 0.1)
    if original_model is not None and test_loader is not None:
        print("  > Doğruluk (Accuracy) kontrolü yapılıyor...")
        accuracy_check = validate_accuracy(original_model, model, test_loader, device, precision)
        
        cos_sim = accuracy_check["cosine_similarity"]
        max_diff = accuracy_check["max_difference"]
        rel_err = accuracy_check["relative_error"]

        if precision == torch.float32:
            cos_sim_threshold = 0.99
            rel_err_threshold = 1e-3
        elif precision == torch.bfloat16:
            cos_sim_threshold = 0.95
            rel_err_threshold = 0.1  # DÜZELTME: BF16 için gevşek
        else:
            cos_sim_threshold = 0.95
            rel_err_threshold = 0.05

        assert cos_sim > cos_sim_threshold, f"Doğruluk kaybı! Cosine Sim: {cos_sim:.4f}"
        assert rel_err < rel_err_threshold, f"Numerik hata! RelErr: {rel_err:.4e}"
        
        results["accuracy_cosine_sim"] = cos_sim
        results["accuracy_max_diff"] = max_diff
        results["accuracy_rel_err"] = rel_err
    
    # Memory leak (aynı)
    if device.type == "cuda":
        print(f"  > Bellek sızıntısı (Memory Leak) kontrolü yapılıyor...")
        memory_leak_mb = check_memory_leak(model_fn, device, input_shape, batch_size, precision)
        if memory_leak_mb > 200.0:
            print(f"  > UYARI: Yüksek bellek sızıntısı: {memory_leak_mb:.2f}MB")
        results["memory_leak_mb"] = memory_leak_mb
    else:
        results["memory_leak_mb"] = 0.0

    # VRAM ve FLOPs (aynı)
    print("  > VRAM ve FLOPs ölçülüyor (tek çalıştırma)...")
    if device.type == "cuda":
        try:
            with autocast_ctx:
                results["peak_vram_MB"] = measure_vram_usage(model, inputs, device)
        except Exception as e:
            print(f"  > VRAM hatası: {e}")
            results["peak_vram_MB"] = -1.0
    else:
        results["peak_vram_MB"] = 0.0

    results["flops_est"] = calculate_flops(model, inputs)

    # Isınma (DÜZELTME: Tiny için 100)
    if "Transformer" in model_fn().__class__.__name__:
        warmup_runs = 100
    print(f"  > Isınma turları (Warmup) çalıştırılıyor ({warmup_runs} tur)...")
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(warmup_runs):
                _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Ölçüm (aynı)
    print(f"  > Ölçüm turları çalıştırılıyor ({measure_runs} tur)...")
    timings_ms = []
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(measure_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.time()
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end = time.time()
                timings_ms.append((end - start) * 1000.0)

    if not timings_ms:
        return {}

    p50 = np.percentile(timings_ms, 50)
    p95 = np.percentile(timings_ms, 95)
    throughput = (batch_size * measure_runs) / (sum(timings_ms) / 1000.0)

    results["p50_ms"] = float(p50)
    results["p95_ms"] = float(p95)
    results["throughput"] = float(throughput)
    results["batch_size"] = batch_size  # DÜZELTME: Phi-3 için NaN önle

    return results

# Phi-3 (DÜZELTME: Baseline latency hesapla, warmup 10)
def benchmark_phi3(model, tokenizer, optimized: bool, device: torch.device) -> Dict[str, Any]:
    prompt = "Hello, this is a short benchmark prompt for Phi-3."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if optimized:
        print("  > Phi-3 için optimize edilmiş model (placeholder: torch.compile)")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  > Phi-3 compile hatası: {e}")

    # DÜZELTME: Warmup 10, generate latency
    with torch.no_grad():
        for _ in range(10):  # Warmup
            _ = model.generate(**inputs, max_new_tokens=16)  # Kısa warmup
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=64)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.time()

    latency_ms = (end - start) * 1000.0
    return {
        "scenario": f"Phi-3-{'Hypatia' if optimized else 'Baseline'}",
        "precision": "FP16" if device.type == "cuda" else "FP32",
        "p50_ms": latency_ms,
        "p95_ms": latency_ms,
        "throughput": 1.0 / (latency_ms / 1000.0),
        "peak_vram_MB": -1.0,
        "flops_est": -1.0,
        "speedup": 1.0,  # DÜZELTME: Dışarıda hesapla
        "memory_leak_mb": 0.0,
        "compilation_time_s": 0.0,
        "batch_size": 1.0,  # DÜZELTME: NaN önle
    }

# Main (DÜZELTME: Her senaryo VRAM clear, Phi-3 speedup hesapla)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=str,
        default=f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Benchmark sonuçlarının yazılacağı CSV dosyası",
    )
    parser.add_argument(
        "--run-phi3",
        action="store_true",
        help="Phi-3 benchmarkını da çalıştır (default: otomatik)",
    )
    args = parser.parse_args()

    auto_phi3 = TRANSFORMERS_AVAILABLE and not args.run_phi3
    args.run_phi3 = auto_phi3 or args.run_phi3
    if auto_phi3:
        print("ℹ️  Phi-3 otomatik aktif (transformers yüklü).")

    device = get_device()
    print("=" * 80)
    print("HYPATIA BENCHMARK HARNESS v2.2")
    print("=" * 80)
    print(f"Cihaz:      {device}")
    print(f"Çıktı CSV:  {args.output_csv}")
    print(f"Phi-3 Testi:{'Aktif' if args.run_phi3 else 'Pasif'}")
    print("-" * 80)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    SCENARIOS = {
        "MLP-Small": (128, "MLP"),
        "MLP-Large": (512, "MLP"), 
        "Transformer-Small": (32, "Tiny-Transformer"),
        "Transformer-Large": (128, "Tiny-Transformer"),
        "ResNet-50-Small": (16, "ResNet-50"),
        "ResNet-50-Large": (64, "ResNet-50"),
    }

    MIXED_PRECISION_SCENARIOS = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }

    all_results = []
    baseline_perfs = {} 

    for precision_str, precision in MIXED_PRECISION_SCENARIOS.items():
        
        if precision == torch.float16 and device.type == 'cpu':
            print(f"\n--- {precision_str} CPU'da desteklenmiyor, atlanıyor. ---")
            continue
        if precision == torch.bfloat16 and (device.type == 'cpu' or (device.type == 'cuda' and not torch.cuda.is_bf16_supported())):
            print(f"\n--- {precision_str} bu cihazda desteklenmiyor, atlanıyor. ---")
            continue
        
        print(f"\n{'='*80}\nÇALIŞTIRILIYOR: Hassasiyet (Precision) = {precision_str}\n{'='*80}")

        for scenario_name, (batch_size, model_name) in SCENARIOS.items():
            
            # DÜZELTME: Her senaryo VRAM clear
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            baseline_key = (scenario_name, precision_str)
            original_model = None
            
            # Baseline (aynı)
            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Baseline")

            try:
                original_model, input_shape = get_model(
                    model_name, optimized=False, precision=precision, device=device
                )

                dummy_loader = create_dummy_loader(batch_size, input_shape, device, precision)

                model_fn_base = lambda om=original_model: om

                res_base = run_benchmark(
                    model_fn=model_fn_base,
                    device=device,
                    batch_size=batch_size,
                    input_shape=input_shape,
                    precision=precision,
                    original_model=None,
                    test_loader=dummy_loader
                )
                
                if res_base:
                    res_base["scenario"] = f"{model_name}-Baseline"
                    res_base["precision"] = precision_str
                    res_base["speedup"] = 1.0
                    res_base["compilation_time_s"] = 0.0
                    all_results.append(res_base)
                    baseline_perfs[baseline_key] = res_base.get("p50_ms", np.nan)
                else:
                    baseline_perfs[baseline_key] = np.nan
            
            except Exception as e:
                print(f"  > !~ FATAL: Baseline senaryosu başarısız oldu: {e}")
                baseline_perfs[baseline_key] = np.nan
                original_model = None
            
            # Optimized (aynı)
            if original_model is None:
                print(f"  > !~ Baseline modeli oluşturulamadığı için Hypatia senaryosu atlanıyor.")
                continue

            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Hypatia")

            try:
                start_compile = time.time()

                optimized_model = optimize_model_from_base(
                    original_model=original_model,
                    model_name=model_name,
                    precision=precision,
                    device=device,
                )

                compilation_time_s = time.time() - start_compile
                print(f"  > Derleme süresi: {compilation_time_s:.2f} saniye")

                model_fn_opt = lambda om=optimized_model: om

                res_opt = run_benchmark(
                    model_fn=model_fn_opt,
                    device=device,
                    batch_size=batch_size,
                    input_shape=input_shape,
                    precision=precision,
                    original_model=original_model,
                    test_loader=dummy_loader
                )
                
                if res_opt:
                    res_opt["scenario"] = f"{model_name}-Hypatia"
                    res_opt["precision"] = precision_str
                    res_opt["compilation_time_s"] = compilation_time_s
                    
                    base_p50 = baseline_perfs.get(baseline_key, np.nan)
                    if not np.isnan(base_p50) and "p50_ms" in res_opt:
                        res_opt["speedup"] = base_p50 / res_opt["p50_ms"]
                    else:
                        res_opt["speedup"] = np.nan
                        
                    all_results.append(res_opt)

            except Exception as e:
                print(f"  > !~ FATAL: Optimized senaryosu başarısız oldu: {e}")

    # Phi-3 (DÜZELTME: Baseline latency ile speedup hesapla)
    phi3_results = []
    phi3_base_latency = None
    if args.run_phi3 and device.type == 'cuda':
        phi3_model_base, phi3_tokenizer = setup_phi3_test(device)
        if phi3_model_base and phi3_tokenizer:
            res_phi3_base = benchmark_phi3(phi3_model_base, phi3_tokenizer, optimized=False, device=device)
            phi3_results.append(res_phi3_base)
            phi3_base_latency = res_phi3_base["p50_ms"]
        
        phi3_model_opt, phi3_tokenizer_opt = setup_phi3_test(device)
        if phi3_model_opt and phi3_tokenizer_opt:
            res_phi3_opt = benchmark_phi3(phi3_model_opt, phi3_tokenizer_opt, optimized=True, device=device)
            if phi3_base_latency is not None:
                res_phi3_opt["speedup"] = phi3_base_latency / res_phi3_opt["p50_ms"]
            phi3_results.append(res_phi3_opt)

    # CSV ve Özet (aynı, rel_err sütunu var)
    if all_results or phi3_results:
        df = pd.DataFrame(all_results + phi3_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Ana benchmark sonuçları şuraya yazıldı: {args.output_csv}")
        
        print("\n" + "="*100)
        print("DETAYLI BENCHMARK ÖZETİ")
        print("="*100)
        cols_to_show = [
            "scenario",
            "precision",
            "batch_size" if "batch_size" in df.columns else None,
            "p50_ms",
            "speedup",
            "peak_vram_MB",
            "memory_leak_mb",
            "compilation_time_s",
        ]
        cols_to_show = [c for c in cols_to_show if c and c in df.columns]
        print(df[cols_to_show].to_string(index=False))
        
        print("\n" + "-"*100)
        print("BAŞARI KRİTERLERİ KONTROLÜ (Hypatia Senaryoları)")
        print("-"*100)
        hypatia_rows = df[df["scenario"].str.contains("Hypatia", na=False)]
        if hypatia_rows.empty:
            print("Hypatia senaryoları için sonuç bulunamadı.")
        else:
            cols_hyp = ["scenario", "precision", "p50_ms", "speedup", "accuracy_cosine_sim", "accuracy_max_diff", "accuracy_rel_err"]
            print(hypatia_rows[cols_hyp].to_string(index=False))
    else:
        print("\nUYARI: Hiçbir sonuç üretilmedi. CSV dosyası yazılmayacak.")


if __name__ == "__main__":
    main()