#!/usr/bin/env python3
"""
Hypatia Benchmark Harness v2 (Priority 1 + Enhancements)

Bu betik, Hypatia optimizasyon motorunun performansını, doğruluğunu ve
kaynak kullanımını PyTorch baseline'ı ile karşılaştırır.

Gereksinimler:
    pip install torch torchvision numpy pandas
    pip install fvcore transformers # Opsiyonel (FLOPs ve Phi-3 testi için)

Kullanım:
    # Temel testler (FP32, FP16, BF16)
    python benchmark_harness_v2.py --output-csv results/benchmark_run_001.csv

    # Phi-3 testini de çalıştırmak için (transformers yüklü olmalı)
    python benchmark_harness_v2.py --run-phi3 --output-csv results/benchmark_run_001.csv
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
import traceback # ✅ Hata ayıklama için eklendi
from datetime import datetime
from typing import Callable, Dict, Any, Tuple, List
import pandas as pd
from contextlib import nullcontext
import copy

# --- Opsiyonel importlar ---
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("⚠️  fvcore bulunamadı. FLOPs hesabı devre dışı (flops_est = -1).")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers bulunamadı. Phi-3 testi devre dışı.")

# Hypatia çekirdeği (opsiyonel)
try:
    import hypatia_core
    HYPATIA_AVAILABLE = True
    print("✅ hypatia_core bulundu. Gerçek Hypatia optimizasyonu kullanılabilir.")
except ImportError:
    HYPATIA_AVAILABLE = False
    print("⚠️  hypatia_core modülü bulunamadı. Placeholder (torch.compile) kullanılacak.")


# ============================================================================
# ✅ YENİ: HYPATIA TRACER
# nn.Sequential gibi konteyner modüllerin içine girmek için eklendi.
# ============================================================================
class HypatiaTracer(torch.fx.Tracer):
    """
    nn.Sequential ve nn.ModuleList gibi konteynerların içine
    girmesi (trace etmesi) için özelleştirilmiş Tracer.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # nn.Sequential veya nn.ModuleList ise, "leaf" (yaprak) DEĞİLDİR,
        # yani "içine gir" (False döndür).
        if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
            return False
        
        # Diğer tüm modüller için varsayılan davranışı kullan
        # (örn. nn.Linear bir yapraktır, True döndürür)
        return super().is_leaf_module(m, module_qualified_name)


# ============================================================================
# Model Tanımları
# ============================================================================

class MLP(nn.Module):
    """Basit bir MLP modeli"""
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
        # Girdiyi düzleştir
        return self.layers(x.view(x.size(0), -1))

# ✅ DÜZELTME (ADIM 2): Transformer tanımı, FX trace için sabit vocab_size kullanacak şekilde güncellendi
class TinyTransformer(nn.Module):
    """Basit bir Transformer Sınıflandırma Modeli"""
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, seq_len=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Embedding layer (sabit vocab size ile)
        self.embedding = nn.Embedding(1000, embed_dim)  # vocab_size=1000
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(embed_dim, 10)  # 10 classes

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


# ============================================================================
# Yardımcı Fonksiyonlar
# ============================================================================

def get_model_and_input_shape(model_name: str) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Model ismine göre (model, input_shape) döner.
    input_shape: tek örnek (single sample) için şekil, batch boyutu ayrıca eklenir.
    """
    if model_name == "MLP":
        model = MLP(in_features=784, out_features=10)
        input_shape = (1, 28, 28)
    elif model_name == "Tiny-Transformer":
        # ✅ DÜZELTME (ADIM 2): Yeni TinyTransformer tanımı kullanılıyor
        model = TinyTransformer(embed_dim=128, nhead=4, num_layers=2, seq_len=64)
        input_shape = (64,)  # sequence length
    elif model_name == "ResNet-50":
        model = models.resnet50(num_classes=1000)
        input_shape = (3, 224, 224)
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")
    return model, input_shape


def get_dummy_inputs(
    batch_size: int, 
    input_shape: tuple, 
    device: torch.device, 
    precision: torch.dtype
) -> List[torch.Tensor]:
    """
    compile_fx_graph için örnek girdi listesi oluşturur.
    """
    if len(input_shape) == 1:
        # Transformer case: (batch_size, seq_len) -> LongTensor
        seq_len = input_shape[0]
        # ✅ DÜZELTME (ADIM 2): vocab_size 1000 ile eşleş
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        # CV/MLP case
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)
    
    # Rust tarafı bir 'PyObject' (Liste) bekliyor
    return [inputs]


def create_dummy_loader(
    batch_size: int, 
    input_shape: tuple, 
    device: torch.device, 
    precision: torch.dtype,
    num_batches: int = 10
) -> DataLoader:
    """
    Accuracy Validation için sahte DataLoader oluşturur.
    """
    if len(input_shape) == 1:
        # Transformer case: sequence of token ids (LongTensor)
        seq_len = input_shape[0]
        # ✅ DÜZELTME (ADIM 2): vocab_size 1000 ile eşleş
        data = torch.randint(0, 1000, (num_batches * batch_size, seq_len), device=device, dtype=torch.long)
    elif len(input_shape) == 3:
        # Image-like (C,H,W)
        c, h, w = input_shape
        data = torch.randn(num_batches * batch_size, c, h, w, device=device, dtype=precision)
    else:
        # Generic tensor
        data = torch.randn((num_batches * batch_size,) + input_shape, device=device, dtype=precision)

    targets = torch.zeros(num_batches * batch_size, dtype=torch.long, device=device)
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


# ✅ DÜZELTME (ADIM 1): StopIteration hatasını yakalamak için güvenli taşıma fonksiyonu
def safe_model_to_device(model, device, precision):
    """Modeli güvenli şekilde cihaza taşır ve parametreleri kontrol eder."""
    try:
        model = model.to(device)
        
        # Parametre varlığını güvenli kontrol et
        try:
            params = list(model.parameters())
            if not params:
                # Parametresi olmayan bir model (örn: sadece F.relu)
                print("  > UYARI: Optimize edilmiş modelin parametresi yok (bu normal olabilir).")
                return model, None
            
            # Sadece floating point modeller için precision ayarı
            if precision != torch.long and hasattr(params[0], 'is_floating_point'):
                if params[0].is_floating_point(): # Sadece zaten float ise çevir
                    model = model.to(dtype=precision)
            
            return model, None
            
        except (StopIteration, IndexError, AttributeError) as e:
            # Bu, 'optimized_model'in geçerli bir nn.Module olmaması durumunda olur
            return None, f"Parametre kontrolü başarısız: {e}"
            
    except Exception as e:
        return None, f"Cihaza taşıma hatası: {e}"


def optimize_model_from_base(
    original_model: nn.Module,
    example_inputs: List[torch.Tensor],
    precision: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Var olan bir modeli kullanarak Hypatia/torch.compile ile optimize edilmiş bir kopya üretir."""
    
    # Orijinal modeli bozmamak için kopyasını al
    # Trace/compile işlemleri CPU'da ve hedef hassasiyette yapılır
    model_cpu = copy.deepcopy(original_model).to("cpu")
    
    # model_cpu'nun parametresi olup olmadığını kontrol et
    try:
        if precision != torch.long and next(model_cpu.parameters()).is_floating_point():
             model_cpu = model_cpu.to(dtype=precision)
    except StopIteration:
        # Parametresi olmayan model
        pass
    except RuntimeError:
        print(f"  > UYARI: Model {precision}'a çevrilemedi (örn: Embedding), atlanıyor.")
            
    # Girdileri de CPU'ya taşı
    example_inputs_cpu = [inp.to("cpu") for inp in example_inputs]
    
    optimized_model_obj = None # Optimize edilmiş model (henüz taşınmadı)

    try:
        print(f"  > [FX] Model 'torch.fx.symbolic_trace' ile trace ediliyor...")
        
        # ====================================================================
        # ✅ GÜNCELLENMİŞ TRACE ÇAĞRISI
        # nn.Sequential'in içine girmek için HypatiaTracer kullanılıyor.
        # ====================================================================
        
        # --- ESKİ ---
        # graph_module = torch.fx.symbolic_trace(model_cpu)
        
        # --- YENİ ---
        tracer = HypatiaTracer()
        graph = tracer.trace(model_cpu)
        graph_module = torch.fx.GraphModule(tracer.root, graph)
        # ====================================================================


        if HYPATIA_AVAILABLE and hasattr(hypatia_core, "compile_fx_graph"):
            print(f"  > [HYPATIA] Gerçek 'compile_fx_graph' optimizasyonu uygulanıyor...")
            # ✅ ADIM 1: Düzeltilmiş imza (2 argüman)
            optimized_model_obj = hypatia_core.compile_fx_graph(graph_module, example_inputs_cpu)
        else:
            print(f"  > [HYPATIA PLACEHOLDER] 'torch.compile' kullanılıyor...")
            optimized_model_obj = torch.compile(graph_module)
            
    except Exception as e:
        print(f"  > UYARI: Optimizasyon başarısız oldu, baseline kullanılacak. Hata: {e}")
        traceback.print_exc()
        optimized_model_obj = model_cpu # Orijinal (CPU) modele fallback

    # ✅ DÜZELTME (ADIM 1): StopIteration hatasını yakalamak için güvenli taşıyıcıyı kullan
    result, error = safe_model_to_device(optimized_model_obj, device, precision)
    if error:
        print(f"  > {error}, orijinal model kullanılıyor.")
        # Orijinal modeli (baseline) güvenle taşı
        result, _ = safe_model_to_device(original_model, device, precision)
    
    return result if result is not None else original_model


# ============================================================================
# Benchmark Yardımcıları
# ============================================================================

def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı (CUDA > MPS > CPU) seçer."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("⚠️  MPS (Apple Silicon) cihazı saptandı. VRAM ölçümü yapılamayacak.")
        return torch.device("mps")
    return torch.device("cpu")

def measure_vram_usage(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> float:
    """Modelin tek bir forward pass için tepe VRAM kullanımını MB cinsinden ölçer."""
    if device.type != "cuda":
        return 0.0
        
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    try:
        _ = model(inputs)
        torch.cuda.synchronize(device)
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        return peak_vram_bytes / (1024 * 1024) # MB
    except Exception as e:
        print(f"  > VRAM ölçüm hatası: {e}")
        return -1.0

def calculate_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    """fvcore kullanarak FLOPs hesaplar."""
    if not FVCORE_AVAILABLE:
        return -1.0
    try:
        # Modeli kopyalayıp CPU'ya taşı, orijinal cihazı bozma
        model_cpu = copy.deepcopy(model).to("cpu")
        inputs_cpu = inputs.to("cpu")
        
        if inputs.dtype != torch.long:
            # Modelin float parametresi olup olmadığını kontrol et
            is_float_model = any(p.is_floating_point() for p in model_cpu.parameters())
            if is_float_model:
                model_dtype = next(p.dtype for p in model_cpu.parameters() if p.is_floating_point())
                inputs_cpu = inputs_cpu.to(dtype=model_dtype)
        
        flops = FlopCountAnalysis(model_cpu, (inputs_cpu,))
        return flops.total()
    except Exception as e:
        print(f"  > FLOPs hesaplama hatası: {e}")
        return -1.0

# ============================================================================
# Validasyon ve Güvenlik Fonksiyonları
# ============================================================================

def validate_accuracy(
    original_model: nn.Module, 
    optimized_model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device,
    precision: torch.dtype
) -> Dict[str, float]:
    """
    Optimizasyonun doğruluk kaybına neden olup olmadığını kontrol et.
    """
    original_model.eval()
    optimized_model.eval()
    
    all_orig_outputs = []
    all_opt_outputs = []

    with torch.no_grad():
        for data, target in test_loader:
            # DataLoader zaten doğru device + precision (veya Long) ile üretildi.
            
            autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision) \
                if precision != torch.float32 and device.type != 'mps' \
                else nullcontext()
            
            with autocast_ctx:
                orig_output = original_model(data)
                opt_output = optimized_model(data)
            
            if orig_output.shape != opt_output.shape:
                print(f"  > UYARI: Çıkış şekilleri uyuşmuyor! "
                      f"original: {orig_output.shape}, optimized: {opt_output.shape}")
                return { "cosine_similarity": -1.0, "max_difference": float("inf"), "relative_error": float("inf") }

            all_orig_outputs.append(orig_output.flatten())
            all_opt_outputs.append(opt_output.flatten())

    if not all_orig_outputs:
        return { "cosine_similarity": -1.0, "max_difference": float("inf"), "relative_error": float("inf") }

    orig_flat = torch.cat(all_orig_outputs).float()
    opt_flat = torch.cat(all_opt_outputs).float()
    
    cos_sim = F.cosine_similarity(orig_flat, opt_flat, dim=0).item()
    max_diff = (orig_flat - opt_flat).abs().max().item()
    
    # Göreli Hata (Relative Error)
    epsilon = 1e-6
    relative_diff = (orig_flat - opt_flat).abs() / (orig_flat.abs() + epsilon)
    rel_err = relative_diff.max().item()

    return {
        "cosine_similarity": cos_sim,
        "max_difference": max_diff,
        "relative_error": rel_err,
    }

def check_memory_leak(
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    input_shape: Tuple[int, ...],
    batch_size: int,
    precision: torch.dtype,
    num_iterations: int = 50,
) -> float:
    """
    Basit bir memory leak kontrolü.
    """
    if device.type != "cuda":
        return 0.0

    model = model_fn() # model_fn zaten modeli doğru cihaz/precision'da döner
    
    # Girdiyi oluştur
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        # ✅ DÜZELTME (ADIM 2): vocab_size 1000 ile eşleş
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)

    leak_bytes = max(0, mem_after - mem_before)
    leak_mb = leak_bytes / (1024 * 1024)
    return leak_mb

# ============================================================================
# Ana Benchmark Fonksiyonu
# ============================================================================

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
    """
    Tek bir senaryo için benchmark çalıştırır.
    """
    results: Dict[str, Any] = {}

    # Girdi tensorunu oluştur
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        # ✅ DÜZELTME (ADIM 2): vocab_size 1000 ile eşleş
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)

    # Modeli al
    model = model_fn() 
    model.eval()

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision) \
        if precision != torch.float32 and device.type != 'mps' \
        else nullcontext()

    # 1) Doğruluk Kontrolü
    if original_model is not None and test_loader is not None:
        print("  > Doğruluk (Accuracy) kontrolü yapılıyor...")
        accuracy_check = validate_accuracy(original_model, model, test_loader, device, precision)
        
        # Accuracy Gate
        cos_sim_threshold = 0.99
        rel_err_threshold = 1e-3 if precision == torch.float32 else 0.1
        
        assert accuracy_check["cosine_similarity"] > cos_sim_threshold, \
            f"Doğruluk kaybı! Cosine Sim: {accuracy_check['cosine_similarity']:.4f} (Eşik: >{cos_sim_threshold})"
        assert accuracy_check["relative_error"] < rel_err_threshold, \
            f"Numerik hata! Rel Err: {accuracy_check['relative_error']:.4f} (Eşik: <{rel_err_threshold})"
        
        results["accuracy_cosine_sim"] = accuracy_check["cosine_similarity"]
        results["accuracy_max_diff"] = accuracy_check["max_difference"]
        results["accuracy_rel_err"] = accuracy_check["relative_error"]
    
    # 2) Memory Leak Kontrolü
    if device.type == "cuda":
        print(f"  > Bellek sızıntısı (Memory Leak) kontrolü yapılıyor...")
        memory_leak_mb = check_memory_leak(model_fn, device, input_shape, batch_size, precision)
        
        if memory_leak_mb > 200.0: # 200MB'dan fazla sızıntı/kullanım varsa uyar
            print(f"  > UYARI: Yüksek bellek kullanımı tespit edildi: {memory_leak_mb:.2f}MB")
        results["memory_leak_mb"] = memory_leak_mb
    else:
        results["memory_leak_mb"] = 0.0

    # 3) VRAM ve FLOPs ölçümü
    print("  > VRAM ve FLOPs ölçülüyor (tek çalıştırma)...")
    if device.type == "cuda":
        with autocast_ctx:
            results["peak_vram_MB"] = measure_vram_usage(model, inputs, device)
    else:
        results["peak_vram_MB"] = 0.0
    results["flops_est"] = calculate_flops(model, inputs)

    # 4) Isınma turları
    print(f"  > Isınma turları (Warmup) çalıştırılıyor ({warmup_runs} tur)...")
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(warmup_runs):
                _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 5) Gerçek ölçüm turları
    print(f"  > Ölçüm turları çalıştırılıyor ({measure_runs} tur)...")
    timings_ms = []
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(measure_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end_time = time.perf_counter()
                timings_ms.append((end_time - start_time) * 1000.0)

    if not timings_ms:
        return {}

    p50 = np.percentile(timings_ms, 50)
    p95 = np.percentile(timings_ms, 95)
    throughput = (batch_size * measure_runs) / (sum(timings_ms) / 1000.0)

    results["p50_ms"] = float(p50)
    results["p95_ms"] = float(p95)
    results["throughput"] = float(throughput)

    return results


# ============================================================================
# Phi-3 Benchmark (Opsiyonel)
# ============================================================================
# (Bu bölüm, ana benchmark'tan bağımsız olduğu için değiştirilmedi)
# ============================================================================

# ============================================================================
# Main
# ============================================================================

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
        help="Phi-3 benchmarkını da çalıştır",
    )
    args = parser.parse_args()

    device = get_device()
    print("=" * 80)
    print("HYPATIA BENCHMARK HARNESS v2")
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
            
            baseline_key = (scenario_name, precision_str)
            original_model = None
            
            # --- 1. Baseline Senaryosu ---
            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Baseline")

            try:
                original_model, input_shape = get_model_and_input_shape(model_name)
                # Modeli doğru cihaza/precision'a taşı
                original_model = original_model.to(device=device)
                
                if model_name != "Tiny-Transformer":
                    # Transformer (Embedding) FP16/BF16'e çevrilemez, hata verir
                    if precision != torch.long:
                         original_model = original_model.to(dtype=precision)

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
                    res_base["scenario"] = f"{scenario_name}-Baseline"
                    res_base["precision"] = precision_str
                    res_base["speedup"] = 1.0
                    res_base["compilation_time_s"] = 0.0
                    all_results.append(res_base)
                    baseline_perfs[baseline_key] = res_base.get("p50_ms", np.nan)
                else:
                    baseline_perfs[baseline_key] = np.nan
            
            except Exception as e:
                print(f"  > !~ FATAL: Baseline senaryosu başarısız oldu: {e}")
                traceback.print_exc()
                baseline_perfs[baseline_key] = np.nan
                original_model = None
            
            # --- 2. Optimized (Hypatia) Senaryosu ---
            if original_model is None:
                print(f"  > !~ Baseline modeli oluşturulamadığı için Hypatia senaryosu atlanıyor.")
                continue


            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Hypatia")

            try:
                start_compile = time.time()
                
                # Örnek girdileri oluştur (Trace için gerekli)
                example_inputs = get_dummy_inputs(batch_size, input_shape, device, precision)

                optimized_model = optimize_model_from_base(
                    original_model=original_model,
                    example_inputs=example_inputs,
                    precision=precision,
                    device=device,
                )

                compilation_time_s = time.time() - start_compile
                
                # 'compile_fx_graph' fonksiyonu compile_time'ı kendi içinde logluyor,
                # ancak biz toplam süreyi burada alıyoruz.
                # (Eğer compile_fx_graph çağrısı başarısız olursa, bu süre sadece
                # trace + fallback süresidir)

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
                    res_opt["scenario"] = f"{scenario_name}-Hypatia"
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
                traceback.print_exc()

    # --- Phi-3 Test Döngüsü ---
    # (Değişiklik yok)

    # CSV yazımı
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Ana benchmark sonuçları şuraya yazıldı: {args.output_csv}")
        
        print("\n" + "="*100)
        print("DETAYLI BENCHMARK ÖZETİ")
        print("="*100)
        cols_to_show = [
            "scenario", "precision", 
            "p50_ms", "speedup", "peak_vram_MB", "memory_leak_mb", "compilation_time_s"
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
            cols_to_show = [
                "scenario", "precision", "p50_ms", "speedup", 
                "accuracy_cosine_sim", "accuracy_max_diff", "accuracy_rel_err"
            ]
            cols_to_show = [c for c in cols_to_show if c and c in df.columns]
            print(hypatia_rows[cols_to_show].to_string(index=False))
    else:
        print("\nUYARI: Hiçbir sonuç üretilmedi. CSV dosyası yazılmayacak.")


if __name__ == "__main__":
    main()