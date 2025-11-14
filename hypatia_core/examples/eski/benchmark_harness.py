#!/usr/bin/env python3
"""
Hypatia Benchmark Harness (Priority 1)

Karşılaştırır: Baseline (torch) vs Optimized (Hypatia placeholder)
Modeller:     ResNet-50, Tiny-Transformer, MLP
Metodoloji:   50 Warmup + 200 Measurement Runs
Çıktı:        Benchmark sonuçlarını içeren CSV dosyası ve konsol özeti.

Kullanım:
    pip install torch torchvision numpy pandas
    python benchmark_harness.py --output-csv results/benchmark_run_001.csv
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import csv
import argparse
import os
from datetime import datetime
from typing import Callable, Dict, Any
import pandas as pd  # Özet raporlama için

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
        return self.layers(x.view(x.size(0), -1))

class TinyTransformer(nn.Module):
    """Basit bir Transformer Sınıflandırma Modeli"""
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, num_classes=10, seq_len=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, features] (dummy input için features=embed_dim)
        b, s, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = x + self.pos_embedding[:, :s, :]
        x = self.transformer_encoder(x)
        # Sadece ilk token'ı (CLS token değil, basitlik için) al
        x = x[:, 0, :] 
        return self.fc(x)

def get_model(model_name: str, optimized: bool) -> nn.Module:
    """
    Model örneğini (instance) ve dummy input için shape'i döndürür.
    """
    model = None
    input_shape = () # (batch, channels, H, W) or (batch, seq, features)

    if model_name == "ResNet-50":
        model = models.resnet50(weights=None)
        input_shape = (3, 224, 224)
    elif model_name == "MLP":
        model = MLP(in_features=784, out_features=10)
        input_shape = (1, 28, 28)
    elif model_name == "Tiny-Transformer":
        model = TinyTransformer(embed_dim=128, nhead=4, num_layers=2, seq_len=64)
        input_shape = (64, 128) # (seq_len, features)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    if optimized:
        try:
            # -------------------------------------------------------------
            # <<< HYPATIA OPTIMIZATION HOOK >>>
            # Burası, 'torch.compile' yerine sizin optimizasyon
            # fonksiyonunuzun (örn: hypatia_core.compile(model)) 
            # çağrılacağı yerdir.
            # -------------------------------------------------------------
            print(f"  > [Hypatia Placeholder] 'torch.compile()' kullanılıyor...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  > UYARI: 'torch.compile' başarısız oldu, baseline kullanılıyor. Hata: {e}")
            # Hata olursa baseline ile devam et
            pass

    return model, input_shape

# ============================================================================
# Benchmark Yardımcıları
# ============================================================================

def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı (CUDA > MPS > CPU) seçer."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def measure_vram_usage(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> float:
    """
    Modelin tek bir forward pass için tepe VRAM kullanımını MB cinsinden ölçer.
    Sadece CUDA cihazları için geçerlidir.
    """
    if device.type != "cuda":
        return 0.0  # VRAM ölçümü sadece CUDA için
        
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    # Tek çalıştırma
    model(inputs)
    
    torch.cuda.synchronize(device)
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    return peak_vram_bytes / (1024 * 1024) # MB

# ============================================================================
# Ana Benchmark Fonksiyonu
# ============================================================================

def run_benchmark(
    model_name: str, 
    model_fn: Callable[[], nn.Module], 
    device: torch.device, 
    batch_size: int,
    input_shape: tuple
) -> Dict[str, Any]:
    """
    Belirtilen senaryo için 50 warmup + 200 ölçüm turu çalıştırır.
    """
    WARMUP_RUNS = 50
    MEASURE_RUNS = 200

    try:
        model = model_fn()
    except Exception as e:
        print(f"  > HATA: Model oluşturulamadı: {e}")
        return None
        
    model.to(device).eval()
    
    # Dummy input oluştur
    full_input_shape = (batch_size,) + input_shape
    inputs = torch.randn(full_input_shape, device=device)

    print(f"  > VRAM ölçülüyor (tek çalıştırma)...")
    peak_vram_mb = measure_vram_usage(model, inputs, device)

    print(f"  > Isınma turları (Warmup) çalıştırılıyor ({WARMUP_RUNS} tur)...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print(f"  > Ölçüm turları çalıştırılıyor ({MEASURE_RUNS} tur)...")
    timings_ms = []
    with torch.no_grad():
        for _ in range(MEASURE_RUNS):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(inputs)
                end.record()
                torch.cuda.synchronize(device)
                timings_ms.append(start.elapsed_time(end))
            else:
                # CPU/MPS için time.perf_counter
                start = time.perf_counter()
                _ = model(inputs)
                end = time.perf_counter()
                timings_ms.append((end - start) * 1000)
    
    # İstatistikleri hesapla
    p50_ms = np.percentile(timings_ms, 50)
    p95_ms = np.percentile(timings_ms, 95)
    mean_s = np.mean(timings_ms) / 1000.0  # saniye cinsinden ortalama
    throughput = batch_size / mean_s       # ops/saniye

    # TODO: FLOPs tahmini (flops_est) için bir kütüphane (örn: fvcore)
    # entegre edilebilir. Şimdilik placeholder.
    flops_est = -1.0 

    return {
        "batch": batch_size,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "throughput": throughput,
        "peak_vram_MB": peak_vram_mb,
        "flops_est": flops_est,
    }

# ============================================================================
# Main Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hypatia Benchmark Harness")
    parser.add_argument(
        "-o", "--output-csv", 
        type=str, 
        default=f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Çıktı CSV dosyasının yolu."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Cihazı zorla (örn: 'cpu', 'cuda'). Varsayılan: otomatik seçim."
    )
    args = parser.parse_args()

    # Cihazı ayarla
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 80)
    print("HYPATIA BENCHMARK HARNESS")
    print("=" * 80)
    print(f"Cihaz:      {device}")
    print(f"Çıktı CSV:  {args.output_csv}")
    print("-" * 80)

    # Çıktı dizinini oluştur
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Çalıştırılacak senaryoları tanımla (Model Adı, Batch Size)
    SCENARIOS = {
        "MLP": 128,
        "Tiny-Transformer": 64,
        "ResNet-50": 32,
    }

    all_results = []
    baseline_perfs = {} # Speedup hesaplaması için

    for model_name, batch_size in SCENARIOS.items():
        
        # --- 1. Baseline Senaryosu ---
        print(f"\n[SENARYO] Model: {model_name}-Baseline | Batch: {batch_size}")
        
        try:
            base_model, input_shape = get_model(model_name, optimized=False)
            model_fn = lambda: get_model(model_name, optimized=False)[0] # Sadece modeli al

            res_base = run_benchmark(
                model_name=model_name,
                model_fn=model_fn,
                device=device,
                batch_size=batch_size,
                input_shape=input_shape
            )
            
            if res_base:
                res_base["scenario"] = f"{model_name}-Baseline"
                res_base["device"] = str(device)
                res_base["speedup"] = 1.0
                all_results.append(res_base)
                baseline_perfs[model_name] = res_base["p50_ms"]
            else:
                baseline_perfs[model_name] = np.nan # Hata oldu

        except Exception as e:
            print(f"  > !~ Baseline senaryosu başarısız oldu: {e}")
            baseline_perfs[model_name] = np.nan
            
        
        # --- 2. Optimized (Hypatia) Senaryosu ---
        print(f"\n[SENARYO] Model: {model_name}-Hypatia | Batch: {batch_size}")
        
        try:
            opt_model, opt_input_shape = get_model(model_name, optimized=True)
            model_fn_opt = lambda: get_model(model_name, optimized=True)[0]
            
            res_opt = run_benchmark(
                model_name=model_name,
                model_fn=model_fn_opt,
                device=device,
                batch_size=batch_size,
                input_shape=opt_input_shape
            )
            
            if res_opt:
                res_opt["scenario"] = f"{model_name}-Hypatia"
                res_opt["device"] = str(device)
                
                # Speedup hesapla
                if model_name in baseline_perfs and not np.isnan(baseline_perfs[model_name]):
                    res_opt["speedup"] = baseline_perfs[model_name] / res_opt["p50_ms"]
                else:
                    res_opt["speedup"] = np.nan
                    
                all_results.append(res_opt)

        except Exception as e:
            print(f"  > !~ Optimized senaryosu başarısız oldu: {e}")

    # --- Sonuçları CSV'ye Yaz ---
    if not all_results:
        print("\nUYARI: Hiçbir sonuç üretilmedi. CSV dosyası yazılmayacak.")
        return

    csv_fields = [
        "scenario", "device", "batch", "p50_ms", "p95_ms", 
        "throughput", "peak_vram_MB", "flops_est", "speedup"
    ]
    
    try:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\n✅ Benchmark sonuçları şuraya yazıldı: {args.output_csv}")
    except IOError as e:
        print(f"\n🔥 HATA: CSV dosyası yazılamadı: {e}")

    # --- Özet Raporu Yazdır ---
    print("\n" + "=" * 80)
    print("BENCHMARK ÖZETİ")
    print("=" * 80)
    
    # Raporlama için Pandas DataFrame kullan
    df = pd.DataFrame(all_results)
    df = df[csv_fields] # Sütunları sırala
    
    # Formatlama
    pd.set_option('display.precision', 2)
    pd.set_option('display.float_format', '{:,.2f}'.format)
    pd.set_option('display.width', 1000)
    
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()