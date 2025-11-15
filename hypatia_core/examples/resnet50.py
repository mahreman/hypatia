import torch
import torch.nn as nn
import torch._dynamo as dynamo
import time
import hypatia_core  # Rust modülünüz
import torchvision.models as models # ResNet-50 için

# --- HYPATIA BACKEND KAYDI ---
print("--- 'hypatia' backend'i PyTorch Dynamo'ya kaydediliyor... ---")

@dynamo.register_backend
def hypatia(gm: torch.fx.GraphModule, example_inputs: list):
    """
    Bu, torch.compile'ın çağıracağı köprü fonksiyondur.
    Rust fonksiyonumuzun beklediği 'module_info_map' argümanını hazırlar.
    """
    print("\n[Hypatia] Backend Wrapper (hypatia) çağrıldı. ModuleInfoMap oluşturuluyor...")
    
    module_info_map = {}
    for name, module in gm.named_modules():
        is_inference = not getattr(module, "training", False)
        has_bias = hasattr(module, "bias") and module.bias is not None
        module_type = type(module).__name__

        module_info_map[name] = {
            "type": module_type,
            "has_bias": has_bias,
            "is_inference": is_inference,
        }
        
    print(f"[Hypatia] {len(module_info_map)} modül bilgisi toplandı.")

    # Hazırlanan argümanlarla asıl Rust fonksiyonunu çağırın
    try:
        compiled_gm = hypatia_core.compile_fx_graph(
            gm, 
            example_inputs, 
            module_info_map
        )
        print("[Hypatia] Rust (compile_fx_graph) başarıyla tamamlandı.")
        return compiled_gm
    except Exception as e:
        print(f"[Hypatia] Rust (compile_fx_graph) Hatası: {e}")
        return gm # Hata durumunda optimize edilmemiş orijinal grafiği geri döndür

print("--- Kayıt başarılı. ---")

# --- MODEL TANIMI (ResNet-50) ---
# (torchvision'dan import edildi)

# --- BENCHMARK AYARLARI ---
print("Cihaz ayarlanıyor...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("ResNet-50 modeli yükleniyor (weights=None)...")
# Performans testi için eğitilmiş ağırlıklara gerek yok (weights=None)
model = models.resnet50(weights=None).to(device)

# Girdi: (batch_size, channels, height, width)
batch_size = 16
inputs = [torch.randn(batch_size, 3, 224, 224, device=device)]
print(f"Cihaz: {device}. Model: ResNet-50. Girdiler hazır (Batch={batch_size}).")

# Iterasyon sayıları (ResNet-50 ağırdır)
WARMUP_ITER = 5
BENCH_ITER = 20

# --- BENCHMARK ÇALIŞTIRMA ---
print(f"\n--- torch.compile(backend='hypatia') çağrılıyor (Model: ResNet-50)... ---")
try:
    optimized = torch.compile(model, backend="hypatia")
    print("--- torch.compile başarılı! ---")

    # Benchmark
    print(f"\n--- Optimize Edilmiş Model (Hypatia) Benchmark ({BENCH_ITER} iterasyon) ---")
    with torch.inference_mode():
        # Warmup
        for _ in range(WARMUP_ITER):
            optimized(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.time()
        for _ in range(BENCH_ITER):
            optimized(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_hypatia = (time.time() - t0) / BENCH_ITER
    print(f"Hypatia (Optimize) Ortalama Süre: {t_hypatia * 1000:.4f} ms")

    # Orijinal (Eager) Benchmark
    print(f"\n--- Orijinal Model (Eager) Benchmark ({BENCH_ITER} iterasyon) ---")
    with torch.inference_mode():
        model(*inputs)  # Warmup
        if device == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.time()
        for _ in range(BENCH_ITER):
            model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_eager = (time.time() - t0) / BENCH_ITER
    print(f"Eager (Orijinal) Ortalama Süre: {t_eager * 1000:.4f} ms")
    
    speedup = (t_eager / t_hypatia - 1) * 100
    print(f"\nSonuç (ResNet-50): Hypatia Hızlanma: ~%{speedup:.2f}")

except Exception as e:
    print(f"\n--- torch.compile BAŞARISIZ ---")
    print(f"Hata: {e}")