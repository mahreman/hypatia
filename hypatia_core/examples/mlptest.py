import torch
import torch.nn as nn
import torch._dynamo as dynamo
import time
import _hypatia_core  # Rust modülünüz

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
        compiled_gm = _hypatia_core.compile_fx_graph(
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

# --- MODEL TANIMI (MLP) ---

class MLP(nn.Module):
    def __init__(self, in_features=256, hidden_dim=512, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- BENCHMARK AYARLARI ---
print("Cihaz ayarlanıyor...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)
# Girdi: (batch_size, in_features)
inputs = [torch.randn(128, 256, device=device)]
print(f"Cihaz: {device}. Model: MLP. Girdiler hazır.")

# Iterasyon sayıları
WARMUP_ITER = 10
BENCH_ITER = 100

# --- BENCHMARK ÇALIŞTIRMA ---
print(f"\n--- torch.compile(backend='hypatia') çağrılıyor (Model: MLP)... ---")
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
    print(f"\nSonuç (MLP): Hypatia Hızlanma: ~%{speedup:.2f}")

except Exception as e:
    print(f"\n--- torch.compile BAŞARISIZ ---")
    print(f"Hata: {e}")