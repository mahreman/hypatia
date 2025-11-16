import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time

# Import hypatia_core - automatically registers the 'hypatia' backend
import hypatia_core


# --- (Geri kalan test kodunuz buradan devam edebilir) ---

# Test Modeli (Dökümandaki gibi)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_a = nn.Parameter(torch.randn(1024, 1024))
        self.p_b = nn.Parameter(torch.randn(1024, 1024))
        self.p_c = nn.Parameter(torch.randn(1024, 1024))

    def forward(self, x):
        # Orijinal "aptal" kod: (x*A)*B + (x*A)*C
        a = torch.matmul(x, self.p_a)
        ab = torch.matmul(a, self.p_b)
        ac = torch.matmul(a, self.p_c)
        return ab + ac


# Model ve Girdiler
print("Cihaz ayarlanıyor...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)
inputs = [torch.randn(1024, 1024, device=device)]

print(f"Cihaz: {device}. Model ve girdiler hazır.")

# 5. torch.compile çağrısı (Artık "hypatia" adını tanımalı)
print("\n--- torch.compile(backend='hypatia') çağrılıyor... ---")
try:
    optimized = torch.compile(model, backend="hypatia")
    print("--- torch.compile başarılı! ---")

    # Benchmark
    print("\n--- Optimize Edilmiş Model (Hypatia) Benchmark ---")
    iters = 100

    with torch.inference_mode():
        # Warmup
        optimized(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            optimized(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_hypatia = (time.time() - t0) / iters
    print(f"Hypatia (Optimize) Ortalama Süre: {t_hypatia * 1000:.4f} ms")

    # Orijinal (Eager) Benchmark
    print("\n--- Orijinal Model (Eager) Benchmark ---")
    with torch.inference_mode():
        model(*inputs)  # Warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_eager = (time.time() - t0) / iters
    print(f"Eager (Orijinal) Ortalama Süre: {t_eager * 1000:.4f} ms")
    
    speedup = (t_eager / t_hypatia - 1) * 100
    print(f"\nSonuç: Hypatia Hızlanma: ~%{speedup:.2f}")

except Exception as e:
    print(f"\n--- torch.compile BAŞARISIZ ---")
    print(f"Hata: {e}")
