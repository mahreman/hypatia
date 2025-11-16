import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time

# Import hypatia_core - automatically registers the 'hypatia' backend
import hypatia_core

# --- MODEL TANIMI (Tiny-Transformer) ---

class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, out_features=10):
        super().__init__()
        self.d_model = d_model
        # TransformerEncoderLayer, batch_first=False varsayılanını kullanır (Seq, Batch, Dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.out_head = nn.Linear(d_model, out_features)

    def forward(self, src):
        # src shape: (SequenceLength, BatchSize, EmbeddingDim)
        x = self.transformer_encoder(src)
        # Ortalama havuzlama (mean pooling) yaparak (BatchSize, EmbeddingDim) elde et
        x = x.mean(dim=0)
        x = self.out_head(x)
        return x

# --- BENCHMARK AYARLARI ---
print("Cihaz ayarlanıyor...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyTransformer().to(device)
# Girdi: (SequenceLength, BatchSize, EmbeddingDim)
inputs = [torch.randn(32, 16, 128, device=device)]
print(f"Cihaz: {device}. Model: TinyTransformer. Girdiler hazır.")

# Iterasyon sayıları
WARMUP_ITER = 10
BENCH_ITER = 50

# --- BENCHMARK ÇALIŞTIRMA ---
print(f"\n--- torch.compile(backend='hypatia') çağrılıyor (Model: TinyTransformer)... ---")
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
    print(f"\nSonuç (TinyTransformer): Hypatia Hızlanma: ~%{speedup:.2f}")

except Exception as e:
    print(f"\n--- torch.compile BAŞARISIZ ---")
    print(f"Hata: {e}")