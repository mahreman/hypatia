import torch
import timeit
import sys
import hypatia_core as hc # Sadece HypatiaError'u yakalamak için

print("=" * 70)
print("HYPATIA AI PIPELINE BENCHMARK (FEZ 9)")
print("=" * 70)
print("Hedef: Hypatia'nın v3.0 E-graph optimizasyonunun (Factoring)")
print("       saf PyTorch üzerinde ne kadar hızlanma sağladığını ölçmek.")

# --- CUDA Kontrolü ---
if not torch.cuda.is_available():
    print("\n!!! HATA: CUDA (GPU) DESTEĞİ BULUNAMADI !!!", file=sys.stderr)
    print("Bu performans benchmark'ı, v1.0 hedefimiz olan GPU optimizasyonunu", file=sys.stderr)
    print("ölçmek için tasarlanmıştır. Lütfen CUDA destekli bir", file=sys.stderr)
    print("PyTorch kurulumu ile tekrar deneyin.", file=sys.stderr)
    sys.exit(1)

print(f"\nDonanım: {torch.cuda.get_device_name(0)}")

# ======================================================================
# BÖLÜM 1: BENCHMARK PARAMETRELERİ
# ======================================================================

# Büyük, gerçekçi matris boyutları kullanıyoruz
N_BATCH = 4096      # Batch boyutu
D_IN = 1024         # Girdi boyutu
H_DIM = 512         # Gizli katman
D_OUT = 256         # Çıktı boyutu

# timeit için çalıştırma sayısı
# (Daha hızlı bir test için 50'ye düşürebilirsiniz)
N_ITERATIONS = 100

print(f"Parametreler: Batch={N_BATCH}, Iterasyon={N_ITERATIONS}")
print(f"Senaryo: (X*W1)*W2 + (X*W1)*W3  vs  (X*W1)*(W2+W3)")

# ======================================================================
# BÖLÜM 2: TENSORLERİN HAZIRLANMASI (SETUP)
# ======================================================================

# timeit modülünün kullanacağı 'setup' kodu
# Tüm tensörleri GPU'ya gönderiyoruz (.cuda())
SETUP_CODE = f"""
import torch

N, D_in, H, D_out = {N_BATCH}, {D_IN}, {H_DIM}, {D_OUT}
DTYPE = torch.float32 # Standart eğitim hassasiyeti

# Tensörleri GPU üzerinde oluştur
x = torch.randn(N, D_in, device='cuda', dtype=DTYPE, requires_grad=True)
w1 = torch.randn(D_in, H, device='cuda', dtype=DTYPE, requires_grad=True)
w2 = torch.randn(H, D_out, device='cuda', dtype=DTYPE, requires_grad=True)
w3 = torch.randn(H, D_out, device='cuda', dtype=DTYPE, requires_grad=True)

# Gradyanları temizle (benchmark döngüsü içinde tekrar yapılacak)
def zero_grads():
    x.grad = None
    w1.grad = None
    w2.grad = None
    w3.grad = None

# Başlamadan önce her şeyin GPU'da hazır olduğundan emin ol
torch.cuda.synchronize()
"""

# ======================================================================
# BÖLÜM 3: ÇALIŞTIRILACAK İFADELER
# ======================================================================

# İfade 1: Orijinal (Optimize Edilmemiş) PyTorch Kodu
# (X*W1)*W2 + (X*W1)*W3
# Toplam 3 MatMul (ileri) + ilişkili geri yayılım
STMT_ORIGINAL = """
zero_grads()

# İleri yayılım (3 MatMul)
xw1 = x @ w1
o1 = xw1 @ w2
o2 = xw1 @ w3
y = o1 + o2

# Geri yayılım
y.sum().backward()

# KRİTİK: GPU'nun işi bitirmesini bekle
torch.cuda.synchronize()
"""

# İfade 2: Hypatia'nın Optimize Ettiği Kod
# (X*W1) * (W2+W3)
# Toplam 2 MatMul (ileri) + ilişkili geri yayılım
STMT_OPTIMIZED = """
zero_grads()

# İleri yayılım (2 MatMul + 1 Add)
xw1 = x @ w1
w2w3 = w2 + w3 # 'Add' işlemi MatMul'a göre çok ucuzdur
y = xw1 @ w2w3

# Geri yayılım
y.sum().backward()

# KRİTİK: GPU'nun işi bitirmesini bekle
torch.cuda.synchronize()
"""

# ======================================================================
# BÖLÜM 4: BENCHMARK'I ÇALIŞTIR
# ======================================================================

try:
    print(f"\n[TEST 1] Orijinal (Optimize Edilmemiş) Kod çalıştırılıyor...")
    print(f"({N_ITERATIONS} iterasyon, 3 MatMul/iterasyon)")
    
    time_orig = timeit.timeit(
        stmt=STMT_ORIGINAL,
        setup=SETUP_CODE,
        number=N_ITERATIONS
    )
    avg_orig = (time_orig / N_ITERATIONS) * 1000 # Saniyeden milisaniyeye çevir
    
    print(f"  Toplam Süre: {time_orig:.4f} saniye")
    print(f"  Ortalama:    {avg_orig:.4f} ms / iterasyon")


    print(f"\n[TEST 2] Hypatia (Optimize Edilmiş) Kod çalıştırılıyor...")
    print(f"({N_ITERATIONS} iterasyon, 2 MatMul/iterasyon)")
    
    time_opt = timeit.timeit(
        stmt=STMT_OPTIMIZED,
        setup=SETUP_CODE,
        number=N_ITERATIONS
    )
    avg_opt = (time_opt / N_ITERATIONS) * 1000 # Saniyeden milisaniyeye çevir
    
    print(f"  Toplam Süre: {time_opt:.4f} saniye")
    print(f"  Ortalama:    {avg_opt:.4f} ms / iterasyon")

    # ======================================================================
    # BÖLÜM 5: SONUÇLAR
    # ======================================================================
    
    print("\n" + "=" * 70)
    print(" BENCHMARK SONUÇLARI (FEZ 9)")
    print("=" * 70)
    
    if time_opt < time_orig:
        speedup = (time_orig / time_opt)
        percentage = (1.0 - (time_opt / time_orig)) * 100
        print(f"  🏆 BAŞARILI: Optimize edilmiş kod {speedup:.2f}x daha hızlı!")
        print(f"  Optimize Edilmemiş Ortalama: {avg_orig:.4f} ms")
        print(f"  Optimize Edilmiş Ortalama:   {avg_opt:.4f} ms")
        print(f"  KAZANÇ: %{percentage:.2f} hızlanma")
    else:
        print(f"  ⚠️ BAŞARISIZ: Optimizasyon kodu yavaşlattı.")
        print(f"  Optimize Edilmemiş Ortalama: {avg_orig:.4f} ms")
        print(f"  Optimize Edilmiş Ortalama:   {avg_opt:.4f} ms")

    print("\n" + "✅" * 20)
    print(" BAŞARILI: FEZ 9 TAMAMLANDI!")
    print("v1.0 MVP'nin (FLOPs optimizasyonu) somut etkisi ölçüldü.")
    print("✅" * 20)

except Exception as e:
    print(f"\n!!! BENCHMARK HATASI: {e}", file=sys.stderr)
    sys.exit(1)