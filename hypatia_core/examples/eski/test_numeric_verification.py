import torch
import sys

print("======================================================================")
print("FEZ 9: PURE PYTORCH SAYISAL DOĞRULAMA (SANITY CHECK)")
print("======================================================================")
print("Hypatia'nın 'factor' optimizasyonunun (A*B+A*C -> A*(B+C))")
print("matematiksel geçerliliğini PURE TORCH ile test etme.")
print("NOT: 'torch.double' (float64) hassasiyeti kullanılıyor.")

try:
    # --- BÖLÜM 1: İLERİ YAYILIM (FORWARD PASS) DOĞRULAMASI ---
    print("\n[TEST 1] İleri Yayılım (float64 / torch.double ile)")
    torch.manual_seed(0)
    
    # DÜZELTME: dtype=torch.double eklendi
    A = torch.randn(7, 5, dtype=torch.double)
    B = torch.randn(5, 3, dtype=torch.double)
    C = torch.randn(5, 3, dtype=torch.double)

    # Orijinal, optimize edilmemiş yol
    lhs = A @ B + A @ C
    
    # Hypatia'nın bulduğu optimize edilmiş yol
    rhs = A @ (B + C)
    
    # Hata miktarını görelim
    diff = (lhs - rhs).abs().max()
    print(f"  Orijinal (lhs) vs Optimize (rhs) Max Hata: {diff.item()}")

    # Tolerans ile kontrol (daha hassas)
    is_allclose_forward = torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)
    print(f"  Sayısal Eşdeğerlik (atol=1e-10): {is_allclose_forward}")
    
    assert is_allclose_forward, "İleri yayılım, 'torch.double' ile bile sayısal olarak eşdeğer DEĞİL!"
    print("  ✅ BAŞARILI: İleri yayılım optimizasyonu 'torch.double' ile doğrulanmıştır.")

    # --- BÖLÜM 2: GERİ YAYILIM (GRADYAN) DOĞRULAMASI ---
    # (Bu bölüm zaten sizin önerinizde de torch.double kullanıyordu)
    print("\n[TEST 2] Geri Yayılım (torch.double ile)")
    A_grad = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    B_grad = torch.randn(6, 5, dtype=torch.double, requires_grad=True)
    C_grad = torch.randn(6, 5, dtype=torch.double, requires_grad=True)

    # Orijinal (Optimize Edilmemiş) Gradyanlar
    f1 = (A_grad @ B_grad + A_grad @ C_grad).sum()
    g1 = torch.autograd.grad(f1, (A_grad, B_grad, C_grad))
    
    # Optimize Edilmiş Gradyanlar
    f2 = (A_grad @ (B_grad + C_grad)).sum()
    g2 = torch.autograd.grad(f2, (A_grad, B_grad, C_grad))
    
    is_allclose_backward = all(torch.allclose(x, y, atol=1e-10, rtol=1e-10) for x, y in zip(g1, g2))
    print(f"  grad(f_orig) vs grad(f_opt) Eşdeğerlik (atol=1e-10): {is_allclose_backward}")
    assert is_allclose_backward, "Geri yayılım gradyanları 'torch.double' ile eşdeğer DEĞİL!"
    print("  ✅ BAŞARILI: Geri yayılım optimizasyonu (AutoDiff) 'torch.double' ile doğrulanmıştır.")

    print("\n" + "🏆" * 20)
    print(" NİHAİ KANIT BAŞARILI: Hypatia'nın temel optimizasyon")
    print(" varsayımı (factor-out) hem ileri hem de geri yayılım")
    print(" için 'torch.double' hassasiyetinde %100 doğrulanmıştır.")
    print("🏆" * 20)

except Exception as e:
    print(f"\n!!! HATA: {e}", file=sys.stderr)
    sys.exit(1)