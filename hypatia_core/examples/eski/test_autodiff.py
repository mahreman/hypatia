import hypatia_core as hc
import sys

print("=" * 70)
print("HYPATIA SEMBOLİK OTOMATİK TÜREV TESTİ (FEZ 7)")
print("=" * 70)
print("Hedef: Optimize edilmiş (string) bir grafiği geri ayrıştır (parse)")
print("       ve sembolik türevini (AutoDiff) al.")

try:
    # ============ ADIM 1: İleri Yayılım (Forward Pass) ============
    # Bu, FEZ 6'da PyTorch'tan gelen string'i temsil ediyor
    original_expr = "(add (mul a b) (mul a c))"
    print(f"\n[ADIM 1] Orijinal İfade (PyTorch'tan):")
    print(f"  {original_expr}")

    # ============ ADIM 2: İleri Yayılım Optimizasyonu (v3.0 Motoru) ============
    optimized_expr_str = hc.optimize_ast(original_expr)
    print(f"\n[ADIM 2] Optimize Edilmiş İfade (E-graph'ten):")
    print(f"  {optimized_expr_str}")
    
    expected_optimized = "(mul a (add b c))"
    assert optimized_expr_str == expected_optimized

    # ============ ADIM 3: Ayrıştırma (Parse) (FEZ 7 Motoru) ============
    # String'i tekrar üzerinde işlem yapabileceğimiz Symbol nesnesine çeviriyoruz
    print(f"\n[ADIM 3] String -> Symbol Nesnesine Ayrıştırma...")
    
    f = hc.parse_expr(optimized_expr_str)
    
    print(f"  Başarılı. '{f}' (PySymbol nesnesi)")
    assert isinstance(f, hc.Symbol)

    # ============ ADIM 4: Geri Yayılım (Backward Pass - AutoDiff) ============
    # f = a * (b + c)
    # d/da [f] = 1 * (b + c) + a * 0 = (b + c)
    print(f"\n[ADIM 4] Sembolik Türev (d/da) alınıyor...")
    
    grad_a = f.derivative("a")
    
    print(f"  Ham Gradyan (d/da): {grad_a}")

    # ============ ADIM 5: Geri Yayılım Optimizasyonu ============
    # 'grad_a' şu anda "(add (mul 1 (add b c)) 0)" gibi ham bir formda.
    # Bunu da optimize etmek için E-graph'e göndermeliyiz.
    print(f"\n[ADIM 5] Gradyan Grafiği Optimize Ediliyor...")
    
    grad_a_optimized = hc.optimize_ast(str(grad_a))
    
    print(f"  Optimize Edilmiş Gradyan: {grad_a_optimized}")

    # ============ ADIM 6: Sonuç Doğrulama ============
    expected_grad = "(add b c)"
    assert grad_a_optimized == expected_grad
    
    print("\n" + "✅" * 20)
    print(" BAŞARILI: FEZ 7 TAMAMLANDI!")
    print("Optimize edilmiş bir ifade (String) başarıyla 'Symbol' nesnesine")
    print("geri dönüştürüldü ve sembolik türevi (AutoDiff) alındı.")
    print("✅" * 20)

except hc.HypatiaError as e:
    print(f"\n!!! HATA: {e}", file=sys.stderr)
    sys.exit(1)
except AttributeError as e:
    print(f"\n!!! HATA: '{e}'. Modül doğru derlenmedi mi?", file=sys.stderr)
    sys.exit(1)
except AssertionError:
    print(f"\n!!! TEST BAŞARISIZ OLDU", file=sys.stderr)
    sys.exit(1)