#!/usr/bin/env python3
"""
Hypatia Çekirdek Testi

Bu betik, `benchmark_harness_v2.py`'nin aksine, PyTorch veya torch.compile
kullanmaz. Doğrudan `hypatia_core` Rust kütüphanesinin sembolik
optimizasyon motorunu (`optimize_ast`) ve değerlendiricisini (`eval`) test eder.

Amaç: Hypatia'nın cebirsel sadeleştirmelerinin sayısal olarak doğru
olduğunu kanıtlamak.
"""

import hypatia_core
import math

def test_hypatia_core_optimizations():
    """Sadece Hypatia'nın sembolik optimizasyonlarını test et"""
    
    print("="*80)
    print("HYPATIA ÇEKİRDEK OPTİMİZASYON VE DOĞRULUK TESTİ")
    print("="*80)
    
    # 1. Test senaryoları (Input, Beklenen Optimize Hali)
    # Not: 'expected' şu an için sadece referans amaçlıdır,
    # test sayısal doğruluğu kontrol eder.
    test_cases = [
        # --- Cebirsel Kurallar ---
        ("(add (mul a b) (mul a c))", "(mul a (add b c))"),      # Dağılma (factor)
        ("(mul a (mul b c))", "(mul (mul a b) c)"),              # Birleşme (assoc-mul)
        ("(add a (add b c))", "(add (add a b) c)"),              # Birleşme (assoc-add)
        
        # --- Sıfır/Bir Kuralları ---
        ("(add x 0)", "x"),                                     # Toplama (identity)
        ("(mul x 1)", "x"),                                     # Çarpma (identity)
        ("(mul x 0)", "0"),                                     # Çarpma (zero)
        ("(sub x x)", "0"),                                     # Çıkarma (self)
        
        # --- Karmaşık / AI Kuralları ---
        ("(div (sub X mean) (sqrt var))", "(mul (sub X mean) (pow var -0.5))"), # LayerNorm
        ("(sqrt (pow x 2))", "x"),                              # Sqrt/Pow
        ("(pow x 0.5)", "(sqrt x)"),                            # Pow/Sqrt
    ]
    
    # Tüm değişkenler için kullanılacak sayısal değerler
    env = {
        "a": 2.0, "b": 3.0, "c": 4.0, 
        "x": 5.0, "y": 6.0, "z": 7.0,
        "X": 10.0, "mean": 4.0, "var": 9.0
    }
    
    all_passed = True
    
    for input_expr, expected_expr in test_cases:
        print(f"Test Senaryosu: {input_expr}")
        
        try:
            # 1. Optimizasyonu Gerçekleştir
            optimized_expr = hypatia_core.optimize_ast(input_expr)
            
            # 2. İki ifadeyi de sayısal olarak değerlendir
            orig_val = hypatia_core.parse_expr(input_expr).eval(env)
            opt_val = hypatia_core.parse_expr(optimized_expr).eval(env)
            
            # 3. Sayısal denkliği kontrol et
            is_accurate = math.isclose(orig_val, opt_val, rel_tol=1e-9)
            
            if not is_accurate:
                all_passed = False
                
            print(f"  > Optimize Edilmiş Hali: {optimized_expr}")
            print(f"  > Orijinal Değer: {orig_val:.6f}")
            print(f"  > Optimize Değer: {opt_val:.6f}")
            print(f"  > Sayısal Doğruluk: {is_accurate}")
            
        except hypatia_core.HypatiaError as e:
            print(f"  > HATA: {e}")
            all_passed = False
        
        print("-" * 40)

    print("="*80)
    if all_passed:
        print("✅ SONUÇ: TÜM ÇEKİRDEK TESTLER BAŞARILI.")
    else:
        print("❌ SONUÇ: BAZI ÇEKİRDEK TESTLER BAŞARISIZ OLDU.")
    print("="*80)

if __name__ == "__main__":
    test_hypatia_core_optimizations()