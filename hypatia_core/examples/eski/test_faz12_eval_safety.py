"""
Hypatia FAZ 12: eval() Panic Safety Test Suite
Test eval() metodunun panic atmak yerine Python exception fırlattığını doğrular
"""

import hypatia_core as hc

print("=" * 70)
print("HYPATIA FAZ 12: eval() PANIC SAFETY TESTS")
print("=" * 70)

# ============ TEST 1: BAŞARILI EVALUATIONS ============
print("\n[TEST 1] Başarılı Değerlendirmeler")

x = hc.Symbol.variable("x")
y = hc.Symbol.variable("y")

# Basit aritmetik
expr1 = x * hc.Symbol.const(2.0) + y
result1 = expr1.eval({"x": 3.0, "y": 5.0})
print(f"✅ (x * 2 + y).eval({{x:3, y:5}}) = {result1}")
assert result1 == 11.0, f"Expected 11.0, got {result1}"

# ReLU
expr2 = hc.Symbol.relu(x * hc.Symbol.const(2.0))
result2 = expr2.eval({"x": 3.0})
print(f"✅ ReLU(x * 2).eval({{x:3}}) = {result2}")
assert result2 == 6.0, f"Expected 6.0, got {result2}"

# Sigmoid
expr3 = hc.Symbol.sigmoid(hc.Symbol.const(0.0))
result3 = expr3.eval({})
print(f"✅ sigmoid(0).eval() = {result3}")
assert abs(result3 - 0.5) < 1e-10, f"Expected 0.5, got {result3}"

print("✅ Tüm başarılı değerlendirmeler doğru")

# ============ TEST 2: DIVISION BY ZERO ============
print("\n[TEST 2] Sıfıra Bölme Kontrolü")

# x / 0
expr_div_zero = x / hc.Symbol.const(0.0)

try:
    result = expr_div_zero.eval({"x": 5.0})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Division by zero should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Division by zero" in str(e), f"Wrong error message: {e}"

# 1 / (x - 5) where x=5
expr_div_zero2 = hc.Symbol.const(1.0) / (x - hc.Symbol.const(5.0))

try:
    result = expr_div_zero2.eval({"x": 5.0})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Division by zero should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Division by zero" in str(e), f"Wrong error message: {e}"

# ============ TEST 3: LOG OF NEGATIVE ============
print("\n[TEST 3] Negatif Sayının Logaritması")

# log(-1)
expr_log_neg = hc.Symbol.log(hc.Symbol.const(-1.0))

try:
    result = expr_log_neg.eval({})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Log of negative should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Log of non-positive" in str(e), f"Wrong error message: {e}"

# log(0)
expr_log_zero = hc.Symbol.log(hc.Symbol.const(0.0))

try:
    result = expr_log_zero.eval({})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Log of zero should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Log of non-positive" in str(e), f"Wrong error message: {e}"

# ============ TEST 4: SQRT OF NEGATIVE ============
print("\n[TEST 4] Negatif Sayının Karekökü")

# sqrt(-1)
expr_sqrt_neg = hc.Symbol.sqrt(hc.Symbol.const(-1.0))

try:
    result = expr_sqrt_neg.eval({})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Sqrt of negative should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Sqrt of negative" in str(e), f"Wrong error message: {e}"

# ============ TEST 5: UNDEFINED VARIABLE ============
print("\n[TEST 5] Tanımsız Değişken")

# x + y but only x is provided
expr_undef = x + y

try:
    result = expr_undef.eval({"x": 5.0})  # y eksik
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Undefined variable should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "not found" in str(e), f"Wrong error message: {e}"

# ============ TEST 6: ZERO TO NEGATIVE POWER ============
print("\n[TEST 6] Sıfırın Negatif Kuvveti")

# 0^(-1)
expr_zero_neg_pow = hc.Symbol.pow(hc.Symbol.const(0.0), hc.Symbol.const(-1.0))

try:
    result = expr_zero_neg_pow.eval({})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Zero to negative power should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Zero to negative power" in str(e), f"Wrong error message: {e}"

# ============ TEST 7: COMPLEX EXPRESSION ERROR ============
print("\n[TEST 7] Karmaşık İfade Hatası")

# exp(log(x)) where x = -1
complex_expr = hc.Symbol.exp(hc.Symbol.log(x))

try:
    result = complex_expr.eval({"x": -1.0})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Log of negative should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    # Log hatası olmalı (çünkü önce log değerlendirilir)
    assert "Log of non-positive" in str(e), f"Wrong error message: {e}"

# ============ TEST 8: ERROR IN DERIVATIVE EVALUATION ============
print("\n[TEST 8] Türev Değerlendirmesinde Hata")

# d/dx[log(x)] = 1/x, x=0'da tanımsız
log_x = hc.Symbol.log(x)
dlog_dx = log_x.derivative("x")

print(f"   d/dx[log(x)] = {dlog_dx}")

try:
    # 1/0 hatası bekliyoruz
    result = dlog_dx.eval({"x": 0.0})
    print("❌ HATA: Exception bekliyorduk ama almadık!")
    assert False, "Division by zero in derivative should raise exception"
except hc.HypatiaError as e:
    print(f"✅ Beklenen exception yakalandı: {e}")
    assert "Division by zero" in str(e), f"Wrong error message: {e}"

# ============ SONUÇ ============
print("\n" + "=" * 70)
print("TÜM PANIC SAFETY TESTLERİ BAŞARIYLA TAMAMLANDI! ✅")
print("=" * 70)

print("\n📊 FAZ 12 ÖZETİ:")
print("✅ Başarılı değerlendirmeler çalışıyor")
print("✅ Division by zero → Python exception")
print("✅ Log of negative → Python exception")
print("✅ Sqrt of negative → Python exception")
print("✅ Undefined variable → Python exception")
print("✅ Zero to negative power → Python exception")
print("✅ Karmaşık ifadelerde hata propagasyonu")
print("✅ Türev değerlendirmesinde hata yakalama")

print("\n🎯 Hypatia artık PANIC-SAFE!")
print("   Python kullanıcıları tüm hataları try/except ile yakalayabilir.")
print("   Rust panic'leri Python interpreter'ı çökertemez.")
print("\n🚀 FAZ 12 - COMPLETE!")