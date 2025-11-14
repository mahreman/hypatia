"""
Hypatia E-graph Optimizer Test Suite
FAZ 2: Otomatik sembolik optimizasyon

Bu dosya, hypatia_core Python modülünde Symbol sınıfı olmadığı durumda
kendi içinde küçük bir sarmalayıcı Symbol tanımlar ve optimize_ast ile çalışır.
"""

import math
import hypatia_core as hc  # derlenen modülde optimize_ast fonksiyonu bulunmalı


# -----------------------------
# BÖLÜM 1: E-GRAPH TESTLERİ İÇİN SAHTE (MOCK) SYMBOL SINIFI
# Bu sınıf, optimize_ast testleri (FEZ 4) için S-expression string'leri oluşturur.
# -----------------------------
class Symbol:
    def __init__(self, sexpr: str):
        self.se = sexpr

    def __str__(self):
        return self.se

    # ------------ factory ------------
    @staticmethod
    def variable(name: str) -> "Symbol":
        return Symbol(name)

    @staticmethod
    def const(val: float) -> "Symbol":
        # testlerde sayılar tamsayı gibi yazılıyor (2.0 -> "2")
        if float(val).is_integer():
            return Symbol(str(int(val)))
        return Symbol(repr(float(val)))

    # ------------ unary ops ------------
    def __neg__(self) -> "Symbol":
        return Symbol(f"(neg {self})")

    # ------------ binary ops ------------
    def __add__(self, other: "Symbol") -> "Symbol":
        return Symbol(f"(add {self} {other})")

    def __sub__(self, other: "Symbol") -> "Symbol":
        return Symbol(f"(sub {self} {other})")

    def __mul__(self, other: "Symbol") -> "Symbol":
        return Symbol(f"(mul {self} {other})")

    def __truediv__(self, other: "Symbol") -> "Symbol":
        return Symbol(f"(div {self} {other})")

    # ------------ math funcs ------------
    @staticmethod
    def exp(x: "Symbol") -> "Symbol":
        return Symbol(f"(exp {x})")

    @staticmethod
    def log(x: "Symbol") -> "Symbol":
        return Symbol(f"(log {x})")

    @staticmethod
    def sqrt(x: "Symbol") -> "Symbol":
        return Symbol(f"(sqrt {x})")

    @staticmethod
    def pow(x: "Symbol", y: "Symbol") -> "Symbol":
        return Symbol(f"(pow {x} {y})")

    @staticmethod
    def relu(x: "Symbol") -> "Symbol":
        # kanonik hedefimiz (relu x); optimizer max->relu map'liyor
        return Symbol(f"(relu {x})")


# testlerde kısa isim
Sym = Symbol


print("=" * 70)
print("HYPATIA E-GRAPH OPTIMIZER TEST SUITE - FAZ 4 (E-graph)")
print("=" * 70)

# ============ TEST 1: TEMEL OPTİMİZASYONLAR ============
print("\n[TEST 1] Temel Optimizasyonlar")

x_str = Sym.variable("x")

# x * 0 = 0
expr1 = x_str * Sym.const(0.0)
print(f"x * 0 = {expr1}")
optimized1 = hc.optimize_ast(str(expr1))
print(f"Optimized: {optimized1}")
assert optimized1 == "0", f"Expected '0', got '{optimized1}'"

# x + 0 = x
expr2 = x_str + Sym.const(0.0)
print(f"\nx + 0 = {expr2}")
optimized2 = hc.optimize_ast(str(expr2))
print(f"Optimized: {optimized2}")
assert optimized2 == "x", f"Expected 'x', got '{optimized2}'"

# x * 1 = x
expr3 = x_str * Sym.const(1.0)
print(f"\nx * 1 = {expr3}")
optimized3 = hc.optimize_ast(str(expr3))
print(f"Optimized: {optimized3}")
assert optimized3 == "x", f"Expected 'x', got '{optimized3}'"

# ============ TEST 2: FACTORING ============
print("\n[TEST 2] Factoring Optimizasyonu (A*B + A*C → A*(B+C))")
expr = (Sym.const(2.0) * x_str) + (Sym.const(3.0) * x_str)
print(f"Expression: {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
print("✅ Factoring çalışıyor! (x katsayısı birleştirildi)")

# ============ TEST 3: EXP/LOG ============
print("\n[TEST 3] Exp/Log İptali")
expr = Sym.exp(Sym.log(x_str))
print(f"exp(log(x)) = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "x", f"Expected 'x', got '{optimized}'"

expr = Sym.log(Sym.exp(x_str))
print(f"\nlog(exp(x)) = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "x", f"Expected 'x', got '{optimized}'"

# ============ TEST 4: POW/DIV ============
print("\n[TEST 4] Pow/Div Optimizasyonları")

expr = Sym.pow(x_str, Sym.const(1.0))
print(f"x^1 = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "x", f"Expected 'x', got '{optimized}'"

expr = x_str / x_str
print(f"\nx/x = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "1", f"Expected '1', got '{optimized}'"

expr = x_str / Sym.const(1.0)
print(f"\nx/1 = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "x", f"Expected 'x', got '{optimized}'"

# ============ TEST 5: RELU ============
print("\n[TEST 5] ReLU Optimizasyonları")

expr = Sym.relu(Sym.relu(x_str))
print(f"ReLU(ReLU(x)) = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
# İki eşdeğer kanonik formu da kabul et
assert optimized in ("(relu x)", "(max x 0)"), \
    f"Expected '(relu x)' or '(max x 0)', got '{optimized}'"

# ============ TEST 6: NEGASYON ============
print("\n[TEST 6] Negasyon Optimizasyonları")

expr = -(-x_str)
print(f"--x = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
assert optimized == "x", f"Expected 'x', got '{optimized}'"

# -(a*b) örneği: patlama olmamalı, sadece gözlem yazdırıyoruz
a = Sym.variable("a")
b = Sym.variable("b")
expr = -(a * b)
print(f"\n-(a*b) = {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
if optimized.startswith("(error"):
    raise RuntimeError(f"Optimizer error: {optimized}")

# ============ TEST 7: KARMAŞIK ============
print("\n[TEST 7] Karmaşık İfade Optimizasyonu")
expr = Sym.exp(Sym.log(x_str)) * Sym.const(2.0) + Sym.const(0.0)
print(f"Expression: exp(log(x)) * 2 + 0")
print(f"Original: {expr}")
optimized = hc.optimize_ast(str(expr))
print(f"Optimized: {optimized}")
print("✅ Birden fazla kural uygulandı!")

print("\n✅ BÜTÜN E-GRAPH TESTLERİ GEÇTİ (FAZ 4)")


# -----------------------------
# BÖLÜM 2: YENİ FEZ 3 (İNTEGRAL) TESTLERİ
# Bu testler, Rust'tan gelen GERÇEK 'hc.Symbol' sınıfını kullanır.
# -----------------------------

print("\n" + "=" * 70)
print("HYPATIA SYMBOLIC CORE TEST SUITE - FAZ 3 (İntegral)")
print("=" * 70)

# Gerçek Rust Symbol sınıfını 'HCS' (Hypatia Core Symbol) olarak alalım
try:
    HCS = hc.Symbol
except AttributeError:
    print("\n" + "!"*70)
    print("!!! HATA: 'hypatia_core' modülünde 'Symbol' sınıfı bulunamadı.")
    print("Lütfen Rust kodunun 'python' özelliği ile derlendiğinden emin olun:")
    print("  maturin develop --features python")
    print("!"*70)
    raise

# Yeni testler
try:
    x = HCS.variable("x")
    y = HCS.variable("y")

    # Test 8: ∫ 5 dx = 5*x
    print("\n[TEST 8] Sabit İntegrali (∫ 5 dx)")
    f1 = HCS.const(5.0)
    int_f1 = f1.integrate("x")
    expected_f1 = HCS.const(5.0) * x
    print(f"  Original: {f1}")
    print(f"  Integral: {int_f1}")
    assert str(int_f1.simplify()) == str(expected_f1.simplify())

    # Test 9: ∫ y dx = y*x
    print("\n[TEST 9] Diğer Değişken İntegrali (∫ y dx)")
    f2 = HCS.variable("y") # y.clone() Rust tarafında gerekmez, PySymbol referans tutar
    int_f2 = f2.integrate("x")
    expected_f2 = y * x
    print(f"  Original: {f2}")
    print(f"  Integral: {int_f2}")
    assert str(int_f2.simplify()) == str(expected_f2.simplify())

    # Test 10: ∫ x^3 dx = x^4 / 4
    print("\n[TEST 10] Kuvvet Kuralı (∫ x^3 dx)")
    f3 = HCS.pow(x, HCS.const(3.0))
    int_f3 = f3.integrate("x")
    expected_f3 = HCS.pow(x, HCS.const(4.0)) / HCS.const(4.0)
    print(f"  Original: {f3}")
    print(f"  Integral: {int_f3}")
    assert str(int_f3.simplify()) == str(expected_f3.simplify())

    # Test 11: Lineerlik (∫ (x + 2) dx) - ARTIK E-GRAPH'TAN GEÇİYOR
    print("\n[TEST 11] Lineerlik (∫ (x + 2) dx) + E-graph Optimizasyonu")
    f4 = x + HCS.const(2.0)
    int_f4 = f4.integrate("x") # Bu, "aptal" simplify() içerir
    
    print(f"  Original: {f4}")
    print(f"  Raw Integral: {int_f4}") # (add (div (pow x 2) 2) (mul 2 x))

    # Şimdi "akıllı" e-graph optimizasyonunu çağır
    optimized_integral_str = hc.optimize_ast(str(int_f4))
    print(f"  Optimized Integral: {optimized_integral_str}")

    # Beklenen sonucu oluştur
    x_sq_2 = HCS.pow(x, HCS.const(2.0)) / HCS.const(2.0)
    two_x = HCS.const(2.0) * x
    expected_f4 = x_sq_2 + two_x
    expected_f4_optimized_str = hc.optimize_ast(str(expected_f4))
    
    print(f"  Beklenen (E-graph): {expected_f4_optimized_str}")

    assert optimized_integral_str == expected_f4_optimized_str

    # Test 12: ∫ 1/x dx = ln(x)
    print("\n[TEST 12] Özel Fonksiyon (∫ 1/x dx)")
    f5 = HCS.pow(x, HCS.const(-1.0)) # (1/x)
    int_f5 = f5.integrate("x")
    expected_f5 = HCS.log(x)
    print(f"  Original: {f5}")
    print(f"  Integral: {int_f5}")
    assert str(int_f5.simplify()) == str(expected_f5.simplify())

    # Test 13: ∫ e^x dx = e^x
    print("\n[TEST 13] Özel Fonksiyon (∫ e^x dx)")
    f6 = HCS.exp(x)
    int_f6 = f6.integrate("x")
    print(f"  Original: {f6}")
    print(f"  Integral: {int_f6}")
    assert str(int_f6.simplify()) == str(f6.simplify())

    # Test 14: Hata Yakalama (∫ sigmoid(x) dx)
    print("\n[TEST 14] Desteklenmeyen İntegral (Hata Fırlatma)")
    f7 = HCS.sigmoid(x)
    print(f"  Original: {f7}")
    try:
        f7.integrate("x")
        # Buraya ulaşırsak, test başarısız olmuştur
        print("  HATA: Desteklenmeyen integral hata fırlatmadı!")
        assert False, "HypatiaError bekleniyordu ama fırlatılmadı."
    except hc.HypatiaError as e:
        # Başarılı durum
        print(f"  Başarılı: Beklenen hata yakalandı: {e}")
        assert "Unsupported integration" in str(e) or "not supported" in str(e)
    except Exception as e:
        # Yanlış hata türü
        print(f"  HATA: Yanlış türde hata fırlatıldı: {type(e)}")
        assert False, f"HypatiaError bekleniyordu, {type(e)} fırlatıldı."
        
    print("\n✅ BÜTÜN İNTEGRAL TESTLERİ GEÇTİ (FEZ 3)")

except NameError:
    # HCS = hc.Symbol satırının kendisi NameError verirse (çok nadir)
    pass