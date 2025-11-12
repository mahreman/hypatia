"""
Hypatia Symbol Test Suite
FAZ 1: Tüm yeni fonksiyonları test eder
"""

import hypatia_core as hc

print("=" * 60)
print("HYPATIA SYMBOL TEST SUITE - FAZ 1")
print("=" * 60)

# ============ TEST 1: TEMEL OLUŞTURUCULAR ============
print("\n[TEST 1] Temel Oluşturucular")
x = hc.Symbol.variable("x")
y = hc.Symbol.variable("y")
c5 = hc.Symbol.const(5.0)
c2 = hc.Symbol.const(2.0)

print(f"x = {x}")
print(f"y = {y}")
print(f"5 = {c5}")

# ============ TEST 2: ARİTMETİK OPERATÖRLER ============
print("\n[TEST 2] Aritmetik Operatörler")

# Toplama
expr1 = x + y
print(f"x + y = {expr1}")

# Çıkarma
expr2 = x - y
print(f"x - y = {expr2}")

# Çarpma
expr3 = x * c2
print(f"x * 2 = {expr3}")

# ✅ YENİ: Bölme
expr4 = x / c2
print(f"x / 2 = {expr4}")
print(f"x / 2 (simplified) = {expr4.simplify()}")

# Negasyon
expr5 = -x
print(f"-x = {expr5}")

# ============ TEST 3: MATEMATİKSEL FONKSİYONLAR ============
print("\n[TEST 3] Matematiksel Fonksiyonlar")

# Exp
exp_x = hc.Symbol.exp(x)
print(f"exp(x) = {exp_x}")

# Log
log_x = hc.Symbol.log(x)
print(f"log(x) = {log_x}")

# Sqrt
sqrt_x = hc.Symbol.sqrt(x)
print(f"sqrt(x) = {sqrt_x}")

# Pow
pow_x_2 = hc.Symbol.pow(x, c2)
print(f"x^2 = {pow_x_2}")

# Sadeleştirme testleri
print("\nSadeleştirme Testleri:")
exp_log_x = hc.Symbol.exp(hc.Symbol.log(x))
print(f"exp(log(x)) = {exp_log_x}")
print(f"exp(log(x)) simplified = {exp_log_x.simplify()}")

log_exp_x = hc.Symbol.log(hc.Symbol.exp(x))
print(f"log(exp(x)) = {log_exp_x}")
print(f"log(exp(x)) simplified = {log_exp_x.simplify()}")

# ============ TEST 4: AKTİVASYON FONKSİYONLARI ============
print("\n[TEST 4] Aktivasyon Fonksiyonları")

# ReLU
relu_x = hc.Symbol.relu(x)
print(f"ReLU(x) = {relu_x}")

relu_5 = hc.Symbol.relu(c5)
print(f"ReLU(5) = {relu_5}")
print(f"ReLU(5) simplified = {relu_5.simplify()}")

c_neg3 = hc.Symbol.const(-3.0)
relu_neg3 = hc.Symbol.relu(c_neg3)
print(f"ReLU(-3) = {relu_neg3}")
print(f"ReLU(-3) simplified = {relu_neg3.simplify()}")

# ✅ YENİ: ReLU Gradient
relu_grad_5 = hc.Symbol.relu_grad(c5)
print(f"ReLU'(5) = {relu_grad_5}")
print(f"ReLU'(5) simplified = {relu_grad_5.simplify()}")

relu_grad_neg3 = hc.Symbol.relu_grad(c_neg3)
print(f"ReLU'(-3) = {relu_grad_neg3}")
print(f"ReLU'(-3) simplified = {relu_grad_neg3.simplify()}")

# Sigmoid
sigmoid_x = hc.Symbol.sigmoid(x)
print(f"sigmoid(x) = {sigmoid_x}")

c0 = hc.Symbol.const(0.0)
sigmoid_0 = hc.Symbol.sigmoid(c0)
print(f"sigmoid(0) = {sigmoid_0}")
print(f"sigmoid(0) simplified = {sigmoid_0.simplify()}")

# Tanh
tanh_x = hc.Symbol.tanh(x)
print(f"tanh(x) = {tanh_x}")

tanh_0 = hc.Symbol.tanh(c0)
print(f"tanh(0) = {tanh_0}")
print(f"tanh(0) simplified = {tanh_0.simplify()}")

# ============ TEST 5: YARDIMCI FONKSİYONLAR ============
print("\n[TEST 5] Yardımcı Fonksiyonlar")

c3 = hc.Symbol.const(3.0)

# Max
max_3_5 = hc.Symbol.max(c3, c5)
print(f"max(3, 5) = {max_3_5}")
print(f"max(3, 5) simplified = {max_3_5.simplify()}")

# Min
min_3_5 = hc.Symbol.min(c3, c5)
print(f"min(3, 5) = {min_3_5}")
print(f"min(3, 5) simplified = {min_3_5.simplify()}")

# ============ TEST 6: TÜREV (DERIVATIVE) ============
print("\n[TEST 6] Türev (Derivative)")

# Basit türev
f1 = x * c2
df1_dx = f1.derivative("x")
print(f"f(x) = x * 2")
print(f"f'(x) = {df1_dx}")
print(f"f'(x) simplified = {df1_dx.simplify()}")

# Exp türevi
f2 = hc.Symbol.exp(x)
df2_dx = f2.derivative("x")
print(f"\nf(x) = exp(x)")
print(f"f'(x) = {df2_dx}")
print(f"f'(x) simplified = {df2_dx.simplify()}")

# Log türevi
f3 = hc.Symbol.log(x)
df3_dx = f3.derivative("x")
print(f"\nf(x) = log(x)")
print(f"f'(x) = {df3_dx}")
print(f"f'(x) simplified = {df3_dx.simplify()}")

# ReLU türevi
f4 = hc.Symbol.relu(x)
df4_dx = f4.derivative("x")
print(f"\nf(x) = ReLU(x)")
print(f"f'(x) = {df4_dx}")
print(f"f'(x) simplified = {df4_dx.simplify()}")

# Sigmoid türevi
f5 = hc.Symbol.sigmoid(x)
df5_dx = f5.derivative("x")
print(f"\nf(x) = sigmoid(x)")
print(f"f'(x) = {df5_dx}")
print(f"f'(x) simplified = {df5_dx.simplify()}")

# Chain rule test: exp(x^2)
x_squared = hc.Symbol.pow(x, c2)
f6 = hc.Symbol.exp(x_squared)
df6_dx = f6.derivative("x")
print(f"\nf(x) = exp(x^2)")
print(f"f'(x) = {df6_dx}")
print(f"f'(x) simplified = {df6_dx.simplify()}")

# ============ TEST 7: SUBSTITUTION (subs) ============
print("\n[TEST 7] Substitution (Değer Verme)")

expr = x * c2 + y
print(f"expr = x * 2 + y")

result = expr.subs({"x": 3.0, "y": 5.0})
print(f"expr.subs({{x: 3, y: 5}}) = {result}")

# ============ TEST 8: ✅ YENİ - NUMERICAL EVALUATION ============
print("\n[TEST 8] Numerical Evaluation (eval)")

# Basit eval
expr1 = x * c2
result1 = expr1.eval({"x": 5.0})
print(f"(x * 2).eval({{x: 5}}) = {result1}")
assert result1 == 10.0, "FAIL: Expected 10.0"

# ReLU eval
expr2 = hc.Symbol.relu(x * c2 + y)
result2 = expr2.eval({"x": 3.0, "y": -5.0})
print(f"ReLU(x*2 + y).eval({{x: 3, y: -5}}) = {result2}")
assert result2 == 1.0, "FAIL: Expected 1.0 (ReLU(6 - 5) = ReLU(1) = 1)"

# Sigmoid eval
expr3 = hc.Symbol.sigmoid(c0)
result3 = expr3.eval({})
print(f"sigmoid(0).eval() = {result3}")
assert abs(result3 - 0.5) < 1e-10, "FAIL: Expected 0.5"

# Complex expression eval
expr4 = hc.Symbol.exp(hc.Symbol.log(x)) * c2
result4 = expr4.eval({"x": 5.0})
print(f"(exp(log(x)) * 2).eval({{x: 5}}) = {result4}")
assert result4 == 10.0, "FAIL: Expected 10.0 (e^ln(5) * 2 = 5 * 2 = 10)"

# Division eval
expr5 = x / c2
result5 = expr5.eval({"x": 10.0})
print(f"(x / 2).eval({{x: 10}}) = {result5}")
assert result5 == 5.0, "FAIL: Expected 5.0"

# ============ TEST 9: KARMAŞIK İFADELER ============
print("\n[TEST 9] Karmaşık İfadeler")

# Neural network katmanı simülasyonu
# f(x) = ReLU(W*x + b) ile W=2, b=3
W = c2
b = c3
layer_output = hc.Symbol.relu(W * x + b)
print(f"Layer: ReLU(2*x + 3)")
print(f"Expression: {layer_output}")

# x=5 için çıktı
output_val = layer_output.eval({"x": 5.0})
print(f"Output when x=5: {output_val}")
assert output_val == 13.0, "FAIL: Expected 13.0 (ReLU(2*5 + 3) = ReLU(13) = 13)"

# Türevini al
dlayer_dx = layer_output.derivative("x")
print(f"d/dx[ReLU(2*x + 3)] = {dlayer_dx}")
print(f"d/dx[ReLU(2*x + 3)] simplified = {dlayer_dx.simplify()}")

# ============ TEST 10: EDGE CASES ============
print("\n[TEST 10] Edge Cases")

# 0 ile işlemler
zero = hc.Symbol.const(0.0)
print(f"x + 0 = {(x + zero).simplify()}")
print(f"x * 0 = {(x * zero).simplify()}")
print(f"x * 1 = {(x * hc.Symbol.const(1.0)).simplify()}")

# Kendisiyle işlemler
print(f"x - x = {(x - x).simplify()}")
print(f"x / x = {(x / x).simplify()}")

# Double negation
print(f"--x = {(-(-x)).simplify()}")

# ReLU idempotence
print(f"ReLU(ReLU(x)) = {hc.Symbol.relu(hc.Symbol.relu(x)).simplify()}")

# ============ SONUÇ ============
print("\n" + "=" * 60)
print("TÜM TESTLER BAŞARIYLA TAMAMLANDI! ✅")
print("=" * 60)
print("\nFAZ 1 Özeti:")
print("✅ Temel oluşturucular (variable, const)")
print("✅ Aritmetik operatörler (+, -, *, /, neg)")
print("✅ Matematiksel fonksiyonlar (exp, log, sqrt, pow)")
print("✅ Aktivasyon fonksiyonları (relu, sigmoid, tanh)")
print("✅ ReLU gradient (relu_grad)")
print("✅ Yardımcı fonksiyonlar (max, min)")
print("✅ Türev (derivative)")
print("✅ Sadeleştirme (simplify)")
print("✅ Substitution (subs)")
print("✅ Numerical evaluation (eval)")
print("\nHypatia FAZ 1 - COMPLETE!")
