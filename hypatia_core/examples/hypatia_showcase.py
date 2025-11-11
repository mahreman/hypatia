# examples/hypatia_showcase.py
import math
import numpy as np
from hypatia_core import (
    PyMultiVector2D, PyMultiVector3D,
    mv2d_from_array, mv2d_to_array, batch_rotate_2d, batch_rotate_3d,
    Symbol, PyMultiVector2D_Symbolic, PyMultiVector3D_Symbolic,
)

print("=== Hypatia Showcase ===")

# 1) Sayısal 2D dönüş (rotor)
r2 = PyMultiVector2D.rotor(math.pi / 2.0)  # 90° CCW
v2 = PyMultiVector2D.vector(1.0, 0.0)
w2 = r2.rotate_vector(v2).grade(1)
print(f"\n[Numeric 2D] (1,0) --90°--> ({w2.e1():.6f}, {w2.e2():.6f})")

# 2) Sayısal 3D dönüş (Z ekseni)
r3 = PyMultiVector3D.rotor(math.pi / 2.0, 0.0, 0.0, 1.0)  # 90° around Z
x3 = PyMultiVector3D.vector(1.0, 0.0, 0.0)
y3 = r3.rotate_vector(x3).grade(1)
print(f"[Numeric 3D] (1,0,0) --90° Z--> ({y3.e1():.6f}, {y3.e2():.6f}, {y3.e3():.6f})")

# 3) NumPy ile toplu (batch) döndürme
arr2 = np.array([
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
], dtype=float)

# Doğru çağrı: theta önce, matris sonra (veya keyword ile arr=...)
out2 = batch_rotate_2d(math.pi / 2.0, arr2)
print("\n[NumPy 2D] batch_rotate_2d:\n", out2)

arr3 = np.array([
    [ 1.0,  0.0, 0.0],
    [ 0.0,  1.0, 0.0],
    [-1.0,  1.0, 0.0],
], dtype=float)

# 3D’de de theta ve eksenler önce, matris en sonda
out3 = batch_rotate_3d(math.pi / 2.0, 0.0, 0.0, 1.0, arr3)
print("\n[NumPy 3D] batch_rotate_3d:\n", out3)

# 4) NumPy <-> PyMultiVector2D tekil dönüştürmeler
mv_from = mv2d_from_array(np.array([[3.0, 4.0]], dtype=float))
back = mv2d_to_array(mv_from)
print("\n[NumPy <-> MV2D] in -> MV -> out:", back)

# 5) Sembolik ifadeler ve türev/sadeleştirme
x = Symbol.variable("x")
y = Symbol.variable("y")
one = Symbol.const_(1.0)   # DÜZELTME: const_ kullan
two = Symbol.const_(2.0)   # DÜZELTME: const_ kullan

f = x * two              # f(x) = 2x
df_dx = f.derivative("x").simplify()
print(f"\n[Symbolic] f = {f}, df/dx = {df_dx}")

g = x * y + two          # g(x,y) = x*y + 2
dg_dx = g.derivative("x").simplify()
dg_dy = g.derivative("y").simplify()
print(f"[Symbolic] g = {g}, dg/dx = {dg_dx}, dg/dy = {dg_dy}")

# 6) Sembolik 3D vektör aritmetiği (GA + Symbol)
v3_sym = PyMultiVector3D_Symbolic.vector(x, y, one)  # (x, y, 1)
u3_sym = PyMultiVector3D_Symbolic.vector(one, one, one)
sum3 = v3_sym + u3_sym
print("\n[Symbolic GA3D] v3 =", v3_sym)
print("[Symbolic GA3D] u3 =", u3_sym)
print("[Symbolic GA3D] v3 + u3 =", sum3)
print("[Symbolic GA3D] (v3 + u3).simplify() =", sum3.simplify())

# 7) (Opsiyonel) 2D sembolik vektör demo
v2_sym = PyMultiVector2D_Symbolic.vector(x, two)  # (x, 2)
w2_sym = v2_sym + PyMultiVector2D_Symbolic.vector(one, one)
print("\n[Symbolic GA2D] v2 =", v2_sym)
print("[Symbolic GA2D] v2 + (1,1) =", w2_sym)
print("[Symbolic GA2D] simplify =", w2_sym.simplify())

print("\n=== Done ===")
