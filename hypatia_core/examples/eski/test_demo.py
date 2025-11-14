from hypatia_core import PyMultiVector2D as V2, PyMultiVector3D as V3

# 2D
a = V2.vector(1,0); b = V2.vector(0,1)
w = a ^ b
print("2D wedge e12:", w.s(), w.e1(), w.e2(), w.e12())   # 0 0 0 1
g1 = (a+b).grade(1)
print("2D grade1:", g1.e1(), g1.e2())                    # 1 1
print("2D mul:", (a * b))                                # e12 yönünde bileşen içermeli
print("2D scalar ops:", (2.0*a), (a/2.0))

# 3D
x = V3.vector(1,0,0); y = V3.vector(0,1,0)
biv = x ^ y
print("3D wedge (e12,e23,e31):", biv.e12(), biv.e23(), biv.e31())  # 1,0,0
print("3D grade2:", (x^y).grade(2))