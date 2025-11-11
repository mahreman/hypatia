import math
from hypatia_core import PyMultiVector2D, PyMultiVector3D, demo_2d_rotation, demo_3d_rotation

print(demo_2d_rotation())
print(demo_3d_rotation())

r2 = PyMultiVector2D.rotor(math.pi/4)
v  = PyMultiVector2D.vector(1.0, 0.0)
w  = r2.rotate_vector(v)
print("2D rot(45°):", w.s(), w.e1(), w.e2(), w.e12())

r3 = PyMultiVector3D.rotor(math.pi/2, 0.0, 0.0, 1.0)
x  = PyMultiVector3D.vector(1.0, 0.0, 0.0)
y  = r3.rotate_vector(x)
print("3D rot(90°, z):", y.s(), y.e1(), y.e2(), y.e3())