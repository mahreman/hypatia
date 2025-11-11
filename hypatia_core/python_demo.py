import math
from hypatia_core import PyMultiVector2D, PyMultiVector3D, demo_2d_rotation, demo_3d_rotation

print(demo_2d_rotation())
print(demo_3d_rotation())

def get(obj, name):
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr

# 2D
r2 = PyMultiVector2D.rotor(math.pi/4)
v  = PyMultiVector2D.vector(1.0, 0.0)
w  = r2.rotate_vector(v)
print("2D rot(45°):", get(w,"s"), get(w,"e1"), get(w,"e2"), get(w,"e12"))

# 3D rotor: önce 4 argüman dener, olmazsa (angle, (ux,uy,uz))
try:
    r3 = PyMultiVector3D.rotor(math.pi/2, 0.0, 0.0, 1.0)
except TypeError:
    r3 = PyMultiVector3D.rotor(math.pi/2, (0.0, 0.0, 1.0))

x  = PyMultiVector3D.vector(1.0, 0.0, 0.0)
y  = r3.rotate_vector(x)
print("3D rot(90°, z):", get(y,"s"), get(y,"e1"), get(y,"e2"), get(y,"e3"))
