import math
import numpy as np
from hypatia_core import (
    PyMultiVector2D, PyMultiVector3D,
    mv2d_from_array, mv2d_to_array, batch_rotate_2d,
    mv3d_from_array, mv3d_to_array, batch_rotate_3d,
)

# 2D toplu rotasyon
r = PyMultiVector2D.rotor(math.pi/2)
pts = np.array([[1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0]], dtype=np.float64, order="C")
rot = batch_rotate_2d(r, pts)
print("2D batch rotated:\n", rot)

# 3D rotor imzası: 4 argüman olmazsa tuple ile dener
try:
    r3 = PyMultiVector3D.rotor(math.pi/2, 0.0, 0.0, 1.0)
except TypeError:
    r3 = PyMultiVector3D.rotor(math.pi/2, (0.0, 0.0, 1.0))

pts3 = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [1.0, 1.0, 0.0]], dtype=np.float64, order="C")
rot3 = batch_rotate_3d(r3, pts3)
print("3D batch rotated:\n", rot3)
