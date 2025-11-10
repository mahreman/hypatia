#!/usr/bin/env python3
"""
Hypatia Python Demo - Geometrik Cebir'in gücü Python'da!
"""

import sys
import os

# Mevcut dizini Python path'ine ekle
sys.path.insert(0, os.path.dirname(__file__))

try:
    import hypatia_core as hypatia
    import math
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you've run: python3 build.py")
    sys.exit(1)

def main():
    print("=== Hypatia Python Binding Demo ===")
    print()

    # 2D Geometrik Cebir Demo
    print("1. 2D Geometrik Cebir:")
    print()

    # Temel vektörler
    e1 = hypatia.PyMultiVector2D.vector(1.0, 0.0)
    e2 = hypatia.PyMultiVector2D.vector(0.0, 1.0)

    print(f"   e1 = {e1}")
    print(f"   e2 = {e2}")
    print(f"   e1 * e1 = {e1 * e1}")
    print(f"   e1 * e2 = {e1 * e2}")
    print(f"   e2 * e1 = {e2 * e1}")
    print()

    # 2D Döndürme
    vector = hypatia.PyMultiVector2D.vector(1.0, 0.0)
    rotor_90 = hypatia.PyMultiVector2D.rotor(math.pi / 2)
    rotated = rotor_90.rotate_vector(vector)

    print(f"   Orijinal vektör: {vector}")
    print(f"   90° döndürülmüş: {rotated}")
    print()

    # 3D Geometrik Cebir Demo
    print("2. 3D Geometrik Cebir:")
    print()

    e1_3d = hypatia.PyMultiVector3D.vector(1.0, 0.0, 0.0)
    e2_3d = hypatia.PyMultiVector3D.vector(0.0, 1.0, 0.0)
    e3_3d = hypatia.PyMultiVector3D.vector(0.0, 0.0, 1.0)

    print(f"   e1 = {e1_3d}")
    print(f"   e2 = {e2_3d}") 
    print(f"   e3 = {e3_3d}")
    print(f"   e1 * e2 = {e1_3d * e2_3d}")
    print(f"   e2 * e3 = {e2_3d * e3_3d}")
    print(f"   e3 * e1 = {e3_3d * e1_3d}")
    print()

    # 3D Döndürme
    vector_3d = hypatia.PyMultiVector3D.vector(1.0, 0.0, 0.0)

    # Z-ekseni etrafında döndürme
    rotor_z = hypatia.PyMultiVector3D.rotor(math.pi / 2, (0.0, 0.0, 1.0))
    rotated_z = rotor_z.rotate_vector(vector_3d)
    print(f"   Z-ekseni 90° döndürme: {vector_3d} → {rotated_z}")

    # Y-ekseni etrafında döndürme  
    rotor_y = hypatia.PyMultiVector3D.rotor(math.pi / 2, (0.0, 1.0, 0.0))
    rotated_y = rotor_y.rotate_vector(vector_3d)
    print(f"   Y-ekseni 90° döndürme: {vector_3d} → {rotated_y}")

    # X-ekseni etrafında döndürme
    vector_y = hypatia.PyMultiVector3D.vector(0.0, 1.0, 0.0)
    rotor_x = hypatia.PyMultiVector3D.rotor(math.pi / 2, (1.0, 0.0, 0.0))
    rotated_x = rotor_x.rotate_vector(vector_y)
    print(f"   X-ekseni 90° döndürme: {vector_y} → {rotated_x}")

    print()
    print("🎉 Hypatia Python'da çalışıyor!")

if __name__ == "__main__":
    main()