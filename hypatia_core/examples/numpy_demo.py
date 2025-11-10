#!/usr/bin/env python3
"""
Hypatia NumPy Demo - Geometrik Cebir + NumPy = ❤️
"""

import sys
import os
import numpy as np

# Mevcut dizini Python path'ine ekle
sys.path.insert(0, os.path.dirname(__file__))

try:
    import hypatia_core as hypatia
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you've run: python3 build.py")
    sys.exit(1)

def main():
    print("=== Hypatia NumPy Integration Demo ===")
    print()

    # 1. NumPy Array'den MultiVector oluşturma
    print("1. NumPy Array'den MultiVector Oluşturma:")
    print()

    # 2D MultiVector için NumPy array
    mv2d_array = np.array([1.0, 2.0, 3.0, 4.0])  # [s, e1, e2, e12]
    mv2d_from_numpy = hypatia.mv2d_from_array(mv2d_array)
    print(f"   NumPy Array: {mv2d_array}")
    print(f"   MultiVector2D: {mv2d_from_numpy}")
    print()

    # MultiVector'den NumPy array'e dönüşüm
    mv2d_to_numpy = hypatia.mv2d_to_array(mv2d_from_numpy)
    print(f"   Back to NumPy: {mv2d_to_numpy}")
    print()

    # 2. Toplu Vektör Döndürme (2D)
    print("2. Toplu 2D Vektör Döndürme:")
    print()

    # 1000 rastgele 2D vektör oluştur
    num_vectors = 1000
    vectors_2d = np.random.randn(num_vectors, 2)
    
    # 45 derece rotor
    rotor_2d = hypatia.PyMultiVector2D.rotor(np.pi / 4)
    
    # Tüm vektörleri döndür
    rotated_vectors = hypatia.batch_rotate_2d(rotor_2d, vectors_2d)
    
    print(f"   Orijinal vektörler şekli: {vectors_2d.shape}")
    print(f"   Döndürülmüş vektörler şekli: {rotated_vectors.shape}")
    print(f"   İlk 5 orijinal vektör:\n{vectors_2d[:5]}")
    print(f"   İlk 5 döndürülmüş vektör:\n{rotated_vectors[:5]}")
    print()

    # 3. 3D NumPy Entegrasyonu
    print("3. 3D NumPy Entegrasyonu:")
    print()

    # 3D MultiVector için NumPy array
    mv3d_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mv3d_from_numpy = hypatia.mv3d_from_array(mv3d_array)
    print(f"   NumPy Array: {mv3d_array}")
    print(f"   MultiVector3D: {mv3d_from_numpy}")
    print()

    # 4. Toplu 3D Vektör Döndürme
    print("4. Toplu 3D Vektör Döndürme:")
    print()

    # 500 rastgele 3D vektör oluştur
    vectors_3d = np.random.randn(500, 3)
    
    # Z-ekseni etrafında 90 derece rotor
    rotor_3d = hypatia.PyMultiVector3D.rotor(np.pi / 2, (0.0, 0.0, 1.0))
    
    # Tüm vektörleri döndür
    rotated_vectors_3d = hypatia.batch_rotate_3d(rotor_3d, vectors_3d)
    
    print(f"   Orijinal 3D vektörler şekli: {vectors_3d.shape}")
    print(f"   Döndürülmüş 3D vektörler şekli: {rotated_vectors_3d.shape}")
    print(f"   İlk 3 orijinal 3D vektör:\n{vectors_3d[:3]}")
    print(f"   İlk 3 döndürülmüş 3D vektör:\n{rotated_vectors_3d[:3]}")
    print()

    # 5. Performans Karşılaştırması
    print("5. Performans Karşılaştırması:")
    print()

    import time

    # Büyük veri seti
    large_vectors = np.random.randn(10000, 2)
    
    # Hypatia ile döndürme
    start_time = time.time()
    hypatia_rotated = hypatia.batch_rotate_2d(rotor_2d, large_vectors)
    hypatia_time = time.time() - start_time
    
    # NumPy ile manual döndürme (karşılaştırma için)
    start_time = time.time()
    theta = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    numpy_rotated = large_vectors @ rotation_matrix.T
    numpy_time = time.time() - start_time
    
    print(f"   Hypatia süresi (10,000 vektör): {hypatia_time:.4f} saniye")
    print(f"   NumPy süresi (10,000 vektör): {numpy_time:.4f} saniye")
    print(f"   Hız farkı: {numpy_time/hypatia_time:.2f}x")
    print()

    # Doğruluk karşılaştırması
    error = np.abs(hypatia_rotated - numpy_rotated).max()
    print(f"   Maksimum hata: {error:.10f}")
    
    print()
    print("🎉 Hypatia NumPy entegrasyonu çalışıyor!")

if __name__ == "__main__":
    main()