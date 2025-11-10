// src/lib.rs

use std::ops::{Add, Mul, Sub, Neg};

/// Hypatia'nın temel 2D MultiVektör yapısı.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiVector2D {
    pub s: f64,      // Skaler bileşen
    pub e1: f64,     // e1 vektör bileşeni  
    pub e2: f64,     // e2 vektör bileşeni
    pub e12: f64,    // e1^e2 bivektör bileşeni
}

impl MultiVector2D {
    /// Yeni bir multivektör oluşturur.
    pub fn new(s: f64, e1: f64, e2: f64, e12: f64) -> Self {
        Self { s, e1, e2, e12 }
    }

    /// Sadece skalerden bir multivektör oluşturur.
    pub fn scalar(s: f64) -> Self {
        Self::new(s, 0.0, 0.0, 0.0)
    }

    /// Sadece vektörden bir multivektör oluşturur.
    pub fn vector(e1: f64, e2: f64) -> Self {
        Self::new(0.0, e1, e2, 0.0)
    }

    /// Sadece bivektörden bir multivektör oluşturur.
    pub fn bivector(e12: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, e12)
    }

    /// Bir Rotor'un tersini (reverse) alır. 
    pub fn reverse(&self) -> Self {
        Self::new(self.s, self.e1, self.e2, -self.e12)
    }

    /// Norm (büyüklük) hesaplar
    pub fn norm(&self) -> f64 {
        (self.s * self.s + self.e1 * self.e1 + self.e2 * self.e2 + self.e12 * self.e12).sqrt()
    }

    /// Normalize eder (birim multivektör)
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            return *self;
        }
        Self::new(self.s / norm, self.e1 / norm, self.e2 / norm, self.e12 / norm)
    }

    /// Belirli bir açıda rotor oluşturur
    pub fn rotor(angle_rad: f64) -> Self {
        let half_angle = angle_rad / 2.0;
        Self::new(
            half_angle.cos(),
            0.0,
            0.0,
            -half_angle.sin()  // DÜZELTME: Eksi işareti eklendi
        )
    }

    /// Bir vektörü bu multivektör (rotor) ile döndürür
    /// DÜZELTME: Doğru formül: R * v * R_reverse
    pub fn rotate_vector(&self, vector: &MultiVector2D) -> Self {
        // v' = R * v * R_reverse
        (*self * *vector) * self.reverse()
    }
}

// Operatör overloading'ler aynı kalacak...
impl Add for MultiVector2D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { s: self.s + rhs.s, e1: self.e1 + rhs.e1, e2: self.e2 + rhs.e2, e12: self.e12 + rhs.e12 }
    }
}

impl Sub for MultiVector2D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self { s: self.s - rhs.s, e1: self.e1 - rhs.e1, e2: self.e2 - rhs.e2, e12: self.e12 - rhs.e12 }
    }
}

impl Mul for MultiVector2D {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self; let b = rhs;
        let s = (a.s * b.s) + (a.e1 * b.e1) + (a.e2 * b.e2) - (a.e12 * b.e12);
        let e1 = (a.s * b.e1) + (a.e1 * b.s) - (a.e2 * b.e12) + (a.e12 * b.e2);
        let e2 = (a.s * b.e2) + (a.e1 * b.e12) + (a.e2 * b.s) - (a.e12 * b.e1);
        let e12 = (a.s * b.e12) + (a.e1 * b.e2) - (a.e2 * b.e1) + (a.e12 * b.s);
        Self::new(s, e1, e2, e12)
    }
}

impl Neg for MultiVector2D {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.s, -self.e1, -self.e2, -self.e12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_approx_eq(a: MultiVector2D, b: MultiVector2D) {
        let epsilon = 1e-10;
        assert!((a.s - b.s).abs() < epsilon, "s: {} != {}", a.s, b.s);
        assert!((a.e1 - b.e1).abs() < epsilon, "e1: {} != {}", a.e1, b.e1);
        assert!((a.e2 - b.e2).abs() < epsilon, "e2: {} != {}", a.e2, b.e2);
        assert!((a.e12 - b.e12).abs() < epsilon, "e12: {} != {}", a.e12, b.e12);
    }

    #[test]
    fn test_basis_multiplication() {
        let s = MultiVector2D::scalar(1.0);
        let e1 = MultiVector2D::vector(1.0, 0.0);
        let e2 = MultiVector2D::vector(0.0, 1.0);
        let e12 = MultiVector2D::bivector(1.0);

        assert_approx_eq(e1 * e1, s);
        assert_approx_eq(e2 * e2, s);
        assert_approx_eq(e1 * e2, e12);
        assert_approx_eq(e2 * e1, MultiVector2D::bivector(-1.0));
        assert_approx_eq(e12 * e12, MultiVector2D::scalar(-1.0));
    }

    #[test]
    fn test_rotation() {
        // 90 derece döndürme rotoru
        let rotor = MultiVector2D::rotor(PI / 2.0);
        let vector = MultiVector2D::vector(1.0, 0.0); // x-ekseni yönünde vektör
        
        let rotated = rotor.rotate_vector(&vector);
        
        // 90 derece döndürülmüş vektör yaklaşık (0, 1) olmalı
        // DÜZELTME: Beklenen değer (0, 1) olmalı
        assert_approx_eq(rotated, MultiVector2D::vector(0.0, 1.0));
    }

    // Yeni test: 45 derece döndürme
    #[test]
    fn test_rotation_45() {
        let rotor = MultiVector2D::rotor(PI / 4.0);
        let vector = MultiVector2D::vector(1.0, 0.0);
        
        let rotated = rotor.rotate_vector(&vector);
        
        // 45 derece döndürülmüş vektör (cos45, sin45) ≈ (0.707, 0.707)
        let expected = MultiVector2D::vector(0.7071067811865475, 0.7071067811865475);
        assert_approx_eq(rotated, expected);
    }
}