// src/lib.rs

use std::ops::{Add, Mul, Sub};

/// Hypatia'nın temel 2D MultiVektör yapısı.
/// 2D Geometrik Cebir'in 4 temel bileşenini içerir:
/// 1 (scalar)
/// e1, e2 (vector components)
/// e1^e2 (bivector/pseudoscalar)
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
    /// Bu, döndürme işlemi için gereklidir.
    pub fn reverse(&self) -> Self {
        // Vektör bileşenleri değil, bivektör bileşenleri işaret değiştirir.
        Self::new(self.s, self.e1, self.e2, -self.e12)
    }
}

// Toplama (+) operatörünü implemente et
impl Add for MultiVector2D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            s: self.s + rhs.s,
            e1: self.e1 + rhs.e1,
            e2: self.e2 + rhs.e2,
            e12: self.e12 + rhs.e12,
        }
    }
}

// Çıkarma (-) operatörünü implemente et
impl Sub for MultiVector2D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            s: self.s - rhs.s,
            e1: self.e1 - rhs.e1,
            e2: self.e2 - rhs.e2,
            e12: self.e12 - rhs.e12,
        }
    }
}

// *** PROJENİN KALBİ: GEOMETRİK ÇARPIM (*) ***
// e1*e1 = 1
// e2*e2 = 1
// e1*e2 = -e2*e1 = e1^e2
impl Mul for MultiVector2D {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // (a.s + a.e1 + a.e2 + a.e12) * (b.s + b.e1 + b.e2 + b.e12)
        
        let a = self;
        let b = rhs;

        // Her terimi tek tek çarpıp topluyoruz
        let s = (a.s * b.s)   + (a.e1 * b.e1) + (a.e2 * b.e2) - (a.e12 * b.e12);
        let e1 = (a.s * b.e1) + (a.e1 * b.s) - (a.e2 * b.e12) + (a.e12 * b.e2);
        let e2 = (a.s * b.e2) + (a.e1 * b.e12) + (a.e2 * b.s) - (a.e12 * b.e1);
        let e12 = (a.s * b.e12) + (a.e1 * b.e2) - (a.e2 * b.e1) + (a.e12 * b.s);

        Self::new(s, e1, e2, e12)
    }
}


// --- UNIT TESTLER ---
// Bu bölüm, kodun doğruluğunu kanıtlar.
#[cfg(test)]
mod tests {
    use super::*;

    // Yaklaşık eşitlik için bir helper
    fn assert_approx_eq(a: MultiVector2D, b: MultiVector2D) {
        let epsilon = 1e-10;
        assert!((a.s - b.s).abs() < epsilon, "Skalerler eşit değil: {} != {}", a.s, b.s);
        assert!((a.e1 - b.e1).abs() < epsilon, "e1'ler eşit değil: {} != {}", a.e1, b.e1);
        assert!((a.e2 - b.e2).abs() < epsilon, "e2'ler eşit değil: {} != {}", a.e2, b.e2);
        assert!((a.e12 - b.e12).abs() < epsilon, "e12'ler eşit değil: {} != {}", a.e12, b.e12);
    }

    #[test]
    fn test_basis_multiplication() {
        let s = MultiVector2D::scalar(1.0);
        let e1 = MultiVector2D::vector(1.0, 0.0);
        let e2 = MultiVector2D::vector(0.0, 1.0);
        let e12 = MultiVector2D::bivector(1.0);

        // e1 * e1 = 1
        assert_approx_eq(e1 * e1, s);

        // e2 * e2 = 1
        assert_approx_eq(e2 * e2, s);

        // e1 * e2 = e1^e2
        assert_approx_eq(e1 * e2, e12);

        // e2 * e1 = -e1^e2
        assert_approx_eq(e2 * e1, MultiVector2D::bivector(-1.0));

        // (e1^e2) * (e1^e2) = -1 (Kompleks sayılardaki i*i = -1 gibi)
        assert_approx_eq(e12 * e12, MultiVector2D::scalar(-1.0));
    }
}
