use std::ops::{Add, Mul, Sub, Neg};
use std::fmt;

/// 3D Geometrik Cebir Multivector'ı
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiVector3D {
    pub s: f64,      // Skaler
    pub e1: f64,     // Vektör bileşenleri
    pub e2: f64, 
    pub e3: f64,
    pub e12: f64,    // Bivector bileşenleri
    pub e23: f64,
    pub e31: f64,
    pub e123: f64,   // Trivector (pseudoscalar)
}

impl MultiVector3D {
    pub fn new(s: f64, e1: f64, e2: f64, e3: f64, e12: f64, e23: f64, e31: f64, e123: f64) -> Self {
        Self { s, e1, e2, e3, e12, e23, e31, e123 }
    }

    pub fn scalar(s: f64) -> Self {
        Self::new(s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn vector(e1: f64, e2: f64, e3: f64) -> Self {
        Self::new(0.0, e1, e2, e3, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn bivector(e12: f64, e23: f64, e31: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, e12, e23, e31, 0.0)
    }

    pub fn trivector(e123: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, e123)
    }

    /// Reverse operasyonu
    pub fn reverse(&self) -> Self {
        Self::new(
            self.s,
            self.e1, self.e2, self.e3,
            -self.e12, -self.e23, -self.e31,
            -self.e123
        )
    }

    /// Norm (büyüklük)
    pub fn norm(&self) -> f64 {
        (self.s * self.s + 
         self.e1 * self.e1 + self.e2 * self.e2 + self.e3 * self.e3 +
         self.e12 * self.e12 + self.e23 * self.e23 + self.e31 * self.e31 +
         self.e123 * self.e123).sqrt()
    }

    /// Normalize eder (birim multivektör)
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            return *self;
        }
        Self::new(
            self.s / norm,
            self.e1 / norm, self.e2 / norm, self.e3 / norm,
            self.e12 / norm, self.e23 / norm, self.e31 / norm,
            self.e123 / norm
        )
    }

    /// Eksene göre rotor oluşturma - DÜZELTİLDİ!
    pub fn rotor(angle_rad: f64, axis: (f64, f64, f64)) -> Self {
        let half_angle = angle_rad / 2.0;
        let (x, y, z) = axis;
        
        // Eksen vektörünü normalize et
        let norm = (x*x + y*y + z*z).sqrt();
        if norm == 0.0 {
            return Self::scalar(1.0);
        }
        
        let (ux, uy, uz) = (x/norm, y/norm, z/norm);
        let sin_half = half_angle.sin();
        
        // DÜZELTME: Doğru rotor formülü
        // R = cos(θ/2) - (ux*e23 + uy*e31 + uz*e12) * sin(θ/2)
        Self::new(
            half_angle.cos(),					// scalar
            0.0, 0.0, 0.0,						// vector components
            -uz * sin_half, 					// e12
            -ux * sin_half, 					// e23  
            -uy * sin_half, 					// e31
            0.0								// trivector
        )
    }

    /// Vektör döndürme
    pub fn rotate_vector(&self, vector: &MultiVector3D) -> Self {
        // v' = R * v * R_reverse
        (*self * *vector) * self.reverse()
    }
}

impl fmt::Display for MultiVector3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiVector3D(s: {:.3}, e1: {:.3}, e2: {:.3}, e3: {:.3}, e12: {:.3}, e23: {:.3}, e31: {:.3}, e123: {:.3})",
            self.s, self.e1, self.e2, self.e3, self.e12, self.e23, self.e31, self.e123
        )
    }
}

// Operatör overloading'leri (Aynı kalıyor)
impl Add for MultiVector3D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s + rhs.s,
            self.e1 + rhs.e1, self.e2 + rhs.e2, self.e3 + rhs.e3,
            self.e12 + rhs.e12, self.e23 + rhs.e23, self.e31 + rhs.e31,
            self.e123 + rhs.e123
        )
    }
}

impl Sub for MultiVector3D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s - rhs.s,
            self.e1 - rhs.e1, self.e2 - rhs.e2, self.e3 - rhs.e3,
            self.e12 - rhs.e12, self.e23 - rhs.e23, self.e31 - rhs.e31,
            self.e123 - rhs.e123
        )
    }
}

impl Neg for MultiVector3D {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(
            -self.s,
            -self.e1, -self.e2, -self.e3,
            -self.e12, -self.e23, -self.e31,
            -self.e123
        )
    }
}

// 3D GEOMETRİK ÇARPIM (Aynı kalıyor)
impl Mul for MultiVector3D {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self;
        let b = rhs;

        let s = a.s*b.s + a.e1*b.e1 + a.e2*b.e2 + a.e3*b.e3 
               - a.e12*b.e12 - a.e23*b.e23 - a.e31*b.e31 - a.e123*b.e123;

        let e1 = a.s*b.e1 + a.e1*b.s - a.e2*b.e12 + a.e3*b.e31 
                + a.e12*b.e2 - a.e23*b.e123 - a.e31*b.e3 - a.e123*b.e23;

        let e2 = a.s*b.e2 + a.e1*b.e12 + a.e2*b.s - a.e3*b.e23 
                - a.e12*b.e1 + a.e23*b.e3 - a.e31*b.e123 - a.e123*b.e31;

        let e3 = a.s*b.e3 - a.e1*b.e31 + a.e2*b.e23 + a.e3*b.s 
                - a.e12*b.e123 - a.e23*b.e2 + a.e31*b.e1 - a.e123*b.e12;

        let e12 = a.s*b.e12 + a.e1*b.e2 - a.e2*b.e1 + a.e3*b.e123 
                 + a.e12*b.s - a.e23*b.e31 + a.e31*b.e23 + a.e123*b.e3;

        let e23 = a.s*b.e23 + a.e1*b.e123 + a.e2*b.e3 - a.e3*b.e2 
                 + a.e12*b.e31 + a.e23*b.s - a.e31*b.e12 + a.e123*b.e1;

        let e31 = a.s*b.e31 - a.e1*b.e3 + a.e2*b.e123 + a.e3*b.e1 
                 - a.e12*b.e23 + a.e23*b.e12 + a.e31*b.s + a.e123*b.e2;

        let e123 = a.s*b.e123 + a.e1*b.e23 + a.e2*b.e31 + a.e3*b.e12 
                  + a.e12*b.e3 + a.e23*b.e1 + a.e31*b.e2 + a.e123*b.s;

        Self::new(s, e1, e2, e3, e12, e23, e31, e123)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_approx_eq_3d(a: MultiVector3D, b: MultiVector3D) {
        let epsilon = 1e-10;
        assert!((a.s - b.s).abs() < epsilon, "s: {} != {}", a.s, b.s);
        assert!((a.e1 - b.e1).abs() < epsilon, "e1: {} != {}", a.e1, b.e1);
        assert!((a.e2 - b.e2).abs() < epsilon, "e2: {} != {}", a.e2, b.e2);
        assert!((a.e3 - b.e3).abs() < epsilon, "e3: {} != {}", a.e3, b.e3);
        assert!((a.e12 - b.e12).abs() < epsilon, "e12: {} != {}", a.e12, b.e12);
        assert!((a.e23 - b.e23).abs() < epsilon, "e23: {} != {}", a.e23, b.e23);
        assert!((a.e31 - b.e31).abs() < epsilon, "e31: {} != {}", a.e31, b.e31);
        assert!((a.e123 - b.e123).abs() < epsilon, "e123: {} != {}", a.e123, b.e123);
    }

    #[test]
    fn test_3d_basis_multiplication() {
        let s = MultiVector3D::scalar(1.0);
        let e1 = MultiVector3D::vector(1.0, 0.0, 0.0);
        let e2 = MultiVector3D::vector(0.0, 1.0, 0.0);
        let e3 = MultiVector3D::vector(0.0, 0.0, 1.0);

        assert_approx_eq_3d(e1 * e1, s);
        assert_approx_eq_3d(e2 * e2, s);
        assert_approx_eq_3d(e3 * e3, s);
        assert_approx_eq_3d(e1 * e2, MultiVector3D::bivector(1.0, 0.0, 0.0));
        assert_approx_eq_3d(e2 * e3, MultiVector3D::bivector(0.0, 1.0, 0.0));
        assert_approx_eq_3d(e3 * e1, MultiVector3D::bivector(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_3d_rotation() {
        // 90 derece z-ekseni etrafında döndürme
        let rotor = MultiVector3D::rotor(PI / 2.0, (0.0, 0.0, 1.0));
        let vector = MultiVector3D::vector(1.0, 0.0, 0.0);
        
        let rotated = rotor.rotate_vector(&vector);
        
        // (1,0,0) vektörü z-ekseni etrafında 90° döndürülmeli → (0,1,0)
        assert_approx_eq_3d(rotated, MultiVector3D::vector(0.0, 1.0, 0.0));
    }

    // Yeni test: X ekseni etrafında döndürme
    #[test]
    fn test_3d_rotation_x_axis() {
        let rotor = MultiVector3D::rotor(PI / 2.0, (1.0, 0.0, 0.0));
        let vector = MultiVector3D::vector(0.0, 1.0, 0.0);
        
        let rotated = rotor.rotate_vector(&vector);
        
        // (0,1,0) vektörü x-ekseni etrafında 90° döndürülmeli → (0,0,1)
        assert_approx_eq_3d(rotated, MultiVector3D::vector(0.0, 0.0, 1.0));
    }
}