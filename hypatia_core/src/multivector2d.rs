use std::ops::{Add, Mul, Sub, Neg};
use num_traits::{Zero, One}; 

#[derive(Debug, Clone, PartialEq)]
pub struct MultiVector2D<T = f64> {
    pub s: T,
    pub e1: T,
    pub e2: T,
    pub e12: T,
}

// ✅ DÜZELTME: Hata E0599. Jenerik (Generic) implementasyon bloğu
impl<T> MultiVector2D<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T> + Zero + One,
{
    pub fn new(s: T, e1: T, e2: T, e12: T) -> Self {
        Self { s, e1, e2, e12 }
    }
    pub fn scalar(s: T) -> Self {
        Self::new(s, T::zero(), T::zero(), T::zero())
    }
    pub fn vector(x: T, y: T) -> Self {
        Self::new(T::zero(), x, y, T::zero())
    }
    pub fn bivector(e12: T) -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), e12)
    }
    pub fn to_coeffs(&self) -> [T; 4] {
        [self.s.clone(), self.e1.clone(), self.e2.clone(), self.e12.clone()]
    }
    pub fn from_coeffs(coeffs: &[T; 4]) -> Self {
        Self::new(coeffs[0].clone(), coeffs[1].clone(), coeffs[2].clone(), coeffs[3].clone())
    }
}

// Sadece f64'e özgü implementasyon bloğu
impl MultiVector2D<f64> {
    pub fn rotor(theta: f64) -> Self {
        let (s, c) = (theta / 2.0).sin_cos();
        Self::new(c, 0.0, 0.0, -s)
    }
    pub fn rotate_vector(&self, v: &Self) -> Self {
        let r_inv = Self::new(self.s, self.e1, self.e2, -self.e12);
        self.clone() * v.clone() * r_inv // ✅ DÜZELTME: Hata E0369
    }
    pub fn grade(&self, k: u8) -> Self {
        match k {
            0 => Self::scalar(self.s),
            1 => Self::vector(self.e1, self.e2),
            2 => Self::bivector(self.e12),
            _ => Self::zero(),
        }
    }
}

// ============================================================================
// Trait Implementasyonları (Jenerik)
// ============================================================================

// ✅ DÜZELTME: Hata E0277. Zero, Add ve Clone gerektirir.
impl<T: Zero + Clone + Add<Output = T>> Zero for MultiVector2D<T> {
    fn zero() -> Self {
        Self {
            s: T::zero(),
            e1: T::zero(),
            e2: T::zero(),
            e12: T::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.s.is_zero() && self.e1.is_zero() && self.e2.is_zero() && self.e12.is_zero()
    }
}

impl<T: Clone + Add<Output = T>> Add for MultiVector2D<T> {
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

// ✅ DÜZELTME: Hata E0599. `impl Mul` artık `Zero` ve `One` kısıtlamalarını içeriyor.
impl<T> Mul for MultiVector2D<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T> + Zero + One,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.to_coeffs();
        let b = rhs.to_coeffs();

        let s = (a[0].clone() * b[0].clone())   // s*s
              + (a[1].clone() * b[1].clone())   // e1*e1
              + (a[2].clone() * b[2].clone())   // e2*e2
              - (a[3].clone() * b[3].clone());  // e12*e12
              
        let e1 = (a[0].clone() * b[1].clone())  // s*e1
               + (a[1].clone() * b[0].clone())  // e1*s
               - (a[2].clone() * b[3].clone())  // e2*e12
               + (a[3].clone() * b[2].clone()); // e12*e2
               
        let e2 = (a[0].clone() * b[2].clone())  // s*e2
               + (a[1].clone() * b[3].clone())  // e1*e12
               + (a[2].clone() * b[0].clone())  // e2*s
               - (a[3].clone() * b[1].clone()); // e12*e1
               
        let e12 = (a[0].clone() * b[3].clone()) // s*e12
                + (a[1].clone() * b[2].clone()) // e1*e2
                - (a[2].clone() * b[1].clone()) // e2*e1
                + (a[3].clone() * b[0].clone());// e12*s

        Self::new(s, e1, e2, e12)
    }
}


// ============================================================================
// TESTLER
// (Değişiklik yok)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_mul() {
        let e1 = MultiVector2D::vector(1.0, 0.0);
        let e2 = MultiVector2D::vector(0.0, 1.0);
        let e12 = MultiVector2D::bivector(1.0);

        let e1_sq = e1.clone() * e1.clone();
        assert_eq!(e1_sq.s, 1.0);
        assert_eq!(e1_sq.e1, 0.0);

        let e1_e2 = e1.clone() * e2.clone();
        assert_eq!(e1_e2.s, 0.0);
        assert_eq!(e1_e2.e12, 1.0);

        let e2_e1 = e2.clone() * e1.clone();
        assert_eq!(e2_e1.e12, -1.0);
        
        let e12_sq = e12.clone() * e12.clone();
        assert_eq!(e12_sq.s, -1.0);
    }

    #[test]
    fn wedge_dot() {
        let v1 = MultiVector2D::vector(1.0, 2.0);
        let v2 = MultiVector2D::vector(3.0, 4.0);
        let wedge = (v1.clone() * v2.clone()).grade(2);
        assert_eq!(wedge.e12, -2.0);
        let dot = (v1.clone() * v2.clone()).grade(0);
        assert_eq!(dot.s, 11.0);
    }
    
    #[test]
    fn rotor_rotation_isometry() {
        let v = MultiVector2D::vector(1.0, 0.0); // e1
        let r = MultiVector2D::rotor(std::f64::consts::PI / 2.0); // 90 derece
        let vp = r.rotate_vector(&v);
        assert!((vp.e2 - 1.0).abs() < 1e-10);
        assert!(vp.e1.abs() < 1e-10);
    }

    #[test]
    fn rotate_90_ccw() {
        let v = MultiVector2D::vector(2.0, 3.0);
        let r = MultiVector2D::rotor(std::f64::consts::PI / 2.0); // 90 derece
        let vp = r.rotate_vector(&v); // (2, 3) -> (-3, 2)
        assert!((vp.e1 - (-3.0)).abs() < 1e-10);
        assert!((vp.e2 - 2.0).abs() < 1e-10);
    }
}