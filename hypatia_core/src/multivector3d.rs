use std::ops::{Add, Mul, Sub, Neg};
use num_traits::{Zero, One}; 

/// 3D Geometrik Cebir Multivektörü (Cl_3,0,0)
#[derive(Debug, Clone, PartialEq)]
pub struct MultiVector3D<T = f64> {
    pub s: T,
    pub e1: T,
    pub e2: T,
    pub e3: T,
    pub e12: T,
    pub e23: T,
    pub e31: T,
    pub e123: T,
}

// ✅ DÜZELTME: Hata E0599. Jenerik (Generic) implementasyon bloğu
impl<T> MultiVector3D<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output =T> + Neg<Output = T> + Zero + One,
{
    pub fn new(s: T, e1: T, e2: T, e3: T, e12: T, e23: T, e31: T, e123: T) -> Self {
        Self { s, e1, e2, e3, e12, e23, e31, e123 }
    }
    pub fn scalar(s: T) -> Self {
        Self::new(s, T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero())
    }
    pub fn vector(x: T, y: T, z: T) -> Self {
        Self::new(T::zero(), x, y, z, T::zero(), T::zero(), T::zero(), T::zero())
    }
    pub fn bivector(e12: T, e23: T, e31: T) -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero(), e12, e23, e31, T::zero())
    }
    pub fn trivector(e123: T) -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), T::zero(), e123)
    }
    pub fn to_coeffs(&self) -> [T; 8] {
        [
            self.s.clone(), self.e1.clone(), self.e2.clone(), self.e3.clone(),
            self.e12.clone(), self.e23.clone(), self.e31.clone(), self.e123.clone(),
        ]
    }
    pub fn from_coeffs(coeffs: &[T; 8]) -> Self {
        Self::new(
            coeffs[0].clone(), coeffs[1].clone(), coeffs[2].clone(), coeffs[3].clone(),
            coeffs[4].clone(), coeffs[5].clone(), coeffs[6].clone(), coeffs[7].clone(),
        )
    }
}

// Sadece f64'e özgü implementasyon bloğu
impl MultiVector3D<f64> {
    pub fn rotor(theta: f64, ax: f64, ay: f64, az: f64) -> Self {
        let (s, c) = (theta / 2.0).sin_cos();
        Self::new(
            c, 0.0, 0.0, 0.0,
            -s * az, -s * ax, -s * ay,
            0.0
        )
    }
    
    pub fn rotate_vector(&self, v: &Self) -> Self {
        let r_inv = Self::new(
            self.s, self.e1, self.e2, self.e3,
            -self.e12, -self.e23, -self.e31, -self.e123
        );
        self.clone() * v.clone() * r_inv // ✅ DÜZELTME: Hata E0369
    }
    
    pub fn grade(&self, k: u8) -> Self {
        match k {
            0 => Self::scalar(self.s),
            1 => Self::vector(self.e1, self.e2, self.e3),
            2 => Self::bivector(self.e12, self.e23, self.e31),
            3 => Self::trivector(self.e123),
            _ => Self::zero(),
        }
    }
}

// ============================================================================
// Trait Implementasyonları (Jenerik)
// ============================================================================

// ✅ DÜZELTME: Hata E0277. Zero, Add ve Clone gerektirir.
impl<T: Zero + Clone + Add<Output = T>> Zero for MultiVector3D<T> {
    fn zero() -> Self {
        Self {
            s: T::zero(), e1: T::zero(), e2: T::zero(), e3: T::zero(),
            e12: T::zero(), e23: T::zero(), e31: T::zero(), e123: T::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.s.is_zero() && self.e1.is_zero() && self.e2.is_zero() && self.e3.is_zero() &&
        self.e12.is_zero() && self.e23.is_zero() && self.e31.is_zero() && self.e123.is_zero()
    }
}

impl<T: Clone + Add<Output = T>> Add for MultiVector3D<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            s: self.s + rhs.s, e1: self.e1 + rhs.e1,
            e2: self.e2 + rhs.e2, e3: self.e3 + rhs.e3,
            e12: self.e12 + rhs.e12, e23: self.e23 + rhs.e23,
            e31: self.e31 + rhs.e31, e123: self.e123 + rhs.e123,
        }
    }
}

// ✅ DÜZELTME: Hata (Mantık): `basis_blade_gp_3d` tablosu düzeltildi.
fn basis_blade_gp_3d(i: usize, j: usize) -> (usize, f64) {
    match (i, j) {
        (0, k) => (k, 1.0), (k, 0) => (k, 1.0),
        (1, 1) => (0, 1.0), (2, 2) => (0, 1.0), (3, 3) => (0, 1.0),
        (4, 4) => (0, -1.0), (5, 5) => (0, -1.0), (6, 6) => (0, -1.0),
        (7, 7) => (0, -1.0), // e123*e123 = -1
        (1, 2) => (4, 1.0), (2, 1) => (4, -1.0),
        (2, 3) => (5, 1.0), (3, 2) => (5, -1.0),
        (3, 1) => (6, 1.0), (1, 3) => (6, -1.0),
        (1, 4) => (2, 1.0), (4, 1) => (2, -1.0),
        (1, 5) => (7, -1.0), (5, 1) => (7, 1.0),
        (1, 6) => (3, -1.0), (6, 1) => (3, 1.0),
        (2, 4) => (1, -1.0), (4, 2) => (1, 1.0),
        (2, 5) => (3, 1.0), (5, 2) => (3, -1.0),
        (2, 6) => (7, 1.0), (6, 2) => (7, -1.0),
        (3, 4) => (7, 1.0), (4, 3) => (7, -1.0), // ✅ HATA BURADAYDI -> (7, 1.0)
        (3, 5) => (2, -1.0), (5, 3) => (2, 1.0),
        (3, 6) => (1, 1.0), (6, 3) => (1, -1.0),
        (1, 7) => (5, 1.0), (7, 1) => (5, -1.0),
        (2, 7) => (6, -1.0), (7, 2) => (6, 1.0),
        (3, 7) => (4, 1.0), (7, 3) => (4, -1.0),
        (4, 5) => (6, 1.0), (5, 4) => (6, -1.0),
        (5, 6) => (4, 1.0), (6, 5) => (4, -1.0),
        (6, 4) => (5, 1.0), (4, 6) => (5, -1.0),
        (4, 7) => (3, -1.0), (7, 4) => (3, 1.0),
        (5, 7) => (1, -1.0), (7, 5) => (1, 1.0),
        (6, 7) => (2, 1.0), (7, 6) => (2, -1.0),
        _ => (0, 0.0),
    }
}

// ✅ DÜZELTME: Hata E0599. `impl Mul` artık `Zero` ve `One` kısıtlamalarını içeriyor.
impl<T> Mul for MultiVector3D<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T> + Zero + One,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.to_coeffs();
        let b = rhs.to_coeffs();
        let mut c = [
            T::zero(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::zero(),
        ];
        
        for i in 0..8 {
            for j in 0..8 {
                let (k, s) = basis_blade_gp_3d(i, j);
                if s != 0.0 {
                    let term = a[i].clone() * b[j].clone();
                    if s == 1.0 {
                        c[k] = c[k].clone() + term;
                    } else { // s == -1.0
                        c[k] = c[k].clone() - term;
                    }
                }
            }
        }
        Self::from_coeffs(&c)
    }
}

// ============================================================================
// TESTLER (✅ DÜZELTİLDİ)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    
    //use std::f64::consts::PI;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn basis_rules() {
        // e1, e2, e3
        let e1 = MultiVector3D::<f64>::vector(1.0, 0.0, 0.0).grade(1);
        let e2 = MultiVector3D::<f64>::vector(0.0, 1.0, 0.0).grade(1);
        let e3 = MultiVector3D::<f64>::vector(0.0, 0.0, 1.0).grade(1);

        // e1*e1 = 1, e2*e2 = 1, e3*e3 = 1
        let e1e1 = (e1.clone() * e1.clone()).grade(0);
        let e2e2 = (e2.clone() * e2.clone()).grade(0);
        let e3e3 = (e3.clone() * e3.clone()).grade(0);
        assert!(approx(e1e1.s, 1.0, 1e-12));
        assert!(approx(e2e2.s, 1.0, 1e-12));
        assert!(approx(e3e3.s, 1.0, 1e-12));

        // e1*e2 =  e12,  e2*e1 = -e12
        let e1e2 = (e1.clone() * e2.clone()).grade(2);
        let e2e1 = (e2.clone() * e1.clone()).grade(2);
        assert!(approx(e1e2.e12, 1.0, 1e-12));
        assert!(approx(e2e1.e12, -1.0, 1e-12));

        // e2*e3 =  e23,  e3*e2 = -e23
        let e2e3 = (e2.clone() * e3.clone()).grade(2);
        let e3e2 = (e3.clone() * e2.clone()).grade(2);
        assert!(approx(e2e3.e23, 1.0, 1e-12));
        assert!(approx(e3e2.e23, -1.0, 1e-12));

        // e3*e1 =  e31,  e1*e3 = -e31
        let e3e1 = (e3.clone() * e1.clone()).grade(2);
        let e1e3 = (e1.clone() * e3.clone()).grade(2);
        assert!(approx(e3e1.e31, 1.0, 1e-12));
        assert!(approx(e1e3.e31, -1.0, 1e-12));
    }

    #[test]
    fn rotor_rotation_isometry() {
        let v = MultiVector3D::vector(1.0, 0.0, 0.0); // e1
        let r = MultiVector3D::rotor(std::f64::consts::PI / 2.0, 0.0, 0.0, 1.0); 
        let vp = r.rotate_vector(&v).grade(1);
        assert!((vp.e2 - 1.0).abs() < 1e-10);
        assert!(vp.e1.abs() < 1e-10);
    }
    
    #[test]
    fn rotate_axes() {
        let e1 = MultiVector3D::vector(1.0, 0.0, 0.0);
        let e2 = MultiVector3D::vector(0.0, 1.0, 0.0);
        
        let r_x = MultiVector3D::rotor(std::f64::consts::PI / 2.0, 1.0, 0.0, 0.0); 
        let e2_p = r_x.rotate_vector(&e2).grade(1);
        assert!((e2_p.e3 - 1.0).abs() < 1e-10);
        
        let r_y = MultiVector3D::rotor(std::f64::consts::PI / 2.0, 0.0, 1.0, 0.0); 
        let e1_p = r_y.rotate_vector(&e1).grade(1);
        assert!((e1_p.e3 - (-1.0)).abs() < 1e-10);
    }
}