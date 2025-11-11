use std::ops::{Add, AddAssign, BitXor, Div, Mul, Neg, Sub, SubAssign};

/// Geometric Algebra (Cl(3,0)) multivector
///
/// Alan sırası (saklanan alanlar):
///   s, e1, e2, e3, e12, e23, e31, e123
///
/// İç hesaplama için kullandığımız dizi sırası:
///   [s, e1, e2, e12, e3, e13, e23, e123]
/// Burada e13 = -e31. to_coeffs / from_coeffs bu dönüşümü yapar.
#[derive(Debug, Copy, Clone, PartialEq, Default)]
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

impl MultiVector3D<f64> {
    #[inline]
    pub fn new(s: f64, e1: f64, e2: f64, e3: f64, e12: f64, e23: f64, e31: f64, e123: f64) -> Self {
        Self {
            s,
            e1,
            e2,
            e3,
            e12,
            e23,
            e31,
            e123,
        }
    }

    #[inline]
    pub fn scalar(s: f64) -> Self {
        Self::new(s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        Self::new(0.0, x, y, z, 0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn bivector(e12: f64, e23: f64, e31: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, e12, e23, e31, 0.0)
    }

    #[inline]
    pub fn trivector(e123: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, e123)
    }

    /// Reverse (grade k -> (-1)^{k(k-1)/2})
    #[inline]
    pub fn reverse(&self) -> Self {
        Self::new(
            self.s, self.e1, self.e2, self.e3, -self.e12, -self.e23, -self.e31, -self.e123,
        )
    }

    /// Grade seçimi (0..=3)
    #[inline]
    pub fn grade(&self, k: u8) -> Self {
        match k {
            0 => Self::new(self.s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1 => Self::new(0.0, self.e1, self.e2, self.e3, 0.0, 0.0, 0.0, 0.0),
            2 => Self::new(0.0, 0.0, 0.0, 0.0, self.e12, self.e23, self.e31, 0.0),
            3 => Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.e123),
            _ => Self::default(),
        }
    }

    // (KALDIRILDI) Kafa karıştırıcı statik norm2 fonksiyonu.
    /*
    /// vektör norm karesi (dot ile)
    #[inline]
    pub fn norm2(v: &Self) -> f64 {
        v.e1 * v.e1 + v.e2 * v.e2 + v.e3 * v.e3
    }
    */

    // --- (EKLENDİ) Norm ve Birim Vektör Metotları ---

    /// Sadece vektör (grade 1) kısmının norm karesi
    #[inline]
    pub fn norm2(&self) -> f64 {
        self.e1 * self.e1 + self.e2 * self.e2 + self.e3 * self.e3
    }

    /// Sadece vektör (grade 1) kısmının normu
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm2().sqrt()
    }

    /// Multivektörü, vektör (grade 1) kısmının normuna böler
    #[inline]
    pub fn unit(&self) -> Self {
        let n = self.norm();
        if n > 0.0 {
            *self / n
        } else {
            Self::default()
        }
    }

    /// Rotor: R = cos(θ/2) - (u_x e23 + u_y e31 + u_z e12) sin(θ/2)
    /// axis = (ax, ay, az)
    #[inline]
    pub fn rotor(theta: f64, ax: f64, ay: f64, az: f64) -> Self {
        let n = (ax * ax + ay * ay + az * az).sqrt();
        let (ux, uy, uz) = if n > 0.0 {
            (ax / n, ay / n, az / n)
        } else {
            // Eksensiz rotasyon (n=0) tanımsızdır, Z eksenini varsayalım
            (0.0, 0.0, 1.0)
        };
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();

        // e23: -ux*s, e31: -uy*s, e12: -uz*s
        Self::new(c, 0.0, 0.0, 0.0, -uz * s, -ux * s, -uy * s, 0.0)
    }

    /// R v R~ (sonra grade(1))
    #[inline]
    pub fn rotate_vector(&self, v: &Self) -> Self {
        let w = (*self * *v * self.reverse()).grade(1);
        w
    }

    // ----------------- İç temsil -----------------

    /// İç dizisel temsile dönüştür (bkz. dosya başındaki açıklama).
    /// Sıra: [s, e1, e2, e12, e3, e13, e23, e123], e13 = -e31
    #[inline]
    fn to_coeffs(&self) -> [f64; 8] {
        [
            self.s, self.e1, self.e2, self.e12, self.e3, -self.e31, // e13
            self.e23, self.e123,
        ]
    }

    /// İçten geri dön (NOT: e31 = -e13)
    /// Girdi sırası: [s, e1, e2, e12, e3, e13, e23, e123]
    #[inline]
    fn from_coeffs(c: [f64; 8]) -> Self {
        Self::new(c[0], c[1], c[2], c[4], c[3], c[6], -c[5], c[7])
    }

    /// bitmask işareti: (-1)^{swap sayısı}, swap sayısı
    /// a'nın her 1 biti için, b'de o bitten küçük indeksli bit sayısı kadar swap olur.
    #[inline]
    fn sign_3d(a: u8, b: u8) -> f64 {
        let mut n = 0u32;
        let mut i = 0u8;
        while i < 3 {
            if (a & (1 << i)) != 0 {
                let mask = (1u8 << i) - 1;
                n += ((b & mask) as u32).count_ones();
            }
            i += 1;
        }
        if (n & 1) == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

// --------- Skalerle çarpma / bölme ---------

impl Mul<f64> for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.s * rhs,
            self.e1 * rhs,
            self.e2 * rhs,
            self.e3 * rhs,
            self.e12 * rhs,
            self.e23 * rhs,
            self.e31 * rhs,
            self.e123 * rhs,
        )
    }
}

impl Div<f64> for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        let inv = 1.0 / rhs;
        self * inv
    }
}

// --------- Toplama / çıkarma / neg ---------

impl Add for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s + rhs.s,
            self.e1 + rhs.e1,
            self.e2 + rhs.e2,
            self.e3 + rhs.e3,
            self.e12 + rhs.e12,
            self.e23 + rhs.e23,
            self.e31 + rhs.e31,
            self.e123 + rhs.e123,
        )
    }
}

impl Sub for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s - rhs.s,
            self.e1 - rhs.e1,
            self.e2 - rhs.e2,
            self.e3 - rhs.e3,
            self.e12 - rhs.e12,
            self.e23 - rhs.e23,
            self.e31 - rhs.e31,
            self.e123 - rhs.e123,
        )
    }
}

impl AddAssign for MultiVector3D<f64> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for MultiVector3D<f64> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(
            -self.s, -self.e1, -self.e2, -self.e3, -self.e12, -self.e23, -self.e31, -self.e123,
        )
    }
}

// --------- Geometric Product (Cl(3,0)) ---------

impl Mul for MultiVector3D<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // İç temsile aktar
        let a = self.to_coeffs(); // [s, e1, e2, e12, e3, e13, e23, e123]
        let b = rhs.to_coeffs();

        let mut r = [0.0f64; 8];

        // mask == index (0..7), e1->1, e2->2, e3->4
        for i in 0..8 {
            let ai = a[i];
            if ai == 0.0 {
                continue;
            }
            let a_mask = i as u8;

            for j in 0..8 {
                let bj = b[j];
                if bj == 0.0 {
                    continue;
                }
                let b_mask = j as u8;

                let k_mask = a_mask ^ b_mask;
                let sign = Self::sign_3d(a_mask, b_mask);
                r[k_mask as usize] += ai * bj * sign;
            }
        }

        Self::from_coeffs(r)
    }
}

// --------- Dış çarpım (vektör ^ vektör -> bivektör) ---------

impl BitXor for MultiVector3D<f64> {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        // Sadece 1-grade ^ 1-grade kısmı (gerekli olan kullanım)
        let e12 = self.e1 * rhs.e2 - self.e2 * rhs.e1;
        let e23 = self.e2 * rhs.e3 - self.e3 * rhs.e2;
        let e31 = self.e3 * rhs.e1 - self.e1 * rhs.e3;
        Self::bivector(e12, e23, e31)
    }
}

// ---------------------------- TESTS ----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // (GÜNCELLENDİ) PI sabiti
    use std::f64::consts::PI;

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
    fn rotate_axes() {
        // Z ekseni etrafında +90°
        // (GÜNCELLENDİ) PI sabiti
        let r = MultiVector3D::<f64>::rotor(PI / 2.0, 0.0, 0.0, 1.0);

        // x=(1,0,0)  -->  y=(0,1,0)
        let x = MultiVector3D::<f64>::vector(1.0, 0.0, 0.0);
        let y = r.rotate_vector(&x).grade(1);
        assert!(approx(y.e1, 0.0, 1e-12));
        assert!(approx(y.e2, 1.0, 1e-12));
        assert!(approx(y.e3, 0.0, 1e-12));

        // y=(0,1,0)  -->  (-1,0,0)
        let yv = MultiVector3D::<f64>::vector(0.0, 1.0, 0.0);
        let x2 = r.rotate_vector(&yv).grade(1);
        assert!(approx(x2.e1, -1.0, 1e-12));
        assert!(approx(x2.e2, 0.0, 1e-12));
        assert!(approx(x2.e3, 0.0, 1e-12));
    }

    #[test]
    fn rotor_rotation_isometry() {
        // (GÜNCELLENDİ) Lokal norm2 tanımı kaldırıldı, self.norm2() kullanılacak
        /*
        fn norm2(v: &MultiVector3D<f64>) -> f64 {
            let g1 = v.grade(1);
            g1.e1 * g1.e1 + g1.e2 * g1.e2 + g1.e3 * g1.e3
        }
        */

        let r = MultiVector3D::<f64>::rotor(PI / 2.0, 0.0, 0.0, 1.0);
        let v = MultiVector3D::<f64>::vector(0.4, -1.3, 2.2);
        let w = r.rotate_vector(&v).grade(1);

        assert!(approx(v.norm2(), w.norm2(), 1e-9));
    }
}