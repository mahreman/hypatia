use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MultiVector2D<T = f64> {
    pub s: T,
    pub e1: T,
    pub e2: T,
    pub e12: T,
}

impl<T> MultiVector2D<T> {
    pub fn new(s: T, e1: T, e2: T, e12: T) -> Self {
        Self { s, e1, e2, e12 }
    }
}

impl<T> Add for MultiVector2D<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s + rhs.s,
            self.e1 + rhs.e1,
            self.e2 + rhs.e2,
            self.e12 + rhs.e12,
        )
    }
}

impl<T> Sub for MultiVector2D<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.s - rhs.s,
            self.e1 - rhs.e1,
            self.e2 - rhs.e2,
            self.e12 - rhs.e12,
        )
    }
}

impl<T> Neg for MultiVector2D<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.s, -self.e1, -self.e2, -self.e12)
    }
}

impl MultiVector2D<f64> {
    #[inline]
    pub fn scalar(s: f64) -> Self {
        Self::new(s, 0.0, 0.0, 0.0)
    }
    #[inline]
    pub fn vector(x: f64, y: f64) -> Self {
        Self::new(0.0, x, y, 0.0)
    }
    #[inline]
    pub fn bivector(e12: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, e12)
    }

    #[inline]
    pub fn reverse(&self) -> Self {
        // k(k-1)/2: grade2 -> -, diğerleri +
        Self::new(self.s, self.e1, self.e2, -self.e12)
    }

    #[inline]
    pub fn grade(&self, k: u8) -> Self {
        match k {
            0 => Self::new(self.s, 0.0, 0.0, 0.0),
            1 => Self::new(0.0, self.e1, self.e2, 0.0),
            2 => Self::new(0.0, 0.0, 0.0, self.e12),
            _ => Self::default(),
        }
    }

    #[inline]
    pub fn rotor(theta: f64) -> Self {
        // R = cos(θ/2) - e12 sin(θ/2)  (CCW)
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        Self::new(c, 0.0, 0.0, -s)
    }

    #[inline]
    pub fn rotate_vector(&self, v: &Self) -> Self {
        // r v r~
        let r = *self;
        let rt = r.reverse();
        (r * *v * rt).grade(1)
    }

    // --- (EKLENDİ) Norm ve Birim Vektör ---

    /// Sadece vektör (grade 1) kısmının norm karesi
    #[inline]
    pub fn norm2(&self) -> f64 {
        self.e1 * self.e1 + self.e2 * self.e2
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
}

impl Mul for MultiVector2D<f64> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // Doğrudan ve hatasız 2D GA formülü (Cl(2,0))
        let s = self.s * rhs.s + self.e1 * rhs.e1 + self.e2 * rhs.e2 - self.e12 * rhs.e12;

        let e1 = self.s * rhs.e1 + self.e1 * rhs.s - self.e2 * rhs.e12 + self.e12 * rhs.e2;

        let e2 = self.s * rhs.e2 + self.e1 * rhs.e12 + self.e2 * rhs.s - self.e12 * rhs.e1;

        let e12 = self.s * rhs.e12 + self.e1 * rhs.e2 - self.e2 * rhs.e1 + self.e12 * rhs.s;

        Self::new(s, e1, e2, e12)
    }
}

impl Mul<f64> for MultiVector2D<f64> {
    type Output = Self;
    fn mul(self, k: f64) -> Self::Output {
        Self::new(self.s * k, self.e1 * k, self.e2 * k, self.e12 * k)
    }
}
impl Div<f64> for MultiVector2D<f64> {
    type Output = Self;
    fn div(self, k: f64) -> Self::Output {
        Self::new(self.s / k, self.e1 / k, self.e2 / k, self.e12 / k)
    }
}

// ---- Testler ----
#[cfg(test)]
mod tests {
    use super::*;

    fn eq(a: &MultiVector2D<f64>, b: &MultiVector2D<f64>) {
        assert!((a.s - b.s).abs() < 1e-12);
        assert!((a.e1 - b.e1).abs() < 1e-12);
        assert!((a.e2 - b.e2).abs() < 1e-12);
        assert!((a.e12 - b.e12).abs() < 1e-12);
    }

    #[test]
    fn basis_mul() {
        let one = MultiVector2D::<f64>::scalar(1.0);
        let e1 = MultiVector2D::<f64>::vector(1.0, 0.0);
        let e2 = MultiVector2D::<f64>::vector(0.0, 1.0);
        let e12 = MultiVector2D::<f64>::bivector(1.0);

        eq(&(e1 * e1), &one);
        eq(&(e2 * e2), &one);
        eq(&(e1 * e2), &e12);
        eq(&(e2 * e1), &-e12);
        eq(&(e12 * e12), &MultiVector2D::<f64>::scalar(-1.0));
    }

    #[test]
    fn wedge_dot() {
        let a = MultiVector2D::<f64>::vector(1.0, 2.0);
        let b = MultiVector2D::<f64>::vector(3.0, -1.0);
        // a.b = 3 - 2 = 1
        // a^b = -1 - 6 = -7
        let wedge = MultiVector2D::<f64>::bivector(-7.0);
        eq(&(a ^ b), &wedge);

        let ab = a * b;
        eq(&ab.grade(0), &MultiVector2D::<f64>::scalar(1.0));
        eq(&ab.grade(2), &wedge);
    }

    #[test]
    fn rotate_90_ccw() {
        // (GÜNCELLENDİ) PI sabiti
        let r = MultiVector2D::<f64>::rotor(std::f64::consts::PI / 2.0);
        let x = MultiVector2D::<f64>::vector(1.0, 0.0);
        let y = r.rotate_vector(&x).grade(1);
        eq(&y, &MultiVector2D::<f64>::vector(0.0, 1.0));
    }

    #[test]
    fn rotor_rotation_isometry() {
        let v = MultiVector2D::<f64>::vector(1.2, -0.7);
        let r = MultiVector2D::<f64>::rotor(1.2345);
        let w = r.rotate_vector(&v).grade(1);
        // (GÜNCELLENDİ) Lokal norm2 tanımı yerine self.norm2() metodu kullanıldı
        assert!((v.norm2() - w.norm2()).abs() < 1e-9);
    }
}

// basit ^ (vektör ^ vektör)
impl std::ops::BitXor for MultiVector2D<f64> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let e12 = self.e1 * rhs.e2 - self.e2 * rhs.e1;
        Self::bivector(e12)
    }
}