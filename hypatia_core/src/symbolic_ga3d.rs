use crate::multivector3d::MultiVector3D;
use crate::symbolic::Symbol;

// --------- Symbolik yapıcılar ---------
impl MultiVector3D<Symbol> {
    pub fn scalar(s: Symbol) -> Self {
        Self {
            s,
            e1: Symbol::c(0.0),
            e2: Symbol::c(0.0),
            e3: Symbol::c(0.0),
            e12: Symbol::c(0.0),
            e23: Symbol::c(0.0),
            e31: Symbol::c(0.0),
            e123: Symbol::c(0.0),
        }
    }
    pub fn vector(x: Symbol, y: Symbol, z: Symbol) -> Self {
        Self {
            s: Symbol::c(0.0),
            e1: x,
            e2: y,
            e3: z,
            e12: Symbol::c(0.0),
            e23: Symbol::c(0.0),
            e31: Symbol::c(0.0),
            e123: Symbol::c(0.0),
        }
    }
    pub fn bivector(e12: Symbol, e23: Symbol, e31: Symbol) -> Self {
        Self {
            s: Symbol::c(0.0),
            e1: Symbol::c(0.0),
            e2: Symbol::c(0.0),
            e3: Symbol::c(0.0),
            e12,
            e23,
            e31,
            e123: Symbol::c(0.0),
        }
    }
    pub fn trivector(e123: Symbol) -> Self {
        Self {
            s: Symbol::c(0.0),
            e1: Symbol::c(0.0),
            e2: Symbol::c(0.0),
            e3: Symbol::c(0.0),
            e12: Symbol::c(0.0),
            e23: Symbol::c(0.0),
            e31: Symbol::c(0.0),
            e123,
        }
    }

    /// Her bileşeni özyinelemeli olarak sadeleştirir.
    pub fn simplify(&self) -> Self {
        Self {
            s: self.s.clone().simplify(),
            e1: self.e1.clone().simplify(),
            e2: self.e2.clone().simplify(),
            e3: self.e3.clone().simplify(),
            e12: self.e12.clone().simplify(),
            e23: self.e23.clone().simplify(),
            e31: self.e31.clone().simplify(),
            e123: self.e123.clone().simplify(),
        }
    }
}

// --------- Bitmask sırası ve dönüşümler ---------
#[inline]
fn to_coeffs(m: &MultiVector3D<Symbol>) -> [Symbol; 8] {
    [
        m.s.clone(),    // 0
        m.e1.clone(),   // 1
        m.e2.clone(),   // 2
        m.e12.clone(),  // 3
        m.e3.clone(),   // 4
        -m.e31.clone(), // 5 (e13 = -e31)
        m.e23.clone(),  // 6
        m.e123.clone(), // 7
    ]
}

#[inline]
fn from_coeffs(c: [Symbol; 8]) -> MultiVector3D<Symbol> {
    MultiVector3D {
        s: c[0].clone(),
        e1: c[1].clone(),
        e2: c[2].clone(),
        e3: c[4].clone(),
        e12: c[3].clone(),
        e23: c[6].clone(),
        e31: -c[5].clone(), // e31 = -e13
        e123: c[7].clone(),
    }
}

// --------- Yardımcılar ---------

#[inline]
fn zero() -> Symbol {
    Symbol::c(0.0)
}

#[inline]
fn add(a: Symbol, b: Symbol) -> Symbol {
    (a + b).simplify()
}

#[inline]
fn mul(a: Symbol, b: Symbol) -> Symbol {
    (a * b).simplify()
}

#[inline]
fn gp_mask_sign_3d(a: u8, b: u8) -> (u8, i32) {
    let k = a ^ b;
    let mut swaps: i32 = 0;
    for i in 0..3 {
        if (a & (1 << i)) != 0 {
            let lower = b & ((1 << i) - 1);
            swaps += lower.count_ones() as i32;
        }
    }
    let sign = if (swaps & 1) == 0 { 1 } else { -1 };
    (k, sign)
}

fn gp_blades_symbolic(a: &[Symbol; 8], b: &[Symbol; 8]) -> [Symbol; 8] {
    let mut out = [
        zero(),
        zero(),
        zero(),
        zero(),
        zero(),
        zero(),
        zero(),
        zero(),
    ];
    for i in 0..8 {
        for j in 0..8 {
            let (rm, sgn) = gp_mask_sign_3d(i as u8, j as u8);
            let mut term = mul(a[i].clone(), b[j].clone());
            if sgn < 0 {
                term = -term;
            }
            out[rm as usize] = add(out[rm as usize].clone(), term);
        }
    }
    out
}

// --------- Geometric Product ---------

impl std::ops::Mul for MultiVector3D<Symbol> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = to_coeffs(&self);
        let b = to_coeffs(&rhs);
        let c = gp_blades_symbolic(&a, &b);
        from_coeffs(c)
    }
}

// --------- Basit test ---------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_3d_basis_rules() {
        // (DÜZELTME) S-expression formatı (değişiklik yok, 1 ve -1 sabit)
        let e1 = MultiVector3D::<Symbol>::vector(
            Symbol::c(1.0),
            Symbol::c(0.0),
            Symbol::c(0.0),
        );
        let one = MultiVector3D::<Symbol>::scalar(Symbol::c(1.0));
        let e1e1 = (e1.clone() * e1.clone()).simplify();
        
        assert_eq!(format!("{}", e1e1.s), format!("{}", one.s));

        let e2 = MultiVector3D::<Symbol>::vector(
            Symbol::c(0.0),
            Symbol::c(1.0),
            Symbol::c(0.0),
        );
        let e1e2 = (e1.clone() * e2.clone()).simplify();
        let e2e1 = (e2.clone() * e1.clone()).simplify();
        assert_eq!(format!("{}", e1e2.e12), "1");
        assert_eq!(format!("{}", e2e1.e12), "-1");
    }

    #[test]
    fn symbolic_3d_simplification() {
        // (DÜZELTME) S-expression formatı (değişiklik yok, 0 sabit)
        let x = Symbol::var("x");
        let _y = Symbol::var("y");
        let _z = Symbol::var("z");

        let zero_mv = MultiVector3D::<Symbol>::scalar(Symbol::c(0.0));
        let vector_x = MultiVector3D::<Symbol>::vector(x.clone(), Symbol::c(0.0), Symbol::c(0.0));
        let product = (zero_mv * vector_x).simplify();
        assert_eq!(format!("{}", product.s), "0");
        assert_eq!(format!("{}", product.e1), "0");
    }
}