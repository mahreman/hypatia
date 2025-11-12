// src/symbolic_ga2d.rs

use crate::multivector2d::MultiVector2D;
use crate::symbolic::Symbol;

impl MultiVector2D<Symbol> {
    #[inline]
    pub fn scalar(s: Symbol) -> Self {
        Self::new(s, Symbol::c(0.0), Symbol::c(0.0), Symbol::c(0.0))
    }

    #[inline]
    pub fn vector(x: Symbol, y: Symbol) -> Self {
        Self::new(Symbol::c(0.0), x, y, Symbol::c(0.0))
    }

    #[inline]
    pub fn bivector(e12: Symbol) -> Self {
        Self::new(Symbol::c(0.0), Symbol::c(0.0), Symbol::c(0.0), e12)
    }

    /// MultiVector için simplify metodu (3D'deki gibi)
    pub fn simplify(&self) -> Self {
        Self {
            s: self.s.clone().simplify(),
            e1: self.e1.clone().simplify(),
            e2: self.e2.clone().simplify(),
            e12: self.e12.clone().simplify(),
        }
    }
}

// ... (to_coeffs, from_coeffs, zero, add, mul değişmedi) ...
#[inline]
fn to_coeffs(m: &MultiVector2D<Symbol>) -> [Symbol; 4] {
    [m.s.clone(), m.e1.clone(), m.e2.clone(), m.e12.clone()]
}

#[inline]
fn from_coeffs(c: [Symbol; 4]) -> MultiVector2D<Symbol> {
    MultiVector2D::new(c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone())
}

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

// ... (gp_sign_and_mask_2d, gp_blades_symbolic, Mul impl değişmedi) ...
#[inline]
fn gp_sign_and_mask_2d(a_mask: u8, b_mask: u8) -> (i32, u8) {
    let k_mask = a_mask ^ b_mask;
    let mut sign = 1;

    let mut swaps: i32 = 0;
    for i in 0..2 {
        if (a_mask & (1 << i)) != 0 {
            let lower = b_mask & ((1 << i) - 1);
            swaps += lower.count_ones() as i32;
        }
    }
    if (swaps & 1) != 0 {
        sign = -sign;
    }

    (sign, k_mask)
}


fn gp_blades_symbolic(a: &[Symbol; 4], b: &[Symbol; 4]) -> [Symbol; 4] {
    let mut out_fixed = [zero(), zero(), zero(), zero()];
    for i in 0..4 {
        for j in 0..4 {
            let (sgn, rm) = gp_sign_and_mask_2d(i as u8, j as u8);
            let mut term = mul(a[i].clone(), b[j].clone());
            if sgn == -1 {
                term = -term;
            }
            out_fixed[rm as usize] = add(out_fixed[rm as usize].clone(), term);
        }
    }
    out_fixed
}

impl std::ops::Mul for MultiVector2D<Symbol> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let a = to_coeffs(&self);
        let b = to_coeffs(&rhs);
        let c = gp_blades_symbolic(&a, &b);
        from_coeffs(c)
    }
}

#[cfg(test)]
mod tests {
    use crate::symbolic::Symbol;
    use crate::multivector2d::MultiVector2D;

    #[test]
    fn symbolic_2d_geometric_product() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");

        let v1 = MultiVector2D::<Symbol>::vector(x.clone(), y.clone());
        let v2 = MultiVector2D::<Symbol>::vector(Symbol::c(2.0), Symbol::c(3.0));

        let result = (v1 * v2).simplify();
        
        // s   = (2*x) + (3*y)
        // e12 = (3*x) - (2*y) = (3*x) + (-2*y)
        
        // (DÜZELTME) S-expression formatı
        assert_eq!(format!("{}", result.s), "(add (mul 2 x) (mul 3 y))");
        assert_eq!(format!("{}", result.e12), "(add (mul 3 x) (mul -2 y))");
    }
    
    /// İşaret mantığını doğrulamak için temel test
    #[test]
    fn symbolic_2d_basis_rules() {
        let e1 = MultiVector2D::<Symbol>::vector(Symbol::c(1.0), Symbol::c(0.0));
        let e2 = MultiVector2D::<Symbol>::vector(Symbol::c(0.0), Symbol::c(1.0));
        let e12 = MultiVector2D::<Symbol>::bivector(Symbol::c(1.0));
        
        let _one = MultiVector2D::<Symbol>::scalar(Symbol::c(1.0));
        let _neg_one = MultiVector2D::<Symbol>::scalar(Symbol::c(-1.0));

        assert_eq!(format!("{}", (e1.clone() * e1.clone()).simplify().s), "1");
        assert_eq!(format!("{}", (e2.clone() * e2.clone()).simplify().s), "1");
        assert_eq!(format!("{}", (e1.clone() * e2.clone()).simplify().e12), "1");
        assert_eq!(format!("{}", (e2.clone() * e1.clone()).simplify().e12), "-1");
        assert_eq!(format!("{}", (e12.clone() * e12.clone()).simplify().s), "-1");
    }
}