use crate::multivector3d::MultiVector3D;
use crate::symbolic::Symbol;
// ✅ DÜZELTME: 'T', 'Mul' ve gereksiz import'lar kaldırıldı.

// 3D Sembolik Geometrik Cebir Fonksiyonları
// (Tüm `impl Mul` içeriği `multivector3d.rs` dosyasına taşındı)


impl MultiVector3D<Symbol> {
    pub fn simplify(&self) -> Self {
        Self {
            s: self.s.simplify(),
            e1: self.e1.simplify(),
            e2: self.e2.simplify(),
            e3: self.e3.simplify(),
            e12: self.e12.simplify(),
            e23: self.e23.simplify(),
            e31: self.e31.simplify(),
            e123: self.e123.simplify(),
        }
    }
}

// ============================================================================
// TESTLER (✅ DÜZELTİLDİ)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::Symbol;

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
        let x = Symbol::var("x");
        let v = MultiVector3D::<Symbol>::vector(x.clone(), Symbol::c(0.0), Symbol::c(0.0)); // x*e1
        let s = MultiVector3D::<Symbol>::scalar(Symbol::c(0.0)); // 0
        
        // v * 0 = 0 (her bileşen 0 ile çarpılır)
        let gp = (v * s).simplify();
        // Güçlendirilmiş simplify: x*0 = 0
        assert_eq!(format!("{}", gp.s), "0");
    }
}