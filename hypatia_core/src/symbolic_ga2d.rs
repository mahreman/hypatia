use crate::multivector2d::MultiVector2D;
use crate::symbolic::Symbol;
// ✅ DÜZELTME: Gereksiz import'lar kaldırıldı.

// 2D Sembolik Geometrik Cebir Fonksiyonları
// (Tüm `impl Mul` içeriği `multivector2d.rs` dosyasına taşındı)

impl MultiVector2D<Symbol> {
    pub fn simplify(&self) -> Self {
        Self {
            s: self.s.simplify(),
            e1: self.e1.simplify(),
            e2: self.e2.simplify(),
            e12: self.e12.simplify(),
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
    fn symbolic_2d_basis_rules() {
        let _s = MultiVector2D::<Symbol>::scalar(Symbol::c(1.0)); // Uyarıyı düzelt
        let e1 = MultiVector2D::<Symbol>::vector(Symbol::c(1.0), Symbol::c(0.0));
        let e2 = MultiVector2D::<Symbol>::vector(Symbol::c(0.0), Symbol::c(1.0));
        let e12 = MultiVector2D::<Symbol>::bivector(Symbol::c(1.0));

        let e1_sq = (e1.clone() * e1.clone()).simplify();
        assert_eq!(e1_sq.s, Symbol::c(1.0));
        assert_eq!(e1_sq.e1, Symbol::c(0.0));

        let e1_e2 = (e1.clone() * e2.clone()).simplify();
        assert_eq!(e1_e2.s, Symbol::c(0.0));
        assert_eq!(e1_e2.e12, Symbol::c(1.0));

        let e2_e1 = (e2.clone() * e1.clone()).simplify();
        assert_eq!(e2_e1.e12, Symbol::c(-1.0));
        
        let e12_sq = (e12.clone() * e12.clone()).simplify();
        assert_eq!(e12_sq.s, Symbol::c(-1.0));
    }

    #[test]
    fn symbolic_2d_geometric_product() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
    
        let v1 = MultiVector2D::<Symbol>::vector(x.clone(), Symbol::c(0.0)); // x*e1
        let v2 = MultiVector2D::<Symbol>::vector(Symbol::c(2.0), y.clone()); // 2*e1 + y*e2
    
        let gp = (v1 * v2).simplify();
        
        // Geometric product hesabı:
        // a = [0, x, 0, 0]
        // b = [0, 2, y, 0]
        
        // s = (0*0) + (x*2) + (0*y) - (0*0) = 2x
        // e1 = (0*2) + (x*0) - (0*0) + (0*y) = 0
        // e2 = (0*y) + (x*0) + (0*0) - (0*2) = 0
        // e12 = (0*0) + (x*y) - (0*2) + (0*0) = xy

        // Güçlendirilmiş simplify sonuçları:
        assert_eq!(format!("{}", gp.s), "(mul x 2)");
        assert_eq!(format!("{}", gp.e1), "0"); 
        assert_eq!(format!("{}", gp.e2), "0"); 
        assert_eq!(format!("{}", gp.e12), "(mul x y)");
    }
}