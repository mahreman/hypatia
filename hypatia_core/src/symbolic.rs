use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Mul, Neg, Sub};

/// Basit sembolik ifade ağacı:
/// - Sabitler
/// - Değişkenler
/// - Toplama, Çarpma, Negasyon
#[derive(Clone, Debug, PartialEq)]
pub enum Symbol {
    Const(f64),
    Variable(String),
    Add(Box<Symbol>, Box<Symbol>),
    Mul(Box<Symbol>, Box<Symbol>),
    Neg(Box<Symbol>),
}

/* --------------------------
   Yardımcı (iç) fonksiyonlar
   -------------------------- */

fn fmt_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

impl From<f64> for Symbol {
    fn from(v: f64) -> Self {
        Symbol::Const(v)
    }
}

/* --------------------------
   Display (yazdırma biçimi)
   -------------------------- */

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use Symbol::*;
        match self {
            Const(c) => write!(f, "{}", fmt_number(*c)),
            Variable(name) => write!(f, "{}", name),
            Add(a, b) => write!(f, "({} + {})", a, b),
            Mul(a, b) => write!(f, "({} * {})", a, b),
            Neg(a) => write!(f, "-{}", a),
        }
    }
}

/* --------------------------
   Operatörler: +, *, - (unary)
   -------------------------- */

impl Add for Symbol {
    type Output = Symbol;
    fn add(self, rhs: Symbol) -> Symbol {
        use Symbol::*;
        match (&self, &rhs) {
            (Const(a), Const(b)) => Const(a + b),
            _ => Add(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Mul for Symbol {
    type Output = Symbol;
    fn mul(self, rhs: Symbol) -> Symbol {
        use Symbol::*;
        match (&self, &rhs) {
            (Const(a), Const(b)) => Const(a * b),
            _ => Mul(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Sub for Symbol {
    type Output = Symbol;
    fn sub(self, rhs: Symbol) -> Symbol {
        use Symbol::*;
        match (&self, &rhs) {
            (Const(a), Const(b)) => Const(a - b),
            // Diğer tüm durumlar için a + (-b) kuralını kullan
            _ => self + (-rhs),
        }
    }
}

impl Neg for Symbol {
    type Output = Symbol;
    fn neg(self) -> Symbol {
        use Symbol::*;
        match self {
            Const(c) => Const(-c),
            Neg(inner) => *inner,
            other => Neg(Box::new(other)),
        }
    }
}

/* --------------------------
   Türev (symbolic differentiation)
   -------------------------- */

impl Symbol {
    /// d/d(var) [self]
    pub fn derivative(&self, var: &str) -> Symbol {
        use Symbol::*;
        match self {
            Const(_) => Const(0.0),
            Variable(name) => {
                if name == var {
                    Const(1.0)
                } else {
                    Const(0.0)
                }
            }
            Add(a, b) => a.derivative(var) + b.derivative(var),
            Mul(a, b) => {
                // (a*b)' = a'*b + a*b'
                (a.derivative(var) * b.as_ref().clone())
                    + (a.as_ref().clone() * b.derivative(var))
            }
            Neg(a) => -(a.derivative(var)),
        }
    }

    /// İfade ağacını özyinelemeli olarak sadeleştirir.
    pub fn simplify(&self) -> Symbol {
        use Symbol::*;
        match self {
            Const(c) => Const(*c),
            Variable(v) => Variable(v.clone()),

            Add(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                match (&a_s, &b_s) {
                    (Const(0.0), x) => x.clone(),
                    (x, Const(0.0)) => x.clone(),
                    (Const(av), Const(bv)) => Const(av + bv),

                    // (DÜZELTME) BU KURAL KALDIRILDI
                    // 'Sub' implementasyonu (a + (-b)) ile çakışıp sonsuz döngü yaratıyordu.
                    // (x, Neg(y)) => (x.clone() - y.as_ref().clone()).simplify(),
                    
                    _ => {
                        // Eğer aynı ifadeler toplanıyorsa: x + x = 2*x
                        if a_s == b_s {
                            (Const(2.0) * a_s).simplify()
                        } else {
                            Add(Box::new(a_s), Box::new(b_s))
                        }
                    }
                }
            }

            Mul(a, b) => {
                let a_s = a.simplify();
                let b_s = b.simplify();
                match (&a_s, &b_s) {
                    (Const(0.0), _) => Const(0.0),
                    (_, Const(0.0)) => Const(0.0),
                    (Const(1.0), x) => x.clone(),
                    (x, Const(1.0)) => x.clone(),
                    (Const(av), Const(bv)) => Const(av * bv),
                    
                    // Kanonikleştirme: Sabitleri başa al (örn: x * 2 -> 2 * x)
                    (x, Const(c)) if !matches!(x, Const(_)) => {
                        Mul(Box::new(Const(*c)), Box::new(x.clone()))
                    }
                    
                    // (-x) * (-y) = x * y
                    (Neg(x), Neg(y)) => (x.as_ref().clone() * y.as_ref().clone()).simplify(),
                    // (-x) * y = -(x * y)
                    (Neg(x), y) => (-(x.as_ref().clone() * y.clone())).simplify(),
                    // x * (-y) = -(x * y)
                    (x, Neg(y)) => (-(x.clone() * y.as_ref().clone())).simplify(),
                    _ => Mul(Box::new(a_s), Box::new(b_s)),
                }
            }

            Neg(a) => {
                let a_s = a.simplify();
                match a_s {
                    Const(c) => Const(-c),
                    Neg(inner) => *inner,

                    // (DÜZELTME) YENİ KANONİKLEŞTİRME KURALI
                    // Negasyonu içeri dağıt: -(A * B) => (-A) * B
                    // Bu, '(-(2 * y))' yerine '(-2 * y)' formatını sağlar.
                    Mul(a_inner, b_inner) => {
                        ( (-a_inner.as_ref().clone()) * b_inner.as_ref().clone() ).simplify()
                    }

                    other => Neg(Box::new(other)),
                }
            }
        }
    }

    /// Değer verme (evaluate) / substitution
    pub fn subs(&self, env: &HashMap<String, f64>) -> Symbol {
        use Symbol::*;
        match self {
            Const(c) => Const(*c),
            Variable(name) => {
                if let Some(v) = env.get(name) {
                    Const(*v)
                } else {
                    Variable(name.clone())
                }
            }
            Add(a, b) => Add(Box::new(a.subs(env)), Box::new(b.subs(env))).simplify(),
            Mul(a, b) => Mul(Box::new(a.subs(env)), Box::new(b.subs(env))).simplify(),
            Neg(a) => Neg(Box::new(a.subs(env))).simplify(),
        }
    }
}

/* --------------------------
   Yardımcı yapım fonksiyonları
   -------------------------- */

impl Symbol {
    pub fn var(name: &str) -> Symbol {
        Symbol::Variable(name.to_string())
    }
    pub fn c(v: f64) -> Symbol {
        Symbol::Const(v)
    }
}

/* --------------------------
   TESTLER
   -------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ast() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let two = Symbol::c(2.0);

        let f = x.clone() * two.clone();
        assert_eq!(format!("{}", f), "(x * 2)");

        let g = x.clone() * y.clone() + two.clone();
        assert_eq!(format!("{}", g), "((x * y) + 2)");
    }

    #[test]
    fn derivative_of_linear_and_product() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let two = Symbol::c(2.0);

        // f(x) = x * 2 => f'(x) = 2
        let f = x.clone() * two.clone();
        let df_dx = f.derivative("x").simplify();
        assert_eq!(format!("{}", df_dx), "2");

        // g(x,y) = x*y => dg/dx = y, dg/dy = x
        let g = x.clone() * y.clone();
        let dg_dx = g.derivative("x").simplify();
        let dg_dy = g.derivative("y").simplify();
        assert_eq!(format!("{}", dg_dx), "y");
        assert_eq!(format!("{}", dg_dy), "x");
    }

    #[test]
    fn simplify_advanced() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        
        // x * 0 = 0
        let expr1 = x.clone() * Symbol::c(0.0);
        assert_eq!(format!("{}", expr1.simplify()), "0");
        
        // x + x = 2*x
        let expr2 = x.clone() + x.clone();
        assert_eq!(format!("{}", expr2.simplify()), "(2 * x)");
        
        // -(-x) = x
        let expr3 = -(-x.clone());
        assert_eq!(format!("{}", expr3.simplify()), "x");
        
        // (x * y) * 0 = 0
        let expr4 = (x.clone() * y.clone()) * Symbol::c(0.0);
        assert_eq!(format!("{}", expr4.simplify()), "0");
    }
}

#[cfg(test)]
mod derivative_tests {
    use super::*;

    #[test]
    fn derivative_simplification() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let two = Symbol::c(2.0);

        // f(x) = x * 2 => f'(x) = 2 (sadeleştirilmiş)
        let f = x.clone() * two.clone();
        let df_dx = f.derivative("x").simplify();
        assert_eq!(format!("{}", df_dx), "2");

        // g(x,y) = x*y => dg/dx = y (sadeleştirilmiş)
        let g = x.clone() * y.clone();
        let dg_dx = g.derivative("x").simplify();
        let dg_dy = g.derivative("y").simplify();
        assert_eq!(format!("{}", dg_dx), "y");
        assert_eq!(format!("{}", dg_dy), "x");
    }
}