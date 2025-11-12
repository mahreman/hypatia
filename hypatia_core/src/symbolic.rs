use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Mul, Neg, Sub, Div};

/// Sembolik ifade ağacı (FAZ 4: Temizlendi)
/// 
/// 'simplify' metodu artık sadece lokal "constant folding" (sabit
/// hesaplama) yapar. Tüm cebirsel optimizasyonlar
/// 'egraph_optimizer.rs' (v3.0 motoru) tarafına taşınmıştır.
#[derive(Clone, Debug, PartialEq)]
pub enum Symbol {
    // ============ TEMEL ============
    Const(f64),
    Variable(String),
    
    // ============ ARİTMETİK ============
    Add(Box<Symbol>, Box<Symbol>),
    Mul(Box<Symbol>, Box<Symbol>),
    Sub(Box<Symbol>, Box<Symbol>),
    Div(Box<Symbol>, Box<Symbol>),
    Neg(Box<Symbol>),
    
    // ============ MATEMATİKSEL FONKSİYONLAR ============
    Exp(Box<Symbol>),
    Log(Box<Symbol>),
    Sqrt(Box<Symbol>),
    Pow(Box<Symbol>, Box<Symbol>),
    
    // ============ AKTİVASYON FONKSİYONLARI ============
    ReLU(Box<Symbol>),
    ReLUGrad(Box<Symbol>),
    Sigmoid(Box<Symbol>),
    Tanh(Box<Symbol>),
    
    // ============ YARDIMCI ============
    Max(Box<Symbol>, Box<Symbol>),
    Min(Box<Symbol>, Box<Symbol>),
    
    // ============ İLERİ SEVİYE (Opsiyonel) ============
    Piecewise(Vec<(Box<Symbol>, Box<Symbol>)>),
}

/* --------------------------
   Yardımcı fonksiyonlar
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
   Display (S-expression format - egg uyumlu)
   (Değişiklik yok)
   -------------------------- */

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use Symbol::*;
        match self {
            Const(c) => write!(f, "{}", fmt_number(*c)),
            Variable(name) => write!(f, "{}", name),
            Add(a, b) => write!(f, "(add {} {})", a, b),
            Mul(a, b) => write!(f, "(mul {} {})", a, b),
            Sub(a, b) => write!(f, "(sub {} {})", a, b),
            Div(a, b) => write!(f, "(div {} {})", a, b),
            Neg(a) => write!(f, "(neg {})", a),
            Exp(a) => write!(f, "(exp {})", a),
            Log(a) => write!(f, "(log {})", a),
            Sqrt(a) => write!(f, "(sqrt {})", a),
            Pow(a, b) => write!(f, "(pow {} {})", a, b),
            ReLU(a) => write!(f, "(relu {})", a),
            ReLUGrad(a) => write!(f, "(relu_grad {})", a),
            Sigmoid(a) => write!(f, "(sigmoid {})", a),
            Tanh(a) => write!(f, "(tanh {})", a),
            Max(a, b) => write!(f, "(max {} {})", a, b),
            Min(a, b) => write!(f, "(min {} {})", a, b),
            Piecewise(cases) => {
                write!(f, "(piecewise")?;
                for (val, cond) in cases {
                    write!(f, " ({} {})", val, cond)?;
                }
                write!(f, ")")
            }
        }
    }
}

/* --------------------------
   Operatörler: +, *, -, / (binary), - (unary)
   (Değişiklik yok)
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
            _ => Sub(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Div for Symbol {
    type Output = Symbol;
    fn div(self, rhs: Symbol) -> Symbol {
        use Symbol::*;
        match (&self, &rhs) {
            (Const(a), Const(b)) if *b != 0.0 => Const(a / b),
            _ => Div(Box::new(self), Box::new(rhs)),
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
   Türev (Symbolic Differentiation)
   (Değişiklik yok)
   -------------------------- */

impl Symbol {
    /// d/d(var) [self]
    pub fn derivative(&self, var: &str) -> Symbol {
        use Symbol::*;
        match self {
            // ============ TEMEL KURALLAR ============
            Const(_) => Const(0.0),
            Variable(name) => {
                if name == var {
                    Const(1.0)
                } else {
                    Const(0.0)
                }
            }
            
            // ============ ARİTMETİK KURALLAR ============
            Add(a, b) => a.derivative(var) + b.derivative(var),
            Sub(a, b) => a.derivative(var) - b.derivative(var),
            
            Mul(a, b) => {
                (a.derivative(var) * b.as_ref().clone())
                    + (a.as_ref().clone() * b.derivative(var))
            }
            
            Div(a, b) => {
                let numer = a.derivative(var) * b.as_ref().clone() 
                          - a.as_ref().clone() * b.derivative(var);
                let denom = Pow(b.clone(), Box::new(Const(2.0)));
                numer / denom
            }
            
            Neg(a) => -(a.derivative(var)),
            
            // ============ MATEMATİKSEL FONKSİYONLAR ============
            Exp(a) => {
                Exp(a.clone()) * a.derivative(var)
            }
            
            Log(a) => {
                a.derivative(var) / a.as_ref().clone()
            }
            
            Sqrt(a) => {
                a.derivative(var) / (Const(2.0) * Sqrt(a.clone()))
            }
            
            Pow(base, exp) => {
                match exp.as_ref() {
                    Const(n) => {
                        Const(*n) * Pow(base.clone(), Box::new(Const(n - 1.0))) 
                            * base.derivative(var)
                    }
                    _ => {
                        let f = base.as_ref();
                        let g = exp.as_ref();
                        Pow(base.clone(), exp.clone()) 
                            * (g.derivative(var) * Log(base.clone()) 
                               + g.clone() * f.derivative(var) / f.clone())
                    }
                }
            }
            
            // ============ AKTİVASYON FONKSİYONLARI ============
            ReLU(a) => {
                ReLUGrad(a.clone()) * a.derivative(var)
            }
            
            ReLUGrad(_a) => {
                Const(0.0)
            }
            
            Sigmoid(a) => {
                let sig = Sigmoid(a.clone());
                sig.clone() * (Const(1.0) - sig) * a.derivative(var)
            }
            
            Tanh(a) => {
                let tanh_val = Tanh(a.clone());
                (Const(1.0) - Pow(Box::new(tanh_val), Box::new(Const(2.0)))) 
                    * a.derivative(var)
            }
            
            // ============ YARDIMCI FONKSİYONLAR ============
            Max(a, b) => {
                (a.derivative(var) + b.derivative(var)) * Const(0.5)
            }
            
            Min(a, b) => {
                (a.derivative(var) + b.derivative(var)) * Const(0.5)
            }
            
            // ============ İLERİ SEVİYE ============
            Piecewise(_cases) => {
                Const(0.0)
            }
        }
    }

    // -----------------------------------------------------------------
    // ✅ YENİ: FEZ 4 - "APTAL" SIMPLIFY (Sadece Constant Folding)
    // -----------------------------------------------------------------
    
    /// İfade ağacını özyinelemeli olarak sadeleştirir.
    /// Sadece "Constant Folding" (sabit değerleri hesaplama) yapar.
    /// Tüm cebirsel kurallar (örn. x+0=x) e-graph motoruna taşınmıştır.
    pub fn simplify(&self) -> Symbol {
        use Symbol::*;
        // Önce çocukları sadeleştir (bottom-up)
        let s = match self {
            Const(c) => Const(*c),
            Variable(v) => Variable(v.clone()),
            
            // Çocukları sadeleştir
            Add(a, b) => Add(Box::new(a.simplify()), Box::new(b.simplify())),
            Sub(a, b) => Sub(Box::new(a.simplify()), Box::new(b.simplify())),
            Mul(a, b) => Mul(Box::new(a.simplify()), Box::new(b.simplify())),
            Div(a, b) => Div(Box::new(a.simplify()), Box::new(b.simplify())),
            Neg(a) => Neg(Box::new(a.simplify())),
            Exp(a) => Exp(Box::new(a.simplify())),
            Log(a) => Log(Box::new(a.simplify())),
            Sqrt(a) => Sqrt(Box::new(a.simplify())),
            Pow(a, b) => Pow(Box::new(a.simplify()), Box::new(b.simplify())),
            ReLU(a) => ReLU(Box::new(a.simplify())),
            ReLUGrad(a) => ReLUGrad(Box::new(a.simplify())),
            Sigmoid(a) => Sigmoid(Box::new(a.simplify())),
            Tanh(a) => Tanh(Box::new(a.simplify())),
            Max(a, b) => Max(Box::new(a.simplify()), Box::new(b.simplify())),
            Min(a, b) => Min(Box::new(a.simplify()), Box::new(b.simplify())),
            
            Piecewise(cases) => {
                let simplified_cases: Vec<_> = cases.iter()
                    .map(|(val, cond)| (Box::new(val.simplify()), Box::new(cond.simplify())))
                    .collect();
                Piecewise(simplified_cases)
            }
        };

        // Çocuklar sadeleştikten sonra, *sadece* sabit değerleri hesapla
        match &s {
            // ============ ARİTMETİK ============
            Add(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av + bv) }
                else { s }
            }
            Sub(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av - bv) }
                else { s }
            }
            Mul(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av * bv) }
                else { s }
            }
            Div(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) {
                    if *bv != 0.0 { Const(av / bv) } else { s }
                } else { s }
            }
            Neg(a) => {
                if let Const(c) = a.as_ref() { Const(-c) }
                else { s }
            }
            
            // ============ MATEMATİKSEL ============
            Exp(a) => {
                if let Const(c) = a.as_ref() { Const(c.exp()) }
                else { s }
            }
            Log(a) => {
                if let Const(c) = a.as_ref() {
                    if *c > 0.0 { Const(c.ln()) } else { s }
                } else { s }
            }
            Sqrt(a) => {
                if let Const(c) = a.as_ref() {
                    if *c >= 0.0 { Const(c.sqrt()) } else { s }
                } else { s }
            }
            Pow(base, exp) => {
                if let (Const(b), Const(e)) = (base.as_ref(), exp.as_ref()) { Const(b.powf(*e)) }
                else { s }
            }
            
            // ============ AKTİVASYONLAR ============
            ReLU(a) => {
                if let Const(c) = a.as_ref() { Const(if *c > 0.0 { *c } else { 0.0 }) }
                else { s }
            }
            ReLUGrad(a) => {
                if let Const(c) = a.as_ref() { Const(if *c >= 0.0 { 1.0 } else { 0.0 }) }
                else { s }
            }
            Sigmoid(a) => {
                if let Const(c) = a.as_ref() { Const(1.0 / (1.0 + (-c).exp())) }
                else { s }
            }
            Tanh(a) => {
                if let Const(c) = a.as_ref() { Const(c.tanh()) }
                else { s }
            }
            
            // ============ YARDIMCI ============
            Max(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av.max(*bv)) }
                else { s }
            }
            Min(a, b) => {
                if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av.min(*bv)) }
                else { s }
            }
            
            // Diğer durumlar (Const, Variable, Piecewise)
            _ => s,
        }
    }


    /// Değer verme (substitution) - Artık simplify() çağırır
    pub fn subs(&self, env: &HashMap<String, f64>) -> Symbol {
        use Symbol::*;
        let s = match self {
            Const(c) => Const(*c),
            Variable(name) => {
                if let Some(v) = env.get(name) {
                    Const(*v)
                } else {
                    Variable(name.clone())
                }
            }
            
            // Aritmetik
            Add(a, b) => Add(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Mul(a, b) => Mul(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Sub(a, b) => Sub(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Div(a, b) => Div(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Neg(a) => Neg(Box::new(a.subs(env))),
            
            // Matematiksel fonksiyonlar
            Exp(a) => Exp(Box::new(a.subs(env))),
            Log(a) => Log(Box::new(a.subs(env))),
            Sqrt(a) => Sqrt(Box::new(a.subs(env))),
            Pow(base, exp) => Pow(Box::new(base.subs(env)), Box::new(exp.subs(env))),
            
            // Aktivasyon fonksiyonları
            ReLU(a) => ReLU(Box::new(a.subs(env))),
            ReLUGrad(a) => ReLUGrad(Box::new(a.subs(env))),
            Sigmoid(a) => Sigmoid(Box::new(a.subs(env))),
            Tanh(a) => Tanh(Box::new(a.subs(env))),
            
            // Yardımcı
            Max(a, b) => Max(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Min(a, b) => Min(Box::new(a.subs(env)), Box::new(b.subs(env))),
            
            // İleri seviye
            Piecewise(cases) => {
                let subbed_cases: Vec<_> = cases.iter()
                    .map(|(val, cond)| (Box::new(val.subs(env)), Box::new(cond.subs(env))))
                    .collect();
                Piecewise(subbed_cases)
            }
        };
        // simplify() artık sadece constant folding yapar, bu yüzden güvenli.
        s.simplify()
    }

    /// Sayısal değerlendirme (numerical evaluation)
    /// (Değişiklik yok)
    pub fn eval(&self, env: &HashMap<String, f64>) -> f64 {
        use Symbol::*;
        match self {
            Const(c) => *c,
            Variable(name) => {
                *env.get(name).unwrap_or_else(|| {
                    panic!("Variable '{}' not found in environment", name)
                })
            }
            Add(a, b) => a.eval(env) + b.eval(env),
            Mul(a, b) => a.eval(env) * b.eval(env),
            Sub(a, b) => a.eval(env) - b.eval(env),
            Div(a, b) => {
                let b_val = b.eval(env);
                if b_val == 0.0 { panic!("Division by zero in eval()"); }
                a.eval(env) / b_val
            }
            Neg(a) => -a.eval(env),
            Exp(a) => a.eval(env).exp(),
            Log(a) => {
                let val = a.eval(env);
                if val <= 0.0 { panic!("Log of non-positive value: {}", val); }
                val.ln()
            }
            Sqrt(a) => {
                let val = a.eval(env);
                if val < 0.0 { panic!("Sqrt of negative value: {}", val); }
                val.sqrt()
            }
            Pow(base, exp) => base.eval(env).powf(exp.eval(env)),
            ReLU(a) => {
                let val = a.eval(env);
                if val > 0.0 { val } else { 0.0 }
            }
            ReLUGrad(a) => {
                let val = a.eval(env);
                if val >= 0.0 { 1.0 } else { 0.0 }
            }
            Sigmoid(a) => {
                let val = a.eval(env);
                1.0 / (1.0 + (-val).exp())
            }
            Tanh(a) => a.eval(env).tanh(),
            Max(a, b) => a.eval(env).max(b.eval(env)),
            Min(a, b) => a.eval(env).min(b.eval(env)),
            Piecewise(cases) => {
                for (val, cond) in cases {
                    if cond.eval(env) != 0.0 { return val.eval(env); }
                }
                0.0
            }
        }
    }

    /// Sembolik İntegral
    /// (Değişiklik yok)
    pub fn contains_var(&self, var: &str) -> bool {
        use Symbol::*;
        match self {
            Variable(name) => name == var,
            Const(_) => false,
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) | Max(a, b) | Min(a, b) => {
                a.contains_var(var) || b.contains_var(var)
            }
            Neg(a) | Exp(a) | Log(a) | Sqrt(a) | ReLU(a) | ReLUGrad(a) | Sigmoid(a) | Tanh(a) => {
                a.contains_var(var)
            }
            Piecewise(cases) => cases.iter().any(|(val, cond)| {
                val.contains_var(var) || cond.contains_var(var)
            }),
        }
    }

    pub fn integrate(&self, var: &str) -> Result<Symbol, String> {
        use Symbol::*;
        
        if !self.contains_var(var) {
            return Ok(self.clone() * Variable(var.to_string()));
        }

        match self {
            Const(_) => unreachable!(), 
            Variable(name) => {
                if name == var {
                    let x = Box::new(self.clone());
                    let two = Box::new(Const(2.0));
                    Ok(Div( Box::new(Pow(x, two.clone())), two ))
                } else {
                    unreachable!() 
                }
            }
            Add(a, b) => Ok(a.integrate(var)? + b.integrate(var)?),
            Sub(a, b) => Ok(a.integrate(var)? - b.integrate(var)?),
            Neg(a) => Ok(-(a.integrate(var)?)),
            Mul(a, b) => {
                if !a.contains_var(var) {
                    Ok(a.as_ref().clone() * b.integrate(var)?)
                } else if !b.contains_var(var) {
                    Ok(b.as_ref().clone() * a.integrate(var)?)
                } else {
                    Err(format!("Unsupported integration by parts: ∫ {} dx", self))
                }
            }
            Div(a, b) => {
                if !b.contains_var(var) {
                    Ok(a.integrate(var)? / b.as_ref().clone())
                } else {
                    Err(format!("Unsupported quotient integration: ∫ {} dx", self))
                }
            }
            Pow(base, exp) => {
                if let (Variable(name), Const(n)) = (base.as_ref(), exp.as_ref()) {
                    if name == var {
                        if *n == -1.0 {
                            return Ok(Symbol::Log(Box::new(Variable(var.to_string()))));
                        }
                        let n_plus_1 = Const(n + 1.0);
                        let new_exp = Box::new(n_plus_1.clone());
                        let new_base = Box::new(Variable(var.to_string()));
                        return Ok(Symbol::Div(
                            Box::new(Symbol::Pow(new_base, new_exp)),
                            Box::new(n_plus_1)
                        ));
                    }
                }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ x^n dx is supported.", self))
            }
            Exp(a) => {
                if let Variable(name) = a.as_ref() {
                    if name == var { return Ok(self.clone()); }
                }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ e^x dx is supported.", self))
            }
            Log(a) => {
                if let Variable(name) = a.as_ref() {
                    if name == var {
                        let x = Box::new(Variable(var.to_string()));
                        return Ok(
                            ((*x).clone() * self.clone()) - (*x).clone()
                        );
                    }
                }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ ln(x) dx is supported.", self))
            }
            _ => Err(format!("Integration for {} is not supported.", self.to_string()))
        }
    }
}

/* --------------------------
   Yardımcı yapım fonksiyonları
   (Değişiklik yok)
   -------------------------- */

impl Symbol {
    pub fn var(name: &str) -> Symbol { Symbol::Variable(name.to_string()) }
    pub fn c(v: f64) -> Symbol { Symbol::Const(v) }
    pub fn exp(a: Symbol) -> Symbol { Symbol::Exp(Box::new(a)) }
    pub fn log(a: Symbol) -> Symbol { Symbol::Log(Box::new(a)) }
    pub fn sqrt(a: Symbol) -> Symbol { Symbol::Sqrt(Box::new(a)) }
    pub fn pow(base: Symbol, exp: Symbol) -> Symbol { Symbol::Pow(Box::new(base), Box::new(exp)) }
    pub fn relu(a: Symbol) -> Symbol { Symbol::ReLU(Box::new(a)) }
    pub fn relu_grad(a: Symbol) -> Symbol { Symbol::ReLUGrad(Box::new(a)) }
    pub fn sigmoid(a: Symbol) -> Symbol { Symbol::Sigmoid(Box::new(a)) }
    pub fn tanh(a: Symbol) -> Symbol { Symbol::Tanh(Box::new(a)) }
    pub fn max(a: Symbol, b: Symbol) -> Symbol { Symbol::Max(Box::new(a), Box::new(b)) }
    pub fn min(a: Symbol, b: Symbol) -> Symbol { Symbol::Min(Box::new(a), Box::new(b)) }
    pub fn piecewise(cases: Vec<(Symbol, Symbol)>) -> Symbol {
        Symbol::Piecewise(
            cases.into_iter()
                .map(|(val, cond)| (Box::new(val), Box::new(cond)))
                .collect()
        )
    }
}

/* --------------------------
   TESTLER (Artık 'simplify' testleri 'aptal' simplify'ı test etmeli)
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
        assert_eq!(format!("{}", f), "(mul x 2)");
        let g = x.clone() * y.clone() + two.clone();
        assert_eq!(format!("{}", g), "(add (mul x y) 2)");
    }

    #[test]
    fn derivative_of_linear_and_product() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let two = Symbol::c(2.0);

        let f = x.clone() * two.clone();
        // Türev sonrası simplify artık (mul 1 2) -> 2 yapar
        let df_dx = f.derivative("x").simplify();
        assert_eq!(format!("{}", df_dx), "2");

        let g = x.clone() * y.clone();
        // Türev sonrası (add (mul 1 y) (mul x 0)) -> (add y 0)
        // Bu, E-graph'te (add y 0) -> y olarak sadeleşir,
        // ancak lokal simplify bunu yapmaz.
        let dg_dx = g.derivative("x").simplify();
        assert_eq!(format!("{}", dg_dx), "(add y 0)"); // BEKLENEN DEĞİŞİKLİK
    }

    #[test]
    fn simplify_basic_constant_folding() {
        let x = Symbol::var("x");
        
        // (x * 0) -> (mul x 0) - Artık cebirsel kural yok
        let expr1 = (x.clone() * Symbol::c(0.0)).simplify();
        assert_eq!(format!("{}", expr1), "(mul x 0)"); // BEKLENEN DEĞİŞİKLİK
        
        // (0 * x) -> 0 (Bu kural Mul(a,b) içinde manuel olarak var)
        let expr1_b = (Symbol::c(0.0) * x.clone()).simplify();
        assert_eq!(format!("{}", expr1_b), "0"); 
        
        // (x + x) -> (add x x)
        let expr2 = (x.clone() + x.clone()).simplify();
        assert_eq!(format!("{}", expr2), "(add x x)"); // BEKLENEN DEĞİŞİKLİK
        
        // -(-x) -> x
        let expr3 = (-(-x.clone())).simplify();
        assert_eq!(format!("{}", expr3), "x"); // Bu kural Neg içinde
    }
    
    #[test]
    fn test_exp_log_no_simplify() {
        let x = Symbol::var("x");
        
        // e^(ln(x)) - Artık sadeleşmez
        let expr1 = Symbol::exp(Symbol::log(x.clone()));
        assert_eq!(format!("{}", expr1.simplify()), "(exp (log x))");
        
        // ln(e^x) - Artık sadeleşmez
        let expr2 = Symbol::log(Symbol::exp(x.clone()));
        assert_eq!(format!("{}", expr2.simplify()), "(log (exp x))");
        
        // e^0 = 1 (Constant folding)
        let expr3 = Symbol::exp(Symbol::c(0.0));
        assert_eq!(format!("{}", expr3.simplify()), "1");
    }
    
    #[test]
    fn test_relu_and_grad_folding() {
        // ReLU(5) = 5 (Constant folding)
        let expr1 = Symbol::relu(Symbol::c(5.0));
        assert_eq!(format!("{}", expr1.simplify()), "5");
        
        // ReLU(-3) = 0 (Constant folding)
        let expr2 = Symbol::relu(Symbol::c(-3.0));
        assert_eq!(format!("{}", expr2.simplify()), "0");
        
        // ReLU(ReLU(x)) - Artık sadeleşmez
        let x = Symbol::var("x");
        let expr3 = Symbol::relu(Symbol::relu(x.clone()));
        assert_eq!(format!("{}", expr3.simplify()), "(relu (relu x))");
    }
    
    #[test]
    fn test_sigmoid_activation_folding() {
        // sigmoid(0) = 0.5 (Constant folding)
        let expr = Symbol::sigmoid(Symbol::c(0.0));
        let result = expr.simplify();
        
        if let Symbol::Const(c) = result {
            assert!((c - 0.5).abs() < 1e-10);
        } else {
            panic!("Expected Const, got {:?}", result);
        }
    }
    
    #[test]
    fn test_eval() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let expr = Symbol::relu(x.clone() * Symbol::c(2.0) + y.clone());
        let mut env = HashMap::new();
        env.insert("x".to_string(), 3.0);
        env.insert("y".to_string(), -5.0);
        assert_eq!(expr.eval(&env), 1.0);
    }

    // ============ FEZ 3 - İNTEGRAL TESTLERİ (Değişiklik yok) ============

    #[test]
    fn test_integration_constants() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let f1 = Symbol::c(5.0);
        assert_eq!(
            f1.integrate("x").unwrap().simplify(),
            (Symbol::c(5.0) * x.clone()).simplify()
        );
        let f2 = y.clone();
        assert_eq!(
            f2.integrate("x").unwrap().simplify(),
            (y.clone() * x.clone()).simplify()
        );
    }

    #[test]
    fn test_integration_pow_rule() {
        let x = Symbol::var("x");
        let f1 = x.clone();
        let expected_f1 = Symbol::pow(x.clone(), Symbol::c(2.0)) / Symbol::c(2.0);
        assert_eq!(
            f1.integrate("x").unwrap().simplify(),
            expected_f1.simplify()
        );
        let f3 = Symbol::pow(x.clone(), Symbol::c(-1.0));
        let expected_f3 = Symbol::log(x.clone());
        assert_eq!(
            f3.integrate("x").unwrap().simplify(),
            expected_f3.simplify()
        );
    }

    #[test]
    fn test_integration_linearity() {
        let x = Symbol::var("x");
        let f = x.clone() + Symbol::c(2.0);
        let int_f = f.integrate("x").unwrap().simplify();
        let x_sq_2 = Symbol::pow(x.clone(), Symbol::c(2.0)) / Symbol::c(2.0);
        let two_x = Symbol::c(2.0) * x.clone();
        let expected = (x_sq_2 + two_x).simplify();
        assert_eq!(format!("{}", int_f), format!("{}", expected));
    }

    #[test]
    fn test_integration_special_functions() {
        let x = Symbol::var("x");
        let f1 = Symbol::exp(x.clone());
        assert_eq!(f1.integrate("x").unwrap(), f1);
        let f2 = Symbol::log(x.clone());
        let expected_f2 = (x.clone() * Symbol::log(x.clone())) - x.clone();
        assert_eq!(f2.integrate("x").unwrap().simplify(), expected_f2.simplify());
    }

    #[test]
    fn test_integration_unsupported() {
        let x = Symbol::var("x");
        let f1 = x.clone() * x.clone();
        assert!(f1.integrate("x").is_err());
        let f2 = Symbol::sigmoid(x.clone());
        assert!(f2.integrate("x").is_err());
    }
}