use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Mul, Neg, Sub, Div};
use num_traits::{Zero, One};

/// Sembolik ifade ağacı (FEZ 10: AI Operatörleri Eklendi)
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

    // ============ ✅ YENİ: FEZ 10 AI / İSTATİSTİK OPERATÖRLERİ ============
    Softmax(Box<Symbol>),
    Mean(Box<Symbol>),
    Variance(Box<Symbol>),
    
    // ✅ YENİ: Transformer Desteği
    Embedding(Box<Symbol>, Box<Symbol>), // (embedding w x)
    
    // ============ YARDIMCI ============
    Max(Box<Symbol>, Box<Symbol>),
    Min(Box<Symbol>, Box<Symbol>),
    Piecewise(Vec<(Box<Symbol>, Box<Symbol>)>),
    
    // ====================================================================
    // ✅ YENİ: ResNet Operatörleri (İstek üzerine eklendi)
    // ====================================================================
    BatchNorm {
        weight: Box<Symbol>,
        bias: Box<Symbol>,
        running_mean: Box<Symbol>,
        running_var: Box<Symbol>,
        input: Box<Symbol>,
        eps: Box<Symbol>,
    },
    MaxPool2d {
        input: Box<Symbol>,
        kernel_size: Box<Symbol>,
        stride: Box<Symbol>,
        padding: Box<Symbol>,
    },
    AddInplace(Box<Symbol>, Box<Symbol>),
}

/* --------------------------
   Yardımcı fonksiyonlar
   (Değişiklik yok)
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
   Display (S-expression format - güncellendi)
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
            Softmax(a) => write!(f, "(softmax {})", a),
            Mean(a) => write!(f, "(mean {})", a),
            Variance(a) => write!(f, "(var {})", a),
            Embedding(w, x) => write!(f, "(embedding {} {})", w, x),
            Max(a, b) => write!(f, "(max {} {})", a, b),
            Min(a, b) => write!(f, "(min {} {})", a, b),
            Piecewise(cases) => {
                write!(f, "(piecewise")?;
                for (val, cond) in cases {
                    write!(f, " ({} {})", val, cond)?;
                }
                write!(f, ")")
            }
            
            // ✅ YENİ: ResNet Operatörleri
            AddInplace(a, b) => write!(f, "(add_inplace {} {})", a, b),
            BatchNorm { weight, bias, running_mean, running_var, input, eps } => {
                write!(f, "(batchnorm {} {} {} {} {} {})", weight, bias, running_mean, running_var, input, eps)
            },
            MaxPool2d { input, kernel_size, stride, padding } => {
                write!(f, "(maxpool2d {} {} {} {})", input, kernel_size, stride, padding)
            },
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
   Türev (Symbolic Differentiation - güncellendi)
   -------------------------- */

impl Symbol {
    /// d/d(var) [self]
    pub fn derivative(&self, var: &str) -> Symbol {
        use Symbol::*;
        match self {
            Const(_) => Const(0.0),
            Variable(name) => {
                if name == var { Const(1.0) } else { Const(0.0) }
            }
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
            Exp(a) => { Exp(a.clone()) * a.derivative(var) },
            Log(a) => { a.derivative(var) / a.as_ref().clone() },
            Sqrt(a) => { a.derivative(var) / (Const(2.0) * Sqrt(a.clone())) },
            Pow(base, exp) => {
                match exp.as_ref() {
                    Const(n) => {
                        Const(*n) * Pow(base.clone(), Box::new(Const(n - 1.0))) 
                            * base.derivative(var)
                    }
                    _ => {
                        let f = base.as_ref(); let g = exp.as_ref();
                        Pow(base.clone(), exp.clone()) 
                            * (g.derivative(var) * Log(base.clone()) 
                               + g.clone() * f.derivative(var) / f.clone())
                    }
                }
            },
            ReLU(a) => { ReLUGrad(a.clone()) * a.derivative(var) },
            ReLUGrad(_a) => { Const(0.0) },
            Sigmoid(a) => {
                let sig = Sigmoid(a.clone());
                sig.clone() * (Const(1.0) - sig) * a.derivative(var)
            },
            Tanh(a) => {
                let tanh_val = Tanh(a.clone());
                (Const(1.0) - Pow(Box::new(tanh_val), Box::new(Const(2.0)))) 
                    * a.derivative(var)
            },
            
            // ✅ YENİ: FEZ 10 AI Operatörleri Türevleri
            Softmax(a) => {
                let sm = Softmax(a.clone());
                sm.clone() * (Const(1.0) - sm) * a.derivative(var)
            },
            Mean(a) => {
                Mean(Box::new(a.derivative(var)))
            },
            Variance(a) => {
                Variance(Box::new(a.derivative(var)))
            },
            
            // ✅ YENİ
            // Girdiler (x) tamsayı indeksleridir, türevleri anlamsızdır (0).
            // Ağırlıklara (w) göre türev, karmaşık bir seyrek matrisdir (one-hot).
            // Şimdilik basitleştirilmiş (0) döndür.
            Embedding(_w, _x) => {
                Const(0.0)
            },
            
            Max(a, b) => { (a.derivative(var) + b.derivative(var)) * Const(0.5) },
            Min(a, b) => { (a.derivative(var) + b.derivative(var)) * Const(0.5) },
            Piecewise(_cases) => { Const(0.0) }

            // ✅ YENİ: ResNet Operatörleri (Türev desteklenmiyor)
            AddInplace(a, b) => a.derivative(var) + b.derivative(var),
            BatchNorm { .. } => Const(0.0), // TODO: Gerçek türev karmaşık
            MaxPool2d { .. } => Const(0.0), // TODO: Gerçek türev karmaşık
        }
    }

    // -----------------------------------------------------------------
    // "Aptal" Simplify (Constant Folding - güncellendi)
    // -----------------------------------------------------------------
    
    pub fn simplify(&self) -> Symbol {
        use Symbol::*;
        let s = match self {
            Const(c) => Const(*c),
            Variable(v) => Variable(v.clone()),
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
            Softmax(a) => Softmax(Box::new(a.simplify())),
            Mean(a) => Mean(Box::new(a.simplify())),
            Variance(a) => Variance(Box::new(a.simplify())),
            Embedding(w, x) => Embedding(Box::new(w.simplify()), Box::new(x.simplify())),
            Max(a, b) => Max(Box::new(a.simplify()), Box::new(b.simplify())),
            Min(a, b) => Min(Box::new(a.simplify()), Box::new(b.simplify())),
            Piecewise(cases) => {
                let simplified_cases: Vec<_> = cases.iter()
                    .map(|(val, cond)| (Box::new(val.simplify()), Box::new(cond.simplify())))
                    .collect();
                Piecewise(simplified_cases)
            }
            
            // ✅ YENİ: ResNet Operatörleri (Simplify)
            AddInplace(a, b) => AddInplace(Box::new(a.simplify()), Box::new(b.simplify())),
            BatchNorm { weight, bias, running_mean, running_var, input, eps } => {
                BatchNorm {
                    weight: Box::new(weight.simplify()),
                    bias: Box::new(bias.simplify()),
                    running_mean: Box::new(running_mean.simplify()),
                    running_var: Box::new(running_var.simplify()),
                    input: Box::new(input.simplify()),
                    eps: Box::new(eps.simplify()),
                }
            },
            MaxPool2d { input, kernel_size, stride, padding } => {
                MaxPool2d {
                    input: Box::new(input.simplify()),
                    kernel_size: Box::new(kernel_size.simplify()),
                    stride: Box::new(stride.simplify()),
                    padding: Box::new(padding.simplify()),
                }
            }
        };

        match &s {
            // Toplama optimizasyonları
            Add(a, b) => {
                match (a.as_ref(), b.as_ref()) {
                    (Const(av), Const(bv)) => Const(av + bv),
                    (Const(0.0), x) | (x, Const(0.0)) => x.clone(),
                    _ => s
                }
            },
            // ✅ YENİ: AddInplace optimizasyonları
            AddInplace(a, b) => {
                match (a.as_ref(), b.as_ref()) {
                    (Const(av), Const(bv)) => Const(av + bv),
                    (Const(0.0), x) | (x, Const(0.0)) => x.clone(),
                    _ => Add(a.clone(), b.clone()) // AddInplace'i Add'e dönüştür
                }
            },
            // Çıkarma optimizasyonları
            Sub(a, b) => {
                match (a.as_ref(), b.as_ref()) {
                    (Const(av), Const(bv)) => Const(av - bv),
                    (x, Const(0.0)) => x.clone(),
                    (Const(0.0), x) => Neg(Box::new(x.clone())),
                    _ => s
                }
            },
            // Çarpma optimizasyonları
            Mul(a, b) => {
                match (a.as_ref(), b.as_ref()) {
                    (Const(av), Const(bv)) => Const(av * bv),
                    (Const(0.0), _) | (_, Const(0.0)) => Const(0.0),
                    (Const(1.0), x) | (x, Const(1.0)) => x.clone(),
                    _ => s
                }
            },
            // Bölme optimizasyonları
            Div(a, b) => {
                match (a.as_ref(), b.as_ref()) {
                    (Const(av), Const(bv)) if *bv != 0.0 => Const(av / bv),
                    (Const(0.0), _) => Const(0.0),
                    (x, Const(1.0)) => x.clone(),
                    _ => s
                }
            },
            Neg(a) => {
                match a.as_ref() {
                    Const(c) => Const(-c),
                    Neg(inner) => inner.as_ref().clone(),
                    _ => s
                }
            },
            Exp(a) => { if let Const(c) = a.as_ref() { Const(c.exp()) } else { s } },
            Log(a) => { if let Const(c) = a.as_ref() { if *c > 0.0 { Const(c.ln()) } else { s } } else { s } },
            Sqrt(a) => { if let Const(c) = a.as_ref() { if *c >= 0.0 { Const(c.sqrt()) } else { s } } else { s } },
            Pow(base, exp) => { if let (Const(b), Const(e)) = (base.as_ref(), exp.as_ref()) { Const(b.powf(*e)) } else { s } },
            ReLU(a) => { if let Const(c) = a.as_ref() { Const(if *c > 0.0 { *c } else { 0.0 }) } else { s } },
            ReLUGrad(a) => { if let Const(c) = a.as_ref() { Const(if *c >= 0.0 { 1.0 } else { 0.0 }) } else { s } },
            Sigmoid(a) => { if let Const(c) = a.as_ref() { Const(1.0 / (1.0 + (-c).exp())) } else { s } },
            Tanh(a) => { if let Const(c) = a.as_ref() { Const(c.tanh()) } else { s } },

            Softmax(a) => { if let Const(_c) = a.as_ref() { Const(1.0) } else { s } },
            Mean(a) => { if let Const(c) = a.as_ref() { Const(*c) } else { s } },
            Variance(a) => { if let Const(_c) = a.as_ref() { Const(0.0) } else { s } },
            
            Embedding(_w, _x) => s,

            Max(a, b) => { if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av.max(*bv)) } else { s } },
            Min(a, b) => { if let (Const(av), Const(bv)) = (a.as_ref(), b.as_ref()) { Const(av.min(*bv)) } else { s } },
            
            // ✅ YENİ: ResNet Operatörleri (Simplify kuralı yok)
            BatchNorm { .. } => s,
            MaxPool2d { .. } => s,

            _ => s,
        }
    }


    /// Değer verme (substitution - güncellendi)
    pub fn subs(&self, env: &HashMap<String, f64>) -> Symbol {
        use Symbol::*;
        let s = match self {
            Const(c) => Const(*c),
            Variable(name) => { if let Some(v) = env.get(name) { Const(*v) } else { Variable(name.clone()) } },
            Add(a, b) => Add(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Mul(a, b) => Mul(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Sub(a, b) => Sub(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Div(a, b) => Div(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Neg(a) => Neg(Box::new(a.subs(env))),
            Exp(a) => Exp(Box::new(a.subs(env))),
            Log(a) => Log(Box::new(a.subs(env))),
            Sqrt(a) => Sqrt(Box::new(a.subs(env))),
            Pow(base, exp) => Pow(Box::new(base.subs(env)), Box::new(exp.subs(env))),
            ReLU(a) => ReLU(Box::new(a.subs(env))),
            ReLUGrad(a) => ReLUGrad(Box::new(a.subs(env))),
            Sigmoid(a) => Sigmoid(Box::new(a.subs(env))),
            Tanh(a) => Tanh(Box::new(a.subs(env))),
            Softmax(a) => Softmax(Box::new(a.subs(env))),
            Mean(a) => Mean(Box::new(a.subs(env))),
            Variance(a) => Variance(Box::new(a.subs(env))),
            Embedding(w, x) => Embedding(Box::new(w.subs(env)), Box::new(x.subs(env))),
            Max(a, b) => Max(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Min(a, b) => Min(Box::new(a.subs(env)), Box::new(b.subs(env))),
            Piecewise(cases) => {
                let subbed_cases: Vec<_> = cases.iter()
                    .map(|(val, cond)| (Box::new(val.subs(env)), Box::new(cond.subs(env))))
                    .collect();
                Piecewise(subbed_cases)
            }
            
            // ✅ YENİ: ResNet Operatörleri (Subs)
            AddInplace(a, b) => AddInplace(Box::new(a.subs(env)), Box::new(b.subs(env))),
            BatchNorm { weight, bias, running_mean, running_var, input, eps } => {
                BatchNorm {
                    weight: Box::new(weight.subs(env)),
                    bias: Box::new(bias.subs(env)),
                    running_mean: Box::new(running_mean.subs(env)),
                    running_var: Box::new(running_var.subs(env)),
                    input: Box::new(input.subs(env)),
                    eps: Box::new(eps.subs(env)),
                }
            },
            MaxPool2d { input, kernel_size, stride, padding } => {
                MaxPool2d {
                    input: Box::new(input.subs(env)),
                    kernel_size: Box::new(kernel_size.subs(env)),
                    stride: Box::new(stride.subs(env)),
                    padding: Box::new(padding.subs(env)),
                }
            }
        };
        s.simplify()
    }

    /// ✅ FAZ 12: Sayısal değerlendirme (PANIC-SAFE)
    /// 
    /// # Hatalar: Sıfıra bölme, log(-), sqrt(-), 0^(-), undefined variable
    pub fn eval(&self, env: &HashMap<String, f64>) -> Result<f64, String> {
        use Symbol::*;
        match self {
            Const(c) => Ok(*c),
            Variable(name) => env.get(name).copied()
                .ok_or_else(|| format!("Variable '{}' not found", name)),
            
            Add(a, b) => Ok(a.eval(env)? + b.eval(env)?),
            Mul(a, b) => Ok(a.eval(env)? * b.eval(env)?),
            Sub(a, b) => Ok(a.eval(env)? - b.eval(env)?),
            
            Div(a, b) => {
                let (a_val, b_val) = (a.eval(env)?, b.eval(env)?);
                if b_val.abs() < 1e-15 {
                    Err(format!("Division by zero: ({}) / ({}) = {}/{}", a, b, a_val, b_val))
                } else {
                    Ok(a_val / b_val)
                }
            }
            
            Neg(a) => Ok(-a.eval(env)?),
            Exp(a) => Ok(a.eval(env)?.exp()),
            
            Log(a) => {
                let val = a.eval(env)?;
                if val <= 0.0 {
                    Err(format!("Log of non-positive: log({}) = log({})", a, val))
                } else {
                    Ok(val.ln())
                }
            }
            
            Sqrt(a) => {
                let val = a.eval(env)?;
                if val < 0.0 {
                    Err(format!("Sqrt of negative: sqrt({}) = sqrt({})", a, val))
                } else {
                    Ok(val.sqrt())
                }
            }
            
            Pow(base, exp) => {
                let (base_val, exp_val) = (base.eval(env)?, exp.eval(env)?);
                if base_val.abs() < 1e-15 && exp_val < 0.0 {
                    Err(format!("Zero to negative power: 0^{}", exp_val))
                } else if base_val < 0.0 && exp_val.fract() != 0.0 {
                    Err(format!("Negative base with fractional exp: {}^{}", base_val, exp_val))
                } else {
                    Ok(base_val.powf(exp_val))
                }
            }
            
            ReLU(a) => Ok(a.eval(env)?.max(0.0)),
            ReLUGrad(a) => Ok(if a.eval(env)? >= 0.0 { 1.0 } else { 0.0 }),
            Sigmoid(a) => { let v = a.eval(env)?; Ok(1.0 / (1.0 + (-v).exp())) }
            Tanh(a) => Ok(a.eval(env)?.tanh()),
            
            Softmax(_a) => Ok(1.0),
            Mean(a) => a.eval(env),
            Variance(_a) => Ok(0.0),

            Embedding(_w, _x) => {
                Err(format!("eval() is not supported for Embedding operator: {}", self))
            },
            
            Max(a, b) => Ok(a.eval(env)?.max(b.eval(env)?)),
            Min(a, b) => Ok(a.eval(env)?.min(b.eval(env)?)),
            
            Piecewise(cases) => {
                for (val, cond) in cases {
                    if cond.eval(env)? != 0.0 { return val.eval(env); }
                }
                Ok(0.0)
            }
            
            // ✅ YENİ: ResNet Operatörleri (Eval)
            AddInplace(a, b) => Ok(a.eval(env)? + b.eval(env)?),
            BatchNorm { input, .. } => {
                // Not: Bu, 'eval' zamanında (eğitim değil) BatchNorm'un
                // basitleştirilmiş bir uygulamasıdır.
                input.eval(env)
                // Err(format!("eval() is not fully supported for BatchNorm operator: {}", self))
            },
            MaxPool2d { input, .. } => {
                // Not: 'eval' bir f64 döndürür, MaxPool tensör -> tensör'dür.
                // Sadece passthrough'a izin ver.
                input.eval(env)
                // Err(format!("eval() is not supported for MaxPool2d operator: {}", self))
            },
        }
    }

    /// Sembolik İntegral (güncellendi)
    pub fn contains_var(&self, var: &str) -> bool { 
        use Symbol::*;
        match self {
            Variable(name) => name == var,
            Const(_) => false,
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) | Max(a, b) | Min(a, b) 
            | Embedding(a, b)
            | AddInplace(a, b) // ✅ YENİ
            => {
                a.contains_var(var) || b.contains_var(var)
            }
            Neg(a) | Exp(a) | Log(a) | Sqrt(a) | ReLU(a) | ReLUGrad(a) | Sigmoid(a) | Tanh(a)
            | Softmax(a) | Mean(a) | Variance(a)
            => {
                a.contains_var(var)
            }
            Piecewise(cases) => cases.iter().any(|(val, cond)| {
                val.contains_var(var) || cond.contains_var(var)
            }),
            
            // ✅ YENİ: ResNet Operatörleri (contains_var)
            BatchNorm { weight, bias, running_mean, running_var, input, eps } => {
                weight.contains_var(var) || bias.contains_var(var) || 
                running_mean.contains_var(var) || running_var.contains_var(var) ||
                input.contains_var(var) || eps.contains_var(var)
            },
            MaxPool2d { input, kernel_size, stride, padding } => {
                input.contains_var(var) || kernel_size.contains_var(var) ||
                stride.contains_var(var) || padding.contains_var(var)
            },
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
                    let x = Box::new(self.clone()); let two = Box::new(Const(2.0));
                    Ok(Div( Box::new(Pow(x, two.clone())), two ))
                } else { unreachable!() }
            }
            Add(a, b) => Ok(a.integrate(var)? + b.integrate(var)?),
            AddInplace(a, b) => Ok(a.integrate(var)? + b.integrate(var)?), // ✅ YENİ
            Sub(a, b) => Ok(a.integrate(var)? - b.integrate(var)?),
            Neg(a) => Ok(-(a.integrate(var)?)),
            Mul(a, b) => {
                if !a.contains_var(var) { Ok(a.as_ref().clone() * b.integrate(var)?) }
                else if !b.contains_var(var) { Ok(b.as_ref().clone() * a.integrate(var)?) }
                else { Err(format!("Unsupported integration by parts: ∫ {} dx", self)) }
            }
            Div(a, b) => {
                if !b.contains_var(var) { Ok(a.integrate(var)? / b.as_ref().clone()) }
                else { Err(format!("Unsupported quotient integration: ∫ {} dx", self)) }
            }
            Pow(base, exp) => {
                if let (Variable(name), Const(n)) = (base.as_ref(), exp.as_ref()) {
                    if name == var {
                        if *n == -1.0 { return Ok(Symbol::Log(Box::new(Variable(var.to_string())))); }
                        let n_plus_1 = Const(n + 1.0);
                        let new_exp = Box::new(n_plus_1.clone());
                        let new_base = Box::new(Variable(var.to_string()));
                        return Ok(Symbol::Div( Box::new(Symbol::Pow(new_base, new_exp)), Box::new(n_plus_1) ));
                    }
                }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ x^n dx is supported.", self))
            }
            Exp(a) => {
                if let Variable(name) = a.as_ref() { if name == var { return Ok(self.clone()); } }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ e^x dx is supported.", self))
            }
            Log(a) => {
                if let Variable(name) = a.as_ref() {
                    if name == var {
                        let x = Box::new(Variable(var.to_string()));
                        return Ok( ((*x).clone() * self.clone()) - (*x).clone() );
                    }
                }
                Err(format!("Unsupported integration: ∫ {} dx. Only ∫ ln(x) dx is supported.", self))
            }
            _ => Err(format!("Integration for {} is not supported.", self.to_string()))
        }
    }
}

/* --------------------------
   Yardımcı yapım fonksiyonları (güncellendi)
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
    
    // ✅ YENİ: FEZ 10 AI Operatörleri
    pub fn softmax(a: Symbol) -> Symbol { Symbol::Softmax(Box::new(a)) }
    pub fn mean(a: Symbol) -> Symbol { Symbol::Mean(Box::new(a)) }
    pub fn variance(a: Symbol) -> Symbol { Symbol::Variance(Box::new(a)) }
    
    // ✅ YENİ
    pub fn embedding(w: Symbol, x: Symbol) -> Symbol { Symbol::Embedding(Box::new(w), Box::new(x)) }

    pub fn max(a: Symbol, b: Symbol) -> Symbol { Symbol::Max(Box::new(a), Box::new(b)) }
    pub fn min(a: Symbol, b: Symbol) -> Symbol { Symbol::Min(Box::new(a), Box::new(b)) }
    pub fn piecewise(cases: Vec<(Symbol, Symbol)>) -> Symbol {
        Symbol::Piecewise( cases.into_iter().map(|(val, cond)| (Box::new(val), Box::new(cond))).collect() )
    }
}

/* --------------------------
   num_traits: Zero ve One implementasyonları
   -------------------------- */

impl Zero for Symbol {
    fn zero() -> Self {
        Symbol::Const(0.0)
    }
    
    fn is_zero(&self) -> bool {
        matches!(self, Symbol::Const(x) if *x == 0.0)
    }
}

impl One for Symbol {
    fn one() -> Self {
        Symbol::Const(1.0)
    }
}

/* --------------------------
   TESTLER (güncellendi)
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
        let df_dx = f.derivative("x").simplify();
        assert_eq!(format!("{}", df_dx), "2");
        let g = x.clone() * y.clone();
        let dg_dx = g.derivative("x").simplify();
        // Güçlendirilmiş simplify: (add y 0) → y
        assert_eq!(format!("{}", dg_dx), "y"); 
    }

    #[test]
    fn simplify_basic_constant_folding() {
        let x = Symbol::var("x");
        // Hem (x * 0) hem de (0 * x) artık 0'a sadeleşmeli
        let expr1 = (x.clone() * Symbol::c(0.0)).simplify();
        assert_eq!(format!("{}", expr1), "0"); 
        let expr1_b = (Symbol::c(0.0) * x.clone()).simplify();
        assert_eq!(format!("{}", expr1_b), "0"); 
        let expr2 = (x.clone() + x.clone()).simplify();
        assert_eq!(format!("{}", expr2), "(add x x)"); 
        let expr3 = (-(-x.clone())).simplify();
        assert_eq!(format!("{}", expr3), "x"); 
    }
    
    #[test]
    fn test_exp_log_no_simplify() {
        let x = Symbol::var("x");
        let expr1 = Symbol::exp(Symbol::log(x.clone()));
        assert_eq!(format!("{}", expr1.simplify()), "(exp (log x))");
        let expr2 = Symbol::log(Symbol::exp(x.clone()));
        assert_eq!(format!("{}", expr2.simplify()), "(log (exp x))");
        let expr3 = Symbol::exp(Symbol::c(0.0));
        assert_eq!(format!("{}", expr3.simplify()), "1");
    }
    #[test]
    fn test_relu_and_grad_folding() {
        let expr1 = Symbol::relu(Symbol::c(5.0));
        assert_eq!(format!("{}", expr1.simplify()), "5");
        let expr2 = Symbol::relu(Symbol::c(-3.0));
        assert_eq!(format!("{}", expr2.simplify()), "0");
        let x = Symbol::var("x");
        let expr3 = Symbol::relu(Symbol::relu(x.clone()));
        assert_eq!(format!("{}", expr3.simplify()), "(relu (relu x))");
    }
    #[test]
    fn test_sigmoid_activation_folding() {
        let expr = Symbol::sigmoid(Symbol::c(0.0));
        let result = expr.simplify();
        if let Symbol::Const(c) = result { assert!((c - 0.5).abs() < 1e-10); } 
        else { panic!("Expected Const, got {:?}", result); }
    }
    #[test]
    fn test_eval() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let expr = Symbol::relu(x.clone() * Symbol::c(2.0) + y.clone());
        let mut env = HashMap::new();
        env.insert("x".to_string(), 3.0);
        env.insert("y".to_string(), -5.0);
        assert_eq!(expr.eval(&env).unwrap(), 1.0);
    }
    
    // ✅ YENİ: eval() için hata testi
    #[test]
    fn test_eval_errors() {
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let mut env = HashMap::new();
        env.insert("x".to_string(), -1.0);
        
        // Tanımsız değişken
        assert!(y.eval(&env).is_err());
        // Sıfıra bölme
        assert!((Symbol::c(1.0) / Symbol::c(0.0)).eval(&env).is_err());
        // Negatif sqrt
        assert!(Symbol::sqrt(x.clone()).eval(&env).is_err());
        // Embedding
        assert!(Symbol::embedding(Symbol::var("w"), x.clone()).eval(&env).is_err());
    }

    #[test]
    fn test_ai_ops_folding() {
        let m = Symbol::mean(Symbol::c(5.0));
        assert_eq!(format!("{}", m.simplify()), "5");
        let v = Symbol::variance(Symbol::c(5.0));
        assert_eq!(format!("{}", v.simplify()), "0");
        let s = Symbol::softmax(Symbol::c(5.0));
        assert_eq!(format!("{}", s.simplify()), "1");
    }

    #[test]
    fn test_ai_ops_display() {
        let x = Symbol::var("x");
        assert_eq!(format!("{}", Symbol::mean(x.clone())), "(mean x)");
        assert_eq!(format!("{}", Symbol::variance(x.clone())), "(var x)");
        assert_eq!(format!("{}", Symbol::softmax(x.clone())), "(softmax x)");
        assert_eq!(format!("{}", Symbol::embedding(Symbol::var("w"), x.clone())), "(embedding w x)");
    }

    // ... (mevcut İntegral testleri) ...
    #[test]
    fn test_integration_constants() { 
        let x = Symbol::var("x");
        let y = Symbol::var("y");
        let f1 = Symbol::c(5.0);
        assert_eq!( f1.integrate("x").unwrap().simplify(), (Symbol::c(5.0) * x.clone()).simplify() );
        let f2 = y.clone();
        assert_eq!( f2.integrate("x").unwrap().simplify(), (y.clone() * x.clone()).simplify() );
    }
    #[test]
    fn test_integration_pow_rule() { 
        let x = Symbol::var("x");
        let f1 = x.clone();
        let expected_f1 = Symbol::pow(x.clone(), Symbol::c(2.0)) / Symbol::c(2.0);
        assert_eq!( f1.integrate("x").unwrap().simplify(), expected_f1.simplify() );
        let f3 = Symbol::pow(x.clone(), Symbol::c(-1.0));
        let expected_f3 = Symbol::log(x.clone());
        assert_eq!( f3.integrate("x").unwrap().simplify(), expected_f3.simplify() );
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
        let f_mean = Symbol::mean(x.clone());
        assert!(f_mean.integrate("x").is_err());
    }
}