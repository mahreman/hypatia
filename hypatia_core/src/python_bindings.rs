use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
// (EKLENDİ) Hata sınıfı oluşturmak için gereken importlar
use pyo3::create_exception;
use pyo3::exceptions::PyException;

use crate::multivector2d::MultiVector2D;
use crate::multivector3d::MultiVector3D;
use crate::symbolic::Symbol;

// (EKLENDİ) Kütüphaneye özel hata sınıfı
// Bu, `hypatia_core.HypatiaError` olarak Python'da erişilebilir olacak.
create_exception!(hypatia_core, HypatiaError, PyException);

// ===============================
// Numeric 2D Wrapper
// ===============================
#[pyclass(name = "PyMultiVector2D")]
#[derive(Clone)]
pub struct PyMultiVector2D {
    pub inner: MultiVector2D<f64>,
}

#[pymethods]
impl PyMultiVector2D {
    #[staticmethod]
    pub fn vector(x: f64, y: f64) -> Self {
        Self { inner: MultiVector2D::<f64>::vector(x, y) }
    }

    #[staticmethod]
    pub fn scalar(s: f64) -> Self {
        Self { inner: MultiVector2D::<f64>::scalar(s) }
    }

    #[staticmethod]
    pub fn bivector(e12: f64) -> Self {
        Self { inner: MultiVector2D::<f64>::bivector(e12) }
    }

    #[staticmethod]
    pub fn rotor(theta: f64) -> Self {
        Self { inner: MultiVector2D::<f64>::rotor(theta) }
    }

    pub fn rotate_vector(&self, v: &PyMultiVector2D) -> Self {
        Self { inner: self.inner.rotate_vector(&v.inner) }
    }

    pub fn grade(&self, k: i32) -> Self {
        Self { inner: self.inner.grade(k as u8) }
    }

    // Bileşen okuyucular
    pub fn e1(&self) -> f64 { self.inner.e1 }
    pub fn e2(&self) -> f64 { self.inner.e2 }
    pub fn e12(&self) -> f64 { self.inner.e12 }
    pub fn s(&self) -> f64 { self.inner.s }

    pub fn __repr__(&self) -> String {
        format!(
            "MV2D(s={:.3}, e1={:.3}, e2={:.3}, e12={:.3})",
            self.inner.s, self.inner.e1, self.inner.e2, self.inner.e12
        )
    }
}

// ===============================
// Numeric 3D Wrapper
// ===============================
#[pyclass(name = "PyMultiVector3D")]
#[derive(Clone)]
pub struct PyMultiVector3D {
    pub inner: MultiVector3D<f64>,
}

#[pymethods]
impl PyMultiVector3D {
    #[staticmethod]
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        Self { inner: MultiVector3D::<f64>::vector(x, y, z) }
    }

    #[staticmethod]
    pub fn scalar(s: f64) -> Self {
        Self { inner: MultiVector3D::<f64>::scalar(s) }
    }

    #[staticmethod]
    pub fn bivector(e12: f64, e23: f64, e31: f64) -> Self {
        Self { inner: MultiVector3D::<f64>::bivector(e12, e23, e31) }
    }

    #[staticmethod]
    pub fn trivector(e123: f64) -> Self {
        Self { inner: MultiVector3D::<f64>::trivector(e123) }
    }

    #[staticmethod]
    pub fn rotor(theta: f64, ax: f64, ay: f64, az: f64) -> Self {
        Self { inner: MultiVector3D::<f64>::rotor(theta, ax, ay, az) }
    }

    pub fn rotate_vector(&self, v: &PyMultiVector3D) -> Self {
        Self { inner: self.inner.rotate_vector(&v.inner) }
    }

    pub fn grade(&self, k: i32) -> Self {
        Self { inner: self.inner.grade(k as u8) }
    }

    // Bileşen okuyucular
    pub fn e1(&self) -> f64 { self.inner.e1 }
    pub fn e2(&self) -> f64 { self.inner.e2 }
    pub fn e3(&self) -> f64 { self.inner.e3 }
    pub fn e12(&self) -> f64 { self.inner.e12 }
    pub fn e23(&self) -> f64 { self.inner.e23 }
    pub fn e31(&self) -> f64 { self.inner.e31 }
    pub fn e123(&self) -> f64 { self.inner.e123 }
    pub fn s(&self) -> f64 { self.inner.s }

    pub fn __repr__(&self) -> String {
        format!(
            "MV3D(s={:.3}, e1={:.3}, e2={:.3}, e3={:.3}, e12={:.3}, e23={:.3}, e31={:.3}, e123={:.3})",
            self.inner.s, self.inner.e1, self.inner.e2, self.inner.e3,
            self.inner.e12, self.inner.e23, self.inner.e31, self.inner.e123
        )
    }
}

// ===============================
// PySymbol (Symbolic scalar)
// ===============================
#[pyclass(name = "Symbol")]
#[derive(Clone)]
pub struct PySymbol {
    pub inner: Symbol,
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    pub fn variable(name: &str) -> Self {
        Self { inner: Symbol::Variable(name.to_string()) }
    }

    #[staticmethod]
    pub fn r#const(v: f64) -> Self {
        Self { inner: Symbol::Const(v) }
    }

    pub fn derivative(&self, var: &str) -> Self {
        Self { inner: self.inner.derivative(var).simplify() } // Otomatik sadeleştirme
    }

    pub fn simplify(&self) -> Self {
        Self { inner: self.inner.simplify() }
    }

    pub fn subs(&self, mapping: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut env = std::collections::HashMap::new();
        for (key, value) in mapping {
            let key_str: String = key.extract()?;
            let value_f64: f64 = value.extract()?;
            env.insert(key_str, value_f64);
        }
        Ok(Self { inner: self.inner.subs(&env) })
    }

    pub fn __str__(&self) -> String { format!("{}", self.inner) }
    pub fn __repr__(&self) -> String { format!("Symbol('{}')", self.inner) }

    pub fn __neg__(&self) -> Self {
        Self { inner: (-self.inner.clone()).simplify() } // Sadeleştirme eklendi
    }
    
    pub fn __add__(&self, rhs: &PySymbol) -> Self {
        Self { inner: (self.inner.clone() + rhs.inner.clone()).simplify() } // Sadeleştirme eklendi
    }
    
    pub fn __sub__(&self, rhs: &PySymbol) -> Self {
        Self { inner: (self.inner.clone() + (-rhs.inner.clone())).simplify() } // Sadeleştirme eklendi
    }
    
    pub fn __mul__(&self, rhs: &PySymbol) -> Self {
        Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() } // Sadeleştirme eklendi
    }
}

// ===============================
// Symbolic 2D Wrapper
// ===============================
#[pyclass(name = "PyMultiVector2D_Symbolic")]
#[derive(Clone)]
pub struct PyMultiVector2dSymbolic {
    pub inner: MultiVector2D<Symbol>,
}

#[pymethods]
impl PyMultiVector2dSymbolic {
    #[staticmethod]
    pub fn scalar(s: PySymbol) -> Self {
        Self { inner: MultiVector2D::<Symbol>::scalar(s.inner) }
    }

    #[staticmethod]
    pub fn vector(x: PySymbol, y: PySymbol) -> Self {
        Self { inner: MultiVector2D::<Symbol>::vector(x.inner, y.inner) }
    }

    #[staticmethod]
    pub fn bivector(e12: PySymbol) -> Self {
        Self { inner: MultiVector2D::<Symbol>::bivector(e12.inner) }
    }

    pub fn simplify(&self) -> Self {
        Self { inner: self.inner.simplify() }
    }

    pub fn geometric_product(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        Self { inner: self.inner.clone() * rhs.inner.clone() }
    }

    pub fn e1(&self) -> PySymbol { PySymbol { inner: self.inner.e1.clone() } }
    pub fn e2(&self) -> PySymbol { PySymbol { inner: self.inner.e2.clone() } }
    pub fn e12(&self) -> PySymbol { PySymbol { inner: self.inner.e12.clone() } }
    pub fn s(&self) -> PySymbol { PySymbol { inner: self.inner.s.clone() } }

    pub fn __add__(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self {
            inner: MultiVector2D::<Symbol> {
                s: (a.s.clone() + b.s.clone()).simplify(),
                e1: (a.e1.clone() + b.e1.clone()).simplify(),
                e2: (a.e2.clone() + b.e2.clone()).simplify(),
                e12: (a.e12.clone() + b.e12.clone()).simplify(),
            },
        }
    }

    pub fn __sub__(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self {
            inner: MultiVector2D::<Symbol> {
                s: (a.s.clone() + (-b.s.clone())).simplify(),
                e1: (a.e1.clone() + (-b.e1.clone())).simplify(),
                e2: (a.e2.clone() + (-b.e2.clone())).simplify(),
                e12: (a.e12.clone() + (-b.e12.clone())).simplify(),
            },
        }
    }

    pub fn __mul__(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() }
    }

    pub fn __repr__(&self) -> String {
        let a = &self.inner;
        format!("MV2D_Symbolic(s={}, e1={}, e2={}, e12={})", a.s, a.e1, a.e2, a.e12)
    }
}

// ===============================
// Symbolic 3D Wrapper
// ===============================
#[pyclass(name = "PyMultiVector3D_Symbolic")]
#[derive(Clone)]
pub struct PyMultiVector3dSymbolic {
    pub inner: MultiVector3D<Symbol>,
}

#[pymethods]
impl PyMultiVector3dSymbolic {
    #[staticmethod]
    pub fn scalar(s: PySymbol) -> Self {
        Self { inner: MultiVector3D::<Symbol>::scalar(s.inner) }
    }

    #[staticmethod]
    pub fn vector(x: PySymbol, y: PySymbol, z: PySymbol) -> Self {
        Self { inner: MultiVector3D::<Symbol>::vector(x.inner, y.inner, z.inner) }
    }

    #[staticmethod]
    pub fn bivector(e12: PySymbol, e23: PySymbol, e31: PySymbol) -> Self {
        Self { inner: MultiVector3D::<Symbol>::bivector(e12.inner, e23.inner, e31.inner) }
    }

    #[staticmethod]
    pub fn trivector(e123: PySymbol) -> Self {
        Self { inner: MultiVector3D::<Symbol>::trivector(e123.inner) }
    }

    pub fn simplify(&self) -> Self {
        Self { inner: self.inner.simplify() }
    }

    pub fn geometric_product(&self, rhs: &PyMultiVector3dSymbolic) -> Self {
        Self { inner: self.inner.clone() * rhs.inner.clone() }
    }

    pub fn e1(&self) -> PySymbol { PySymbol { inner: self.inner.e1.clone() } }
    pub fn e2(&self) -> PySymbol { PySymbol { inner: self.inner.e2.clone() } }
    pub fn e3(&self) -> PySymbol { PySymbol { inner: self.inner.e3.clone() } }
    pub fn e12(&self) -> PySymbol { PySymbol { inner: self.inner.e12.clone() } }
    pub fn e23(&self) -> PySymbol { PySymbol { inner: self.inner.e23.clone() } }
    pub fn e31(&self) -> PySymbol { PySymbol { inner: self.inner.e31.clone() } }
    pub fn e123(&self) -> PySymbol { PySymbol { inner: self.inner.e123.clone() } }
    pub fn s(&self) -> PySymbol { PySymbol { inner: self.inner.s.clone() } }

    pub fn __add__(&self, rhs: &PyMultiVector3dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self {
            inner: MultiVector3D::<Symbol> {
                s: (a.s.clone() + b.s.clone()).simplify(),
                e1: (a.e1.clone() + b.e1.clone()).simplify(),
                e2: (a.e2.clone() + b.e2.clone()).simplify(),
                e3: (a.e3.clone() + b.e3.clone()).simplify(),
                e12: (a.e12.clone() + b.e12.clone()).simplify(),
                e23: (a.e23.clone() + b.e23.clone()).simplify(),
                e31: (a.e31.clone() + b.e31.clone()).simplify(),
                e123: (a.e123.clone() + b.e123.clone()).simplify(),
            },
        }
    }

    pub fn __sub__(&self, rhs: &PyMultiVector3dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self {
            inner: MultiVector3D::<Symbol> {
                s: (a.s.clone() + (-b.s.clone())).simplify(),
                e1: (a.e1.clone() + (-b.e1.clone())).simplify(),
                e2: (a.e2.clone() + (-b.e2.clone())).simplify(),
                e3: (a.e3.clone() + (-b.e3.clone())).simplify(),
                e12: (a.e12.clone() + (-b.e12.clone())).simplify(),
                e23: (a.e23.clone() + (-b.e23.clone())).simplify(),
                e31: (a.e31.clone() + (-b.e31.clone())).simplify(),
                e123: (a.e123.clone() + (-b.e123.clone())).simplify(),
            },
        }
    }

    pub fn __mul__(&self, rhs: &PyMultiVector3dSymbolic) -> Self {
        Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() }
    }

    pub fn __repr__(&self) -> String {
        let a = &self.inner;
        format!(
            "MV3D_Symbolic(s={}, e1={}, e2={}, e3={}, e12={}, e23={}, e31={}, e123={})",
            a.s, a.e1, a.e2, a.e3, a.e12, a.e23, a.e31, a.e123
        )
    }
}