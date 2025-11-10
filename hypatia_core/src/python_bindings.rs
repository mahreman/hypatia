use pyo3::prelude::*;
use pyo3::types::PyModule;
use crate::{MultiVector2D, MultiVector3D};
use std::f64::consts::PI;

// -----------------------------------------------------------------
// 1. HATA DÜZELTMESİ: Struct'ları buraya, modülün en üstüne taşı.
// -----------------------------------------------------------------

/// Python için 2D MultiVector sınıfı
#[pyclass(name = "PyMultiVector2D")]
#[derive(Clone)]
pub struct PyMultiVector2D {
    inner: MultiVector2D,
}

#[pymethods]
impl PyMultiVector2D {
    #[new]
    pub fn new(s: f64, e1: f64, e2: f64, e12: f64) -> Self {
        Self { inner: MultiVector2D::new(s, e1, e2, e12) }
    }

    #[staticmethod]
    pub fn scalar(s: f64) -> Self {
        Self { inner: MultiVector2D::scalar(s) }
    }

    #[staticmethod]
    pub fn vector(e1: f64, e2: f64) -> Self {
        Self { inner: MultiVector2D::vector(e1, e2) }
    }

    #[staticmethod]
    pub fn bivector(e12: f64) -> Self {
        Self { inner: MultiVector2D::bivector(e12) }
    }

    #[staticmethod]
    pub fn rotor(angle_rad: f64) -> Self {
        Self { inner: MultiVector2D::rotor(angle_rad) }
    }

    pub fn rotate_vector(&self, other: &Self) -> Self {
        Self { inner: self.inner.rotate_vector(&other.inner) }
    }

    pub fn __add__(&self, other: &Self) -> Self {
        Self { inner: self.inner + other.inner }
    }

    pub fn __mul__(&self, other: &Self) -> Self {
        Self { inner: self.inner * other.inner }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
    
    // __repr__ metodunu eklemek Python'da daha iyi görünür
    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    #[getter]
    pub fn s(&self) -> f64 { self.inner.s }

    #[getter]
    pub fn e1(&self) -> f64 { self.inner.e1 }

    #[getter]
    pub fn e2(&self) -> f64 { self.inner.e2 }

    #[getter]
    pub fn e12(&self) -> f64 { self.inner.e12 }
}

/// Python için 3D MultiVector sınıfı
#[pyclass(name = "PyMultiVector3D")]
#[derive(Clone)]
pub struct PyMultiVector3D {
    inner: MultiVector3D,
}

#[pymethods]
impl PyMultiVector3D {
    #[new]
    pub fn new(s: f64, e1: f64, e2: f64, e3: f64, e12: f64, e23: f64, e31: f64, e123: f64) -> Self {
        Self { inner: MultiVector3D::new(s, e1, e2, e3, e12, e23, e31, e123) }
    }

    #[staticmethod]
    pub fn scalar(s: f64) -> Self {
        Self { inner: MultiVector3D::scalar(s) }
    }

    #[staticmethod]
    pub fn vector(e1: f64, e2: f64, e3: f64) -> Self {
        Self { inner: MultiVector3D::vector(e1, e2, e3) }
    }

    #[staticmethod]
    pub fn bivector(e12: f64, e23: f64, e31: f64) -> Self {
        Self { inner: MultiVector3D::bivector(e12, e23, e31) }
    }

    #[staticmethod]
    pub fn rotor(angle_rad: f64, axis: (f64, f64, f64)) -> Self {
        Self { inner: MultiVector3D::rotor(angle_rad, axis) }
    }

    pub fn rotate_vector(&self, other: &Self) -> Self {
        Self { inner: self.inner.rotate_vector(&other.inner) }
    }

    pub fn __add__(&self, other: &Self) -> Self {
        Self { inner: self.inner + other.inner }
    }

    pub fn __mul__(&self, other: &Self) -> Self {
        Self { inner: self.inner * other.inner }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
    
    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    #[getter]
    pub fn s(&self) -> f64 { self.inner.s }

    #[getter]
    pub fn e1(&self) -> f64 { self.inner.e1 }

    #[getter]
    pub fn e2(&self) -> f64 { self.inner.e2 }

    #[getter]
    pub fn e3(&self) -> f64 { self.inner.e3 }

    #[getter]
    pub fn e12(&self) -> f64 { self.inner.e12 }

    #[getter]
    pub fn e23(&self) -> f64 { self.inner.e23 }

    #[getter]
    pub fn e31(&self) -> f64 { self.inner.e31 }

    #[getter]
    pub fn e123(&self) -> f64 { self.inner.e123 }
}


/// Python için Hypatia binding'leri
#[pymodule]
fn hypatia_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    // Modül fonksiyonları
    #[pyfunction]
    fn demo_2d_rotation() -> PyResult<String> {
        let rotor = MultiVector2D::rotor(PI / 2.0);
        let vector = MultiVector2D::vector(1.0, 0.0);
        let rotated = rotor.rotate_vector(&vector);
        Ok(format!("2D Rotation: (1,0) rotated 90° = ({:.3},{:.3})", rotated.e1, rotated.e2))
    }

    #[pyfunction]
    fn demo_3d_rotation() -> PyResult<String> {
        let rotor = MultiVector3D::rotor(PI / 2.0, (0.0, 0.0, 1.0));
        let vector = MultiVector3D::vector(1.0, 0.0, 0.0);
        let rotated = rotor.rotate_vector(&vector);
        Ok(format!("3D Rotation: (1,0,0) rotated 90° around Z = ({:.3},{:.3},{:.3})", 
                  rotated.e1, rotated.e2, rotated.e3))
    }

    // Sınıfları ve fonksiyonları modüle ekle
    m.add_class::<PyMultiVector2D>()?;
    m.add_class::<PyMultiVector3D>()?;
    m.add_function(wrap_pyfunction!(demo_2d_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(demo_3d_rotation, m)?)?;

    // NumPy entegrasyonunu ekle
    crate::numpy_integration::register_numpy_integration(m)?;

    Ok(())
}