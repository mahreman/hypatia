mod multivector2d;
mod multivector3d;
mod symbolic;
mod symbolic_ga2d;
mod symbolic_ga3d;

pub use multivector2d::MultiVector2D;
pub use multivector3d::MultiVector3D;
pub use symbolic::Symbol;

#[cfg(feature = "python")]
mod numpy_integration;

#[cfg(feature = "python")]
mod python_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn hypatia_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Sayısal sınıflar ---
    m.add_class::<python_bindings::PyMultiVector2D>()?;
    m.add_class::<python_bindings::PyMultiVector3D>()?;

    // --- (EKLENDİ) Özel Hata Sınıfı ---
    // python_bindings.rs'de tanımlanan HypatiaError'u modüle ekler
    m.add(
        "HypatiaError",
        py.get_type_bound::<python_bindings::HypatiaError>(),
    )?;

    // --- NumPy entegrasyon fonksiyonları ---
    m.add_function(wrap_pyfunction!(numpy_integration::mv2d_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_integration::mv2d_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_integration::batch_rotate_2d, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_integration::batch_rotate_3d, m)?)?;

    // --- Sembolik sınıflar ---
    m.add_class::<python_bindings::PySymbol>()?;
    m.add_class::<python_bindings::PyMultiVector2dSymbolic>()?;
    m.add_class::<python_bindings::PyMultiVector3dSymbolic>()?;

    Ok(())
}