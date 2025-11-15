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
mod egraph_optimizer;  // E-graph mod declare
#[cfg(feature = "python")]
mod fx_bridge;  // ✅ YENİ: FX Graph bridge module

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
pub fn hypatia_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ✅ Task 2.4: Initialize logging system
    // Default to "info" level, can be overridden with RUST_LOG environment variable
    // e.g., RUST_LOG=debug python script.py
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init()
        .ok(); // Ignore error if already initialized

    // --- Sayısal sınıflar ---
    m.add_class::<python_bindings::PyMultiVector2D>()?;
    m.add_class::<python_bindings::PyMultiVector3D>()?;
    
    // --- Özel Hata Sınıfı ---
    m.add(
        "HypatiaError",
        py.get_type_bound::<python_bindings::HypatiaError>(),
    )?;
    
    // --- NumPy entegrasyon fonksiyonları ---
    m.add_function(wrap_pyfunction!(crate::numpy_integration::mv2d_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(crate::numpy_integration::mv2d_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(crate::numpy_integration::batch_rotate_2d, m)?)?;
    m.add_function(wrap_pyfunction!(crate::numpy_integration::batch_rotate_3d, m)?)?;
    
    // --- Sembolik sınıflar ---
    m.add_class::<python_bindings::PySymbol>()?;
    m.add_class::<python_bindings::PyMultiVector2dSymbolic>()?;
    m.add_class::<python_bindings::PyMultiVector3dSymbolic>()?;
    
    // --- Modül Fonksiyonları (Optimizasyon ve Ayrıştırma) ---
    m.add_function(wrap_pyfunction!(crate::python_bindings::optimize_ast, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::parse_expr, m)?)?;
    
    // ✅ FEZ 11: Sembolik denklik kontrolü
    m.add_function(wrap_pyfunction!(crate::python_bindings::is_equivalent, m)?)?;
    
    // ✅ YENİ: FX GRAPH COMPILATION
    // Phase 1: Identity pass-through (benchmark harness'i aktive et)
    // Phase 2+: Gerçek FX → S-expr → optimize → FX pipeline
    m.add_function(wrap_pyfunction!(crate::python_bindings::compile_fx_graph, m)?)?;
    
    Ok(())
}