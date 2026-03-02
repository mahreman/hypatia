mod multivector2d;
mod multivector3d;
mod symbolic;
mod symbolic_ga2d;
mod symbolic_ga3d;

pub use multivector2d::MultiVector2D;
pub use multivector3d::MultiVector3D;
pub use symbolic::Symbol;

mod numpy_integration;
mod python_bindings;
mod egraph_optimizer;  // E-graph mod declare
mod fx_bridge;  // ✅ YENİ: FX Graph bridge module
mod native_ops;  // Native fused GEMM operations
mod quantize;    // INT4 block quantization for large models
mod gpu_backend; // GPU acceleration (CUDA/Metal)
mod geometric_ops; // Geometric algebra neural network operations

use pyo3::prelude::*;

#[pymodule]
pub fn _hypatia_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ✅ Task 2.4: Initialize logging system
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init()
        .ok();

    // Default to single-threaded OpenBLAS. The INT4 quantized path uses
    // Rayon+SIMD for parallelism (11-16x speedup on LLaMA-7B/13B).
    // Training with many sequential GEMMs benefits from single-threaded BLAS.
    // Users can override with OPENBLAS_NUM_THREADS env var.
    if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }

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

    // ✅ Görev 1.3: Compilation result class
    m.add_class::<python_bindings::HypatiaCompileResult>()?;

    // --- Modül Fonksiyonları (Optimizasyon ve Ayrıştırma) ---
    m.add_function(wrap_pyfunction!(crate::python_bindings::optimize_ast, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::parse_expr, m)?)?;
    
    // ✅ FEZ 11: Sembolik denklik kontrolü
    m.add_function(wrap_pyfunction!(crate::python_bindings::is_equivalent, m)?)?;
    
    // ✅ YENİ: FX GRAPH COMPILATION
    // Phase 1: Identity pass-through (benchmark harness'i aktive et)
    // Phase 2+: Gerçek FX → S-expr → optimize → FX pipeline
    m.add_function(wrap_pyfunction!(crate::python_bindings::compile_fx_graph, m)?)?;

    // ✅ Task 2.5: Python Logging Control
    m.add_function(wrap_pyfunction!(crate::python_bindings::set_log_level, m)?)?;

    // Native fused forward/training (bypass PyTorch dispatch)
    m.add_function(wrap_pyfunction!(crate::python_bindings::native_forward, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::native_train_step, m)?)?;

    // Fused native kernels for torch.compile (Phase 3 reconstruction)
    m.add_function(wrap_pyfunction!(crate::python_bindings::fused_gelu_mlp_forward, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::fused_linear_relu_forward, m)?)?;

    // INT4 quantized inference (for large models)
    m.add_function(wrap_pyfunction!(crate::python_bindings::quantize_weights, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::quantized_forward, m)?)?;

    // Transformer forward (LayerNorm + Attention + MLP)
    m.add_function(wrap_pyfunction!(crate::python_bindings::transformer_forward_py, m)?)?;

    // Quantization-Aware Training (QAT)
    m.add_function(wrap_pyfunction!(crate::python_bindings::quantized_train_step, m)?)?;

    // Geometric Algebra neural network operations
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga_batch_rotate_2d, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga_batch_rotate_3d, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga2d_product_layer, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga3d_product_layer, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga2d_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::ga3d_normalize, m)?)?;

    // GPU backend info
    m.add_function(wrap_pyfunction!(crate::python_bindings::gpu_info, m)?)?;

    Ok(())
}