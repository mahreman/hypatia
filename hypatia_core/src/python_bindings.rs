use pyo3::prelude::*;
use pyo3::types::{PyDict, PyAny}; // PyAny eklendi
use pyo3::Bound;

use crate::multivector2d::MultiVector2D;
use crate::multivector3d::MultiVector3D;
use crate::symbolic::Symbol;
use std::collections::HashMap;

// Hypatia özel hata sınıfı
pyo3::create_exception!(hypatia_core, HypatiaError, pyo3::exceptions::PyException);

// ====================================================================
// ✅ GÜNCEL: Manuel FromPyObject implementasyonu
// ====================================================================

// #[derive(FromPyObject, Debug, Clone)] // Otomatik türetmeyi kaldır
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub module_type: String,
    pub has_bias: bool,
    pub is_inference: bool, // Tekrar bool oldu, çünkü varsayılan değer atayacağız
}

// KeyError'ı önlemek için manuel olarak FromPyObject uygulayın
impl<'source> FromPyObject<'source> for ModuleInfo {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let dict = ob.downcast::<PyDict>()?;

        // 'type' zorunludur, eksikse hata verir
        let module_type: String = dict
            .get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'type' key missing in ModuleInfo"))?
            .extract()?;

        // 'has_bias' opsiyoneldir, eksikse 'true' varsayılır
        let has_bias: bool = dict
            .get_item("has_bias")?
            .map_or(Ok(true), |item| item.extract())?; // Eksikse true

        // 'is_inference' opsiyoneldir, eksikse 'true' (çıkarım modu) varsayılır
        let is_inference: bool = dict
            .get_item("is_inference")?
            .map_or(Ok(true), |item| item.extract())?; // Eksikse true

        Ok(ModuleInfo {
            module_type,
            has_bias,
            is_inference,
        })
    }
}
// ====================================================================


// ===============================
// Numeric 2D Wrapper (Değişiklik yok)
// ===============================
#[pyclass(name = "PyMultiVector2D")]
#[derive(Clone)]
pub struct PyMultiVector2D { pub inner: MultiVector2D<f64>, }
#[pymethods]
impl PyMultiVector2D {
    #[staticmethod]
    pub fn vector(x: f64, y: f64) -> Self { Self { inner: MultiVector2D::<f64>::vector(x, y) } }
    #[staticmethod]
    pub fn scalar(s: f64) -> Self { Self { inner: MultiVector2D::<f64>::scalar(s) } }
    #[staticmethod]
    pub fn bivector(e12: f64) -> Self { Self { inner: MultiVector2D::<f64>::bivector(e12) } }
    #[staticmethod]
    pub fn rotor(theta: f64) -> Self { Self { inner: MultiVector2D::<f64>::rotor(theta) } }
    pub fn rotate_vector(&self, v: &PyMultiVector2D) -> Self { Self { inner: self.inner.rotate_vector(&v.inner) } }
    pub fn grade(&self, k: i32) -> Self { Self { inner: self.inner.grade(k as u8) } }
    pub fn e1(&self) -> f64 { self.inner.e1 }
    pub fn e2(&self) -> f64 { self.inner.e2 }
    pub fn e12(&self) -> f64 { self.inner.e12 }
    pub fn s(&self) -> f64 { self.inner.s }
    pub fn __repr__(&self) -> String {
        format!("MV2D(s={:.3}, e1={:.3}, e2={:.3}, e12={:.3})", self.inner.s, self.inner.e1, self.inner.e2, self.inner.e12)
    }
}

// ===============================
// Numeric 3D Wrapper (Değişiklik yok)
// ===============================
#[pyclass(name = "PyMultiVector3D")]
#[derive(Clone)]
pub struct PyMultiVector3D { pub inner: MultiVector3D<f64>, }
#[pymethods]
impl PyMultiVector3D {
    #[staticmethod]
    pub fn vector(x: f64, y: f64, z: f64) -> Self { Self { inner: MultiVector3D::<f64>::vector(x, y, z) } }
    #[staticmethod]
    pub fn scalar(s: f64) -> Self { Self { inner: MultiVector3D::<f64>::scalar(s) } }
    #[staticmethod]
    pub fn bivector(e12: f64, e23: f64, e31: f64) -> Self { Self { inner: MultiVector3D::<f64>::bivector(e12, e23, e31) } }
    #[staticmethod]
    pub fn trivector(e123: f64) -> Self { Self { inner: MultiVector3D::<f64>::trivector(e123) } }
    #[staticmethod]
    pub fn rotor(theta: f64, ax: f64, ay: f64, az: f64) -> Self { Self { inner: MultiVector3D::<f64>::rotor(theta, ax, ay, az) } }
    pub fn rotate_vector(&self, v: &PyMultiVector3D) -> Self { Self { inner: self.inner.rotate_vector(&v.inner) } }
    pub fn grade(&self, k: i32) -> Self { Self { inner: self.inner.grade(k as u8) } }
    pub fn e1(&self) -> f64 { self.inner.e1 } pub fn e2(&self) -> f64 { self.inner.e2 }
    pub fn e3(&self) -> f64 { self.inner.e3 } pub fn e12(&self) -> f64 { self.inner.e12 }
    pub fn e23(&self) -> f64 { self.inner.e23 } pub fn e31(&self) -> f64 { self.inner.e31 }
    pub fn e123(&self) -> f64 { self.inner.e123 } pub fn s(&self) -> f64 { self.inner.s }
    pub fn __repr__(&self) -> String {
        format!("MV3D(s={:.3}, e1={:.3}, e2={:.3}, e3={:.3}, e12={:.3}, e23={:.3}, e31={:.3}, e123={:.3})",
            self.inner.s, self.inner.e1, self.inner.e2, self.inner.e3,
            self.inner.e12, self.inner.e23, self.inner.e31, self.inner.e123
        )
    }
}

// ===============================
// PySymbol (Değişiklik yok)
// ===============================
#[pyclass(name = "Symbol")]
#[derive(Clone)]
pub struct PySymbol {
    pub inner: Symbol,
}
#[pymethods]
impl PySymbol {
    // ============ TEMEL OLUŞTURUCULAR ============
    #[staticmethod]
    pub fn variable(name: &str) -> Self { Self { inner: Symbol::Variable(name.to_string()) } }
    #[staticmethod]
    pub fn r#const(v: f64) -> Self { Self { inner: Symbol::Const(v) } }
    // ============ MATEMATİKSEL FONKSİYONLAR ============
    #[staticmethod]
    pub fn exp(x: &PySymbol) -> Self { Self { inner: Symbol::exp(x.inner.clone()) } }
    #[staticmethod]
    pub fn log(x: &PySymbol) -> Self { Self { inner: Symbol::log(x.inner.clone()) } }
    #[staticmethod]
    pub fn sqrt(x: &PySymbol) -> Self { Self { inner: Symbol::sqrt(x.inner.clone()) } }
    #[staticmethod]
    pub fn pow(base: &PySymbol, exp: &PySymbol) -> Self { Self { inner: Symbol::pow(base.inner.clone(), exp.inner.clone()) } }
    // ============ AKTİVASYON FONKSİYONLARI ============
    #[staticmethod]
    pub fn relu(x: &PySymbol) -> Self { Self { inner: Symbol::relu(x.inner.clone()) } }
    #[staticmethod]
    pub fn relu_grad(x: &PySymbol) -> Self { Self { inner: Symbol::relu_grad(x.inner.clone()) } }
    #[staticmethod]
    pub fn sigmoid(x: &PySymbol) -> Self { Self { inner: Symbol::sigmoid(x.inner.clone()) } }
    #[staticmethod]
    pub fn tanh(x: &PySymbol) -> Self { Self { inner: Symbol::tanh(x.inner.clone()) } }
    // ============ YENİ: FEZ 10 AI Operatörleri ============
    #[staticmethod]
    pub fn softmax(x: &PySymbol) -> Self { Self { inner: Symbol::softmax(x.inner.clone()) } }
    #[staticmethod]
    pub fn mean(x: &PySymbol) -> Self { Self { inner: Symbol::mean(x.inner.clone()) } }
    #[staticmethod]
    pub fn variance(x: &PySymbol) -> Self { Self { inner: Symbol::variance(x.inner.clone()) } }
    // ============ YARDIMCI FONKSİYONLAR ============
    #[staticmethod]
    pub fn max(a: &PySymbol, b: &PySymbol) -> Self { Self { inner: Symbol::max(a.inner.clone(), b.inner.clone()) } }
    #[staticmethod]
    pub fn min(a: &PySymbol, b: &PySymbol) -> Self { Self { inner: Symbol::min(a.inner.clone(), b.inner.clone()) } }
    // ============ SEMBOL MANİPÜLASYONU ============
    pub fn derivative(&self, var: &str) -> Self { Self { inner: self.inner.derivative(var).simplify() } }
    pub fn integrate(&self, var: &str) -> PyResult<Self> {
        match self.inner.integrate(var) {
            Ok(integrated_symbol) => Ok(Self { inner: integrated_symbol.simplify() }),
            Err(e) => Err(HypatiaError::new_err(format!("IntegrationError: {}", e))),
        }
    }
    pub fn simplify(&self) -> Self { Self { inner: self.inner.simplify() } }
    pub fn subs(&self, mapping: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut env = HashMap::new();
        for (key, value) in mapping.iter() {
            let key_str: String = key.extract()?;
            let value_f64: f64 = value.extract()?;
            env.insert(key_str, value_f64);
        }
        Ok(Self { inner: self.inner.subs(&env) })
    }
    pub fn eval(&self, mapping: &Bound<'_, PyDict>) -> PyResult<f64> {
        let mut env = HashMap::new();
        for (key, value) in mapping.iter() {
            let key_str: String = key.extract()?;
            let value_f64: f64 = value.extract()?;
            env.insert(key_str, value_f64);
        }
        self.inner.eval(&env)
            .map_err(|e| HypatiaError::new_err(format!("EvalError: {}", e)))
    }
    // ============ PYTHON OPERATÖRLER ============
    pub fn __str__(&self) -> String { format!("{}", self.inner) }
    pub fn __repr__(&self) -> String { format!("Symbol('{}')", self.inner) }
    pub fn __neg__(&self) -> Self { Self { inner: (-self.inner.clone()).simplify() } }
    pub fn __add__(&self, rhs: &PySymbol) -> Self { Self { inner: (self.inner.clone() + rhs.inner.clone()).simplify() } }
    pub fn __sub__(&self, rhs: &PySymbol) -> Self { Self { inner: (self.inner.clone() - rhs.inner.clone()).simplify() } }
    pub fn __mul__(&self, rhs: &PySymbol) -> Self { Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() } }
    pub fn __truediv__(&self, rhs: &PySymbol) -> Self { Self { inner: (self.inner.clone() / rhs.inner.clone()).simplify() } }
}

// ===============================
// Symbolic 2D/3D Wrappers (Değişiklik yok)
// ===============================
#[pyclass(name = "PyMultiVector2dSymbolic")]
#[derive(Clone)]
pub struct PyMultiVector2dSymbolic { pub inner: MultiVector2D<Symbol>, }
#[pymethods]
impl PyMultiVector2dSymbolic {
    #[staticmethod]
    pub fn scalar(s: PySymbol) -> Self { Self { inner: MultiVector2D::<Symbol>::scalar(s.inner) } }
    #[staticmethod]
    pub fn vector(x: PySymbol, y: PySymbol) -> Self { Self { inner: MultiVector2D::<Symbol>::vector(x.inner, y.inner) } }
    #[staticmethod]
    pub fn bivector(e12: PySymbol) -> Self { Self { inner: MultiVector2D::<Symbol>::bivector(e12.inner) } }
    pub fn simplify(&self) -> Self { Self { inner: self.inner.simplify() } }
    pub fn __add__(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self { inner: MultiVector2D::<Symbol> {
                s: (a.s.clone() + b.s.clone()).simplify(), e1: (a.e1.clone() + b.e1.clone()).simplify(),
                e2: (a.e2.clone() + b.e2.clone()).simplify(), e12: (a.e12.clone() + b.e12.clone()).simplify(),
        }}
    }
    pub fn __sub__(&self, rhs: &PyMultiVector2dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self { inner: MultiVector2D::<Symbol> {
                s: (a.s.clone() + (-b.s.clone())).simplify(), e1: (a.e1.clone() + (-b.e1.clone())).simplify(),
                e2: (a.e2.clone() + (-b.e2.clone())).simplify(), e12: (a.e12.clone() + (-b.e12.clone())).simplify(),
        }}
    }
    pub fn __mul__(&self, rhs: &PyMultiVector2dSymbolic) -> Self { Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() } }
    pub fn __repr__(&self) -> String { let a = &self.inner;
        format!("MV2D_Symbolic(s={}, e1={}, e2={}, e12={})", a.s, a.e1, a.e2, a.e12)
    }
}

#[pyclass(name = "PyMultiVector3D_Symbolic")]
#[derive(Clone)]
pub struct PyMultiVector3dSymbolic { pub inner: MultiVector3D<Symbol>, }
#[pymethods]
impl PyMultiVector3dSymbolic {
    #[staticmethod]
    pub fn scalar(s: PySymbol) -> Self { Self { inner: MultiVector3D::<Symbol>::scalar(s.inner) } }
    #[staticmethod]
    pub fn vector(x: PySymbol, y: PySymbol, z: PySymbol) -> Self { Self { inner: MultiVector3D::<Symbol>::vector(x.inner, y.inner, z.inner) } }
    #[staticmethod]
    pub fn bivector(e12: PySymbol, e23: PySymbol, e31: PySymbol) -> Self { Self { inner: MultiVector3D::<Symbol>::bivector(e12.inner, e23.inner, e31.inner) } }
    #[staticmethod]
    pub fn trivector(e123: PySymbol) -> Self { Self { inner: MultiVector3D::<Symbol>::trivector(e123.inner) } }
    pub fn simplify(&self) -> Self { Self { inner: self.inner.simplify() } }
    pub fn geometric_product(&self, rhs: &PyMultiVector3dSymbolic) -> Self { Self { inner: self.inner.clone() * rhs.inner.clone() } }
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
        Self { inner: MultiVector3D::<Symbol> {
                s: (a.s.clone() + b.s.clone()).simplify(), e1: (a.e1.clone() + b.e1.clone()).simplify(),
                e2: (a.e2.clone() + b.e2.clone()).simplify(), e3: (a.e3.clone() + b.e3.clone()).simplify(),
                e12: (a.e12.clone() + b.e12.clone()).simplify(), e23: (a.e23.clone() + b.e23.clone()).simplify(),
                e31: (a.e31.clone() + b.e31.clone()).simplify(), e123: (a.e123.clone() + b.e123.clone()).simplify(),
        }}
    }
    pub fn __sub__(&self, rhs: &PyMultiVector3dSymbolic) -> Self {
        let a = &self.inner; let b = &rhs.inner;
        Self { inner: MultiVector3D::<Symbol> {
                s: (a.s.clone() + (-b.s.clone())).simplify(), e1: (a.e1.clone() + (-b.e1.clone())).simplify(),
                e2: (a.e2.clone() + (-b.e2.clone())).simplify(), e3: (a.e3.clone() + (-b.e3.clone())).simplify(),
                e12: (a.e12.clone() + (-b.e12.clone())).simplify(), e23: (a.e23.clone() + (-b.e23.clone())).simplify(),
                e31: (a.e31.clone() + (-b.e31.clone())).simplify(), e123: (a.e123.clone() + (-b.e123.clone())).simplify(),
        }}
    }
    pub fn __mul__(&self, rhs: &PyMultiVector3dSymbolic) -> Self { Self { inner: (self.inner.clone() * rhs.inner.clone()).simplify() } }
    pub fn __repr__(&self) -> String { let a = &self.inner;
        format!("MV3D_Symbolic(s={}, e1={}, e2={}, e3={}, e12={}, e23={}, e31={}, e123={})",
            a.s, a.e1, a.e2, a.e3, a.e12, a.e23, a.e31, a.e123
        )
    }
}

// ===============================
// Python Modül Fonksiyonları
// ===============================

/// Optimizasyon motorunu (v3.0) çalıştırır. (String -> String)
#[pyfunction]
pub fn optimize_ast(expr_str: String) -> PyResult<String> {
    // optimize_to_ast_internal'ın varsayılan is_inference=true değerini kullanır
    let info = ModuleInfo {
        module_type: "Unknown".to_string(),
        has_bias: false,
        is_inference: true // Manuel olarak ayarlandı
    };
    match crate::egraph_optimizer::optimize_to_ast_with_info(&expr_str, &info) {
        Ok(ast) => Ok(crate::egraph_optimizer::rec_to_string(&ast)),
        Err(e) => Err(HypatiaError::new_err(format!("OptimizationError: {}", e))),
    }
}

/// S-expression string'ini PySymbol nesnesine ayrıştırır (FEZ 7)
#[pyfunction]
pub fn parse_expr(expr_str: String) -> PyResult<PySymbol> {
    match crate::egraph_optimizer::parse_expr_to_symbol(&expr_str) {
        Ok(symbol) => Ok(PySymbol { inner: symbol }),
        Err(e) => Err(HypatiaError::new_err(format!("ParseError: {}", e))),
    }
}

/// ✅ FEZ 11: Sembolik Denklik Kontrolü
#[pyfunction]
pub fn is_equivalent(expr1_str: String, expr2_str: String) -> PyResult<bool> {
    match crate::egraph_optimizer::is_equivalent(&expr1_str, &expr2_str) {
        Ok(is_equiv) => Ok(is_equiv),
        Err(e) => Err(HypatiaError::new_err(format!("EquivalenceCheckError: {}", e))),
    }
}

/// ✅ GÜNCELLENDİ: FX GRAPH COMPILATION (4 Argümanlı)
/// Python'dan gelen 4 argümanlı çağrıya uyacak şekilde imza güncellendi.
#[pyfunction]
pub fn compile_fx_graph(
    py_graph_module: PyObject,      // Arg 1: GraphModule (graph için)
    py_original_model: PyObject,    // Arg 2: ✅ YENİ: Orijinal model (parametreler için)
    _example_inputs: PyObject,      // Arg 3: example_inputs (List)
    module_info_map: &Bound<'_, PyDict>, // Arg 4: module_info_map
) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        let gm = py_graph_module.bind(py);
        let model_bound = py_original_model.bind(py); // ✅ Orijinal model parametreler için
        
        eprintln!("[DEBUG] compile_fx_graph called (4-arg: gm, original_model, example_inputs, module_info)");

        // ✅ DÜZELTME: `module_info_map` (Dict) 3. argümandır ve zorunludur.
        let info_map: HashMap<String, ModuleInfo> = module_info_map
            .iter()
            .map(|(k, v)| {
                Ok((k.extract::<String>()?, v.extract::<ModuleInfo>()?))
            })
            .collect::<PyResult<HashMap<String, ModuleInfo>>>()?;

        // Modelin genel çıkarım modunda olup olmadığını belirle
        let general_info = info_map.values().next().cloned().unwrap_or(ModuleInfo {
            module_type: "Unknown".to_string(),
            has_bias: true,
            is_inference: true // Varsayılan
        });
        
        // Bu çağrı artık `gm`'yi doğru şekilde almalı
        let conversion_result = match crate::fx_bridge::fx_graph_to_sexpr(py, &gm, &info_map) {
            Ok(result) => result,
            Err(e) => {
                log::warn!("FX graph parsing failed: {}. Falling back to original GraphModule.", e);
                return Ok(gm.to_object(py)); // Fallback
            }
        };

        let sexpr = &conversion_result.sexpr;
        let param_var_map = &conversion_result.param_var_map;
        eprintln!("[DEBUG] S-expression (ilk 500 karakter): {:.500}...", sexpr);
        eprintln!("[DEBUG] Parameter mapping: {} entries", param_var_map.len());
        
        // optimize_to_ast_with_info'yu kullanarak genel is_inference bayrağını iletin
        let optimized_ast = match crate::egraph_optimizer::optimize_to_ast_with_info(&sexpr, &general_info) {
            Ok(ast) => ast,
            Err(e) => {
                log::warn!("E-graph optimization failed: {}. Falling back to original GraphModule.", e);
                return Ok(gm.to_object(py)); // Fallback
            }
        };
        
        let optimized_sexpr = crate::egraph_optimizer::rec_to_string(&optimized_ast);
        eprintln!("[DEBUG] Optimized AST: {}", optimized_sexpr);

        // Detect if structure changed (optimization modified the graph)
        let structure_changed = sexpr != &optimized_sexpr;
        if structure_changed {
            log::info!("Graph structure changed during optimization (e-graph applied rewrites)");
        } else {
            log::info!("Graph structure preserved (no e-graph rewrites applied)");
        }

        // ✅ DÜZELTME: `sexpr_to_fx_graph`'a `model` ve `gm` olarak `gm`'nin kendisini (farklı binding'lerle) ver
        match crate::fx_bridge::sexpr_to_fx_graph(py, model_bound, &gm, optimized_ast, param_var_map) {
            Ok(optimized_gm) => {
                // ✅ YENİ: Gelişmiş parametre doğrulama mekanizması

                // Read checksum validation mode from environment
                let checksum_mode = crate::fx_bridge::ChecksumMode::from_env();

                // 1. Orijinal ve optimize edilmiş parametreleri al
                let original_params: Vec<_> = gm.call_method0("parameters")?
                    .iter()?
                    .collect::<PyResult<_>>()?;
                let optimized_params: Vec<_> = optimized_gm.bind(py).call_method0("parameters")?
                    .iter()?
                    .collect::<PyResult<_>>()?;

                // 2. Always check parameter count (critical structural validation)
                if original_params.len() != optimized_params.len() {
                    log::error!("Parameter count mismatch during reconstruction: {} → {}. Falling back to original model.",
                                original_params.len(), optimized_params.len());
                    return Ok(gm.to_object(py));
                }

                // 3. Always check for empty model (critical structural validation)
                if optimized_params.is_empty() && !original_params.is_empty() {
                    log::error!("Optimized model has no parameters (original had {}). Falling back to original model.",
                                original_params.len());
                    return Ok(gm.to_object(py));
                }

                // 4. Parametre checksum kontrolü (mode-dependent, structure-aware)
                match checksum_mode {
                    crate::fx_bridge::ChecksumMode::Off => {
                        // Skip checksum validation entirely
                        log::info!("Checksum validation skipped (mode: Off). Accepting optimized model with {} params.",
                                   original_params.len());
                        Ok(optimized_gm)
                    }
                    crate::fx_bridge::ChecksumMode::Soft => {
                        // Soft mode: Skip checksum if structure changed
                        if structure_changed {
                            log::warn!(
                                "ChecksumMode::Soft + structural change detected → skipping parameter checksum validation. \
                                 Accepting optimized model with {} params.",
                                original_params.len()
                            );
                            Ok(optimized_gm)
                        } else {
                            // No structural change → validate checksums
                            log::info!("ChecksumMode::Soft + no structural change → validating parameter checksums");
                            let orig_checksum = crate::fx_bridge::compute_param_checksum(py, &gm)?;
                            let opt_checksum = crate::fx_bridge::compute_param_checksum(py, &optimized_gm.bind(py))?;

                            if orig_checksum == opt_checksum {
                                log::info!("Parameter checksum matched: {:016x}. Accepting optimized model.", orig_checksum);
                                Ok(optimized_gm)
                            } else {
                                log::error!(
                                    "Parameter checksum mismatch: {:016x} vs {:016x}. Falling back to original model.",
                                    orig_checksum, opt_checksum
                                );
                                Ok(gm.to_object(py))
                            }
                        }
                    }
                    crate::fx_bridge::ChecksumMode::Strict => {
                        // Strict mode: Always validate checksums regardless of structure change
                        log::info!("ChecksumMode::Strict → validating parameter checksums (structure_changed: {})",
                                   structure_changed);
                        let orig_checksum = crate::fx_bridge::compute_param_checksum(py, &gm)?;
                        let opt_checksum = crate::fx_bridge::compute_param_checksum(py, &optimized_gm.bind(py))?;

                        if orig_checksum == opt_checksum {
                            log::info!("Parameter checksum matched: {:016x}. Accepting optimized model.", orig_checksum);
                            Ok(optimized_gm)
                        } else {
                            log::error!(
                                "Parameter checksum mismatch: {:016x} vs {:016x}. Falling back to original model.",
                                orig_checksum, opt_checksum
                            );
                            Ok(gm.to_object(py))
                        }
                    }
                }
            }
            Err(err) => {
                log::error!("Graph reconstruction failed: {:?}. Falling back to original GraphModule.", err);
                Ok(gm.to_object(py))
            }
        }
    })
}

// ============================================================================
// ✅ Task 2.5: Python Logging Control
// ============================================================================

/// Set the Rust logging level from Python
///
/// # Arguments
/// * `level` - Log level: "debug", "info", "warn", or "error"
///
/// # Example
/// ```python
/// import hypatia_core
/// hypatia_core.set_log_level("debug")  # Enable debug logging
/// ```
#[pyfunction]
pub fn set_log_level(level: &str) -> PyResult<()> {
    use log::LevelFilter;
    let filter = match level.to_lowercase().as_str() {
        "debug" => LevelFilter::Debug,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _ => return Err(HypatiaError::new_err(format!("Invalid log level: {}", level))),
    };

    env_logger::Builder::new()
        .filter_level(filter)
        .try_init()
        .ok();

    Ok(())
}