use pyo3::prelude::*;
use pyo3::types::{PyDict, PyAny, PyList, PyTuple}; // PyAny eklendi
use pyo3::Bound;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyArray2, PyUntypedArrayMethods, PyArrayMethods};

use crate::multivector2d::MultiVector2D;
use crate::multivector3d::MultiVector3D;
use crate::symbolic::Symbol;
use crate::native_ops;
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

/// ✅ Görev 1.3: Compilation result with structure_changed flag
#[pyclass(name = "HypatiaCompileResult")]
#[derive(Clone)]
pub struct HypatiaCompileResult {
    /// The optimized GraphModule
    #[pyo3(get)]
    pub optimized_gm: PyObject,

    /// Whether the graph structure changed during optimization
    /// (true = e-graph applied rewrites like fusion, false = only parameter/constant changes)
    #[pyo3(get)]
    pub structure_changed: bool,
}

#[pymethods]
impl HypatiaCompileResult {
    fn __repr__(&self) -> String {
        format!(
            "HypatiaCompileResult(structure_changed={})",
            self.structure_changed
        )
    }
}

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

/// ✅ REFACTORED: FX GRAPH COMPILATION - example_inputs üzerinden parametre çözme
/// Python'dan gelen 3 argümanlı çağrıya uyacak şekilde imza güncellendi.
/// ✅ Görev 1.3: Returns HypatiaCompileResult with structure_changed flag
#[pyfunction]
pub fn compile_fx_graph(
    py_graph_module: PyObject,      // Arg 1: GraphModule (graph için)
    py_example_inputs: PyObject,    // Arg 2: ✅ YENİ: example_inputs (placeholder sırası ile eşleşir)
    module_info_map: &Bound<'_, PyDict>, // Arg 3: module_info_map
) -> PyResult<Py<HypatiaCompileResult>> {
    Python::with_gil(|py| {
        let gm = py_graph_module.bind(py);
        let example_inputs_bound = py_example_inputs.bind(py);

        eprintln!("[DEBUG] compile_fx_graph called (3-arg: gm, example_inputs, module_info)");

        // example_inputs'i Vec<Bound<PyAny>>'e çevir
        let inputs: Vec<Bound<PyAny>> = if let Ok(tuple) = example_inputs_bound.downcast::<pyo3::types::PyTuple>() {
            tuple.iter().collect()
        } else if let Ok(list) = example_inputs_bound.downcast::<pyo3::types::PyList>() {
            list.iter().collect()
        } else {
            return Err(HypatiaError::new_err("example_inputs must be a tuple or list"));
        };

        eprintln!("[DEBUG] example_inputs count: {}", inputs.len());

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

        // ✅ YENİ: example_inputs'i fx_graph_to_sexpr'a gönder
        let conversion_result = match crate::fx_bridge::fx_graph_to_sexpr(py, &gm, &inputs, &info_map) {
            Ok(result) => result,
            Err(e) => {
                log::warn!("FX graph parsing failed: {}. Falling back to original GraphModule.", e);
                // Fallback: return original model with structure_changed=false
                return Ok(Py::new(py, HypatiaCompileResult {
                    optimized_gm: gm.to_object(py),
                    structure_changed: false,
                })?);
            }
        };

        let sexpr = &conversion_result.sexpr;
        eprintln!("[DEBUG] S-expression (ilk 500 karakter): {:.500}...", sexpr);
        eprintln!("[DEBUG] Placeholder mapping: {} entries", conversion_result.placeholder_map.len());
        
        // optimize_to_ast_with_info'yu kullanarak genel is_inference bayrağını iletin
        let optimized_ast = match crate::egraph_optimizer::optimize_to_ast_with_info(&sexpr, &general_info) {
            Ok(ast) => ast,
            Err(e) => {
                log::warn!("E-graph optimization failed: {}. Falling back to original GraphModule.", e);
                // Fallback: return original model with structure_changed=false
                return Ok(Py::new(py, HypatiaCompileResult {
                    optimized_gm: gm.to_object(py),
                    structure_changed: false,
                })?);
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

        // ✅ YENİ: sexpr_to_fx_graph'a placeholder_map'i gönder
        match crate::fx_bridge::sexpr_to_fx_graph(py, &gm, optimized_ast, &conversion_result.placeholder_map) {
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
                    return Ok(Py::new(py, HypatiaCompileResult {
                        optimized_gm: gm.to_object(py),
                        structure_changed: false,
                    })?);
                }

                // 3. Always check for empty model (critical structural validation)
                if optimized_params.is_empty() && !original_params.is_empty() {
                    log::error!("Optimized model has no parameters (original had {}). Falling back to original model.",
                                original_params.len());
                    return Ok(Py::new(py, HypatiaCompileResult {
                        optimized_gm: gm.to_object(py),
                        structure_changed: false,
                    })?);
                }

                // 4. Parametre checksum kontrolü (mode-dependent, structure-aware)
                match checksum_mode {
                    crate::fx_bridge::ChecksumMode::Off => {
                        // Skip checksum validation entirely
                        log::info!("Checksum validation skipped (mode: Off). Accepting optimized model with {} params.",
                                   original_params.len());
                        Ok(Py::new(py, HypatiaCompileResult {
                            optimized_gm,
                            structure_changed,
                        })?)
                    }
                    crate::fx_bridge::ChecksumMode::Soft => {
                        // Soft mode: Skip checksum if structure changed
                        if structure_changed {
                            log::warn!(
                                "ChecksumMode::Soft + structural change detected → skipping parameter checksum validation. \
                                 Accepting optimized model with {} params.",
                                original_params.len()
                            );
                            Ok(Py::new(py, HypatiaCompileResult {
                                optimized_gm,
                                structure_changed,
                            })?)
                        } else {
                            // No structural change → validate checksums
                            log::info!("ChecksumMode::Soft + no structural change → validating parameter checksums");
                            let orig_checksum = crate::fx_bridge::compute_param_checksum(py, &gm)?;
                            let opt_checksum = crate::fx_bridge::compute_param_checksum(py, &optimized_gm.bind(py))?;

                            if orig_checksum == opt_checksum {
                                log::info!("Parameter checksum matched: {:016x}. Accepting optimized model.", orig_checksum);
                                Ok(Py::new(py, HypatiaCompileResult {
                                    optimized_gm,
                                    structure_changed,
                                })?)
                            } else {
                                log::error!(
                                    "Parameter checksum mismatch: {:016x} vs {:016x}. Falling back to original model.",
                                    orig_checksum, opt_checksum
                                );
                                Ok(Py::new(py, HypatiaCompileResult {
                                    optimized_gm: gm.to_object(py),
                                    structure_changed: false,
                                })?)
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
                            Ok(Py::new(py, HypatiaCompileResult {
                                optimized_gm,
                                structure_changed,
                            })?)
                        } else {
                            log::error!(
                                "Parameter checksum mismatch: {:016x} vs {:016x}. Falling back to original model.",
                                orig_checksum, opt_checksum
                            );
                            Ok(Py::new(py, HypatiaCompileResult {
                                optimized_gm: gm.to_object(py),
                                structure_changed: false,
                            })?)
                        }
                    }
                }
            }
            Err(err) => {
                log::error!("Graph reconstruction failed: {:?}. Falling back to original GraphModule.", err);
                Ok(Py::new(py, HypatiaCompileResult {
                    optimized_gm: gm.to_object(py),
                    structure_changed: false,
                })?)
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

// ============================================================================
// Native Forward Pass - Bypasses PyTorch dispatch overhead
// ============================================================================

/// Native MLP forward pass: all layers fused in a single Rust call.
/// ZERO-COPY: weight and bias data is read directly from numpy arrays.
///
/// Args:
///     input: numpy array [batch, in_features] f32
///     layers: Python list of (weight_np, bias_np_or_None, activation_str) tuples
///
/// Returns:
///     numpy array [batch, out_features] f32
#[pyfunction]
pub fn native_forward<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    layers: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input_shape = input.shape();
    let batch = input_shape[0];
    let in_features = input_shape[1];

    let input_slice = input
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;

    // Process layers with zero-copy weight access
    let mut current = input_slice.to_vec(); // Only copy input once
    let mut current_feat = in_features;

    for item in layers.iter() {
        let tuple = item
            .downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Each layer must be a tuple (weight, bias, activation)"))?;

        // Zero-copy: borrow weight data directly from numpy array
        let weight: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let out_feat = weight.shape()[0];
        let w_slice = weight
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Weight must be C-contiguous: {}", e)))?;

        let bias_obj = tuple.get_item(1)?;
        let activation: String = tuple.get_item(2)?.extract()?;
        let is_relu = activation.to_lowercase() == "relu";

        // Call GEMM with zero-copy references - branch on bias presence
        if bias_obj.is_none() {
            current = native_ops::fused_linear(
                &current, w_slice, None, batch, current_feat, out_feat, is_relu,
            );
        } else {
            let bias: PyReadonlyArray1<f32> = bias_obj.extract()?;
            let b_slice = bias
                .as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Bias must be C-contiguous: {}", e)))?;
            current = native_ops::fused_linear(
                &current, w_slice, Some(b_slice), batch, current_feat, out_feat, is_relu,
            );
        }

        current_feat = out_feat;
    }

    // Convert to numpy 2D array
    let result_2d: Vec<Vec<f32>> = current
        .chunks(current_feat)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2_bound(py, &result_2d)
        .map_err(|e| HypatiaError::new_err(format!("Failed to create output array: {}", e)))
}

/// Native training step: forward -> MSE loss -> backward -> SGD update.
/// All computation in Rust, single Python crossing.
///
/// Args:
///     input: numpy [batch, in_features] f32
///     target: numpy [batch, out_features] f32
///     weights: list of numpy [out, in] f32 arrays (will be modified in-place!)
///     biases: list of numpy [out] f32 arrays or None (will be modified in-place!)
///     activations: list of activation strings ("relu" or "none")
///     lr: learning rate
///
/// Returns:
///     loss value (float)
#[pyfunction]
pub fn native_train_step<'py>(
    _py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    target: PyReadonlyArray2<'py, f32>,
    weights: &Bound<'py, PyList>,
    biases: &Bound<'py, PyList>,
    activations: &Bound<'py, PyList>,
    lr: f32,
) -> PyResult<f32> {
    let batch = input.shape()[0];
    let in_features = input.shape()[1];
    let out_features_final = target.shape()[1];

    let input_slice = input
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;
    let target_slice = target
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Target must be C-contiguous: {}", e)))?;

    let num_layers = weights.len();

    // Extract weight/bias data as mutable owned Vecs
    let mut weight_vecs: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
    let mut bias_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_layers);
    let mut dims: Vec<(usize, usize, bool)> = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let w: PyReadonlyArray2<f32> = weights.get_item(i)?.extract()?;
        let out_f = w.shape()[0];
        let in_f = w.shape()[1];

        weight_vecs.push(
            w.as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Weight must be C-contiguous: {}", e)))?
                .to_vec(),
        );

        let b_obj = biases.get_item(i)?;
        if b_obj.is_none() {
            bias_vecs.push(None);
        } else {
            let b: PyReadonlyArray1<f32> = b_obj.extract()?;
            bias_vecs.push(Some(
                b.as_slice()
                    .map_err(|e| HypatiaError::new_err(format!("Bias must be C-contiguous: {}", e)))?
                    .to_vec(),
            ));
        }

        let act: String = activations.get_item(i)?.extract()?;
        let is_relu = act.to_lowercase() == "relu";

        dims.push((in_f, out_f, is_relu));
    }

    // Run training step
    let loss = native_ops::train_step_sgd(
        input_slice,
        target_slice,
        batch,
        in_features,
        out_features_final,
        &mut weight_vecs,
        &mut bias_vecs,
        &dims,
        lr,
    );

    // Write updated weights/biases back to numpy arrays
    for i in 0..num_layers {
        let w_py = weights.get_item(i)?;
        let mut w_arr: PyReadwriteArray2<f32> = w_py.extract()?;
        let w_slice = w_arr
            .as_slice_mut()
            .map_err(|e| HypatiaError::new_err(format!("Weight write-back failed: {}", e)))?;
        w_slice.copy_from_slice(&weight_vecs[i]);

        if let Some(ref bias_data) = bias_vecs[i] {
            let b_py = biases.get_item(i)?;
            if !b_py.is_none() {
                let mut b_arr: PyReadwriteArray1<f32> = b_py.extract()?;
                let b_slice = b_arr.as_slice_mut()
                    .map_err(|e| HypatiaError::new_err(format!("Bias write-back failed: {}", e)))?;
                b_slice.copy_from_slice(bias_data);
            }
        }
    }

    Ok(loss)
}

// ============================================================================
// INT4 Quantized Inference - For Large Models (1B+ params)
// ============================================================================

/// Quantize weight matrices to INT4 block format.
/// Returns an opaque handle (list of quantized layer data) for use with quantized_forward.
///
/// Args:
///     layers: list of (weight_np, bias_np_or_None, activation_str) tuples
///     group_size: quantization group size (default: 128)
///
/// Returns:
///     list of (packed_data_bytes, scales_np, zeros_np, bias_np_or_None, activation_str,
///              rows, cols, group_size, orig_bytes, quant_bytes) tuples
#[pyfunction]
#[pyo3(signature = (layers, group_size=128))]
pub fn quantize_weights<'py>(
    py: Python<'py>,
    layers: &Bound<'py, PyList>,
    group_size: usize,
) -> PyResult<PyObject> {
    let result = pyo3::types::PyList::empty_bound(py);

    for item in layers.iter() {
        let tuple = item
            .downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Each layer must be a tuple"))?;

        let weight: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let rows = weight.shape()[0];
        let cols = weight.shape()[1];
        let w_slice = weight
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Weight must be C-contiguous: {}", e)))?;

        // Quantize to INT4
        let qt = crate::quantize::QuantizedTensor::quantize(w_slice, rows, cols, group_size);

        let orig_bytes = qt.original_bytes();
        let quant_bytes = qt.memory_bytes();

        // Convert to Python objects
        let scales_np = numpy::PyArray1::from_vec_bound(py, qt.scales);
        let zeros_np = numpy::PyArray1::from_vec_bound(py, qt.zeros);
        let data_np = numpy::PyArray1::from_vec_bound(py, qt.data);

        let bias_obj = tuple.get_item(1)?;
        let activation = tuple.get_item(2)?;

        let layer_tuple = PyTuple::new_bound(
            py,
            &[
                data_np.as_any().clone(),
                scales_np.as_any().clone(),
                zeros_np.as_any().clone(),
                bias_obj.clone(),
                activation.clone(),
                rows.into_py(py).bind(py).clone(),
                cols.into_py(py).bind(py).clone(),
                group_size.into_py(py).bind(py).clone(),
                orig_bytes.into_py(py).bind(py).clone(),
                quant_bytes.into_py(py).bind(py).clone(),
            ],
        );

        result.append(layer_tuple)?;
    }

    Ok(result.into())
}

/// Quantized forward pass: dequantize + GEMM fused, reducing memory bandwidth by ~7x.
///
/// Args:
///     input: numpy [batch, in_features] f32
///     quantized_layers: output from quantize_weights()
///
/// Returns:
///     numpy [batch, out_features] f32
#[pyfunction]
pub fn quantized_forward<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    quantized_layers: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    let input_slice = input
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;

    let mut current = input_slice.to_vec();
    let mut current_feat = input.shape()[1];

    for item in quantized_layers.iter() {
        let tuple = item
            .downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Invalid quantized layer format"))?;

        // ZERO-COPY: borrow data directly from numpy arrays via as_slice()
        let data: numpy::PyReadonlyArray1<u8> = tuple.get_item(0)?.extract()?;
        let scales: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
        let zeros: PyReadonlyArray1<f32> = tuple.get_item(2)?.extract()?;
        let bias_obj = tuple.get_item(3)?;
        let activation: String = tuple.get_item(4)?.extract()?;
        let rows: usize = tuple.get_item(5)?.extract()?;
        let cols: usize = tuple.get_item(6)?.extract()?;
        let group_size: usize = tuple.get_item(7)?.extract()?;

        let is_relu = activation.to_lowercase() == "relu";

        // Zero-copy slices from numpy
        let data_slice = data
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Data not contiguous: {}", e)))?;
        let scales_slice = scales
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Scales not contiguous: {}", e)))?;
        let zeros_slice = zeros
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Zeros not contiguous: {}", e)))?;

        // Call quantized_linear_ref with zero-copy slices (no .to_vec()!)
        if bias_obj.is_none() {
            current = crate::quantize::quantized_linear_ref(
                &current, data_slice, scales_slice, zeros_slice,
                rows, cols, group_size, None, batch, is_relu,
            );
        } else {
            let bias: PyReadonlyArray1<f32> = bias_obj.extract()?;
            let b_slice = bias
                .as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Bias not contiguous: {}", e)))?;
            current = crate::quantize::quantized_linear_ref(
                &current, data_slice, scales_slice, zeros_slice,
                rows, cols, group_size, Some(b_slice), batch, is_relu,
            );
        }
        current_feat = rows;
    }

    let result_2d: Vec<Vec<f32>> = current
        .chunks(current_feat)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2_bound(py, &result_2d)
        .map_err(|e| HypatiaError::new_err(format!("Failed to create quantized output: {}", e)))
}

/// Execute a transformer forward pass with mixed operation types.
///
/// ops_list: List of operation tuples:
///   ("linear", weight_np, bias_np_or_None, activation_str)
///   ("layernorm", gamma_np, beta_np, eps_float)
///   ("attention", wq, bq, wk, bk, wv, bv, wo, bo, n_heads)
///   ("residual_start",)
///   ("residual_end",)
///   ("gelu",)
#[pyfunction]
pub fn transformer_forward_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    ops_list: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    let features = input.shape()[1];
    let input_slice = input
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;

    let mut current = input_slice.to_vec();
    let mut current_feat = features;
    let mut residual_stack: Vec<Vec<f32>> = Vec::new();

    for item in ops_list.iter() {
        let tuple = item
            .downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Op must be a tuple"))?;

        let op_type: String = tuple.get_item(0)?.extract()?;

        match op_type.as_str() {
            "linear" => {
                let weight: PyReadonlyArray2<f32> = tuple.get_item(1)?.extract()?;
                let bias_obj = tuple.get_item(2)?;
                let activation: String = tuple.get_item(3)?.extract()?;

                let w_slice = weight.as_slice()
                    .map_err(|e| HypatiaError::new_err(format!("Weight not contiguous: {}", e)))?;
                let out_feat = weight.shape()[0];
                let in_feat = weight.shape()[1];
                let is_relu = activation == "relu";

                if bias_obj.is_none() {
                    current = native_ops::fused_linear(&current, w_slice, None, batch, in_feat, out_feat, is_relu);
                } else {
                    let bias: PyReadonlyArray1<f32> = bias_obj.extract()?;
                    let b_slice = bias.as_slice()
                        .map_err(|e| HypatiaError::new_err(format!("Bias not contiguous: {}", e)))?;
                    current = native_ops::fused_linear(&current, w_slice, Some(b_slice), batch, in_feat, out_feat, is_relu);
                }
                current_feat = out_feat;
            }
            "layernorm" => {
                let gamma: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
                let beta: PyReadonlyArray1<f32> = tuple.get_item(2)?.extract()?;
                let eps: f32 = tuple.get_item(3)?.extract()?;

                let g_slice = gamma.as_slice()
                    .map_err(|e| HypatiaError::new_err(format!("Gamma not contiguous: {}", e)))?;
                let b_slice = beta.as_slice()
                    .map_err(|e| HypatiaError::new_err(format!("Beta not contiguous: {}", e)))?;
                let feat = gamma.shape()[0];

                current = native_ops::layer_norm(&current, g_slice, b_slice, batch, feat, eps);
                current_feat = feat;
            }
            "attention" => {
                let wq: PyReadonlyArray2<f32> = tuple.get_item(1)?.extract()?;
                let bq_obj = tuple.get_item(2)?;
                let wk: PyReadonlyArray2<f32> = tuple.get_item(3)?.extract()?;
                let bk_obj = tuple.get_item(4)?;
                let wv: PyReadonlyArray2<f32> = tuple.get_item(5)?.extract()?;
                let bv_obj = tuple.get_item(6)?;
                let wo: PyReadonlyArray2<f32> = tuple.get_item(7)?.extract()?;
                let bo_obj = tuple.get_item(8)?;
                let n_heads: usize = tuple.get_item(9)?.extract()?;

                let wq_s = wq.as_slice().map_err(|e| HypatiaError::new_err(format!("{}", e)))?;
                let wk_s = wk.as_slice().map_err(|e| HypatiaError::new_err(format!("{}", e)))?;
                let wv_s = wv.as_slice().map_err(|e| HypatiaError::new_err(format!("{}", e)))?;
                let wo_s = wo.as_slice().map_err(|e| HypatiaError::new_err(format!("{}", e)))?;

                let hidden = wq.shape()[0];

                let extract_bias = |obj: Bound<'py, PyAny>| -> PyResult<Option<Vec<f32>>> {
                    if obj.is_none() {
                        Ok(None)
                    } else {
                        let b: PyReadonlyArray1<f32> = obj.extract()?;
                        Ok(Some(b.as_slice()
                            .map_err(|e| HypatiaError::new_err(format!("{}", e)))?.to_vec()))
                    }
                };

                let bq_v = extract_bias(bq_obj)?;
                let bk_v = extract_bias(bk_obj)?;
                let bv_v = extract_bias(bv_obj)?;
                let bo_v = extract_bias(bo_obj)?;

                current = native_ops::multi_head_attention(
                    &current,
                    wq_s, bq_v.as_deref(), wk_s, bk_v.as_deref(),
                    wv_s, bv_v.as_deref(), wo_s, bo_v.as_deref(),
                    batch, hidden, n_heads,
                );
                current_feat = hidden;
            }
            "residual_start" => {
                residual_stack.push(current.clone());
            }
            "residual_end" => {
                if let Some(residual) = residual_stack.pop() {
                    for (c, r) in current.iter_mut().zip(residual.iter()) {
                        *c += r;
                    }
                }
            }
            "gelu" => {
                native_ops::gelu(&mut current);
            }
            _ => {
                return Err(HypatiaError::new_err(format!("Unknown op type: {}", op_type)));
            }
        }
    }

    let result_2d: Vec<Vec<f32>> = current
        .chunks(current_feat)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2_bound(py, &result_2d)
        .map_err(|e| HypatiaError::new_err(format!("Failed to create transformer output: {}", e)))
}