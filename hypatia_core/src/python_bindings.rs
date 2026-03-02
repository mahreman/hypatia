use pyo3::prelude::*;
use pyo3::types::{PyDict, PyAny, PyList, PyTuple}; // PyAny eklendi
use pyo3::Bound;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyArray1, PyArray2, PyUntypedArrayMethods, PyArrayMethods};

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

    // Zero-copy: transfer Vec ownership directly to numpy, then reshape
    let flat = PyArray1::from_vec_bound(py, current);
    flat.reshape([batch, current_feat])
        .map_err(|e| HypatiaError::new_err(format!("Failed to reshape output: {}", e)))
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
// Fused Native Kernels for torch.compile (Phase 3 Reconstruction)
// ============================================================================

/// Fast tensor→numpy conversion: tries .numpy() first (zero-copy for contiguous f32 CPU tensors),
/// falls back to .detach().float().contiguous().numpy() if needed.
#[inline]
fn tensor_to_numpy_2d<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray2<'py, f32>> {
    // Fast path: torch.compile always passes contiguous f32 CPU tensors
    if let Ok(arr) = tensor.call_method0("numpy").and_then(|a| a.extract::<PyReadonlyArray2<f32>>()) {
        return Ok(arr);
    }
    // Slow path: ensure correct dtype/layout
    tensor
        .call_method0("detach")?
        .call_method0("float")?
        .call_method0("contiguous")?
        .call_method0("numpy")?
        .extract()
}

/// Fast tensor→numpy conversion for 1D tensors.
#[inline]
fn tensor_to_numpy_1d<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray1<'py, f32>> {
    if let Ok(arr) = tensor.call_method0("numpy").and_then(|a| a.extract::<PyReadonlyArray1<f32>>()) {
        return Ok(arr);
    }
    tensor
        .call_method0("detach")?
        .call_method0("contiguous")?
        .call_method0("numpy")?
        .extract()
}

/// Fused GELU MLP: Linear → GELU → Linear in a single Rust call.
/// Uses MKL sgemm_ for GEMM and MKL VML vsTanh for vectorized GELU.
///
/// Called by torch.compile's reconstructed FX graph for the fused_gelu_mlp pattern.
/// Accepts torch tensors (CPU, f32), returns torch tensor.
///
/// Args:
///     input: torch.Tensor [batch, in_features]
///     w1: torch.Tensor [hidden, in_features]
///     b1: torch.Tensor [hidden] or None
///     w2: torch.Tensor [out_features, hidden]
///     b2: torch.Tensor [out_features] or None
///
/// Returns:
///     torch.Tensor [batch, out_features]
#[pyfunction]
pub fn fused_gelu_mlp_forward<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    w1: &Bound<'py, PyAny>,
    b1: &Bound<'py, PyAny>,
    w2: &Bound<'py, PyAny>,
    b2: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let torch = PyModule::import_bound(py, "torch")?;

    let input_np = tensor_to_numpy_2d(input)?;
    let input_slice = input_np
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;
    let batch = input_np.shape()[0];
    let in_feat = input_np.shape()[1];

    let w1_np = tensor_to_numpy_2d(w1)?;
    let w1_slice = w1_np
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("W1 must be C-contiguous: {}", e)))?;
    let hidden = w1_np.shape()[0];

    // Layer 1: linear(input, w1, b1) — no activation
    let mut hidden_out = if b1.is_none() {
        native_ops::fused_linear(input_slice, w1_slice, None, batch, in_feat, hidden, false)
    } else {
        let b1_np = tensor_to_numpy_1d(b1)?;
        let b1_slice = b1_np
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("B1 must be C-contiguous: {}", e)))?;
        native_ops::fused_linear(input_slice, w1_slice, Some(b1_slice), batch, in_feat, hidden, false)
    };

    // GELU activation (MKL VML vsTanh vectorized, 12x faster than scalar)
    native_ops::gelu(&mut hidden_out);

    let w2_np = tensor_to_numpy_2d(w2)?;
    let w2_slice = w2_np
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("W2 must be C-contiguous: {}", e)))?;
    let out_feat = w2_np.shape()[0];

    // Layer 2: linear(gelu_out, w2, b2) — no activation
    let output = if b2.is_none() {
        native_ops::fused_linear(&hidden_out, w2_slice, None, batch, hidden, out_feat, false)
    } else {
        let b2_np = tensor_to_numpy_1d(b2)?;
        let b2_slice = b2_np
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("B2 must be C-contiguous: {}", e)))?;
        native_ops::fused_linear(&hidden_out, w2_slice, Some(b2_slice), batch, hidden, out_feat, false)
    };

    // Zero-copy: Vec ownership → numpy → torch.from_numpy
    let flat = PyArray1::from_vec_bound(py, output);
    let reshaped = flat.reshape([batch, out_feat])
        .map_err(|e| HypatiaError::new_err(format!("Failed to reshape output: {}", e)))?;
    let result = torch.call_method1("from_numpy", (reshaped,))?;
    Ok(result.to_object(py))
}

/// Fused Linear + ReLU in a single Rust call.
/// Uses MKL sgemm_ for GEMM with fused bias + ReLU in one memory pass.
///
/// Args:
///     input: torch.Tensor [batch, in_features]
///     weight: torch.Tensor [out_features, in_features]
///     bias: torch.Tensor [out_features] or None
///
/// Returns:
///     torch.Tensor [batch, out_features]
#[pyfunction]
pub fn fused_linear_relu_forward<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    weight: &Bound<'py, PyAny>,
    bias: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let torch = PyModule::import_bound(py, "torch")?;

    let input_np = tensor_to_numpy_2d(input)?;
    let input_slice = input_np
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input must be C-contiguous: {}", e)))?;
    let batch = input_np.shape()[0];
    let in_feat = input_np.shape()[1];

    let w_np = tensor_to_numpy_2d(weight)?;
    let w_slice = w_np
        .as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weight must be C-contiguous: {}", e)))?;
    let out_feat = w_np.shape()[0];

    let output = if bias.is_none() {
        native_ops::fused_linear(input_slice, w_slice, None, batch, in_feat, out_feat, true)
    } else {
        let b_np = tensor_to_numpy_1d(bias)?;
        let b_slice = b_np
            .as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Bias must be C-contiguous: {}", e)))?;
        native_ops::fused_linear(input_slice, w_slice, Some(b_slice), batch, in_feat, out_feat, true)
    };

    let flat = PyArray1::from_vec_bound(py, output);
    let reshaped = flat.reshape([batch, out_feat])
        .map_err(|e| HypatiaError::new_err(format!("Failed to reshape output: {}", e)))?;
    let result = torch.call_method1("from_numpy", (reshaped,))?;
    Ok(result.to_object(py))
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

    // Zero-copy: transfer Vec ownership directly to numpy
    let flat = PyArray1::from_vec_bound(py, current);
    flat.reshape([batch, current_feat])
        .map_err(|e| HypatiaError::new_err(format!("Failed to reshape quantized output: {}", e)))
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
#[pyo3(signature = (input, ops_list, seq_len=1))]
pub fn transformer_forward_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    ops_list: &Bound<'py, PyList>,
    seq_len: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let total_rows = input.shape()[0];
    let features = input.shape()[1];
    let batch = total_rows / seq_len;
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
                    current = native_ops::fused_linear(&current, w_slice, None, total_rows, in_feat, out_feat, is_relu);
                } else {
                    let bias: PyReadonlyArray1<f32> = bias_obj.extract()?;
                    let b_slice = bias.as_slice()
                        .map_err(|e| HypatiaError::new_err(format!("Bias not contiguous: {}", e)))?;
                    current = native_ops::fused_linear(&current, w_slice, Some(b_slice), total_rows, in_feat, out_feat, is_relu);
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

                current = native_ops::layer_norm(&current, g_slice, b_slice, total_rows, feat, eps);
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
                    batch, seq_len, hidden, n_heads,
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

    // Zero-copy: transfer Vec ownership directly to numpy
    let flat = PyArray1::from_vec_bound(py, current);
    flat.reshape([total_rows, current_feat])
        .map_err(|e| HypatiaError::new_err(format!("Failed to reshape transformer output: {}", e)))
}

/// Quantization-Aware Training step.
/// Forward with INT4 quantized weights, backward with f32 weights (STE).
///
/// Args:
///   input: [batch, in_features] f32 array
///   target: [batch, out_features] f32 array
///   weights: list of [out, in] f32 arrays (mutable, updated in-place)
///   biases: list of [out] f32 arrays or None (mutable)
///   activations: list of activation name strings ("relu" or "none")
///   lr: learning rate
///   group_size: INT4 quantization group size (default 128)
///
/// Returns: loss value (float)
#[pyfunction]
pub fn quantized_train_step<'py>(
    _py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    target: PyReadonlyArray2<'py, f32>,
    weights: &Bound<'py, PyList>,
    biases: &Bound<'py, PyList>,
    activations: &Bound<'py, PyList>,
    lr: f32,
    group_size: Option<usize>,
) -> PyResult<f32> {
    let gs = group_size.unwrap_or(128);
    let batch = input.shape()[0];
    let in_features = input.shape()[1];

    let x = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let y = target.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Target: {}", e)))?;

    let out_features = target.shape()[1];

    // Extract mutable weights
    let n_layers = weights.len();
    let mut w_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut b_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_layers);
    let mut layer_dims: Vec<(usize, usize, bool)> = Vec::with_capacity(n_layers);
    let mut current_in = in_features;

    for i in 0..n_layers {
        let w_arr: numpy::PyReadonlyArray2<f32> = weights.get_item(i)?.extract()?;
        let out_f = w_arr.shape()[0];
        let in_f = w_arr.shape()[1];
        w_vecs.push(w_arr.as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Weight {}: {}", i, e)))?
            .to_vec());

        let bias_obj = biases.get_item(i)?;
        if bias_obj.is_none() {
            b_vecs.push(None);
        } else {
            let b_arr: PyReadonlyArray1<f32> = bias_obj.extract()?;
            b_vecs.push(Some(b_arr.as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Bias {}: {}", i, e)))?
                .to_vec()));
        }

        let act: String = activations.get_item(i)?.extract()?;
        let is_relu = act == "relu";
        layer_dims.push((in_f, out_f, is_relu));
        current_in = out_f;
    }

    // Run QAT training step
    let loss = crate::quantize::quantized_train_step_sgd(
        x, y, batch, in_features, out_features,
        &mut w_vecs, &mut b_vecs, &layer_dims, gs, lr,
    );

    // Copy updated weights back to numpy arrays
    for i in 0..n_layers {
        let w_obj = weights.get_item(i)?;
        let mut w_arr: numpy::PyReadwriteArray2<f32> = w_obj.extract()?;
        let w_slice = w_arr.as_slice_mut()
            .map_err(|e| HypatiaError::new_err(format!("Write weight {}: {}", i, e)))?;
        w_slice.copy_from_slice(&w_vecs[i]);

        let bias_obj = biases.get_item(i)?;
        if !bias_obj.is_none() {
            if let Some(ref b_data) = b_vecs[i] {
                let mut b_arr: PyReadwriteArray1<f32> = bias_obj.extract()?;
                let b_slice = b_arr.as_slice_mut()
                    .map_err(|e| HypatiaError::new_err(format!("Write bias {}: {}", i, e)))?;
                b_slice.copy_from_slice(b_data);
            }
        }
    }

    Ok(loss)
}

/// Batch 2D rotation using geometric algebra rotor.
/// input: [batch, 2], theta: rotation angle in radians
/// Returns: [batch, 2] rotated vectors
#[pyfunction]
pub fn ga_batch_rotate_2d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    theta: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    let cols = input.shape()[1];
    if cols != 2 {
        return Err(HypatiaError::new_err("Input must have 2 columns (e1, e2)"));
    }
    let data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;

    let result = crate::geometric_ops::batch_rotor_2d(data, batch, theta);

    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, 2])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// Batch 3D rotation using geometric algebra rotor.
/// input: [batch, 3], axis: [3] normalized rotation axis, theta: rotation angle
/// Returns: [batch, 3] rotated vectors
#[pyfunction]
pub fn ga_batch_rotate_3d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    axis: PyReadonlyArray1<'py, f32>,
    theta: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    if input.shape()[1] != 3 {
        return Err(HypatiaError::new_err("Input must have 3 columns (e1, e2, e3)"));
    }
    let ax_slice = axis.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Axis: {}", e)))?;
    if ax_slice.len() != 3 {
        return Err(HypatiaError::new_err("Axis must have 3 elements"));
    }
    let data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let axis_arr = [ax_slice[0], ax_slice[1], ax_slice[2]];

    let result = crate::geometric_ops::batch_rotor_3d(data, batch, &axis_arr, theta);

    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, 3])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// 2D Geometric Product layer: computes geometric product of input multivectors
/// with learned weight multivectors.
/// input: [batch, 4] (s, e1, e2, e12), weights: [out_features, 4]
/// Returns: [batch, out_features * 4]
#[pyfunction]
pub fn ga2d_product_layer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weights: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    if input.shape()[1] != 4 {
        return Err(HypatiaError::new_err("Input must have 4 columns (s, e1, e2, e12)"));
    }
    let out_features = weights.shape()[0];
    if weights.shape()[1] != 4 {
        return Err(HypatiaError::new_err("Weights must have 4 columns"));
    }

    let in_data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let w_data = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;

    let result = crate::geometric_ops::ga2d_geometric_product_layer(in_data, w_data, batch, out_features);

    let out_cols = out_features * 4;
    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, out_cols])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// 3D Geometric Product layer.
/// input: [batch, 8] (s, e1, e2, e3, e12, e23, e31, e123)
/// weights: [out_features, 8]
/// Returns: [batch, out_features * 8]
#[pyfunction]
pub fn ga3d_product_layer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weights: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    if input.shape()[1] != 8 {
        return Err(HypatiaError::new_err("Input must have 8 columns"));
    }
    let out_features = weights.shape()[0];
    if weights.shape()[1] != 8 {
        return Err(HypatiaError::new_err("Weights must have 8 columns"));
    }

    let in_data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let w_data = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;

    let result = crate::geometric_ops::ga3d_geometric_product_layer(in_data, w_data, batch, out_features);

    let out_cols = out_features * 8;
    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, out_cols])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// Normalize 2D multivectors.
/// input: [batch, 4] -> output: [batch, 4] (unit multivectors)
#[pyfunction]
pub fn ga2d_normalize<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    if input.shape()[1] != 4 {
        return Err(HypatiaError::new_err("Input must have 4 columns"));
    }
    let data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;

    let result = crate::geometric_ops::ga2d_normalize(data, batch);

    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, 4])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// Normalize 3D multivectors.
/// input: [batch, 8] -> output: [batch, 8] (unit multivectors)
#[pyfunction]
pub fn ga3d_normalize<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let batch = input.shape()[0];
    if input.shape()[1] != 8 {
        return Err(HypatiaError::new_err("Input must have 8 columns"));
    }
    let data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;

    let result = crate::geometric_ops::ga3d_normalize(data, batch);

    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([batch, 8])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

/// Get GPU backend info
#[pyfunction]
pub fn gpu_info() -> String {
    crate::gpu_backend::gpu_info()
}

// ============================================================================
// Fused Attention: Multi-Head Self-Attention (Rust Native)
// ============================================================================

/// Fused multi-head self-attention forward pass.
///
/// Computes Q/K/V projections, scaled dot-product attention with causal mask,
/// and output projection in a single call using optimized GEMM kernels.
///
/// Args:
///     input: [total_rows, hidden] f32 tensor (total_rows = batch * seq_len)
///     wq, wk, wv, wo: [hidden, hidden] weight matrices
///     bq, bk, bv, bo: [hidden] bias vectors (or None)
///     batch: number of sequences
///     seq_len: tokens per sequence
///     n_heads: number of attention heads
///
/// Returns:
///     [total_rows, hidden] output tensor
#[pyfunction]
#[pyo3(signature = (input, wq, bq, wk, bk, wv, bv, wo, bo, batch, seq_len, n_heads))]
pub fn fused_attention_forward<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    wq: PyReadonlyArray2<'py, f32>,
    bq: Option<PyReadonlyArray1<'py, f32>>,
    wk: PyReadonlyArray2<'py, f32>,
    bk: Option<PyReadonlyArray1<'py, f32>>,
    wv: PyReadonlyArray2<'py, f32>,
    bv: Option<PyReadonlyArray1<'py, f32>>,
    wo: PyReadonlyArray2<'py, f32>,
    bo: Option<PyReadonlyArray1<'py, f32>>,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let total_rows = input.shape()[0];
    let hidden = input.shape()[1];

    if total_rows != batch * seq_len {
        return Err(HypatiaError::new_err(
            format!("input rows ({}) != batch ({}) * seq_len ({})", total_rows, batch, seq_len)
        ));
    }
    if hidden % n_heads != 0 {
        return Err(HypatiaError::new_err(
            format!("hidden ({}) must be divisible by n_heads ({})", hidden, n_heads)
        ));
    }

    let x = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let wq_s = wq.as_slice().map_err(|e| HypatiaError::new_err(format!("Wq: {}", e)))?;
    let wk_s = wk.as_slice().map_err(|e| HypatiaError::new_err(format!("Wk: {}", e)))?;
    let wv_s = wv.as_slice().map_err(|e| HypatiaError::new_err(format!("Wv: {}", e)))?;
    let wo_s = wo.as_slice().map_err(|e| HypatiaError::new_err(format!("Wo: {}", e)))?;

    let bq_v: Option<Vec<f32>> = bq.map(|b| b.as_slice().unwrap().to_vec());
    let bk_v: Option<Vec<f32>> = bk.map(|b| b.as_slice().unwrap().to_vec());
    let bv_v: Option<Vec<f32>> = bv.map(|b| b.as_slice().unwrap().to_vec());
    let bo_v: Option<Vec<f32>> = bo.map(|b| b.as_slice().unwrap().to_vec());

    let result = crate::native_ops::multi_head_attention(
        x,
        wq_s, bq_v.as_deref(),
        wk_s, bk_v.as_deref(),
        wv_s, bv_v.as_deref(),
        wo_s, bo_v.as_deref(),
        batch, seq_len, hidden, n_heads,
    );

    let flat = PyArray1::from_vec_bound(py, result);
    flat.reshape([total_rows, hidden])
        .map_err(|e| HypatiaError::new_err(format!("{}", e)))
}

// ============================================================================
// Neuromorphic Computing: ANN→SNN, LIF Neuron Simulation
// ============================================================================

/// Convert ANN weights to SNN and run neuromorphic inference.
///
/// Args:
///     layer_weights: List of (weight_2d, bias_1d_or_None) tuples
///     input: 1D input array
///     v_threshold: Spike threshold (default 1.0)
///     beta: Membrane decay factor (default 0.95)
///     timesteps: Simulation timesteps (default 32)
///
/// Returns:
///     output: 1D output array (decoded firing rates)
#[pyfunction]
#[pyo3(signature = (layer_weights, input, v_threshold=1.0, beta=0.95, timesteps=32))]
pub fn neuromorphic_forward<'py>(
    py: Python<'py>,
    layer_weights: &Bound<'py, PyList>,
    input: PyReadonlyArray1<f32>,
    v_threshold: f32,
    beta: f32,
    timesteps: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::neuromorphic::{ANNLayer, LIFParams, ResetMode, ann_to_snn};

    let input_data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;

    // Parse layer weights from Python
    let mut layers = Vec::new();
    for i in 0..layer_weights.len() {
        let item = layer_weights.get_item(i)?;
        let tuple = item.downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Each layer must be a (weight, bias) tuple"))?;

        let weight: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let w_shape = weight.shape().to_vec();
        let out_feat = w_shape[0];
        let in_feat = w_shape[1];
        let w_data: Vec<f32> = weight.as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Weight: {}", e)))?
            .to_vec();

        let bias_item = tuple.get_item(1)?;
        let bias_data: Option<Vec<f32>> = if bias_item.is_none() {
            None
        } else {
            let b: PyReadonlyArray1<f32> = bias_item.extract()?;
            Some(b.as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Bias: {}", e)))?
                .to_vec())
        };

        layers.push(ANNLayer {
            weights: w_data,
            bias: bias_data,
            out_features: out_feat,
            in_features: in_feat,
        });
    }

    let params = LIFParams {
        v_threshold,
        beta,
        timesteps,
        reset_mode: ResetMode::Soft,
    };

    let snn = ann_to_snn(layers, params);
    let output = snn.forward(input_data);

    Ok(PyArray1::from_vec_bound(py, output))
}

/// Run neuromorphic inference with spike statistics.
///
/// Returns: (output, stats_list)
///   where stats_list is a list of dicts with keys:
///   layer_idx, total_spikes, total_possible, firing_rate
#[pyfunction]
#[pyo3(signature = (layer_weights, input, v_threshold=1.0, beta=0.95, timesteps=32))]
pub fn neuromorphic_forward_with_stats<'py>(
    py: Python<'py>,
    layer_weights: &Bound<'py, PyList>,
    input: PyReadonlyArray1<f32>,
    v_threshold: f32,
    beta: f32,
    timesteps: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyList>)> {
    use crate::neuromorphic::{ANNLayer, LIFParams, ResetMode, ann_to_snn};

    let input_data = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;

    let mut layers = Vec::new();
    for i in 0..layer_weights.len() {
        let item = layer_weights.get_item(i)?;
        let tuple = item.downcast::<PyTuple>()
            .map_err(|_| HypatiaError::new_err("Each layer must be a (weight, bias) tuple"))?;

        let weight: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let w_shape = weight.shape().to_vec();
        let out_feat = w_shape[0];
        let in_feat = w_shape[1];
        let w_data: Vec<f32> = weight.as_slice()
            .map_err(|e| HypatiaError::new_err(format!("Weight: {}", e)))?
            .to_vec();

        let bias_item = tuple.get_item(1)?;
        let bias_data: Option<Vec<f32>> = if bias_item.is_none() {
            None
        } else {
            let b: PyReadonlyArray1<f32> = bias_item.extract()?;
            Some(b.as_slice()
                .map_err(|e| HypatiaError::new_err(format!("Bias: {}", e)))?
                .to_vec())
        };

        layers.push(ANNLayer {
            weights: w_data,
            bias: bias_data,
            out_features: out_feat,
            in_features: in_feat,
        });
    }

    let params = LIFParams {
        v_threshold,
        beta,
        timesteps,
        reset_mode: ResetMode::Soft,
    };

    let snn = ann_to_snn(layers, params);
    let (output, stats) = snn.forward_with_stats(input_data);

    let output_arr = PyArray1::from_vec_bound(py, output);

    let stats_list = PyList::empty_bound(py);
    for stat in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("layer_idx", stat.layer_idx)?;
        dict.set_item("total_spikes", stat.total_spikes)?;
        dict.set_item("total_possible", stat.total_possible)?;
        dict.set_item("firing_rate", stat.firing_rate)?;
        stats_list.append(dict)?;
    }

    Ok((output_arr, stats_list))
}

/// Optimize an S-expression for neuromorphic hardware target.
/// Applies ReLU→LIF rewrite rules via e-graph equality saturation.
///
/// Args:
///     expr_str: S-expression string (e.g. "(relu (linear w b x))")
///
/// Returns:
///     Optimized S-expression with neuromorphic operators
#[pyfunction]
pub fn optimize_for_neuromorphic(expr_str: &str) -> PyResult<String> {
    match crate::egraph_optimizer::optimize_for_neuromorphic(expr_str) {
        Ok(expr) => Ok(crate::egraph_optimizer::rec_to_string(&expr)),
        Err(e) => Err(HypatiaError::new_err(format!("Neuromorphic optimization failed: {}", e))),
    }
}

/// Estimate energy consumption for neuromorphic vs conventional execution.
///
/// Args:
///     in_features: Input dimension
///     out_features: Output dimension
///     timesteps: Number of LIF simulation timesteps
///     avg_firing_rate: Average neuron firing rate (0.0-1.0)
///
/// Returns:
///     Dict with keys: neuromorphic_nj, conventional_nj, energy_ratio
#[pyfunction]
#[pyo3(signature = (in_features, out_features, timesteps=32, avg_firing_rate=0.1))]
pub fn estimate_neuromorphic_energy<'py>(
    py: Python<'py>,
    in_features: usize,
    out_features: usize,
    timesteps: usize,
    avg_firing_rate: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::neuromorphic::NeuromorphicEnergy;

    let energy_model = NeuromorphicEnergy::default();
    let (neuro_nj, conv_nj) = energy_model.compare_energy(
        in_features, out_features, timesteps, avg_firing_rate,
    );

    let dict = PyDict::new_bound(py);
    dict.set_item("neuromorphic_nj", neuro_nj)?;
    dict.set_item("conventional_nj", conv_nj)?;
    dict.set_item("energy_ratio", neuro_nj / conv_nj)?;
    dict.set_item("savings_pct", (1.0 - neuro_nj / conv_nj) * 100.0)?;
    Ok(dict)
}

// ============================================================================
// Sparse Tensor IR
// ============================================================================

/// Convert a dense weight matrix to CSR sparse format.
///
/// Args:
///     weights: 2D numpy array [out_features, in_features]
///     threshold: minimum absolute value to keep (magnitude pruning)
///
/// Returns:
///     dict with keys: row_ptrs, col_indices, values, rows, cols, nnz,
///           sparsity, compression_ratio, memory_bytes, dense_memory_bytes
#[pyfunction]
#[pyo3(signature = (weights, threshold))]
pub fn to_sparse_csr<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::sparse_ops;

    let shape = weights.shape();
    let rows = shape[0];
    let cols = shape[1];
    let w = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;

    let csr = sparse_ops::to_sparse_csr(w, rows, cols, threshold);

    let dict = PyDict::new_bound(py);
    dict.set_item("row_ptrs", csr.row_ptrs.iter().map(|&v| v as u64).collect::<Vec<_>>())?;
    dict.set_item("col_indices", csr.col_indices.iter().map(|&v| v as u64).collect::<Vec<_>>())?;
    dict.set_item("values", PyArray1::from_slice_bound(py, &csr.values))?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("nnz", csr.nnz())?;
    dict.set_item("sparsity", csr.sparsity())?;
    dict.set_item("compression_ratio", csr.compression_ratio())?;
    dict.set_item("memory_bytes", csr.memory_bytes())?;
    dict.set_item("dense_memory_bytes", csr.dense_memory_bytes())?;
    Ok(dict)
}

/// Sparse linear forward: output = input @ sparse_weight.T + bias
///
/// Args:
///     input: 2D numpy array [batch, in_features]
///     row_ptrs: CSR row pointers (list of ints)
///     col_indices: CSR column indices (list of ints)
///     values: CSR non-zero values (1D numpy array)
///     bias: optional 1D numpy array [out_features]
///     rows: number of output features
///     cols: number of input features
///     relu: apply ReLU activation
///
/// Returns:
///     2D numpy array [batch, out_features]
#[pyfunction]
#[pyo3(signature = (input, row_ptrs, col_indices, values, bias, rows, cols, relu))]
pub fn sparse_linear_forward<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: PyReadonlyArray1<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    rows: usize,
    cols: usize,
    relu: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use crate::sparse_ops::{SparseWeightCSR, sparse_linear, sparse_linear_parallel};

    let batch = input.shape()[0];
    let in_feat = input.shape()[1];

    if in_feat != cols {
        return Err(HypatiaError::new_err(
            format!("Input features ({}) != CSR cols ({})", in_feat, cols)
        ));
    }

    let x = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let vals = values.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Values: {}", e)))?;

    let csr = SparseWeightCSR {
        row_ptrs,
        col_indices,
        values: vals.to_vec(),
        rows,
        cols,
    };

    let bias_vec: Option<Vec<f32>> = bias.map(|b| {
        b.as_slice().unwrap_or(&[]).to_vec()
    });
    let bias_ref = bias_vec.as_deref();

    let output = if batch >= 4 {
        sparse_linear_parallel(x, &csr, bias_ref, batch, relu)
    } else {
        sparse_linear(x, &csr, bias_ref, batch, relu)
    };

    Ok(PyArray2::from_vec2_bound(py, &output.chunks(rows).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| HypatiaError::new_err(format!("Output array: {}", e)))?)
}

/// Compute magnitude pruning threshold for a target sparsity.
///
/// Args:
///     weights: 2D numpy array
///     sparsity: target sparsity ratio (0.0-1.0), e.g. 0.5 = 50% zeros
///
/// Returns:
///     float threshold value
#[pyfunction]
#[pyo3(signature = (weights, sparsity))]
pub fn compute_sparsity_threshold(
    weights: PyReadonlyArray2<f32>,
    sparsity: f32,
) -> PyResult<f32> {
    let w = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;
    Ok(crate::sparse_ops::compute_pruning_threshold(w, sparsity))
}

/// Get sparsity statistics for a weight matrix.
///
/// Args:
///     weights: 2D numpy array
///
/// Returns:
///     dict with sparsity stats
#[pyfunction]
#[pyo3(signature = (weights,))]
pub fn sparsity_stats<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;
    let stats = crate::sparse_ops::sparsity_stats(w);

    let dict = PyDict::new_bound(py);
    dict.set_item("total_elements", stats.total_elements)?;
    dict.set_item("nonzero_elements", stats.nonzero_elements)?;
    dict.set_item("zero_elements", stats.zero_elements)?;
    dict.set_item("sparsity_ratio", stats.sparsity_ratio)?;
    dict.set_item("dense_bytes", stats.dense_bytes)?;
    dict.set_item("sparse_bytes_estimate", stats.sparse_bytes_estimate)?;
    Ok(dict)
}

// ============================================================================
// Mixed Precision: FP16/BF16
// ============================================================================

/// Convert a dense weight matrix to half-precision (FP16 or BF16).
///
/// Args:
///     weights: 2D numpy array [out_features, in_features]
///     format: "fp16" or "bf16"
///
/// Returns:
///     dict with keys: data (u16 array), rows, cols, format,
///           memory_bytes, fp32_memory_bytes, compression_ratio
#[pyfunction]
#[pyo3(signature = (weights, format))]
pub fn to_half_precision<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
    format: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::mixed_precision::{self, HalfPrecision};

    let precision = match format.to_lowercase().as_str() {
        "fp16" => HalfPrecision::FP16,
        "bf16" => HalfPrecision::BF16,
        _ => return Err(HypatiaError::new_err(
            format!("Invalid format '{}'. Use 'fp16' or 'bf16'", format)
        )),
    };

    let shape = weights.shape();
    let rows = shape[0];
    let cols = shape[1];
    let w = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;

    let hw = mixed_precision::to_half_weights(w, rows, cols, precision);

    let dict = PyDict::new_bound(py);
    // Store raw u16 data as numpy array for efficient Python<->Rust transfer
    let data_u16: Vec<u16> = hw.data.clone();
    dict.set_item("data", PyArray1::from_vec_bound(py, data_u16))?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("format", format.to_lowercase())?;
    dict.set_item("memory_bytes", hw.memory_bytes())?;
    dict.set_item("fp32_memory_bytes", hw.fp32_memory_bytes())?;
    dict.set_item("compression_ratio", hw.compression_ratio())?;
    Ok(dict)
}

/// Mixed-precision linear forward: output = input_f32 @ half_weight.T + bias
///
/// Weights are stored in half-precision (u16), computation in FP32.
///
/// Args:
///     input: 2D numpy array [batch, in_features] (float32)
///     data: 1D numpy array of u16 raw half-precision weights
///     bias: optional 1D numpy array [out_features] (float32)
///     rows: out_features
///     cols: in_features
///     format: "fp16" or "bf16"
///     relu: apply ReLU activation
///
/// Returns:
///     2D numpy array [batch, out_features] (float32)
#[pyfunction]
#[pyo3(signature = (input, data, bias, rows, cols, format, relu))]
pub fn mixed_precision_forward<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    data: PyReadonlyArray1<'py, u16>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    rows: usize,
    cols: usize,
    format: &str,
    relu: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use crate::mixed_precision::{self, HalfPrecision, HalfWeights, mixed_precision_linear, mixed_precision_linear_parallel};

    let precision = match format.to_lowercase().as_str() {
        "fp16" => HalfPrecision::FP16,
        "bf16" => HalfPrecision::BF16,
        _ => return Err(HypatiaError::new_err(
            format!("Invalid format '{}'. Use 'fp16' or 'bf16'", format)
        )),
    };

    let batch = input.shape()[0];
    let in_feat = input.shape()[1];

    if in_feat != cols {
        return Err(HypatiaError::new_err(
            format!("Input features ({}) != weight cols ({})", in_feat, cols)
        ));
    }

    let x = input.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Input: {}", e)))?;
    let raw_data = data.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Data: {}", e)))?;

    let hw = HalfWeights {
        data: raw_data.to_vec(),
        rows,
        cols,
        format: precision,
    };

    let bias_vec: Option<Vec<f32>> = bias.map(|b| b.as_slice().unwrap_or(&[]).to_vec());
    let bias_ref = bias_vec.as_deref();

    let output = if batch >= 4 {
        mixed_precision_linear_parallel(x, &hw, bias_ref, batch, relu)
    } else {
        mixed_precision_linear(x, &hw, bias_ref, batch, relu)
    };

    Ok(PyArray2::from_vec2_bound(py, &output.chunks(rows).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| HypatiaError::new_err(format!("Output array: {}", e)))?)
}

/// Analyze precision loss from FP32→half conversion.
///
/// Args:
///     weights: 2D numpy array (float32)
///     format: "fp16" or "bf16"
///
/// Returns:
///     dict with precision stats
#[pyfunction]
#[pyo3(signature = (weights, format))]
pub fn mixed_precision_stats<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
    format: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::mixed_precision::{self, HalfPrecision};

    let precision = match format.to_lowercase().as_str() {
        "fp16" => HalfPrecision::FP16,
        "bf16" => HalfPrecision::BF16,
        _ => return Err(HypatiaError::new_err(
            format!("Invalid format '{}'. Use 'fp16' or 'bf16'", format)
        )),
    };

    let w = weights.as_slice()
        .map_err(|e| HypatiaError::new_err(format!("Weights: {}", e)))?;
    let stats = mixed_precision::precision_stats(w, precision);

    let dict = PyDict::new_bound(py);
    dict.set_item("total_elements", stats.total_elements)?;
    dict.set_item("max_abs_error", stats.max_abs_error)?;
    dict.set_item("rmse", stats.rmse)?;
    dict.set_item("overflow_count", stats.overflow_count)?;
    dict.set_item("underflow_count", stats.underflow_count)?;
    dict.set_item("format", format.to_lowercase())?;
    Ok(dict)
}

// ============================================================================
// Visualization
// ============================================================================

/// Convert an S-expression to GraphViz DOT format for visualization.
///
/// Args:
///     expr: S-expression string, e.g. "(relu (linear w b x))"
///     graph_name: name for the DOT graph (default: "hypatia")
///
/// Returns:
///     DOT format string (can be rendered with `dot -Tpng`)
#[pyfunction]
#[pyo3(signature = (expr, graph_name = "hypatia"))]
pub fn expr_to_dot(
    expr: &str,
    graph_name: &str,
) -> PyResult<String> {
    crate::visualization::sexpr_to_dot(expr, graph_name)
        .map_err(|e| HypatiaError::new_err(e))
}

/// Convert an S-expression to an ASCII tree visualization.
///
/// Args:
///     expr: S-expression string
///
/// Returns:
///     ASCII tree string
#[pyfunction]
#[pyo3(signature = (expr,))]
pub fn expr_to_ascii_tree(
    expr: &str,
) -> PyResult<String> {
    crate::visualization::sexpr_to_ascii_tree(expr)
        .map_err(|e| HypatiaError::new_err(e))
}

/// Build an optimization report comparing before/after S-expressions.
///
/// Args:
///     input_expr: original S-expression
///     output_expr: optimized S-expression
///
/// Returns:
///     dict with report fields: input/output node counts, fusions, rewrites, etc.
#[pyfunction]
#[pyo3(signature = (input_expr, output_expr))]
pub fn optimization_report<'py>(
    py: Python<'py>,
    input_expr: &str,
    output_expr: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let report = crate::visualization::build_optimization_report(input_expr, output_expr)
        .map_err(|e| HypatiaError::new_err(e))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("input_expr", &report.input_expr)?;
    dict.set_item("output_expr", &report.output_expr)?;
    dict.set_item("input_node_count", report.input_node_count)?;
    dict.set_item("output_node_count", report.output_node_count)?;
    dict.set_item("node_reduction", report.node_reduction)?;
    dict.set_item("fusions_found", report.fusions_found)?;
    dict.set_item("rewrites_applied", report.rewrites_applied)?;

    // Convert node type maps to Python dicts
    let input_types = PyDict::new_bound(py);
    for (k, v) in &report.input_node_types {
        input_types.set_item(k, v)?;
    }
    dict.set_item("input_node_types", input_types)?;

    let output_types = PyDict::new_bound(py);
    for (k, v) in &report.output_node_types {
        output_types.set_item(k, v)?;
    }
    dict.set_item("output_node_types", output_types)?;

    Ok(dict)
}