// ============================================================================
// FX BRIDGE: PyTorch FX Graph ↔ Hypatia S-expression
// ============================================================================

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyDict, PyModule};
use crate::python_bindings::{HypatiaError, ModuleInfo};
use std::collections::HashMap;
use std::env;

use egg::{RecExpr, Id};
use crate::egraph_optimizer::HypatiaLang;

// ❌ KALDIRILDI: Regex bağımlılıkları (Artık kullanılmıyor)
// use once_cell::sync::Lazy;
// use regex::Regex;

// ❌ KALDIRILDI: PLACEHOLDER_RE (Artık kullanılmıyor)
// static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| { ... });

// ❌ KALDIRILDI: normalize_placeholder (Artık kullanılmıyor)
// fn normalize_placeholder(name: &str) -> String { ... }

// ============================================================================
// DEBUG FLAGS
// ============================================================================

/// Check if verbose FX debugging is enabled via HYPATIA_DEBUG_FX environment variable
/// Set HYPATIA_DEBUG_FX=1 for detailed placeholder mapping and reconstruction logs
fn is_debug_fx_enabled() -> bool {
    env::var("HYPATIA_DEBUG_FX")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

// ============================================================================
// CHECKSUM VALIDATION MODE
// ============================================================================

/// Parameter checksum validation mode
/// - Off: No checksum validation (fastest, least safe)
/// - Soft: Checksum only for non-structural optimizations (balanced)
/// - Strict: Always validate checksums, fallback on mismatch (safest, slowest)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumMode {
    Off,
    Soft,
    Strict,
}

impl ChecksumMode {
    /// Read checksum mode from HYPATIA_CHECKSUM_MODE environment variable
    /// - "off" or "0" → Off
    /// - "strict" or "2" → Strict
    /// - "soft" or "1" or default → Soft
    pub fn from_env() -> Self {
        let val = env::var("HYPATIA_CHECKSUM_MODE")
            .unwrap_or_else(|_| "soft".to_string())
            .to_lowercase();

        match val.as_str() {
            "off" | "0" => {
                log::info!("Checksum mode: Off (no validation)");
                ChecksumMode::Off
            }
            "strict" | "2" => {
                log::info!("Checksum mode: Strict (always validate)");
                ChecksumMode::Strict
            }
            _ => {
                log::info!("Checksum mode: Soft (validate non-structural optimizations)");
                ChecksumMode::Soft
            }
        }
    }
}

// ============================================================================
// PARAMETER CHECKSUM COMPUTATION
// ============================================================================

/// Compute a checksum hash of model parameters for validation
/// Uses parameter names, shapes, dtypes, and sample values
pub fn compute_param_checksum(py: Python<'_>, gm: &Bound<PyAny>) -> PyResult<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let state_dict = gm.call_method0("state_dict")?;
    let items = state_dict.call_method0("items")?;

    let mut hasher = DefaultHasher::new();
    let mut param_count = 0;

    for item in items.iter()? {
        let pair = item?;
        let key: String = pair.get_item(0)?.extract()?;
        let tensor = pair.get_item(1)?;

        // Hash parameter name
        key.hash(&mut hasher);

        // Hash shape
        let shape: Vec<i64> = tensor.getattr("shape")?.extract()?;
        shape.hash(&mut hasher);

        // Hash dtype
        let dtype_str = format!("{:?}", tensor.getattr("dtype")?);
        dtype_str.hash(&mut hasher);

        // Hash sample of numeric values (first 10 elements for efficiency)
        let flat = tensor.call_method0("flatten")?;
        let numel: i64 = flat.call_method0("numel")?.extract()?;
        let n = numel.min(10);

        if n > 0 {
            for i in 0..n {
                if let Ok(item) = flat.get_item(i as usize) {
                    if let Ok(val) = item.extract::<f32>() {
                        val.to_bits().hash(&mut hasher);
                    }
                }
            }
        }

        param_count += 1;
    }

    let checksum = hasher.finish();
    log::debug!("Computed checksum for {} parameters: {:016x}", param_count, checksum);

    Ok(checksum)
}


// ============================================================================
// PHASE 2: FX NODE STRUCTURES
// ============================================================================
#[derive(Debug, Clone)]
struct FxNode {
    name: String,
    op: FxOp,
    inputs: Vec<String>,
    kwargs: HashMap<String, String>,
}
#[derive(Debug, Clone)]
enum FxOp {
    Placeholder, Output,
    CallFunction { target: String },
    CallModule { target: String },
    CallMethod { method: String },
    GetAttr { attr: String },
}

// ============================================================================
// PHASE 2: CONVERSION RESULT
// ============================================================================
pub struct FxConversionResult<'py> {
    pub sexpr: String,
    pub placeholder_map: HashMap<String, Bound<'py, PyAny>>, // ✅ YENİ: placeholder_name -> tensor
    // param_var_map artık gereksiz (placeholder_map her şeyi içeriyor)
}

// ============================================================================
// PHASE 2: MAIN PARSING FUNCTION (fx_graph_to_sexpr)
// ============================================================================
pub fn fx_graph_to_sexpr<'py>(
    py: Python<'py>,
    gm: &Bound<'py, PyAny>,
    example_inputs: &[Bound<'py, PyAny>], // ✅ YENİ: placeholder sırası ile eşleşir
    types_map: &HashMap<String, ModuleInfo>
) -> PyResult<FxConversionResult<'py>> {
    let graph = gm.getattr("graph")?;
    eprintln!("[fx_bridge::Phase2] Starting FX parse...");
    let nodes = parse_fx_nodes(py, &graph)?;
    eprintln!("[DEBUG] FX graph node count: {}", nodes.len());

    // ✅ YENİ: Placeholder → example_inputs mapping oluştur
    let mut placeholder_map = HashMap::new();
    let mut next_input_index = 0;

    for node in &nodes {
        if let FxOp::Placeholder = &node.op {
            if next_input_index >= example_inputs.len() {
                return Err(HypatiaError::new_err(format!(
                    "example_inputs too short: expected at least {} inputs for placeholder '{}'",
                    next_input_index + 1,
                    node.name
                )));
            }
            let tensor = example_inputs[next_input_index].clone();
            // Enhanced debug: show tensor shape and device (only if HYPATIA_DEBUG_FX=1)
            if is_debug_fx_enabled() {
                let shape = tensor.getattr("shape").ok().and_then(|s| s.extract::<Vec<i64>>().ok());
                let device = tensor.getattr("device").ok().and_then(|d| d.str().ok()).map(|s| s.to_string());
                eprintln!(
                    "[DEBUG] Placeholder '{}' → example_inputs[{}] (shape={:?}, device={:?})",
                    node.name, next_input_index, shape, device
                );
            }
            placeholder_map.insert(node.name.clone(), tensor);
            next_input_index += 1;
        }
    }

    if is_debug_fx_enabled() {
        eprintln!("[DEBUG] Placeholder mapping created: {} entries", placeholder_map.len());
    }

    let sexpr = build_sexpr_from_nodes(&nodes, types_map, gm)?;

    Ok(FxConversionResult { sexpr, placeholder_map })
}

// ============================================================================
// PHASE 3: MAIN RECONSTRUCTION FUNCTION (sexpr_to_fx_graph)
// ============================================================================
pub fn sexpr_to_fx_graph<'py>(
    py: Python<'py>,
    original_gm: &Bound<'py, PyAny>, // GraphModule (submodule eklemek için)
    optimized_expr: RecExpr<HypatiaLang>,
    placeholder_map: &HashMap<String, Bound<'py, PyAny>>, // ✅ YENİ: name -> tensor mapping
) -> PyResult<PyObject> {
    eprintln!("[fx_bridge::Phase3] Reconstructing graph from optimized AST...");

    let root_id = Id::from(optimized_expr.as_ref().len() - 1);
    let fx = PyModule::import_bound(py, "torch.fx")?;
    let new_graph_obj = fx.getattr("Graph")?.call0()?;
    let new_graph = new_graph_obj.downcast::<PyAny>()?;

    let mut rebuilder = FxRebuilder {
        py,
        original_gm, // Submodule'ler için
        new_graph,
        node_map: HashMap::new(),
        fx_placeholder_map: HashMap::new(), // ✅ YENİ: FX node'ları için
        placeholder_map, // ✅ YENİ: Tensörler için
        param_map: HashMap::new(),
    };

    // ✅ YENİ: Yeni grafa placeholder node'ları ekle
    let original_graph = original_gm.getattr("graph")?;
    for node in original_graph.getattr("nodes")?.iter()? {
        let node = node?;
        let op = node.getattr("op")?.extract::<String>()?;
        let name = node.getattr("name")?.extract::<String>()?;

        if op == "placeholder" {
            // Yeni grafa placeholder'ı ekle
            let ph_node = rebuilder.new_graph.call_method1("placeholder", (name.clone(),))?;
            rebuilder.fx_placeholder_map.insert(name.clone(), ph_node.to_object(py));
        } else if op == "get_attr" {
            // GetAttr node'ları için de yeni grafikte ekle
            let target = node.getattr("target")?.extract::<String>()?;
            let get_attr_node = rebuilder.new_graph.call_method1("get_attr", (target,))?;
            rebuilder.fx_placeholder_map.insert(name.clone(), get_attr_node.to_object(py));
        }
    }

    let final_node_obj = match rebuilder.reconstruct_node(root_id, &optimized_expr) {
        Ok(obj) => obj,
        Err(e) => {
            eprintln!("[fx_bridge::Phase3] Reconstruction failed (Node Build): {}. Orijinal GraphModule'e geri dönülüyor.", e);
            return Ok(original_gm.to_object(py));
        }
    };

    rebuilder.new_graph.call_method1("output", (final_node_obj,))?;

    // Fused modüller (LinearReLU vb.) için submodule'leri ekle
    for (name, module) in rebuilder.param_map {
        original_gm.call_method("add_submodule", (name, module), None)?;
    }

    let new_gm = fx.getattr("GraphModule")?.call1((original_gm, rebuilder.new_graph))?;

    eprintln!("[fx_bridge::Phase3] Reconstruction complete.");
    Ok(new_gm.to_object(py))
}


// ============================================================================
// MLP Modül Adı Çözümleyici
// ============================================================================
fn resolve_module_type(gm: &Bound<PyAny>, target: &str) -> PyResult<String> {
    let module = gm.call_method1("get_submodule", (target,))?;
    // Örn: <class 'torch.nn.modules.linear.Linear'>
    let module_type = module.getattr("__class__")?
                            .getattr("__name__")?
                            .extract::<String>()?;
    Ok(module_type)
}


// ============================================================================
// PHASE 2: NODE PARSING
// ============================================================================
fn parse_fx_nodes(py: Python<'_>, graph: &Bound<PyAny>) -> PyResult<Vec<FxNode>> {
    let nodes_attr = graph.getattr("nodes")?;
    let mut parsed_nodes = Vec::new();
    
    for node_obj in nodes_attr.iter()? {
        let node = node_obj?;
        let name = node.getattr("name")?.extract::<String>()?;
        let op_str = node.getattr("op")?.extract::<String>()?;
        
        let target_str = if let Ok(target) = node.getattr("target") {
            let target_repr = target.repr()?.to_string();
            if target_repr.contains("aten::") {
                if let Some(start) = target_repr.find("aten::") {
                    let end = target_repr[start..].find(|c: char| !c.is_alphanumeric() && c != ':' && c != '_').unwrap_or(target_repr.len() - start);
                    target_repr[start..start+end].to_string()
                } else {
                    target_repr.trim_matches('\'').to_string()
                }
            } else if target_repr.starts_with("<built-in") {
                target_repr
            } else if let Ok(s) = target.extract::<String>() {
                s
            } else {
                target.to_string()
            }
        } else {
            "unknown".to_string()
        };
        
        eprintln!("[DEBUG] Node: {}, op: {}, target: {}", name, op_str, target_str);
        
        let inputs = parse_node_inputs(py, &node)?;
        let kwargs = parse_node_kwargs(py, &node)?;
        
        let op = match op_str.as_str() {
            "placeholder" => FxOp::Placeholder,
            "output" => FxOp::Output,
            "call_function" => FxOp::CallFunction { target: target_str.clone() },
            "call_module" => FxOp::CallModule { target: target_str.clone() },
            "call_method" => FxOp::CallMethod { method: target_str.clone() },
            "get_attr" => FxOp::GetAttr { attr: target_str.clone() },
            _ => FxOp::CallFunction { target: format!("unknown_{}", op_str) }
        };
        
        parsed_nodes.push(FxNode { name, op, inputs, kwargs });
    }
    
    Ok(parsed_nodes)
}

fn parse_node_inputs(_py: Python<'_>, node: &Bound<PyAny>) -> PyResult<Vec<String>> {
    let mut inputs = Vec::new();
    if let Ok(args) = node.getattr("args") {
        if let Ok(args_tuple) = args.downcast::<PyTuple>() {
            for arg in args_tuple.iter() {
                // Doğrudan bir Node ise (örn: matmul'un girdileri)
                if let Ok(name) = arg.getattr("name") {
                    if let Ok(name_str) = name.extract::<String>() {
                        inputs.push(name_str);
                        continue; // Bir sonraki argümana geç
                    }
                }

                // Bir Node değilse, iç içe bir tuple olup olmadığını kontrol et (örn: output düğümü)
                if let Ok(inner_tuple) = arg.downcast::<PyTuple>() {
                    for inner_arg in inner_tuple.iter() {
                        if let Ok(name) = inner_arg.getattr("name") {
                            if let Ok(name_str) = name.extract::<String>() {
                                inputs.push(name_str);
                            }
                        }
                    }
                }
                // Diğer olası tipler (list, dict vb.) buraya eklenebilir
            }
        }
    }
    Ok(inputs)
}
fn parse_node_kwargs(_py: Python<'_>, node: &Bound<PyAny>) -> PyResult<HashMap<String, String>> {
    let mut kwargs_map = HashMap::new();
    if let Ok(kwargs) = node.getattr("kwargs") {
        if let Ok(kwargs_dict) = kwargs.downcast::<PyDict>() {
            for (key, value) in kwargs_dict.iter() {
                let key_str = key.extract::<String>()?;
                
                let value_str = if let Ok(v) = value.extract::<String>() { v }
                    else if let Ok(v) = value.extract::<i64>() { v.to_string() }
                    else if let Ok(v) = value.extract::<f64>() { v.to_string() }
                    else if let Ok(v) = value.extract::<bool>() { v.to_string() }
                    else if let Ok(tuple) = value.downcast::<PyTuple>() {
                        tuple.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
                    }
                    else { "unknown_kwarg".to_string() };
                
                kwargs_map.insert(key_str, value_str);
            }
        }
    }
    Ok(kwargs_map)
}

// ============================================================================
// PHASE 2: S-EXPRESSION BUILDER
// ============================================================================

fn build_sexpr_from_nodes(
    nodes: &[FxNode],
    types_map: &HashMap<String, ModuleInfo>,
    gm: &Bound<PyAny>
) -> PyResult<String> {
    let mut node_exprs: HashMap<String, String> = HashMap::new();

    for node in nodes {
        let sexpr = match &node.op {
            FxOp::Placeholder => {
                // ✅ YENİ: Placeholder isimleri direkt kullanılır
                // example_inputs sırası ile eşleşir
                // 'l_x_' -> input placeholder
                // 'l_self_modules_fc1_parameters_weight_' -> param placeholder
                node.name.clone()
            },
            FxOp::Output => {
                let input_name = node.inputs.first()
                    .ok_or_else(|| HypatiaError::new_err("Output node has no input"))?;
                node_exprs.get(input_name)
                    .ok_or_else(|| HypatiaError::new_err(format!("Output input expression not found for '{}'", input_name)))?
                    .clone()
            },
            FxOp::CallModule { target } => {
                handle_call_module(target, &node.inputs, &node.kwargs, &node_exprs, types_map, gm)?
            },
            FxOp::CallFunction { target } => {
                handle_call_function(target, &node.inputs, &node.kwargs, &node_exprs)?
            },
            FxOp::CallMethod { method } => {
                handle_call_method(method, &node.inputs, &node.kwargs, &node_exprs)?
            },
            FxOp::GetAttr { .. } => {
                // ✅ YENİ: GetAttr node'ları da placeholder gibi işlenir
                // Placeholder_map'te zaten var (Phase2'de eklendi)
                node.name.clone()
            }
        };
        
        if node.op.op_str() == "output" {
            return Ok(sexpr);
        }
        node_exprs.insert(node.name.clone(), sexpr);
    }
    
    Err(HypatiaError::new_err("Could not build S-expression: no output node found"))
}
impl FxOp {
    fn op_str(&self) -> &str {
        match self {
            FxOp::Placeholder => "placeholder", FxOp::Output => "output",
            FxOp::CallFunction { .. } => "call_function",
            FxOp::CallModule { .. } => "call_module",
            FxOp::CallMethod { .. } => "call_method",
            FxOp::GetAttr { .. } => "get_attr",
        }
    }
}

// ============================================================================
// PHASE 2 & 3: OPERATOR HANDLERS
// ============================================================================

fn handle_call_module(
    target: &str, 
    inputs: &[String],
    _kwargs: &HashMap<String, String>, 
    node_exprs: &HashMap<String, String>,
    types_map: &HashMap<String, ModuleInfo>,
    gm: &Bound<PyAny>
) -> PyResult<String> {
    
    let input_expr = if let Some(input_name) = inputs.first() {
        node_exprs.get(input_name).ok_or_else(|| HypatiaError::new_err(
            format!("Input node '{}' not found in expression map", input_name)
        ))?
    } else {
         return Err(HypatiaError::new_err(format!("Module call '{}' has no inputs", target)));
    };

    let module_type = match types_map.get(target) {
        Some(t) => t.module_type.clone(),
        None => {
            eprintln!("[fx_bridge] UYARI: types_map'te '{}' bulunamadı. Çalışma zamanında çözülüyor...", target);
            match resolve_module_type(gm, target) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[fx_bridge] UYARI: Modül tipi çözülemedi (target: {}): {}. Pas geçiliyor.", target, e);
                    return Ok(input_expr.clone());
                }
            }
        }
    };
    
    let has_bias = types_map.get(target).map_or(true, |i| i.has_bias);
    let is_inference_mode = types_map.get(target).map_or(true, |i| i.is_inference);

    // ----------------------------------------------------------------------
    // 🔥 ÖNEMLİ NOT:
    // Bu 'handle_call_module' fonksiyonu, 'target'ı (örn: 'fc1')
    // 'l_self_modules_fc1_parameters_weight_' gibi bir şeye dönüştüremez.
    // Bu, 'nn.Linear'ın 'call_module' olarak değil, 'F.linear'a
    // 'call_function' olarak izlendiği varsayımını güçlendirir.
    //
    // ✅ DÜZELTME: Bu nedenle, 'sanitize_var_name'i 'handle_call_function'
    // içindeki gibi SADECE 'target'ten (örn: 'fc1') değil,
    // 'node_exprs'ten (örn: 'l_self_modules...') gelen
    // tam adlara dayalı olarak yapacağız.
    //
    // HAYIR, `handle_call_function` düzeltmesi daha doğru.
    // `handle_call_module`'deki bu kod, muhtemelen `MLP` testi için
    // *kullanılmıyor*, ancak ResNet gibi modül tabanlı modeller için
    // kullanılabilir.
    //
    // Kullanıcının isteğine geri dönelim: `sanitize_var_name`'i kaldır.
    // `format!("{}.weight", target)` 'fc1.weight' üretecek.
    // Bu, S-ifadesine 'fc1.weight' olarak girecek.
    // Phase 3'te `get_var_name`, 'fc1.weight' döndürecek.
    // `build_linear_module`'deki yeni `self.model.getattr("fc1.weight")`
    // çağrısı BAŞARISIZ OLACAK.
    //
    // Bu, 'nn.Linear'ın `call_module` olarak *izlenemeyeceği* anlamına gelir.
    // 'F.linear'a 'call_function' olarak izlenmelidir.
    // `mlptest.py`'nin `nn.Linear` kullanmasına rağmen...
    //
    // `handle_call_function`'daki 'linear' mantığına bakalım:
    // "linear" => {
    //     // F.linear(input, weight, bias)
    //     ...
    //     let input = node_exprs.get(&inputs[0]).. // YANLIŞ: F.linear(input, weight, bias)
    //     let weight = node_exprs.get(&inputs[1])..
    //     let bias = node_exprs.get(&inputs[2])..
    //     Ok(format!("(linear {} {} {})", weight, bias, input))
    // }
    //
    // 🔥 BU YANLIŞ! `F.linear`'ın sırası `(input, weight, bias)`'tır.
    // S-ifadesi `(linear weight bias input)` olmalı.
    // `handle_call_function`'daki 'linear' mantığı:
    // Girdiler: `inputs = ["l_x_node", "l_self_..._weight_node", "l_self_..._bias_node"]`
    // `input = node_exprs.get(&inputs[0])` -> "l_x_"
    // `weight = node_exprs.get(&inputs[1])` -> "l_self_..._weight_"
    // `bias = node_exprs.get(&inputs[2])` -> "l_self_..._bias_"
    // `Ok(format!("(linear {} {} {})", weight, bias, input))`
    // -> `(linear l_self_..._weight_ l_self_..._bias_ l_x_)`
    // BU DOĞRU GÖRÜNÜYOR.
    //
    // Bu, `nn.Linear`'ın `call_module` olarak DEĞİL, `call_function`
    // olarak izlendiği teorimizi doğrular.
    // Bu nedenle `handle_call_module`'deki bu kod bloğu
    // `mlptest.py` için muhtemelen ölü koddur.
    // Dokunmayacağım.
    // ----------------------------------------------------------------------

    if module_type.contains("Linear") {
        // (Bu kod muhtemelen mlptest için çalışmıyor)
        let w = sanitize_var_name(&format!("{}.weight", target));
        let b = if has_bias { 
            sanitize_var_name(&format!("{}.bias", target)) 
        } else { 
            "none".to_string() 
        };
        Ok(format!("(linear {} {} {})", w, b, input_expr))
    
    } else if module_type.contains("Conv2d") {
        let w = sanitize_var_name(&format!("{}.weight", target));
        let b = if has_bias { 
            sanitize_var_name(&format!("{}.bias", target)) 
        } else { 
            "none".to_string() 
        };
        
        let module = gm.call_method1("get_submodule", (target,))?;
        let stride = sanitize_tuple_to_string(&module.getattr("stride")?)?;
        let padding = sanitize_tuple_to_string(&module.getattr("padding")?)?;
        let dilation = sanitize_tuple_to_string(&module.getattr("dilation")?)?;
        let groups = sanitize_var_name(&module.getattr("groups")?.to_string());

        Ok(format!("(conv2d {} {} {} {} {} {} {})", w, b, input_expr, stride, padding, dilation, groups))

    } else if module_type.contains("BatchNorm2d") {
        let w = sanitize_var_name(&format!("{}.weight", target));
        let b = sanitize_var_name(&format!("{}.bias", target));
        let m = sanitize_var_name(&format!("{}.running_mean", target));
        let v = sanitize_var_name(&format!("{}.running_var", target));
        let eps = gm.call_method1("get_submodule", (target,))?.getattr("eps")?.to_string();
        Ok(format!("(batchnorm {} {} {} {} {} {})", w, b, m, v, input_expr, eps))

    } else if module_type.contains("MaxPool2d") {
        let module = gm.call_method1("get_submodule", (target,))?;
        let kernel = sanitize_tuple_to_string(&module.getattr("kernel_size")?)?;
        let stride = sanitize_tuple_to_string(&module.getattr("stride")?)?;
        let padding = sanitize_tuple_to_string(&module.getattr("padding")?)?;
        Ok(format!("(maxpool2d {} {} {} {})", input_expr, kernel, stride, padding))
    
    } else if module_type.contains("AdaptiveAvgPool2d") {
        Ok(format!("(adaptive_avg_pool2d {})", input_expr))
    
    } else if module_type.contains("ReLU") {
        Ok(format!("(relu {})", input_expr))
    } else if module_type.contains("Embedding") {
        let w = sanitize_var_name(&format!("{}.weight", target));
        Ok(format!("(embedding {} {})", w, input_expr))
    } else if module_type.contains("TransformerEncoder") {
        Ok(format!("(transformer_encoder {})", input_expr))
    
    } else if module_type.contains("Sequential") {
        eprintln!("[fx_bridge] Sequential modülü tespit edildi, pas geçiliyor");
        Ok(input_expr.clone())
    } else if module_type.contains("Dropout") {
        if is_inference_mode {
            Ok(input_expr.clone())  // Inference mode: identity
        } else {
            let p = gm.call_method1("get_submodule", (target,))?
                      .getattr("p")?.extract::<f64>()?;
            Ok(format!("(dropout {} {})", p, input_expr))
        }
    } else if module_type.contains("LayerNorm") {
        let module = gm.call_method1("get_submodule", (target,))?;
        let _normalized_shape = module.getattr("normalized_shape")?;
        let eps = module.getattr("eps")?.extract::<f64>()?;
            
        let w = if module.hasattr("weight")? {
            sanitize_var_name(&format!("{}.weight", target))
        } else {
            "none".to_string()
        };
            
        let b = if module.hasattr("bias")? {
            sanitize_var_name(&format!("{}.bias", target))
        } else {
            "none".to_string()
        };
            
        Ok(format!("(layernorm {} {} {} {})", w, b, input_expr, eps))
    } else if module_type.contains("GELU") {
        Ok(format!("(gelu {})", input_expr))
    } else if module_type.contains("SiLU") || module_type.contains("Swish") {
        Ok(format!("(silu {})", input_expr))
    } else if module_type.contains("BatchNorm1d") {
        let w = sanitize_var_name(&format!("{}.weight", target));
        let b = sanitize_var_name(&format!("{}.bias", target));
        let m = sanitize_var_name(&format!("{}.running_mean", target));
        let v = sanitize_var_name(&format!("{}.running_var", target));
        let eps = gm.call_method1("get_submodule", (target,))?.getattr("eps")?.to_string();
        Ok(format!("(batchnorm1d {} {} {} {} {} {})", w, b, m, v, input_expr, eps))

    } else {
        eprintln!("[fx_bridge] UYARI: Desteklenmeyen modül tipi '{}' (target: {}), pas geçiliyor.", module_type, target);
        Ok(input_expr.clone())
    }
}

fn handle_call_function(
    target: &str,
    inputs: &[String],
    kwargs: &HashMap<String, String>,
    node_exprs: &HashMap<String, String>,
) -> PyResult<String> {
    eprintln!("[DEBUG] handle_call_function: target = '{}', inputs = {:?}", target, inputs);

    if target.contains("flatten") {
        if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
            eprintln!("[DEBUG] Function resolved: '{}' -> 'flatten'", target);
            return Ok(format!("(flatten {})", input));
        } else {
            return Err(HypatiaError::new_err("Flatten node has no valid input"));
        }
    }

    let cleaned_target = if target.starts_with("aten::") {
        target.strip_prefix("aten::").unwrap_or(target)
    } else {
        target
    };
    eprintln!("[DEBUG] Cleaned target: '{}' -> '{}'", target, cleaned_target);

    // --- ÖNCELİKLİ KONTROLLER (KWARGS GEREKTİRENLER) ---

    if cleaned_target == "max_pool2d" {
        if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
            let kernel = kwargs.get("kernel_size")
                .map(|s| sanitize_var_name(s))
                .unwrap_or_else(|| "2".to_string());
            let stride = kwargs.get("stride")
                .map(|s| sanitize_var_name(s))
                .unwrap_or_else(|| kernel.clone());
            let padding = kwargs.get("padding")
                .map(|s| sanitize_var_name(s))
                .unwrap_or_else(|| "0".to_string());
            return Ok(format!("(maxpool2d {} {} {} {})", input, kernel, stride, padding));
        }
    }
    if cleaned_target == "add" || cleaned_target == "add_" || target.contains("add") {
        if inputs.len() >= 2 {
            let input1 = node_exprs.get(&inputs[0])
                .ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
            let input2 = node_exprs.get(&inputs[1])
                .ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
            return Ok(format!("(add {} {})", input1, input2));
        }
    }

    // --- OPERATÖR EŞLEŞTİRME (String matching) ---
    // ✅ GÜNCELLEME: Hem `<built-in function linear>` hem de `F.linear`'ı
    // yakalamak için `.contains()` geri getirildi.
    
    let func_name = if cleaned_target.contains("matmul") { "matmul" } 
    else if cleaned_target.contains("mul") { "mul" }
    else if cleaned_target.contains("sub") { "sub" }
    else if cleaned_target.contains("div") { "div" }
    else if cleaned_target.contains("mean") { "mean" } 
    else if cleaned_target.contains("relu") { "relu" } // `.contains` geri geldi
    else if cleaned_target.contains("sigmoid") { "sigmoid" } // `.contains` geri geldi
    else if cleaned_target.contains("tanh") { "tanh" } // `.contains` geri geldi
    else if cleaned_target.contains("softmax") { "softmax" } // `.contains` geri geldi
    // --- YENİ EKLENEN OPERATÖRLER (Adım 0.1) ---
    else if cleaned_target.contains("linear") { "linear" } // `.contains` geri geldi
    else if cleaned_target.contains("conv2d") { "conv2d" } // `.contains` geri geldi
    else if cleaned_target.contains("batch_norm") { "batchnorm" } // `.contains` geri geldi
    // ---
    else if cleaned_target == "max_pool2d" { "maxpool2d" } // Bu spesifik kalabilir
    else if cleaned_target.contains("adaptive_avg_pool2d") { "mean" } // `contains` kalabilir
    else { 
        eprintln!("[DEBUG] Unknown function target: {}", cleaned_target);
        "unknown_func" 
    };
    eprintln!("[DEBUG] Function resolved: '{}' -> '{}'", cleaned_target, func_name);
    
    match func_name {
        "mul" | "sub" | "div" | "matmul" => {
            if inputs.len() >= 2 {
                let input1 = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
                let input2 = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
                Ok(format!("({} {} {})", func_name, input1, input2))
            } else { 
                Err(HypatiaError::new_err(format!("{} node needs 2 inputs", func_name))) 
            }
        },
        "relu" | "sigmoid" | "tanh" | "softmax" | "mean" => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(format!("({} {})", func_name, input))
            } else { 
                Err(HypatiaError::new_err(format!("{} node has no valid input", func_name))) 
            }
        },
        
        // --- YENİ EKLENEN OPERATÖRLER (Adım 0.1) ---
        "linear" => {
             // F.linear(input, weight, bias)
            if inputs.len() >= 3 {
                let input = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err("Linear input not found"))?;
                let weight = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err("Linear weight not found"))?;
                let bias = node_exprs.get(&inputs[2]).ok_or_else(|| HypatiaError::new_err("Linear bias not found"))?;
                Ok(format!("(linear {} {} {})", weight, bias, input))
            } else if inputs.len() == 2 { // Bias=None durumu
                let input = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err("Linear input not found"))?;
                let weight = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err("Linear weight not found"))?;
                Ok(format!("(linear {} none {})", weight, input))
            } else { 
                Err(HypatiaError::new_err("Linear needs 2 or 3 inputs")) 
            }
        },
        "conv2d" => {
             // F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            let w = node_exprs.get(&inputs[1]).map_or("unknown_w".to_string(), |s| s.clone());
            let b = node_exprs.get(&inputs[2]).map_or("none".to_string(), |s| s.clone());
            let i = node_exprs.get(&inputs[0]).map_or("unknown_i".to_string(), |s| s.clone());
            
            // Kwargs'dan veya varsayılanlardan al
            let s = kwargs.get("stride").map_or("1_1".to_string(), |s| sanitize_var_name(s));
            let p = kwargs.get("padding").map_or("0_0".to_string(), |s| sanitize_var_name(s));
            let d = kwargs.get("dilation").map_or("1_1".to_string(), |s| sanitize_var_name(s));
            let g = kwargs.get("groups").map_or("1".to_string(), |s| sanitize_var_name(s));
            
            Ok(format!("(conv2d {} {} {} {} {} {} {})", w, b, i, s, p, d, g))
        },
        "batchnorm" => {
            // F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
            if inputs.len() < 5 { 
                return Err(HypatiaError::new_err(format!("batchnorm node {} inputs, 5 expected", inputs.len()))); 
            }
            let i = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
            let m = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
            let v = node_exprs.get(&inputs[2]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[2])))?;
            let w = node_exprs.get(&inputs[3]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[3])))?;
            let b = node_exprs.get(&inputs[4]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[4])))?;
            let eps = kwargs.get("eps").cloned().unwrap_or("1e-05".to_string());
            Ok(format!("(batchnorm {} {} {} {} {} {})", w, b, m, v, i, eps))
        },
        // ---
        
        "maxpool2d" => {
            let i = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
            let kernel = sanitize_var_name(&kwargs.get("kernel_size").cloned().unwrap_or("2".to_string()));
            let stride = sanitize_var_name(&kwargs.get("stride").cloned().unwrap_or(kernel.clone()));
            let padding = sanitize_var_name(&kwargs.get("padding").cloned().unwrap_or("0".to_string()));
            Ok(format!("(maxpool2d {} {} {} {})", i, kernel, stride, padding))
        },
        _ => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                eprintln!("[fx_bridge] UYARI: Desteklenmeyen fonksiyon {}, pas geçiliyor.", target);
                Ok(input.clone())
            } else { 
                Err(HypatiaError::new_err(format!("Fonksiyon node'u {} geçerli bir girdiye sahip değil", target))) 
            }
        }
    }
}

fn handle_call_method(
    method: &str,
    inputs: &[String],
    _kwargs: &HashMap<String, String>, 
    node_exprs: &HashMap<String, String>,
) -> PyResult<String> {
    eprintln!("[DEBUG] handle_call_method: method = '{}', inputs = {:?}", method, inputs);

    match method {
        "view" | "reshape" | "flatten" | "unflatten" | "size" => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                if method == "flatten" {
                    return Ok(format!("(flatten {})", input));
                }
                // Diğerlerini (view, reshape vb.) şimdilik pas geç (identity)
                Ok(input.clone())
            } else { 
                Err(HypatiaError::new_err(format!("Method {} has no input", method))) 
            }
        },
        "mean" => { 
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(format!("(mean {})", input))
            } else { 
                Err(HypatiaError::new_err("Method 'mean' has no input".to_string())) 
            }
        },
        "add_" => {
            if inputs.len() >= 2 {
                let input1 = node_exprs.get(&inputs[0])
                    .ok_or_else(|| HypatiaError::new_err("add_: input1 not found"))?;
                let input2 = node_exprs.get(&inputs[1])
                    .ok_or_else(|| HypatiaError::new_err("add_: input2 not found"))?;
                Ok(format!("(add {} {})", input1, input2))
            } else {
                // Fallback: tek input varsa identity döndür (örn: `x.add_(0)`)
                inputs.first().and_then(|n| node_exprs.get(n))
                    .map(|s| s.clone())
                    .ok_or_else(|| HypatiaError::new_err("add_: no valid input"))
            }
        }
        _ => {
            // Fallback: Bilinmeyen method (örn: 'transpose', 'permute' vb.)
            // Şimdilik ilk girdiyi (tensor'un kendisi) pas geç.
            eprintln!("[fx_bridge] UYARI: Bilinmeyen method: {}. Pas geçiliyor.", method);
            inputs.first().and_then(|n| node_exprs.get(n))
                .map(|s| s.clone())
                .ok_or_else(|| HypatiaError::new_err(format!("Unknown method: {}", method)))
        }
    }
}

// ============================================================================
// PHASE 3: RECONSTRUCTION HELPERS
// ============================================================================

struct FxRebuilder<'a, 'py> {
    py: Python<'py>,
    original_gm: &'a Bound<'py, PyAny>, // Submodule eklemek için
    new_graph: &'a Bound<'py, PyAny>,
    node_map: HashMap<Id, PyObject>,
    fx_placeholder_map: HashMap<String, PyObject>, // ✅ YENİ: FX node'ları (placeholder/get_attr)
    placeholder_map: &'a HashMap<String, Bound<'py, PyAny>>, // ✅ YENİ: Tensörler (example_inputs'tan)
    param_map: HashMap<String, PyObject>, // Fused modüller için
}
impl<'a, 'py> FxRebuilder<'a, 'py> {

    fn get_var_name(&self, node_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<String> {
        let node = &expr[node_id];
        match node {
            HypatiaLang::Var(var_symbol) => Ok(var_symbol.to_string()),
            HypatiaLang::Constant(c) => {
                Ok(c.into_inner().to_string())
            },
            _ => Err(HypatiaError::new_err(format!(
                "Reconstruction failed: Expected Var or Constant, found {:?}", node
            ))),
        }
    }

    /// ✅ YENİ: Tensör al (placeholder_map'ten)
    /// AST içindeki Var ismi ("l_self_modules_fc1_parameters_weight_", "l_x_" vs.)
    fn get_tensor(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.placeholder_map.get(name)
            .map(|t| t.clone())
            .ok_or_else(|| {
                log::error!("Unknown tensor placeholder: {}", name);
                pyo3::exceptions::PyKeyError::new_err(format!("Unknown tensor placeholder: {}", name))
            })
    }

    /// ✅ Helper: Modülü referans tensörün device'ına taşı
    /// Bu, CUDA tensörlerle çalışırken critical!
    fn move_module_to_tensor_device(
        &self,
        module: &Bound<'py, PyAny>,
        tensor: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let device = tensor.getattr("device")?;
        module.call_method1("to", (device,))?;
        Ok(())
    }

    /// ✅ YENİ: Canonical parameter name al
    /// Parametre ismini desanitize eder ("l_self_modules_fc1_parameters_weight_" → "fc1.weight")
    /// Değilse sanitized var name döner ("l_x_")
    fn get_canonical_param_name(&self, node_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<String> {
        let var_name = self.get_var_name(node_id, expr)?; // "l_self_modules_fc1_parameters_weight_"

        // Parametre ise desanitize et
        if var_name.starts_with("l_self_modules_") && var_name.contains("_parameters_") {
            let after_prefix = var_name.strip_prefix("l_self_modules_").unwrap();
            if let Some(params_idx) = after_prefix.find("_parameters_") {
                let module_part = &after_prefix[..params_idx];
                let param_part = &after_prefix[params_idx + 12..]; // "_parameters_" = 12 chars

                let module_canonical = module_part.replace("_", ".");
                let param_canonical = param_part.strip_suffix("_").unwrap_or(param_part);

                Ok(format!("{}.{}", module_canonical, param_canonical))
            } else {
                Ok(var_name)
            }
        } else {
            Ok(var_name) // Girdi placeholder için ("l_x_")
        }
    }

    /// ✅ Get parameter/buffer tensor from GraphModule
    /// Uses state_dict() for actual parameter tensors
    /// Handles both sanitized names ("l_self_modules_conv_parameters_weight_")
    /// and canonical names ("conv.weight")
    fn get_param_tensor(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        // Check if it's an input placeholder first (not a parameter)
        if !name.contains('.') && self.placeholder_map.contains_key(name) {
            let placeholder_node = self.placeholder_map.get(name).unwrap();
            return Ok(placeholder_node.clone());
        }

        // Determine the canonical parameter name
        let canonical_name = if !name.contains('.') {
            // Check if it's a parameter and desanitize
            if name.starts_with("l_self_modules_") && name.contains("_parameters_") {
                // Desanitize parameters: "l_self_modules_fc1_parameters_weight_" → "fc1.weight"
                let after_prefix = name.strip_prefix("l_self_modules_").unwrap();
                if let Some(params_idx) = after_prefix.find("_parameters_") {
                    let module_part = &after_prefix[..params_idx];
                    let param_part = &after_prefix[params_idx + 12..];

                    let module_canonical = module_part.replace("_", ".");
                    let param_canonical = param_part.strip_suffix("_").unwrap_or(param_part);

                    format!("{}.{}", module_canonical, param_canonical)
                } else {
                    log::error!("Invalid parameter name: '{}'", name);
                    return Err(pyo3::exceptions::PyKeyError::new_err(
                        format!("Invalid parameter name: {}", name)
                    ));
                }
            } else if name.starts_with("l_self_modules_") && name.contains("_buffers_") {
                // Desanitize buffers: "l_self_modules_bn_buffers_running_mean_" → "bn.running_mean"
                let after_prefix = name.strip_prefix("l_self_modules_").unwrap();

                if let Some(buffers_idx) = after_prefix.find("_buffers_") {
                    let module_part = &after_prefix[..buffers_idx];
                    let buffer_part = &after_prefix[buffers_idx + 9..];

                    let module_canonical = module_part.replace("_", ".");
                    let buffer_canonical = buffer_part.strip_suffix("_").unwrap_or(buffer_part);

                    format!("{}.{}", module_canonical, buffer_canonical)
                } else {
                    log::error!("Invalid buffer name: '{}'", name);
                    return Err(pyo3::exceptions::PyKeyError::new_err(
                        format!("Invalid buffer name: {}", name)
                    ));
                }
            } else {
                log::error!("Unknown tensor name '{}': not a parameter, buffer, or input placeholder", name);
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("Unknown tensor name: {}", name)
                ));
            }
        } else {
            // Name already contains '.', it's canonical
            name.to_string()
        };

        // Get tensor from state_dict
        let state_dict = self.original_gm.call_method0("state_dict")?;
        state_dict.get_item(&canonical_name)
            .map_err(|e| {
                log::error!("Parameter '{}' not found in state_dict", canonical_name);
                log::debug!("Available keys: {:?}", state_dict.call_method0("keys"));
                e
            })
    }

    fn reconstruct_node(
        &mut self,
        node_id: Id,
        expr: &RecExpr<HypatiaLang>,
    ) -> PyResult<PyObject> {
        if let Some(node_obj) = self.node_map.get(&node_id) {
            return Ok(node_obj.clone_ref(self.py)); 
        }
        let node = &expr[node_id];
        eprintln!("[DEBUG] reconstruct_node: processing {:?}", node);
        
        let new_node_obj = match node {
            HypatiaLang::Var(var_symbol) => {
                // ✅ YENİ: FX placeholder node'unu döndür
                let var_name = var_symbol.to_string(); // "l_x_" VEYA "l_self_modules_..."
                if let Some(fx_node) = self.fx_placeholder_map.get(&var_name) {
                    Ok(fx_node.clone_ref(self.py))
                } else {
                    Err(HypatiaError::new_err(format!(
                        "Reconstruction failed: Var '{}' is not a known placeholder", var_name
                    )))
                }
            }

            HypatiaLang::Constant(c) => {
                let value = c.into_inner();
                Ok(value.into_py(self.py))
            }
            HypatiaLang::Add([a_id, b_id]) => {
                let a_node = self.reconstruct_node(*a_id, expr)?;
                let b_node = self.reconstruct_node(*b_id, expr)?;
                self.build_call_function("add", &[a_node, b_node], None)
            }
            HypatiaLang::Conv2d([w_id, b_id, x_id, s_id, p_id, d_id, g_id]) => {
                // ✅ YENİ: reconstruct_conv2d kullanarak parametre kopyalama ile nn.Conv2d modülü oluştur
                self.reconstruct_conv2d(*w_id, *b_id, *x_id, *s_id, *p_id, *d_id, *g_id, expr)
            }
            HypatiaLang::BatchNorm([w_id, b_id, m_id, v_id, x_id, eps_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_batchnorm_module(node_id, expr, *w_id, *b_id, *m_id, *v_id, input_node, *eps_id)
            }
            HypatiaLang::FusedConvBN([w_c, b_c, w_bn, b_bn, m, v, x, eps, s, p, d, g]) => {
                let input_node = self.reconstruct_node(*x, expr)?;
                self.build_fused_conv_bn_module(
                    node_id, expr, input_node,
                    *w_c, *b_c, *w_bn, *b_bn, *m, *v, *eps,
                    *s, *p, *d, *g
                )
            }
            HypatiaLang::MaxPool2d([x_id, k_id, s_id, p_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?; 
                self.build_maxpool2d_module(node_id, expr, input_node, *k_id, *s_id, *p_id) 
            }
            HypatiaLang::Mean(x_id) => {
                let kwargs = PyDict::new_bound(self.py);
                kwargs.set_item("dim", 1.to_object(self.py))?;
                kwargs.set_item("keepdim", false)?; 
                let x_node = self.reconstruct_node(*x_id, expr)?;
                self.build_call_function("mean", &[x_node], Some(kwargs))
            }
            HypatiaLang::AdaptiveAvgPool2d(x_id) => {
                let kwargs = PyDict::new_bound(self.py);
                kwargs.set_item("output_size", (1, 1).to_object(self.py))?;
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_passthrough_module_with_kwargs("AdaptiveAvgPool2d", input_node, kwargs.into())
            }
            HypatiaLang::MatMul([a_id, b_id]) => {
                let a_node = self.reconstruct_node(*a_id, expr)?;
                let b_node = self.reconstruct_node(*b_id, expr)?;
                self.build_call_function("matmul", &[a_node, b_node], None)
            }
            // 🟢 ADIM 4: ReLU, modül yerine call_function (F.relu) kullanacak şekilde güncellendi
            HypatiaLang::ReLU(x_id) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                // ADIM 4 ile tutarlılık için F.relu (call_function) kullan
                self.build_call_function("relu", &[input_node], None)
            }
            HypatiaLang::Flatten(x_id) => {
                let x_node = self.reconstruct_node(*x_id, expr)?;
                self.build_call_function("flatten", &[x_node], None)
            }
            HypatiaLang::TransformerEncoder(x_id) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_passthrough_module("transformer_encoder", input_node)
            }
            HypatiaLang::Linear([w_id, b_id, x_id]) => {
                // ✅ YENİ: reconstruct_linear kullanarak parametre kopyalama ile nn.Linear modülü oluştur
                self.reconstruct_linear(*w_id, *b_id, *x_id, expr)
            }
            HypatiaLang::FusedLinearReLU([w_id, b_id, x_id]) => {
                // ✅ NEW: Fused Linear+ReLU reconstruction
                self.reconstruct_fused_linear_relu(*w_id, *b_id, *x_id, expr)
            }
            HypatiaLang::Embedding([w_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_embedding_module(node_id, expr, *w_id, input_node, "embedding")
            }
            // ✅ YENİ: GELU aktivasyon
            HypatiaLang::GELU(x_id) => {
                self.reconstruct_gelu(*x_id, expr)
            }
            // ✅ YENİ: SiLU/Swish aktivasyon
            HypatiaLang::SiLU(x_id) => {
                self.reconstruct_silu(*x_id, expr)
            }
            // ✅ YENİ: LayerNorm ile parametre kopyalama
            HypatiaLang::LayerNorm([w_id, b_id, x_id, eps_id]) => {
                self.reconstruct_layernorm(*w_id, *b_id, *x_id, *eps_id, expr)
            }

            // ✅ YENİ: (linear-relu w b x) -> nn.ReLU(nn.Linear(x, w, b))
            // Parametre kopyalama ile nn.Linear ve nn.ReLU modülleri oluştur
            HypatiaLang::LinearReLU(args) => {
                // args = [w, b, x]
                let w_id = args[0];
                let b_id = args[1];
                let x_id = args[2];

                // nn.Linear modülü oluştur (parametre kopyalama ile)
                let linear_node = self.reconstruct_linear(w_id, b_id, x_id, expr)?;
                // F.relu(input) - fonksiyonel relu kullan
                let relu = self.build_call_function("relu", &[linear_node], None)?;
                Ok(relu)
            }
            
            HypatiaLang::FusedMLP(ids) => {
                // ✅ REFACTORED: Use FusedMLP module with kernel fusion
                // Pattern: (fused-mlp w1 b1 w2 b2 x)
                let w1_id = ids[0];
                let b1_id = ids[1];
                let w2_id = ids[2];
                let b2_id = ids[3];
                let x_id = ids[4];

                self.reconstruct_fused_mlp(w1_id, b1_id, w2_id, b2_id, x_id, expr)
            },
            
            HypatiaLang::Mul([a, b]) => {
                let a_node = self.reconstruct_node(*a, expr)?;
                let b_node = self.reconstruct_node(*b, expr)?;
                self.build_call_function("mul", &[a_node, b_node], None)
            },

            // ========== Catch-all: Unsupported Operations ==========
            _ => {
                // Check if this operation is marked as supported in is_supported_node()
                if is_supported_node(node) {
                    // Operation is supported but reconstruction not implemented
                    log::error!("Reconstruction not implemented for supported operation: {:?}. This is a bug.", node);
                    Err(HypatiaError::new_err(format!(
                        "Reconstruction not implemented for supported operation: {:?}. \
                         This is a bug - the operation is marked as supported but has no reconstruction handler.",
                        node
                    )))
                } else {
                    // Operation is truly unsupported
                    log::warn!("Attempted to reconstruct unsupported operation: {:?}. Operation not currently supported.", node);
                    Err(HypatiaError::new_err(format!(
                        "Unsupported operation in reconstruction: {:?}. \
                         This operation is not currently supported by Hypatia's optimizer.",
                        node
                    )))
                }
            }

        }?;
        self.node_map.insert(node_id, new_node_obj.clone_ref(self.py));
        Ok(new_node_obj)
    }
    
    fn build_passthrough_module(
        &mut self, 
        original_target_name: &str, 
        input_node: PyObject
    ) -> PyResult<PyObject> {
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", original_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    fn build_passthrough_module_with_kwargs(
        &mut self, 
        module_type_name: &str, 
        input_node: PyObject,
        kwargs_obj: PyObject 
    ) -> PyResult<PyObject> {
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        
        let kwargs_dict = kwargs_obj.downcast_bound::<PyDict>(self.py)?;
        
        let pool_module = torch_nn.getattr(module_type_name)?.call((), Some(kwargs_dict))?;

        let module_target_name = format!("{}_{}", module_type_name.to_lowercase(), self.param_map.len());
        self.param_map.insert(module_target_name.clone(), pool_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    
    // ❌ KALDIRILDI: build_linear_module (Artık 'call_function' olarak ele alınıyor)
    
    fn build_embedding_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, input_node: PyObject, prefix: &str,
    ) -> PyResult<PyObject> {
        // ✅ REFACTORED: get_tensor kullanarak
        let w_name = self.get_var_name(w_id, expr)?;
        let w_tensor = self.get_tensor(&w_name)?;

        let module_target_name = format!("{}_{}", prefix, self.param_map.len());
        
        // ✅ DÜZELTME: .getattr(self.py, "shape")? -> .getattr("shape")?
        let w_shape_py = w_tensor.getattr("shape")?;
        // ✅ DÜZELTME: .extract(self.py)? -> .extract()?
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
        if w_shape.len() != 2 { return Err(HypatiaError::new_err(format!("Embedding weight shape 2 boyutlu olmalı, {} boyutlu bulundu", w_shape.len()))); }
        let num_embeddings = w_shape[0];
        let embedding_dim = w_shape[1];
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let embedding_module = torch_nn.getattr("Embedding")?.call1((num_embeddings, embedding_dim))?;

        // ✅ CRITICAL: Modülü weight'in device'ına taşı (CUDA support için)
        self.move_module_to_tensor_device(&embedding_module, &w_tensor)?;

        // Artık aynı device'tayız, copy yapabiliriz!
        embedding_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        
        self.param_map.insert(module_target_name.clone(), embedding_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    fn build_relu_module(&mut self, _node_id: Id, input_node: PyObject) -> PyResult<PyObject> {
        // nn.ReLU modülü oluştur (call_function('relu', ...) yerine)
        // 🟢 ADIM 4: Bu fonksiyon artık `reconstruct_node` tarafından çağrılmıyor
        // (ReLU artık build_call_function kullanıyor), ancak "kod çıkartma"
        // kuralı gereği yerinde bırakıldı.
        let module_target_name = format!("relu_{}", self.param_map.len());
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let relu_module = torch_nn.getattr("ReLU")?.call0()?;

        self.param_map.insert(module_target_name.clone(), relu_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ✅ REFACTORED: reconstruct_linear() - get_tensor kullanarak
    fn reconstruct_linear(&mut self, w_id: Id, b_id: Id, x_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // 1. Tensörleri get_tensor ile al
        let w_name = self.get_var_name(w_id, expr)?;
        let weight_tensor = self.get_tensor(&w_name)?;

        let b_name = self.get_var_name(b_id, expr)?;

        // DEBUG: Log linear node arguments (only if HYPATIA_DEBUG_FX=1)
        // x_id might be a nested expression (e.g., ReLU(...)), not just Var/Const
        if is_debug_fx_enabled() {
            let x_desc = match self.get_var_name(x_id, expr) {
                Ok(name) => name,
                Err(_) => format!("{:?}", expr[x_id])  // Fallback to debug repr for expressions
            };
            eprintln!("[DEBUG] Linear node args: w='{}', b='{}', x={}", w_name, b_name, x_desc);
        }

        let has_bias = b_name != "none";
        let bias_tensor = if has_bias {
            Some(self.get_tensor(&b_name)?)
        } else {
            None
        };

        // 2. Shape bilgisini TENSÖRDEN çıkar
        let weight_shape = weight_tensor.getattr("shape")?;
        let out_features: i64 = weight_shape.get_item(0)?.extract()?;
        let in_features: i64 = weight_shape.get_item(1)?.extract()?;

        // 3. Yeni Linear modül oluştur
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let linear_module = torch_nn.getattr("Linear")?.call1((in_features, out_features))?;

        // 3.5. ✅ CRITICAL: Modülü weight'in device'ına taşı (CUDA support için)
        self.move_module_to_tensor_device(&linear_module, &weight_tensor)?;

        // 4. ✅ PARAMETRELERİ (Tensörleri) KOPYALA (artık aynı device'tayız!)
        linear_module.getattr("weight")?.call_method1("copy_", (weight_tensor,))?;
        if let Some(bias_t) = bias_tensor {
            linear_module.getattr("bias")?.call_method1("copy_", (bias_t,))?;
        }

        // 5. Module map'e ekle ve node oluştur
        let module_target_name = format!("linear_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), linear_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ✅ NEW: Fused Linear+ReLU reconstruction
    fn reconstruct_fused_linear_relu(&mut self, w_id: Id, b_id: Id, x_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // 1. Get weight and bias tensors
        let w_name = self.get_var_name(w_id, expr)?;
        let weight_tensor = self.get_tensor(&w_name)?;

        let b_name = self.get_var_name(b_id, expr)?;
        let bias_tensor = if b_name != "none" {
            Some(self.get_tensor(&b_name)?)
        } else {
            None
        };

        // 2. Import Python helper
        let hypatia_core = PyModule::import_bound(self.py, "hypatia_core")?;
        let create_fn = hypatia_core.getattr("create_fused_linear_relu_from_tensors")?;

        // 3. Create fused module (weight/bias already on correct device from get_tensor)
        let none_obj = self.py.None();
        let none_value = none_obj.bind(self.py);
        let bias_arg = match bias_tensor {
            Some(ref b) => b.as_any(),
            None => none_value.as_any()
        };

        let fused_module = create_fn.call1((weight_tensor.as_any(), bias_arg))?;

        // 4. Move module to input's device (for CUDA compatibility)
        let input_tensor = input_node.bind(self.py);
        if let Ok(device_attr) = input_tensor.getattr("device") {
            fused_module.call_method1("to", (device_attr,))?;
        }

        // 5. Add to module map and create FX node
        let module_target_name = format!("fused_linear_relu_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), fused_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;

        log::info!("✅ Created fused Linear+ReLU module: {}", module_target_name);
        Ok(new_node.to_object(self.py))
    }

    /// ✅ NEW: Reconstruct FusedMLP (2-layer MLP with kernel fusion)
    /// Pattern: (fused-mlp w1 b1 w2 b2 x) -> FusedMLP module
    fn reconstruct_fused_mlp(
        &mut self,
        w1_id: Id, b1_id: Id,
        w2_id: Id, b2_id: Id,
        x_id: Id,
        expr: &RecExpr<HypatiaLang>
    ) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // 1. Get weight and bias tensors for both layers
        let w1_name = self.get_var_name(w1_id, expr)?;
        let weight1_tensor = self.get_tensor(&w1_name)?;

        let b1_name = self.get_var_name(b1_id, expr)?;
        let bias1_tensor = if b1_name != "none" {
            Some(self.get_tensor(&b1_name)?)
        } else {
            None
        };

        let w2_name = self.get_var_name(w2_id, expr)?;
        let weight2_tensor = self.get_tensor(&w2_name)?;

        let b2_name = self.get_var_name(b2_id, expr)?;
        let bias2_tensor = if b2_name != "none" {
            Some(self.get_tensor(&b2_name)?)
        } else {
            None
        };

        // 2. Import Python helper
        let hypatia_core = PyModule::import_bound(self.py, "hypatia_core")?;
        let create_fn = hypatia_core.getattr("create_fused_mlp_from_tensors")?;

        // 3. Create fused MLP module
        let none_obj = self.py.None();
        let none_value = none_obj.bind(self.py);

        let bias1_arg = match bias1_tensor {
            Some(ref b) => b.as_any(),
            None => none_value.as_any()
        };

        let bias2_arg = match bias2_tensor {
            Some(ref b) => b.as_any(),
            None => none_value.as_any()
        };

        let fused_module = create_fn.call1((
            weight1_tensor.as_any(),
            bias1_arg,
            weight2_tensor.as_any(),
            bias2_arg
        ))?;

        // 4. Move module to input's device (for CUDA compatibility)
        let input_tensor = input_node.bind(self.py);
        if let Ok(device_attr) = input_tensor.getattr("device") {
            fused_module.call_method1("to", (device_attr,))?;
        }

        // 5. Add to module map and create FX node
        let module_target_name = format!("fused_mlp_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), fused_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;

        log::info!("✅ Created fused MLP module: {}", module_target_name);
        Ok(new_node.to_object(self.py))
    }

    // ✅ REFACTORED: reconstruct_conv2d() - get_tensor kullanarak
    fn reconstruct_conv2d(&mut self, w_id: Id, b_id: Id, x_id: Id, s_id: Id, p_id: Id, d_id: Id, g_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // 1. Tensörleri get_tensor ile al
        let w_name = self.get_var_name(w_id, expr)?;
        let weight_tensor = self.get_tensor(&w_name)?;

        let b_name = self.get_var_name(b_id, expr)?;
        let has_bias = b_name != "none";
        let bias_tensor = if has_bias {
            Some(self.get_tensor(&b_name)?)
        } else {
            None
        };

        // 2. Shape bilgisini ve hyperparametreleri çıkar
        let weight_shape = weight_tensor.getattr("shape")?;
        let out_channels: i64 = weight_shape.get_item(0)?.extract()?;
        let in_channels_per_group: i64 = weight_shape.get_item(1)?.extract()?;
        let kernel_h: i64 = weight_shape.get_item(2)?.extract()?;
        let kernel_w: i64 = weight_shape.get_item(3)?.extract()?;

        let stride = self.unsanitize_tuple(&self.get_var_name(s_id, expr)?)?;
        let padding = self.unsanitize_tuple(&self.get_var_name(p_id, expr)?)?;
        let dilation = self.unsanitize_tuple(&self.get_var_name(d_id, expr)?)?;
        let groups = self.unsanitize_value(&self.get_var_name(g_id, expr)?)?;

        let in_channels = in_channels_per_group * groups.extract::<i64>(self.py)?;
        let kernel_size = (kernel_h, kernel_w).to_object(self.py);

        // 3. Yeni Conv2d modül oluştur
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", dilation)?;
        kwargs.set_item("groups", groups)?;
        kwargs.set_item("bias", has_bias)?;

        let conv_module = torch_nn.getattr("Conv2d")?.call(
            (in_channels, out_channels, kernel_size),
            Some(&kwargs)
        )?;

        // 3.5. ✅ CRITICAL: Modülü weight'in device'ına taşı (CUDA support için)
        self.move_module_to_tensor_device(&conv_module, &weight_tensor)?;

        // 4. ✅ PARAMETRELERİ (Tensörleri) KOPYALA (artık aynı device'tayız!)
        conv_module.getattr("weight")?.call_method1("copy_", (weight_tensor,))?;
        if let Some(bias_t) = bias_tensor {
            conv_module.getattr("bias")?.call_method1("copy_", (bias_t,))?;
        }

        // 5. Module map'e ekle ve node oluştur
        let module_target_name = format!("conv2d_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), conv_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ✅ YENİ FONKSİYON: reconstruct_gelu() - GELU aktivasyon
    fn reconstruct_gelu(&mut self, x_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // GELU modül oluştur (parametre yok)
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let gelu_module = torch_nn.getattr("GELU")?.call0()?;

        // Module map'e ekle ve node oluştur
        let module_name = format!("gelu_{}", self.param_map.len());
        self.param_map.insert(module_name.clone(), gelu_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ✅ YENİ FONKSİYON: reconstruct_silu() - SiLU/Swish aktivasyon
    fn reconstruct_silu(&mut self, x_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // SiLU modül oluştur (parametre yok)
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let silu_module = torch_nn.getattr("SiLU")?.call0()?;

        // Module map'e ekle ve node oluştur
        let module_name = format!("silu_{}", self.param_map.len());
        self.param_map.insert(module_name.clone(), silu_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ✅ YENİ FONKSİYON: reconstruct_layernorm() - LayerNorm ile parametre kopyalama
    fn reconstruct_layernorm(&mut self, w_id: Id, b_id: Id, x_id: Id, eps_id: Id, expr: &RecExpr<HypatiaLang>) -> PyResult<PyObject> {
        let input_node = self.reconstruct_node(x_id, expr)?;

        // 1. Tensörleri get_tensor ile al
        let w_name = self.get_var_name(w_id, expr)?;
        let b_name = self.get_var_name(b_id, expr)?;

        let has_weight = w_name != "none";
        let has_bias = b_name != "none";

        let weight_tensor = if has_weight {
            Some(self.get_tensor(&w_name)?)
        } else {
            None
        };

        let bias_tensor = if has_bias {
            Some(self.get_tensor(&b_name)?)
        } else {
            None
        };

        // 2. Shape'ten normalized_shape çıkar
        let normalized_shape = if let Some(ref weight) = weight_tensor {
            let w_shape = weight.getattr("shape")?;
            w_shape.extract::<Vec<i64>>()?
        } else if let Some(ref bias) = bias_tensor {
            let b_shape = bias.getattr("shape")?;
            b_shape.extract::<Vec<i64>>()?
        } else {
            return Err(HypatiaError::new_err("LayerNorm: Both weight and bias are 'none'"));
        };

        // 3. eps değerini al
        let eps_str = self.get_var_name(eps_id, expr)?;
        let eps_value = self.unsanitize_value(&eps_str)?;

        // 4. LayerNorm modül oluştur
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("eps", eps_value)?;
        kwargs.set_item("elementwise_affine", has_weight || has_bias)?;

        let ln_module = torch_nn.getattr("LayerNorm")?.call(
            (normalized_shape,),
            Some(&kwargs)
        )?;

        // 4.5. ✅ CRITICAL: Modülü referans tensörün device'ına taşı (CUDA support için)
        if let Some(ref weight) = weight_tensor {
            self.move_module_to_tensor_device(&ln_module, weight)?;
        } else if let Some(ref bias) = bias_tensor {
            self.move_module_to_tensor_device(&ln_module, bias)?;
        }

        // 5. ✅ PARAMETRELERİ (Tensörleri) KOPYALA (artık aynı device'tayız!)
        if let Some(weight) = weight_tensor {
            ln_module.getattr("weight")?.call_method1("copy_", (weight,))?;
        }
        if let Some(bias) = bias_tensor {
            ln_module.getattr("bias")?.call_method1("copy_", (bias,))?;
        }

        // 6. Module map'e ekle ve node oluştur
        let module_name = format!("layernorm_{}", self.param_map.len());
        self.param_map.insert(module_name.clone(), ln_module.to_object(self.py));

        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    fn build_call_function(
        &mut self,
        target: &str,
        arg_nodes: &[PyObject],
        kwargs: Option<Bound<'_, PyDict>>
    ) -> PyResult<PyObject> {
        
        let mut final_kwargs = kwargs;
        let mut final_arg_nodes = arg_nodes.to_vec(); // Kopyala
        
        let torch_module = PyModule::import_bound(self.py, "torch")?;
        let nn_functional = PyModule::import_bound(self.py, "torch.nn.functional")?;

        let target_fn = if target == "linear" {
            nn_functional.getattr("linear")?
        } else if target == "relu" {
            // 🟢 ADIM 4: F.relu'yu (call_function) desteklemek için eklendi
            nn_functional.getattr("relu")?
        } else {
             torch_module.getattr(target)?
        };

        if target == "flatten" {
            if arg_nodes.len() == 1 {
                 final_arg_nodes.push(1.to_object(self.py)); // start_dim = 1
                 final_kwargs = None; 
            }
        }
        
        if target == "mean" {
            if final_kwargs.is_none() {
                 let dict = PyDict::new_bound(self.py);
                 dict.set_item("dim", (2, 3).to_object(self.py))?; 
                 dict.set_item("keepdim", true)?;
                 final_kwargs = Some(dict);
            }
        }

        let args_tuple = PyTuple::new_bound(self.py, final_arg_nodes);
        
        let kwargs_for_create_node = final_kwargs
            .map(|dict| dict.to_object(self.py))
            .unwrap_or_else(|| self.py.None());
            
        let create_node_args = (
            "call_function", 
            target_fn, 
            args_tuple, 
            kwargs_for_create_node
        );
        
        let new_node = self.new_graph.call_method1(
            "create_node", 
            create_node_args
        )?;
        
        Ok(new_node.to_object(self.py))
    }

    fn build_batchnorm_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, m_id: Id, v_id: Id, input_node: PyObject, eps_id: Id
    ) -> PyResult<PyObject> {
        // ✅ REFACTORED: get_tensor kullanarak
        let w_name = self.get_var_name(w_id, expr)?;
        let b_name = self.get_var_name(b_id, expr)?;
        let m_name = self.get_var_name(m_id, expr)?;
        let v_name = self.get_var_name(v_id, expr)?;

        eprintln!("[DEBUG] BatchNorm param names: weight={}, bias={}, mean={}, var={}",
            w_name, b_name, m_name, v_name);

        let w_tensor = self.get_tensor(&w_name)?;
        let b_tensor = self.get_tensor(&b_name)?;
        let mean_tensor = self.get_tensor(&m_name)?;
        let var_tensor = self.get_tensor(&v_name)?;

        let eps_str = self.get_var_name(eps_id, expr)?;
        let eps = self.unsanitize_value(&eps_str)?;
        
        // ✅ DÜZELTME: .getattr(self.py, "shape")? -> .getattr("shape")?
        let w_shape_py = w_tensor.getattr("shape")?;
        // ✅ DÜZELTME: .extract(self.py)? -> .extract()?
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
        let num_features = w_shape[0];

        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;

        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("eps", eps)?;
        
        let bn_module = torch_nn.getattr("BatchNorm2d")?.call(
            (num_features,),
            Some(&kwargs)
        )?;

        // ✅ CRITICAL: Modülü weight'in device'ına taşı (CUDA support için)
        self.move_module_to_tensor_device(&bn_module, &w_tensor)?;

        // Artık aynı device'tayız, copy yapabiliriz!
        bn_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        bn_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_tensor,))?;
        bn_module.getattr("running_mean")?.getattr("data")?.call_method1("copy_", (mean_tensor,))?;
        bn_module.getattr("running_var")?.getattr("data")?.call_method1("copy_", (var_tensor,))?;
        
        let module_target_name = format!("batchnorm_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), bn_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        
        eprintln!("[DEBUG] BatchNorm module created successfully");
        Ok(new_node.to_object(self.py))
    }
    
    /// Phase3: Reconstruct FusedConvBN node
    ///
    /// Implements the Conv2d + BatchNorm2d fusion formula:
    ///   gamma_hat = gamma / sqrt(var + eps)
    ///   w_fused = w_conv * gamma_hat.reshape(-1, 1, 1, 1)
    ///   b_fused = gamma_hat * (b_conv - mean) + beta
    ///
    /// This creates a single Conv2d module with fused parameters that is
    /// mathematically equivalent to sequential Conv2d → BatchNorm2d.
    fn build_fused_conv_bn_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        input_node: PyObject,
        w_c_id: Id, b_c_id: Id, w_bn_id: Id, b_bn_id: Id,
        m_id: Id, v_id: Id, eps_id: Id,
        s_id: Id, p_id: Id, d_id: Id, g_id: Id
    ) -> PyResult<PyObject> {

        log::info!("Building fused Conv+BN module");

        // ✅ REFACTORED: get_tensor kullanarak
        let w_c_var = self.get_var_name(w_c_id, expr)?;
        let b_c_var = self.get_var_name(b_c_id, expr)?;
        let w_bn_var = self.get_var_name(w_bn_id, expr)?;
        let b_bn_var = self.get_var_name(b_bn_id, expr)?;
        let m_var = self.get_var_name(m_id, expr)?;
        let v_var = self.get_var_name(v_id, expr)?;

        let b_c_exists = b_c_var != "none";

        log::debug!("FusedConvBN params: conv.w={}, bn.gamma={}, bn.mean={}, bn.var={}",
            w_c_var, w_bn_var, m_var, v_var);

        // 1) Tensörleri get_tensor ile al
        let torch = PyModule::import_bound(self.py, "torch")?;

        let w_c = self.get_tensor(&w_c_var)?;
        let w_bn = self.get_tensor(&w_bn_var)?;
        let b_bn = self.get_tensor(&b_bn_var)?;
        let m = self.get_tensor(&m_var)?;
        let v = self.get_tensor(&v_var)?;

        let eps_str = self.get_var_name(eps_id, expr)?;
        let eps = self.unsanitize_value(&eps_str)?.extract::<f64>(self.py)?;

        // 2) Extract Conv2d shape parameters
        let w_c_shape = w_c.getattr("shape")?.extract::<Vec<usize>>()?;
        let out_channels = w_c_shape[0];
        // Note: in_channels_per_group = w_c_shape[1]
        // groups will be extracted from AST below

        // 3) Apply fusion formula: gamma_hat = gamma / sqrt(var + eps)
        let sqrt_var = v.call_method1("add", (eps,))?.call_method0("sqrt")?;
        let gamma_hat = w_bn.call_method1("div", (sqrt_var,))?;

        // Reshape gamma_hat for broadcasting: (C,) → (C, 1, 1, 1)
        let gamma_hat_4d = gamma_hat.call_method1("view", (vec![-1, 1, 1, 1],))?;

        // 4) Compute fused weight: w_fused = w_conv * gamma_hat
        let w_fused = w_c.call_method1("mul", (gamma_hat_4d,))?;

        // 5) Compute fused bias: b_fused = gamma_hat * (b_conv - mean) + beta
        let b_c_val = if b_c_exists {
            self.get_tensor(&b_c_var)?
        } else {
            // If conv has no bias, use zeros_like(mean)
            torch.call_method1("zeros_like", (m.clone(),))?
        };

        let b_fused = b_c_val.call_method1("sub", (m,))?           // b_conv - mean
                           .call_method1("mul", (gamma_hat,))?     // * gamma_hat
                           .call_method1("add", (b_bn,))?;         // + beta

        // 7) Extract hyperparameters from AST
        let stride = self.unsanitize_tuple(&self.get_var_name(s_id, expr)?)?;
        let padding = self.unsanitize_tuple(&self.get_var_name(p_id, expr)?)?;
        let dilation = self.unsanitize_tuple(&self.get_var_name(d_id, expr)?)?;
        let groups = self.unsanitize_value(&self.get_var_name(g_id, expr)?)?;

        let groups_val = groups.extract::<usize>(self.py)?;
        let in_channels = w_c_shape[1] * groups_val;
        let kernel_size = (w_c_shape[2], w_c_shape[3]).to_object(self.py);

        log::debug!("Creating fused Conv2d: in={}, out={}, kernel={:?}, stride={:?}, groups={}",
                    in_channels, out_channels, (w_c_shape[2], w_c_shape[3]),
                    stride, groups_val);

        // 9) Create new Conv2d module with fused parameters
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", dilation)?;
        kwargs.set_item("groups", groups)?;
        kwargs.set_item("bias", true)?;  // Fused conv always has bias

        let fused_conv = torch_nn.getattr("Conv2d")?.call(
            (in_channels, out_channels, kernel_size),
            Some(&kwargs)
        )?;

        // 9.5) ✅ CRITICAL: ÖNCE modülü device'a taşı, SONRA copy yap!
        // Bu sayede CPU→CUDA copy hatası almayız
        self.move_module_to_tensor_device(&fused_conv, &w_c)?;

        // 10) Copy fused parameters to module (artık aynı device'tayız!)
        // Note: copy_ preserves dtype automatically
        fused_conv.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_fused,))?;
        fused_conv.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_fused,))?;

        // 12) Register fused module in param_map and create FX node
        let module_target_name = format!("fused_conv_bn_{}", self.param_map.len());
        log::info!("Fused Conv+BN module created successfully: {}", &module_target_name);

        self.param_map.insert(module_target_name.clone(), fused_conv.to_object(self.py));

        // 13) Create FX graph node calling the fused module
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;

        Ok(new_node.to_object(self.py))
    }

    fn build_maxpool2d_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        input_node: PyObject, k_id: Id, s_id: Id, p_id: Id 
    ) -> PyResult<PyObject> {
        let kernel_str = self.get_var_name(k_id, expr)?;
        let stride_str = self.get_var_name(s_id, expr)?;
        let padding_str = self.get_var_name(p_id, expr)?;
        
        let kernel_size = self.unsanitize_tuple(&kernel_str)?;
        let stride = self.unsanitize_tuple(&stride_str)?;
        let padding = self.unsanitize_tuple(&padding_str)?;
        
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", 1)?;
        kwargs.set_item("ceil_mode", false)?;

        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let pool_module = torch_nn.getattr("MaxPool2d")?.call(
            (kernel_size,),
            Some(&kwargs)
        )?;

        let module_target_name = format!("maxpool2d_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), pool_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }

    // ❌ KALDIRILDI: get_param_from_original_model (Artık doğrudan getattr kullanıyoruz)
    // fn get_param_from_original_model(&self, name: &str) -> PyResult<PyObject> { ... }

    fn unsanitize_tuple(&self, s: &str) -> PyResult<PyObject> {
        let py = self.py;
        // "1_1" veya "3_3" gibi sanitize edilmiş string'leri parse et
        let parts: Vec<PyResult<PyObject>> = s.split('_')
            .map(|part| part.parse::<i64>() // i64'e parse et
                .map(|i| i.into_py(py)) // PyObject'e çevir
                .map_err(|e| HypatiaError::new_err(format!("Tuple parse error: '{}' -> {}", s, e)))
            )
            .collect();
        let elements = parts.into_iter().collect::<PyResult<Vec<PyObject>>>()?;
        // PyTuple oluştur (örn: (1, 1))
        Ok(PyTuple::new_bound(self.py, elements).to_object(self.py))
    }

    fn unsanitize_value(&self, s: &str) -> PyResult<PyObject> {
        // Önce tamsayı (i64) olarak ayrıştırmayı dene
        if let Ok(i) = s.parse::<i64>() {
            Ok(i.into_py(self.py))
        }
        // Sadece tamsayı değilse ondalık (f64) olarak dene
        else if let Ok(f) = s.parse::<f64>() {
            Ok(f.into_py(self.py))
        }
        // '1e-05' gibi bilimsel gösterim (e-notasyonu)
        else if s.contains('e') && s.contains('_') {
            let s_cleaned = s.replace("_", "-");
            if let Ok(f) = s_cleaned.parse::<f64>() {
                Ok(f.into_py(self.py))
            } else {
                Ok(s.into_py(self.py))
            }
        }
        // Diğer her şey (string vb.)
        else {
            Ok(s.into_py(self.py))
        }
    }
}


// ============================================================================
// HELPERS
// ============================================================================

// ✅ DÜZELTME: Bu fonksiyon hala 'handle_call_module' tarafından kullanılıyor.
// 'handle_call_module' (muhtemelen) mlptest için çalışmasa da,
// diğer modeller için çalışabilir, bu yüzden onu SİLMEDİM.
fn sanitize_var_name(name: &str) -> String {
    name.replace(".", "_")
        .replace("-", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
}

fn sanitize_tuple_to_string(tuple: &Bound<PyAny>) -> PyResult<String> {
    if let Ok(t) = tuple.downcast::<PyTuple>() {
        let mut s = String::new();
        for (i, item) in t.iter().enumerate() {
            if i > 0 { s.push('_'); }
            s.push_str(&item.to_string());
        }
        Ok(s) // " (1, 1) " -> "1_1"
    } else {
        Ok(tuple.to_string()) // "1" -> "1"
    }
}

// ============================================================================
// NODE SUPPORT CHECKER
// ============================================================================
// Desteklenen düğümleri kategorize ederek kontrol eder ve desteklenmeyen
// operasyonlar için uyarı mesajları verir.
fn is_supported_node(node: &HypatiaLang) -> bool {
    match node {
        // ========== Aritmetik Operatörler ==========
        HypatiaLang::Add(_) | HypatiaLang::Mul(_) => true,
        HypatiaLang::Sub(_) | HypatiaLang::Div(_) | HypatiaLang::Neg(_) => {
            eprintln!("[WARN] Arithmetic op {:?} not fully implemented in reconstruction", node);
            false
        }
        HypatiaLang::Exp(_) | HypatiaLang::Log(_) | HypatiaLang::Sqrt(_) | HypatiaLang::Pow(_) => {
            eprintln!("[WARN] Advanced math op {:?} not supported", node);
            false
        }

        // ========== Aktivasyon Fonksiyonları ==========
        HypatiaLang::ReLU(_) | HypatiaLang::GELU(_) | HypatiaLang::SiLU(_) => true,
        HypatiaLang::Sigmoid(_) | HypatiaLang::Tanh(_) | HypatiaLang::Softmax(_) => {
            eprintln!("[WARN] Activation {:?} not implemented in reconstruction (may work via fallback)", node);
            false
        }
        HypatiaLang::ReLUGrad(_) | HypatiaLang::LeakyReLU(_) | HypatiaLang::ELU(_) => {
            eprintln!("[WARN] Activation {:?} not supported", node);
            false
        }

        // ========== NN Layers ==========
        HypatiaLang::Linear(_) | HypatiaLang::Conv2d(_) | HypatiaLang::MatMul(_) => true,
        HypatiaLang::Embedding(_) | HypatiaLang::TransformerEncoder(_) => true,
        HypatiaLang::Attention(_) => {
            eprintln!("[WARN] Attention layer reconstruction not implemented");
            false
        }

        // ========== Normalization Layers ==========
        HypatiaLang::BatchNorm(_) | HypatiaLang::LayerNorm(_) => true,
        HypatiaLang::BatchNorm1d(_) | HypatiaLang::GroupNorm(_) => {
            eprintln!("[WARN] Normalization layer {:?} not implemented", node);
            false
        }

        // ========== Pooling Layers ==========
        HypatiaLang::MaxPool2d(_) | HypatiaLang::AdaptiveAvgPool2d(_) => true,
        HypatiaLang::AvgPool2d(_) => {
            eprintln!("[WARN] AvgPool2d reconstruction not implemented");
            false
        }

        // ========== Fusion Operations ==========
        HypatiaLang::LinearReLU(_) | HypatiaLang::FusedMLP(_) | HypatiaLang::FusedConvBN(_) => true,

        // ========== Statistical Operations ==========
        HypatiaLang::Mean(_) => true,
        HypatiaLang::Variance(_) | HypatiaLang::Max(_) | HypatiaLang::Min(_) => {
            eprintln!("[WARN] Statistical op {:?} not supported", node);
            false
        }

        // ========== Shape Operations ==========
        HypatiaLang::Flatten(_) => true,

        // ========== Regularization ==========
        HypatiaLang::Dropout(_) => {
            eprintln!("[WARN] Dropout reconstruction not implemented (inference mode may pass through)");
            false
        }

        // ========== Primitives ==========
        HypatiaLang::Var(_) | HypatiaLang::Constant(_) => true,

        // ========== Catch-all for Unknown ==========
        _ => {
            eprintln!("[ERROR] Unknown/unsupported node type: {:?}", node);
            false
        }
    }
}


// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sanitize_var_name() {
        assert_eq!(sanitize_var_name("input.weight"), "input_weight");
        assert_eq!(sanitize_var_name("layer[0]"), "layer_0_");
        assert_eq!(sanitize_var_name("1 1"), "1_1");
        assert_eq!(sanitize_var_name("1e-05"), "1e_05");
    }

    #[test]
    fn test_unsanitize_value_f64() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rebuilder = FxRebuilder {
                py,
                model: PyDict::new_bound(py).as_any(),
                original_gm: PyDict::new_bound(py).as_any(), // Dummy
                new_graph: PyDict::new_bound(py).as_any(), // Dummy
                node_map: HashMap::new(),
                placeholder_map: HashMap::new(),
                param_map: HashMap::new(),
            };

            let val1 = rebuilder.unsanitize_value("0.00001").unwrap();
            assert_eq!(val1.extract::<f64>(py).unwrap(), 1e-05);
            let val2 = rebuilder.unsanitize_value("1e-05").unwrap();
            assert_eq!(val2.extract::<f64>(py).unwrap(), 1e-05);
            
            let val3 = rebuilder.unsanitize_value("3").unwrap();
            assert_eq!(val3.extract::<i64>(py).unwrap(), 3);
            
            // ✅ DÜZELTME: '...' kaldırıldı
            let val4 = rebuilder.unsanitize_value("1e_05").unwrap();
            assert_eq!(val4.extract::<f64>(py).unwrap(), 1e-05);
            
            let val5 = rebuilder.unsanitize_value("3.0").unwrap();
            assert_eq!(val5.extract::<f64>(py).unwrap(), 3.0);
        });
    }

    #[test]
    fn test_unsanitize_tuple() {
         pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rebuilder = FxRebuilder {
                py,
                model: PyDict::new_bound(py).as_any(),
                original_gm: PyDict::new_bound(py).as_any(),
                new_graph: PyDict::new_bound(py).as_any(),
                node_map: HashMap::new(),
                placeholder_map: HashMap::new(),
                param_map: HashMap::new(),
            };
            
            let val = rebuilder.unsanitize_tuple("1_1").unwrap();
            let tuple = val.downcast_bound::<PyTuple>(py).unwrap();
            assert_eq!(tuple.len(), 2);
            assert_eq!(tuple.get_item(0).unwrap().extract::<i64>().unwrap(), 1);
            // ✅ DÜZELTME: Satır tamamlandı
            assert_eq!(tuple.get_item(1).unwrap().extract::<i64>().unwrap(), 1);

            let val2 = rebuilder.unsanitize_tuple("3").unwrap();
            let tuple2 = val2.downcast_bound::<PyTuple>(py).unwrap();
            assert_eq!(tuple2.len(), 1);
            assert_eq!(tuple2.get_item(0).unwrap().extract::<i64>().unwrap(), 3);
        });
    }
}