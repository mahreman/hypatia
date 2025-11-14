// ============================================================================
// FX BRIDGE: PyTorch FX Graph ↔ Hypatia S-expression
// ============================================================================

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyDict, PyModule};
use crate::python_bindings::{HypatiaError, ModuleInfo};
use std::collections::HashMap;

use egg::{RecExpr, Id};
use crate::egraph_optimizer::HypatiaLang;

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
// PHASE 2: MAIN PARSING FUNCTION (fx_graph_to_sexpr)
// ============================================================================
pub fn fx_graph_to_sexpr(
    py: Python<'_>, 
    gm: &Bound<PyAny>,
    types_map: &HashMap<String, ModuleInfo>
) -> PyResult<String> {
    let graph = gm.getattr("graph")?;
    eprintln!("[fx_bridge::Phase2] Starting FX parse...");
    let nodes = parse_fx_nodes(py, &graph)?;
    eprintln!("[DEBUG] FX graph node count: {}", nodes.len());

    // NOT: Varsayılan değerler (is_inference vb.)
    // python_bindings.rs'de ModuleInfo'nun FromPyObject implementasyonunda halledildi.
    
    let sexpr = build_sexpr_from_nodes(&nodes, types_map, gm)?;
    Ok(sexpr)
}

// ============================================================================
// PHASE 3: MAIN RECONSTRUCTION FUNCTION (sexpr_to_fx_graph)
// ============================================================================
pub fn sexpr_to_fx_graph(
    py: Python<'_>,
    original_gm: &Bound<PyAny>,
    optimized_expr: RecExpr<HypatiaLang>,
) -> PyResult<PyObject> {
    eprintln!("[fx_bridge::Phase3] Reconstructing graph from optimized AST...");

    let root_id = Id::from(optimized_expr.as_ref().len() - 1);
    let fx = PyModule::import_bound(py, "torch.fx")?;
    let new_graph_obj = fx.getattr("Graph")?.call0()?;
    let new_graph = new_graph_obj.downcast::<PyAny>()?;

    let mut rebuilder = FxRebuilder {
        py,
        original_gm,
        new_graph,
        node_map: HashMap::new(), 
        placeholder_map: HashMap::new(),
        param_map: HashMap::new(),
    };

    let original_graph = original_gm.getattr("graph")?;
    for node in original_graph.getattr("nodes")?.iter()? {
        let node = node?;
        if node.getattr("op")?.extract::<String>()? == "placeholder" {
            let name = node.getattr("name")?.extract::<String>()?;
            let ph_node = rebuilder.new_graph.call_method1("placeholder", (name.clone(),))?;
            rebuilder.placeholder_map.insert(sanitize_var_name(&name), ph_node.to_object(py));
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
                if let Ok(name) = arg.getattr("name") {
                    if let Ok(name_str) = name.extract::<String>() {
                        inputs.push(name_str);
                    }
                }
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
            FxOp::Placeholder => sanitize_var_name(&node.name),
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
            FxOp::GetAttr { attr } => sanitize_var_name(attr),
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

    if module_type.contains("Linear") {
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

    let func_name = if cleaned_target.contains("mul") { "mul" }
    else if cleaned_target.contains("sub") { "sub" }
    else if cleaned_target.contains("div") { "div" }
    else if cleaned_target.contains("matmul") { "matmul" } 
    else if cleaned_target.contains("mean") { "mean" } 
    else if cleaned_target.contains("relu") { "relu" }
    else if cleaned_target.contains("sigmoid") { "sigmoid" }
    else if cleaned_target.contains("tanh") { "tanh" }
    else if cleaned_target.contains("softmax") { "softmax" }
    else if cleaned_target.contains("batch_norm") { "batchnorm" }
    else if cleaned_target.contains("max_pool2d") { "maxpool2d" }
    else if cleaned_target.contains("adaptive_avg_pool2d") { "mean" }
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
        "batchnorm" => {
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
                    .ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
                let input2 = node_exprs.get(&inputs[1])
                    .ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
                return Ok(format!("(add {} {})", input1, input2));
            } else {
                Err(HypatiaError::new_err("add_ method needs 2 inputs".to_string()))
            }
        }
        _ => {
            eprintln!("[fx_bridge] UYARI: Bilinmeyen method: {}", method);
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(input.clone())
            } else { Err(HypatiaError::new_err(format!("Method node'u {} geçerli bir girdiye sahip değil", method))) }
        }
    }
}

// ============================================================================
// PHASE 3: RECONSTRUCTION HELPERS
// ============================================================================
struct FxRebuilder<'a, 'py> {
    py: Python<'py>,
    original_gm: &'a Bound<'py, PyAny>,
    new_graph: &'a Bound<'py, PyAny>,
    node_map: HashMap<Id, PyObject>,
    placeholder_map: HashMap<String, PyObject>,
    param_map: HashMap<String, PyObject>,
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
                let var_name = var_symbol.to_string();
                if let Some(placeholder_node) = self.placeholder_map.get(&var_name) {
                    Ok(placeholder_node.clone_ref(self.py)) 
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
                self.build_call_function("add", &[*a_id, *b_id], expr, None)
            }
            HypatiaLang::Conv2d([w_id, b_id, x_id, s_id, p_id, d_id, g_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_conv2d_module(node_id, expr, *w_id, *b_id, input_node, *s_id, *p_id, *d_id, *g_id)
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
                self.build_call_function("mean", &[*x_id], expr, Some(kwargs))
            }
            HypatiaLang::AdaptiveAvgPool2d(x_id) => {
                let kwargs = PyDict::new_bound(self.py);
                kwargs.set_item("output_size", (1, 1).to_object(self.py))?;
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_passthrough_module_with_kwargs("AdaptiveAvgPool2d", input_node, kwargs.into())
            }
            HypatiaLang::MatMul([a_id, b_id]) => {
                self.build_call_function("matmul", &[*a_id, *b_id], expr, None)
            }
            HypatiaLang::ReLU(x_id) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_relu_module(node_id, input_node)
            }
            HypatiaLang::Flatten(x_id) => {
                self.build_call_function("flatten", &[*x_id], expr, None)
            }
            HypatiaLang::TransformerEncoder(x_id) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_passthrough_module("transformer_encoder", input_node)
            }
            HypatiaLang::Linear([w_id, b_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_linear_module(node_id, expr, *w_id, *b_id, input_node, "linear")
            }
            HypatiaLang::Embedding([w_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_embedding_module(node_id, expr, *w_id, input_node, "embedding")
            }
            HypatiaLang::LinearReLU([w_id, b_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                let linear_node = self.build_linear_module(
                    node_id, expr, *w_id, *b_id, input_node, "fused_linear_relu_linear"
                )?;
                self.build_relu_module(node_id, linear_node)
            }
            HypatiaLang::FusedMLP([w1_id, b1_id, w2_id, b2_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                let linear1_node = self.build_linear_module(
                    node_id, expr, *w1_id, *b1_id, input_node, "fused_mlp_linear1"
                )?;
                let relu1_node = self.build_relu_module(node_id, linear1_node)?;
                self.build_linear_module(
                    *w2_id, expr, *w2_id, *b2_id, relu1_node, "fused_mlp_linear2"
                )
            }
            _ => {
                Err(HypatiaError::new_err(
                    format!("Desteklenmeyen S-ifadesi node'u (reconstruct): {:?}", node))
                )
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
    
    fn build_linear_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, input_node: PyObject, prefix: &str,
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?; 
        let b_name = self.get_var_name(b_id, expr)?;
        let module_target_name = format!("{}_{}", prefix, self.param_map.len());
        
        let w_tensor = self.get_param_from_original_gm(&w_name.replace("_", "."))?;
        
        let has_bias = b_name != "none";

        let w_shape_py = w_tensor.getattr(self.py, "shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>(self.py)?;
        let in_features = w_shape[1];
        let out_features = w_shape[0];

        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("bias", has_bias)?;
        
        let linear_module = torch_nn.getattr("Linear")?.call(
            (in_features, out_features), Some(&kwargs)
        )?;
        
        linear_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        
        if has_bias {
            let b_tensor = self.get_param_from_original_gm(&b_name.replace("_", "."))?;
            linear_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_tensor,))?;
        }
        
        self.param_map.insert(module_target_name.clone(), linear_module.to_object(self.py));
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    
    fn build_embedding_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, input_node: PyObject, prefix: &str,
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?;
        let module_target_name = format!("{}_{}", prefix, self.param_map.len());
        let w_tensor = self.get_param_from_original_gm(&w_name.replace("_", "."))?;
        let w_shape_py = w_tensor.getattr(self.py, "shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>(self.py)?;
        if w_shape.len() != 2 { return Err(HypatiaError::new_err(format!("Embedding weight shape 2 boyutlu olmalı, {} boyutlu bulundu", w_shape.len()))); }
        let num_embeddings = w_shape[0];
        let embedding_dim = w_shape[1];
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let embedding_module = torch_nn.getattr("Embedding")?.call1((num_embeddings, embedding_dim))?;
        embedding_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        self.param_map.insert(module_target_name.clone(), embedding_module.to_object(self.py));
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    fn build_relu_module(&mut self, _node_id: Id, input_node: PyObject) -> PyResult<PyObject> {
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
    
    fn build_call_function(
        &mut self,
        target: &str,
        arg_ids: &[Id],
        expr: &RecExpr<HypatiaLang>,
        kwargs: Option<Bound<'_, PyDict>>
    ) -> PyResult<PyObject> {
        let mut arg_nodes = Vec::new();
        for arg_id in arg_ids {
            let node_obj = self.reconstruct_node(*arg_id, expr)?;
            arg_nodes.push(node_obj);
        }
        
        let mut final_kwargs = kwargs;
        
        if target == "flatten" {
            let start_dim_arg = 1.to_object(self.py);
            arg_nodes.push(start_dim_arg);
            final_kwargs = None; 
        }
        
        if target == "mean" {
            if final_kwargs.is_none() {
                 let dict = PyDict::new_bound(self.py);
                 dict.set_item("dim", (2, 3).to_object(self.py))?; 
                 dict.set_item("keepdim", true)?;
                 final_kwargs = Some(dict);
            }
        }

        let args_tuple = PyTuple::new_bound(self.py, arg_nodes);
        
        let torch_module = PyModule::import_bound(self.py, "torch")?;
        let target_fn = torch_module.getattr(target)?;
        
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

    
    fn build_conv2d_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, input_node: PyObject, 
        s_id: Id, p_id: Id, d_id: Id, g_id: Id 
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?;
        let b_name = self.get_var_name(b_id, expr)?;
        let has_bias = b_name != "none";

        let w_tensor = self.get_param_from_original_gm(&w_name.replace("_", "."))?;
        let b_tensor = if has_bias {
            Some(self.get_param_from_original_gm(&b_name.replace("_", "."))?)
        } else {
            None
        };
        
        let stride = self.unsanitize_tuple(&self.get_var_name(s_id, expr)?)?;
        let padding = self.unsanitize_tuple(&self.get_var_name(p_id, expr)?)?;
        let dilation = self.unsanitize_tuple(&self.get_var_name(d_id, expr)?)?;
        let groups = self.unsanitize_value(&self.get_var_name(g_id, expr)?)?;

        let w_shape_py = w_tensor.getattr(self.py, "shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>(self.py)?;
        let out_channels = w_shape[0];
        let in_channels_per_group = w_shape[1];
        let kernel_size = (w_shape[2], w_shape[3]).to_object(self.py);
        let in_channels = in_channels_per_group * groups.extract::<usize>(self.py)?;

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
        
        conv_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        if let Some(b_tensor_val) = b_tensor {
            conv_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_tensor_val,))?;
        }
        
        let module_target_name = format!("conv2d_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), conv_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    
    fn build_batchnorm_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, m_id: Id, v_id: Id, input_node: PyObject, eps_id: Id
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?;
        let b_name = self.get_var_name(b_id, expr)?;
        let mean_name = self.get_var_name(m_id, expr)?;
        let var_name = self.get_var_name(v_id, expr)?;
        let eps_str = self.get_var_name(eps_id, expr)?; 

        eprintln!("[DEBUG] Building BatchNorm with params: w={}, b={}, mean={}, var={}, eps={}", 
            w_name, b_name, mean_name, var_name, eps_str);

        let base_name = if let Some(stripped) = w_name.strip_suffix("_weight") {
            stripped
        } else {
            w_name.strip_suffix(".weight").unwrap_or(&w_name)
        }.replace("_", ".");

        let weight_fx_name = format!("{}.weight", base_name);
        let bias_fx_name = format!("{}.bias", base_name);
        let mean_fx_name = format!("{}.running_mean", base_name);
        let var_fx_name = format!("{}.running_var", base_name);

        eprintln!("[DEBUG] BatchNorm resolved names: weight={}, bias={}, mean={}, var={}", 
            weight_fx_name, bias_fx_name, mean_fx_name, var_fx_name);

        let w_tensor = self.get_param_from_original_gm(&weight_fx_name)?;
        let b_tensor = self.get_param_from_original_gm(&bias_fx_name)?;
        let mean_tensor = self.get_buffer_from_original_gm(&mean_fx_name)?;
        let var_tensor = self.get_buffer_from_original_gm(&var_fx_name)?;

        let eps = self.unsanitize_value(&eps_str)?;
        
        let w_shape_py = w_tensor.getattr(self.py, "shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>(self.py)?;
        let num_features = w_shape[0];

        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;

        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("eps", eps)?;
        
        let bn_module = torch_nn.getattr("BatchNorm2d")?.call(
            (num_features,),
            Some(&kwargs)
        )?;

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
    
    // ✅ DÜZELTME: Uyarıları gidermek için kullanılmayan ID'lerin başına _ eklendi
    fn build_fused_conv_bn_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        input_node: PyObject, 
        w_c_id: Id, b_c_id: Id, w_bn_id: Id, _b_bn_id: Id, 
        _m_id: Id, _v_id: Id, eps_id: Id,
        s_id: Id, p_id: Id, d_id: Id, g_id: Id 
    ) -> PyResult<PyObject> {
        
        eprintln!("[DEBUG] Building Fused Conv-BN module...");

        let w_c_name_sanitized = self.get_var_name(w_c_id, expr)?;
        let b_c_name = self.get_var_name(b_c_id, expr)?; 
        let w_bn_name_sanitized = self.get_var_name(w_bn_id, expr)?; 
        
        let eps_str = self.get_var_name(eps_id, expr)?; 
        
        let torch = PyModule::import_bound(self.py, "torch")?;
        
        let w_c_name = w_c_name_sanitized.replace("_", ".");
        let w_c = self.get_param_from_original_gm(&w_c_name)?;
        let b_c_exists = b_c_name != "none";

        let base_name_bn = if let Some(stripped) = w_bn_name_sanitized.strip_suffix("_weight") {
            stripped 
        } else {
            return Err(HypatiaError::new_err(format!("FusedConvBN: BN weight adı '_weight' ile bitmiyor: {}", w_bn_name_sanitized)));
        }.replace("_", "."); 

        let w_bn_name = format!("{}.weight", base_name_bn);
        let b_bn_name = format!("{}.bias", base_name_bn);
        let m_name = format!("{}.running_mean", base_name_bn);
        let v_name = format!("{}.running_var", base_name_bn);

        eprintln!("[DEBUG] FusedConvBN resolved names: conv_w={}, bn_w={}, bn_m={}, bn_v={}", 
            w_c_name, w_bn_name, m_name, v_name);

        let w_bn = self.get_param_from_original_gm(&w_bn_name)?;
        let b_bn = self.get_param_from_original_gm(&b_bn_name)?;
        let m = self.get_buffer_from_original_gm(&m_name)?;
        let v = self.get_buffer_from_original_gm(&v_name)?;
        let eps = self.unsanitize_value(&eps_str)?.extract::<f64>(self.py)?;

        let sqrt_var = v.call_method1(self.py, "add", (eps,))?.call_method0(self.py, "sqrt")?;
        let scale = w_bn.call_method1(self.py, "div", (sqrt_var,))?;

        let w_c_shape = w_c.getattr(self.py, "shape")?.extract::<Vec<usize>>(self.py)?;
        let scale_shape: Vec<isize> = vec![-1, 1, 1, 1];
        let scale_broadcast = scale.call_method1(self.py, "view", (scale_shape,))?;
        
        let w_fused = w_c.call_method1(self.py, "mul", (scale_broadcast,))?;

        let b_c_val = if b_c_exists {
            self.get_param_from_original_gm(&b_c_name.replace("_", "."))?
        } else {
            torch.call_method1("zeros_like", (m.clone(),))?.to_object(self.py)
        };
        
        let b_fused = b_c_val.call_method1(self.py, "sub", (m,))?
                           .call_method1(self.py, "mul", (scale,))?
                           .call_method1(self.py, "add", (b_bn,))?;

        let stride = self.unsanitize_tuple(&self.get_var_name(s_id, expr)?)?;
        let padding = self.unsanitize_tuple(&self.get_var_name(p_id, expr)?)?;
        let dilation = self.unsanitize_tuple(&self.get_var_name(d_id, expr)?)?;
        let groups = self.unsanitize_value(&self.get_var_name(g_id, expr)?)?;
        
        let out_channels = w_c_shape[0];
        let in_channels = w_c_shape[1] * groups.extract::<usize>(self.py)?;
        
        // ✅ DÜZELTME: E0425 (yazım hatası) düzeltildi. w_shape[3] -> w_c_shape[3]
        let kernel_size = (w_c_shape[2], w_c_shape[3]).to_object(self.py);
        
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", dilation)?;
        kwargs.set_item("groups", groups)?;
        kwargs.set_item("bias", true)?; 

        let conv_module = torch_nn.getattr("Conv2d")?.call(
            (in_channels, out_channels, kernel_size),
            Some(&kwargs)
        )?;
        
        conv_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_fused,))?;
        conv_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_fused,))?;

        let module_target_name = format!("fused_conv_bn_{}", self.param_map.len());
        self.param_map.insert(module_target_name.clone(), conv_module.to_object(self.py));
        
        let args_tuple = PyTuple::new_bound(self.py, &[input_node]);
        let new_node = self.new_graph.call_method(
            "create_node", ("call_module", module_target_name, args_tuple), None
        )?;
        
        eprintln!("[DEBUG] Fused Conv-BN module created successfully");
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

    fn get_param_from_original_gm(&self, param_name: &str) -> PyResult<PyObject> {
        Ok(self.original_gm.call_method1("get_parameter", (param_name,))?.to_object(self.py))
    }
    
    fn get_buffer_from_original_gm(&self, buffer_name: &str) -> PyResult<PyObject> {
        eprintln!("[DEBUG] Getting buffer/parameter: {}", buffer_name);

        let buffer_result = self.original_gm.call_method1("get_buffer", (buffer_name,));
        if let Ok(buffer) = buffer_result {
            eprintln!("[DEBUG] Found as buffer: {}", buffer_name);
            return Ok(buffer.to_object(self.py));
        }

        let param_result = self.original_gm.call_method1("get_parameter", (buffer_name,));
        if let Ok(param) = param_result {
            eprintln!("[DEBUG] Found as parameter: {}", buffer_name);
            return Ok(param.to_object(self.py));
        }
        
        let mut alternatives = Vec::new();
        if buffer_name.contains(".") {
            alternatives.push(buffer_name.replace(".", "_"));
        }
        if buffer_name.contains("_") {
            alternatives.push(buffer_name.replace("_", "."));
        }

        for alt_name in alternatives.iter() {
            eprintln!("[DEBUG] Trying alternative: {}", alt_name);
            if let Ok(result) = self.original_gm.call_method1("get_buffer", (alt_name,)) {
                eprintln!("[DEBUG] Found alternative as buffer: {}", alt_name);
                return Ok(result.to_object(self.py));
            }
            if let Ok(result) = self.original_gm.call_method1("get_parameter", (alt_name,)) {
                eprintln!("[DEBUG] Found alternative as parameter: {}", alt_name);
                return Ok(result.to_object(self.py));
            }
        }

        Err(HypatiaError::new_err(format!("Buffer veya parameter bulunamadı: {} (denenen alternatifler: {:?})", buffer_name, alternatives)))
    }
    
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

    // ✅ DÜZELTME: TypeError'ı çözmek için önce i64 (int) kontrolü yap
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
            
            // ✅ GÜNCEL TEST: '3' artık i64 (int) olmalı
            let val3 = rebuilder.unsanitize_value("3").unwrap();
            assert_eq!(val3.extract::<i64>(py).unwrap(), 3);
            
            let val4 = rebuilder.unsanitize_value("1e_05").unwrap();
            assert_eq!(val4.extract::<f64>(py).unwrap(), 1e-05);
            
            // ✅ GÜNCEL TEST: '3.0' f64 (float) olmalı
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
            assert_eq!(tuple.get_item(1).unwrap().extract::<i64>().unwrap(), 1);

            let val2 = rebuilder.unsanitize_tuple("3").unwrap();
            let tuple2 = val2.downcast_bound::<PyTuple>(py).unwrap();
            assert_eq!(tuple2.len(), 1);
            assert_eq!(tuple2.get_item(0).unwrap().extract::<i64>().unwrap(), 3);
        });
    }
}