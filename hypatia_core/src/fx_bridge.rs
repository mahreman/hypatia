// ============================================================================
// FX BRIDGE: PyTorch FX Graph ↔ Hypatia S-expression
// ============================================================================
// (Diğer kısımlar değişmedi)

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyDict, PyModule};
use crate::python_bindings::HypatiaError;
use std::collections::HashMap;

use egg::{RecExpr, Id};
use crate::egraph_optimizer::HypatiaLang;

// ============================================================================
// PHASE 2: FX NODE STRUCTURES
// (Değişiklik yok)
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
// (Değişiklik yok)
// ============================================================================
pub fn fx_graph_to_sexpr(
    py: Python<'_>, 
    gm: &Bound<PyAny>,
    types_map: &HashMap<String, String>
) -> PyResult<String> {
    let graph = gm.getattr("graph")?;
    eprintln!("[fx_bridge::Phase2] Starting FX parse...");
    let nodes = parse_fx_nodes(py, &graph)?;
    eprintln!("[DEBUG] FX graph node count: {}", nodes.len());
    let sexpr = build_sexpr_from_nodes(&nodes, types_map)?;
    Ok(sexpr)
}

// ============================================================================
// PHASE 3: MAIN RECONSTRUCTION FUNCTION (sexpr_to_fx_graph)
// (Değişiklik yok)
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
// PHASE 2: NODE PARSING
// (Değişiklik yok)
// ============================================================================
fn parse_fx_nodes(py: Python<'_>, graph: &Bound<PyAny>) -> PyResult<Vec<FxNode>> {
    let nodes_attr = graph.getattr("nodes")?;
    let mut parsed_nodes = Vec::new();
    
    for node_obj in nodes_attr.iter()? {
        let node = node_obj?;
        let name = node.getattr("name")?.extract::<String>()?;
        let op_str = node.getattr("op")?.extract::<String>()?;
        
        let target_str = if let Ok(target) = node.getattr("target") {
            if let Ok(s) = target.extract::<String>() { s }
            else if let Ok(s) = target.str() { s.to_string_lossy().to_string() }
            else { "unknown".to_string() }
        } else { "unknown".to_string() };
        
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
// (Değişiklik yok)
// ============================================================================

fn build_sexpr_from_nodes(
    nodes: &[FxNode],
    types_map: &HashMap<String, String>
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
                handle_call_module(target, &node.inputs, &node.kwargs, &node_exprs, types_map)?
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
// (Değişiklik yok)
// ============================================================================

fn handle_call_module(
    target: &str, 
    inputs: &[String],
    _kwargs: &HashMap<String, String>,
    node_exprs: &HashMap<String, String>,
    types_map: &HashMap<String, String>
) -> PyResult<String> {
    
    let input_expr = if let Some(input_name) = inputs.first() {
        node_exprs.get(input_name).ok_or_else(|| HypatiaError::new_err(
            format!("Input node '{}' not found in expression map", input_name)
        ))?
    } else {
         return Err(HypatiaError::new_err(format!("Module call '{}' has no inputs", target)));
    };

    let module_type = match types_map.get(target) {
        Some(t) => t.as_str(),
        None => "Unknown",
    };

    if module_type == "Linear" {
        let w = sanitize_var_name(&format!("{}.weight", target));
        let b = sanitize_var_name(&format!("{}.bias", target));
        Ok(format!("(linear {} {} {})", w, b, input_expr))
    } else if module_type == "ReLU" {
        Ok(format!("(relu {})", input_expr))
    } else if module_type == "Embedding" {
        let w = sanitize_var_name(&format!("{}.weight", target));
        Ok(format!("(embedding {} {})", w, input_expr))
    } else if module_type == "TransformerEncoder" {
        Ok(format!("(transformer_encoder {})", input_expr))
    } else {
        eprintln!("[fx_bridge] UYARI: Desteklenmeyen modül tipi '{}' (target: {}), pas geçiliyor.", module_type, target);
        Ok(input_expr.clone()) // Passthrough
    }
}
fn handle_call_function(
    target: &str,
    inputs: &[String],
    kwargs: &HashMap<String, String>,
    node_exprs: &HashMap<String, String>,
) -> PyResult<String> {
    
    if target.contains("flatten") {
        if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
            return Ok(format!("(flatten {})", input));
        } else {
            return Err(HypatiaError::new_err("Flatten node has no valid input"));
        }
    }
    
    let func_name = if target.contains("add") || target.contains("__add__") { "add" }
    else if target.contains("mul") || target.contains("__mul__") { "mul" }
    else if target.contains("sub") || target.contains("__sub__") { "sub" }
    else if target.contains("div") || target.contains("__truediv__") { "div" }
    else if target.contains("matmul") { "matmul" } 
    else if target.contains("mean") { "mean" } 
    else if target.contains("relu") { "relu" }
    else if target.contains("sigmoid") { "sigmoid" }
    else if target.contains("tanh") { "tanh" }
    else if target.contains("softmax") { "softmax" }
    else if target.contains("batch_norm") { "batchnorm" }
    else if target.contains("max_pool2d") { "maxpool2d" }
    else { "unknown_func" };
    
    match func_name {
        "add" | "mul" | "sub" | "div" | "matmul" => {
            if inputs.len() >= 2 {
                let input1 = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
                let input2 = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
                Ok(format!("({} {} {})", func_name, input1, input2))
            } else { Err(HypatiaError::new_err(format!("{} node needs 2 inputs", func_name))) }
        },
        "relu" | "sigmoid" | "tanh" | "softmax" | "mean" => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(format!("({} {})", func_name, input))
            } else { Err(HypatiaError::new_err(format!("{} node has no valid input", func_name))) }
        },
        "batchnorm" => {
            if inputs.len() < 5 { return Err(HypatiaError::new_err(format!("batchnorm node {} inputs, 5 expected", inputs.len()))); }
            let i = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
            let m = node_exprs.get(&inputs[1]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[1])))?;
            let v = node_exprs.get(&inputs[2]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[2])))?;
            let w = node_exprs.get(&inputs[3]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[3])))?;
            let b = node_exprs.get(&inputs[4]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[4])))?;
            let eps = sanitize_var_name(&kwargs.get("eps").cloned().unwrap_or("1e-05".to_string()));
            Ok(format!("(batchnorm {} {} {} {} {} {})", w, b, m, v, i, eps))
        },
        "maxpool2d" => {
            let i = node_exprs.get(&inputs[0]).ok_or_else(|| HypatiaError::new_err(format!("Undefined input: {}", inputs[0])))?;
            let kernel = sanitize_var_name(&kwargs.get("kernel_size").cloned().unwrap_or("2 2".to_string()));
            let stride = sanitize_var_name(&kwargs.get("stride").cloned().unwrap_or(kernel.clone()));
            let padding = sanitize_var_name(&kwargs.get("padding").cloned().unwrap_or("0 0".to_string()));
            Ok(format!("(maxpool2d {} {} {} {})", i, kernel, stride, padding))
        },
        _ => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                eprintln!("[fx_bridge] UYARI: Desteklenmeyen fonksiyon {}, pas geçiliyor.", target);
                Ok(input.clone())
            } else { Err(HypatiaError::new_err(format!("Fonksiyon node'u {} geçerli bir girdiye sahip değil", target))) }
        }
    }
}
fn handle_call_method(
    method: &str,
    inputs: &[String],
    _kwargs: &HashMap<String, String>, 
    node_exprs: &HashMap<String, String>,
) -> PyResult<String> {
    match method {
        "view" | "reshape" | "flatten" | "unflatten" | "size" => {
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(input.clone()) // Passthrough
            } else { Err(HypatiaError::new_err(format!("Method {} has no input", method))) }
        },
        "mean" => { 
            if let Some(input) = inputs.first().and_then(|n| node_exprs.get(n)) {
                Ok(format!("(mean {})", input))
            } else { 
                Err(HypatiaError::new_err("Method 'mean' has no input".to_string())) 
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
// (✅ GÜNCELLENDİ: 'build_call_function' ve 'reconstruct_node' düzeltildi)
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
            _ => Err(HypatiaError::new_err(format!(
                "Reconstruction failed: Expected Var for parameter, found {:?}", node
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
            HypatiaLang::Add([a_id, b_id]) => {
                self.build_call_function("add", &[*a_id, *b_id], expr, None)
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
            HypatiaLang::Mean(x_id) => {
                // ====================================================================
                // ✅ CRITICAL FIX: 'mean' için kwargs'ı buradan oluştur
                // ====================================================================
                let kwargs = PyDict::new_bound(self.py);
                kwargs.set_item("dim", 1)?; // Transformer'daki 'mean(dim=1)' varsayımı
                self.build_call_function("mean", &[*x_id], expr, Some(kwargs))
                // ====================================================================
            }
            HypatiaLang::TransformerEncoder(x_id) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_passthrough_module("transformer_encoder", input_node)
            }
            HypatiaLang::Linear([w_id, b_id, x_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_linear_module(node_id, expr, *w_id, *b_id, input_node, "linear")
            }
            HypatiaLang::Conv2d([w_id, b_id, x_id, s_id, p_id, d_id, g_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_conv2d_module(node_id, expr, *w_id, *b_id, input_node, *s_id, *p_id, *d_id, *g_id)
            }
            HypatiaLang::BatchNorm([w_id, b_id, m_id, v_id, x_id, eps_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_batchnorm_module(node_id, expr, *w_id, *b_id, *m_id, *v_id, input_node, *eps_id)
            }
            HypatiaLang::MaxPool2d([x_id, k_id, s_id, p_id]) => {
                let input_node = self.reconstruct_node(*x_id, expr)?;
                self.build_maxpool2d_module(node_id, expr, input_node, *k_id, *s_id, *p_id)
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
    fn build_linear_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, input_node: PyObject, prefix: &str,
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?; 
        let b_name = self.get_var_name(b_id, expr)?;
        let module_target_name = format!("{}_{}", prefix, self.param_map.len());
        let w_tensor = self.get_param_from_original_gm(&w_name)?;
        let b_tensor = self.get_param_from_original_gm(&b_name)?;
        let w_shape_py = w_tensor.bind(self.py).getattr("shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
        let in_features = w_shape[1];
        let out_features = w_shape[0];
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let linear_module = torch_nn.getattr("Linear")?.call1((in_features, out_features))?;
        linear_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        linear_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_tensor,))?;
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
        let w_tensor = self.get_param_from_original_gm(&w_name)?;
        let w_shape_py = w_tensor.bind(self.py).getattr("shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
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
            "create_node", ("call_module", module_target_name.clone(), args_tuple), None
        )?;
        Ok(new_node.to_object(self.py))
    }
    
    // ====================================================================
    // ✅ CRITICAL FIX: `flatten` ve `mean` için `kwargs` mantığı düzeltildi
    // ====================================================================
    fn build_call_function(
        &mut self,
        target: &str,
        arg_ids: &[Id],
        expr: &RecExpr<HypatiaLang>,
        kwargs: Option<Bound<'_, PyDict>> // `reconstruct_node`'dan gelen opsiyonel kwargs
    ) -> PyResult<PyObject> {
        let mut arg_nodes = Vec::new();
        for arg_id in arg_ids {
            let node_obj = self.reconstruct_node(*arg_id, expr)?;
            arg_nodes.push(node_obj);
        }
        
        // 'kwargs'ı, 'reconstruct_node'dan geleni kullanarak başlat
        let mut final_kwargs = kwargs; 
        
        if target == "flatten" {
            // 'start_dim=1' (PyObject) oluştur ve 'args'a ekle
            let start_dim_arg = 1.to_object(self.py);
            arg_nodes.push(start_dim_arg);
            final_kwargs = None; // 'flatten' kwargs almaz
        }
        
        if target == "mean" {
            // 'mean' için kwargs'ın 'reconstruct_node'dan geldiğini varsay
            if final_kwargs.is_none() {
                 // Güvenlik önlemi (eğer reconstruct_node'da ayarlanmadıysa)
                 let dict = PyDict::new_bound(self.py);
                 dict.set_item("dim", 1)?;
                 final_kwargs = Some(dict);
            }
        }

        let args_tuple = PyTuple::new_bound(self.py, arg_nodes);
        
        let torch_module = PyModule::import_bound(self.py, "torch")?;
        let target_fn = torch_module.getattr(target)?;
        
        // ====================================================================
        // ✅ CRITICAL FIX: `kwargs`'ı 4. pozisyonel argüman olarak geçir
        // ====================================================================
        
        // 1. kwargs dict'ini (veya PyNone) oluştur
        let kwargs_for_create_node = final_kwargs
            .map(|dict| dict.to_object(self.py))
            .unwrap_or_else(|| self.py.None()); // kwargs None ise PyNone kullan
            
        // 2. Ana 'args' tuple'ını 4 elementli olarak oluştur
        let create_node_args = (
            "call_function", 
            target_fn, 
            args_tuple, 
            kwargs_for_create_node // 4. element
        );
        
        // 3. 'call_method1' (tek argümanlı) kullanarak 'create_node'u çağır
        let new_node = self.new_graph.call_method1(
            "create_node", 
            create_node_args
        )?;
        // ====================================================================
        
        Ok(new_node.to_object(self.py))
    }
    // ====================================================================


    fn build_conv2d_module(
        &mut self, _node_id: Id, expr: &RecExpr<HypatiaLang>,
        w_id: Id, b_id: Id, input_node: PyObject, s_id: Id, p_id: Id, d_id: Id, g_id: Id
    ) -> PyResult<PyObject> {
        let w_name = self.get_var_name(w_id, expr)?;
        let b_name = self.get_var_name(b_id, expr)?;
        let w_tensor = self.get_param_from_original_gm(&w_name)?;
        let b_tensor = self.get_param_from_original_gm(&b_name)?;
        let stride_str = self.get_var_name(s_id, expr)?;
        let padding_str = self.get_var_name(p_id, expr)?;
        let dilation_str = self.get_var_name(d_id, expr)?;
        let groups_str = self.get_var_name(g_id, expr)?;
        let stride = self.unsanitize_tuple(&stride_str)?;
        let padding = self.unsanitize_tuple(&padding_str)?;
        let dilation = self.unsanitize_tuple(&dilation_str)?;
        let groups = self.unsanitize_value(&groups_str)?;
        let w_shape_py = w_tensor.bind(self.py).getattr("shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
        let out_channels = w_shape[0];
        let in_channels_per_group = w_shape[1];
        let kernel_h = w_shape[2];
        let kernel_w = w_shape[3];
        let kernel_size = (kernel_h, kernel_w).to_object(self.py);
        let in_channels = in_channels_per_group * groups.extract::<usize>(self.py)?;
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("stride", stride.bind(self.py))?;
        kwargs.set_item("padding", padding.bind(self.py))?;
        kwargs.set_item("dilation", dilation.bind(self.py))?;
        kwargs.set_item("groups", groups.bind(self.py))?;
        let conv_module = torch_nn.getattr("Conv2d")?.call(
            (in_channels, out_channels, kernel_size),
            Some(&kwargs)
        )?;
        conv_module.getattr("weight")?.getattr("data")?.call_method1("copy_", (w_tensor,))?;
        conv_module.getattr("bias")?.getattr("data")?.call_method1("copy_", (b_tensor,))?;
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
        let w_tensor = self.get_param_from_original_gm(&w_name)?;
        let b_tensor = self.get_param_from_original_gm(&b_name)?;
        let mean_tensor = self.get_param_from_original_gm(&mean_name)?;
        let var_tensor = self.get_param_from_original_gm(&var_name)?;
        let eps = self.unsanitize_value(&eps_str)?;
        let w_shape_py = w_tensor.bind(self.py).getattr("shape")?;
        let w_shape = w_shape_py.extract::<Vec<usize>>()?;
        let num_features = w_shape[0];
        let torch_nn = PyModule::import_bound(self.py, "torch.nn")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("eps", eps.bind(self.py))?;
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
        kwargs.set_item("stride", stride.bind(self.py))?;
        kwargs.set_item("padding", padding.bind(self.py))?;
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
        let fx_name = param_name.replace("_", ".");
        Ok(self.original_gm.call_method1("get_parameter", (fx_name,))?.to_object(self.py))
    }
    fn unsanitize_tuple(&self, s: &str) -> PyResult<PyObject> {
        let py = self.py;
        let parts: Vec<PyResult<PyObject>> = s.split('_')
            .map(|part| part.parse::<i64>()
                .map(|i| i.into_py(py))
                .map_err(|e| HypatiaError::new_err(format!("Tuple parse error: {}", e)))
            )
            .collect();
        let elements = parts.into_iter().collect::<PyResult<Vec<PyObject>>>()?;
        Ok(PyTuple::new_bound(self.py, elements).to_object(self.py))
    }
    fn unsanitize_value(&self, s: &str) -> PyResult<PyObject> {
        if let Ok(i) = s.parse::<i64>() {
            Ok(i.into_py(self.py))
        } else if let Ok(f) = s.parse::<f64>() {
            Ok(f.into_py(self.py))
        } else {
            Ok(s.into_py(self.py))
        }
    }
}


// ============================================================================
// HELPERS
// (Değişiklik yok)
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

// ============================================================================
// TESTS
// (Değişiklik yok)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sanitize_var_name() {
        assert_eq!(sanitize_var_name("input.weight"), "input_weight");
        assert_eq!(sanitize_var_name("layer[0]"), "layer_0_");
        assert_eq!(sanitize_var_name("1 1"), "1_1");
        assert_eq!(sanitize_var_name("1e-05"), "1e-05");
    }
}