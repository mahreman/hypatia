//! Visualization: DOT graph export and optimization report generation
//!
//! Provides structured data for visualizing the optimization pipeline:
//! - S-expression → DOT graph (tree structure)
//! - Before/after comparison
//! - Optimization statistics and cost breakdown
//! - Node-type frequency analysis

use crate::egraph_optimizer::HypatiaLang;
use egg::{Language, RecExpr};
use std::collections::HashMap;
use std::fmt::Write;

// ============================================================================
// S-expression → DOT Graph
// ============================================================================

/// Convert an S-expression string to GraphViz DOT format.
/// Each operator becomes a node; children become edges.
pub fn sexpr_to_dot(expr_str: &str, graph_name: &str) -> Result<String, String> {
    let expr: RecExpr<HypatiaLang> = expr_str
        .parse()
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(rec_expr_to_dot(&expr, graph_name))
}

/// Convert a RecExpr to DOT format.
pub fn rec_expr_to_dot(expr: &RecExpr<HypatiaLang>, graph_name: &str) -> String {
    let nodes = expr.as_ref();
    let mut dot = String::new();

    writeln!(dot, "digraph {} {{", graph_name).unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];").unwrap();
    writeln!(dot, "    edge [arrowsize=0.7];").unwrap();
    writeln!(dot).unwrap();

    for (i, node) in nodes.iter().enumerate() {
        let (label, color) = node_style(node);
        writeln!(
            dot,
            "    n{} [label=\"{}\", fillcolor=\"{}\"];",
            i, label, color
        )
        .unwrap();

        // Add edges to children
        for child_id in node.children() {
            let child_idx = usize::from(*child_id);
            writeln!(dot, "    n{} -> n{};", i, child_idx).unwrap();
        }
    }

    writeln!(dot, "}}").unwrap();
    dot
}

/// Get display label and color for a HypatiaLang node.
fn node_style(node: &HypatiaLang) -> (String, &'static str) {
    use HypatiaLang::*;

    match node {
        // Arithmetic: light blue
        Add(_) => ("add".into(), "#B3D9FF"),
        Mul(_) => ("mul".into(), "#B3D9FF"),
        Sub(_) => ("sub".into(), "#B3D9FF"),
        Div(_) => ("div".into(), "#B3D9FF"),
        Neg(_) => ("neg".into(), "#B3D9FF"),
        MatMul(_) => ("matmul".into(), "#B3D9FF"),

        // Activations: light green
        ReLU(_) => ("ReLU".into(), "#B3FFB3"),
        Sigmoid(_) => ("Sigmoid".into(), "#B3FFB3"),
        Tanh(_) => ("Tanh".into(), "#B3FFB3"),
        GELU(_) => ("GELU".into(), "#B3FFB3"),
        SiLU(_) => ("SiLU".into(), "#B3FFB3"),
        Softmax(_) => ("Softmax".into(), "#B3FFB3"),
        LeakyReLU(_) => ("LeakyReLU".into(), "#B3FFB3"),
        ELU(_) => ("ELU".into(), "#B3FFB3"),

        // Linear/Conv: light orange
        Linear(_) => ("Linear".into(), "#FFD9B3"),
        Conv2d(_) => ("Conv2d".into(), "#FFD9B3"),

        // Normalization: light purple
        LayerNorm(_) => ("LayerNorm".into(), "#D9B3FF"),
        BatchNorm(_) => ("BatchNorm".into(), "#D9B3FF"),
        BatchNorm1d(_) => ("BatchNorm1d".into(), "#D9B3FF"),
        GroupNorm(_) => ("GroupNorm".into(), "#D9B3FF"),

        // Attention: light yellow
        Attention(_) => ("Attention".into(), "#FFFFB3"),

        // Fused ops: light red/pink (highlight optimizations)
        LinearReLU(_) => ("Linear+ReLU".into(), "#FFB3B3"),
        FusedLinearReLU(_) => ("FusedLinearReLU".into(), "#FFB3B3"),
        FusedMLP(_) => ("FusedMLP".into(), "#FFB3B3"),
        FusedGeluMLP(_) => ("FusedGeluMLP".into(), "#FFB3B3"),
        FusedConvBN(_) => ("FusedConvBN".into(), "#FFB3B3"),
        FusedAttention(_) => ("FusedAttention".into(), "#FFB3B3"),
        FusedLNAttention(_) => ("FusedLNAttn".into(), "#FFB3B3"),

        // Sparse: light teal
        SparseLinear(_) => ("SparseLinear".into(), "#B3FFE6"),
        FusedSparseLinearReLU(_) => ("FusedSparseReLU".into(), "#B3FFE6"),
        ToSparse(_) => ("ToSparse".into(), "#B3FFE6"),

        // Mixed precision: light cyan
        CastFP16(_) => ("CastFP16".into(), "#B3F0FF"),
        CastBF16(_) => ("CastBF16".into(), "#B3F0FF"),
        CastFP32(_) => ("CastFP32".into(), "#B3F0FF"),
        MixedPrecisionLinear(_) => ("MPLinear".into(), "#B3F0FF"),
        FusedMPLinearReLU(_) => ("FusedMPReLU".into(), "#B3F0FF"),

        // Neuromorphic: light pink
        LIF(_) => ("LIF".into(), "#FFB3E6"),
        SpikeEncode(_) => ("SpikeEncode".into(), "#FFB3E6"),
        SpikeDecode(_) => ("SpikeDecode".into(), "#FFB3E6"),
        LIFLinear(_) => ("LIFLinear".into(), "#FFB3E6"),
        NeuromorphicLinear(_) => ("NeuroLinear".into(), "#FFB3E6"),

        // Terminals: light gray
        Constant(v) => (format!("{:.3}", v), "#E0E0E0"),
        Var(s) => (format!("{}", s), "#FFFFFF"),

        // Default
        _ => (format!("{:?}", node).chars().take(20).collect(), "#F0F0F0"),
    }
}

// ============================================================================
// Optimization Report
// ============================================================================

/// Structured optimization report.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub input_expr: String,
    pub output_expr: String,
    pub input_node_count: usize,
    pub output_node_count: usize,
    pub input_node_types: HashMap<String, usize>,
    pub output_node_types: HashMap<String, usize>,
    pub rewrites_applied: Vec<String>,
    pub fusions_found: Vec<String>,
    pub node_reduction: i32,
}

/// Analyze a RecExpr and count node types.
pub fn count_node_types(expr: &RecExpr<HypatiaLang>) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in expr.as_ref() {
        let name = node_type_name(node);
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

/// Get human-readable name for a node type.
fn node_type_name(node: &HypatiaLang) -> String {
    use HypatiaLang::*;
    match node {
        Add(_) => "Add", Mul(_) => "Mul", Sub(_) => "Sub", Div(_) => "Div",
        Neg(_) => "Neg", Exp(_) => "Exp", Log(_) => "Log", Sqrt(_) => "Sqrt",
        Pow(_) => "Pow", MatMul(_) => "MatMul",
        ReLU(_) => "ReLU", Sigmoid(_) => "Sigmoid", Tanh(_) => "Tanh",
        GELU(_) => "GELU", SiLU(_) => "SiLU", Softmax(_) => "Softmax",
        LeakyReLU(_) => "LeakyReLU", ELU(_) => "ELU",
        ReLUGrad(_) => "ReLUGrad",
        Linear(_) => "Linear", Conv2d(_) => "Conv2d",
        LayerNorm(_) => "LayerNorm", BatchNorm(_) => "BatchNorm",
        BatchNorm1d(_) => "BatchNorm1d", GroupNorm(_) => "GroupNorm",
        Dropout(_) => "Dropout",
        Mean(_) => "Mean", Variance(_) => "Variance",
        Max(_) => "Max", Min(_) => "Min",
        Flatten(_) => "Flatten",
        Attention(_) => "Attention", Embedding(_) => "Embedding",
        TransformerEncoder(_) => "TransformerEncoder",
        MaxPool2d(_) => "MaxPool2d", AvgPool2d(_) => "AvgPool2d",
        AdaptiveAvgPool2d(_) => "AdaptiveAvgPool2d",
        // Fused
        LinearReLU(_) => "LinearReLU", FusedLinearReLU(_) => "FusedLinearReLU",
        FusedMLP(_) => "FusedMLP", FusedGeluMLP(_) => "FusedGeluMLP",
        FusedConvBN(_) => "FusedConvBN",
        FusedAttention(_) => "FusedAttention", FusedLNAttention(_) => "FusedLNAttention",
        // Sparse
        SparseLinear(_) => "SparseLinear", FusedSparseLinearReLU(_) => "FusedSparseLinearReLU",
        ToSparse(_) => "ToSparse",
        // Mixed precision
        CastFP16(_) => "CastFP16", CastBF16(_) => "CastBF16", CastFP32(_) => "CastFP32",
        MixedPrecisionLinear(_) => "MPLinear", FusedMPLinearReLU(_) => "FusedMPLinearReLU",
        // Neuromorphic
        LIF(_) => "LIF", SpikeEncode(_) => "SpikeEncode", SpikeDecode(_) => "SpikeDecode",
        LIFLinear(_) => "LIFLinear", NeuromorphicLinear(_) => "NeuromorphicLinear",
        // Terminals
        Constant(_) => "Constant", Var(_) => "Var",
    }.to_string()
}

/// Build an optimization report comparing before/after expressions.
pub fn build_optimization_report(
    input_str: &str,
    output_str: &str,
) -> Result<OptimizationReport, String> {
    let input_expr: RecExpr<HypatiaLang> = input_str
        .parse()
        .map_err(|e| format!("Parse input error: {}", e))?;
    let output_expr: RecExpr<HypatiaLang> = output_str
        .parse()
        .map_err(|e| format!("Parse output error: {}", e))?;

    let input_types = count_node_types(&input_expr);
    let output_types = count_node_types(&output_expr);

    let input_count = input_expr.as_ref().len();
    let output_count = output_expr.as_ref().len();

    // Detect which fusions occurred
    let mut fusions = Vec::new();
    let fusion_names = [
        "FusedLinearReLU", "FusedMLP", "FusedGeluMLP", "FusedConvBN",
        "FusedAttention", "FusedLNAttention",
        "FusedSparseLinearReLU", "FusedMPLinearReLU",
        "NeuromorphicLinear", "LIFLinear",
        "SparseLinear", "MPLinear",
    ];

    for name in &fusion_names {
        if output_types.contains_key(*name) && !input_types.contains_key(*name) {
            let count = output_types[*name];
            fusions.push(format!("{} (x{})", name, count));
        }
    }

    // Detect which rewrites were applied (by comparing type changes)
    let mut rewrites = Vec::new();
    for (name, &in_count) in &input_types {
        let out_count = output_types.get(name).copied().unwrap_or(0);
        if out_count < in_count {
            rewrites.push(format!("{}: {} → {}", name, in_count, out_count));
        }
    }

    Ok(OptimizationReport {
        input_expr: input_str.to_string(),
        output_expr: output_str.to_string(),
        input_node_count: input_count,
        output_node_count: output_count,
        input_node_types: input_types,
        output_node_types: output_types,
        rewrites_applied: rewrites,
        fusions_found: fusions,
        node_reduction: input_count as i32 - output_count as i32,
    })
}

/// Generate ASCII visualization of an S-expression tree.
pub fn sexpr_to_ascii_tree(expr_str: &str) -> Result<String, String> {
    let expr: RecExpr<HypatiaLang> = expr_str
        .parse()
        .map_err(|e| format!("Parse error: {}", e))?;

    let nodes = expr.as_ref();
    if nodes.is_empty() {
        return Ok("(empty)".to_string());
    }

    let root = nodes.len() - 1;
    let mut output = String::new();
    ascii_tree_recursive(nodes, root, "", true, &mut output);
    Ok(output)
}

fn ascii_tree_recursive(
    nodes: &[HypatiaLang],
    idx: usize,
    prefix: &str,
    is_last: bool,
    output: &mut String,
) {
    let connector = if is_last { "└── " } else { "├── " };
    let node = &nodes[idx];
    let label = node_type_name(node);

    // Add constant/var value info
    let detail = match node {
        HypatiaLang::Constant(v) => format!(" = {:.4}", v),
        HypatiaLang::Var(s) => format!(" '{}'", s),
        _ => String::new(),
    };

    writeln!(output, "{}{}{}{}", prefix, connector, label, detail).unwrap();

    let children: Vec<usize> = node.children().iter().map(|id| usize::from(*id)).collect();
    let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

    for (i, &child_idx) in children.iter().enumerate() {
        let child_is_last = i == children.len() - 1;
        ascii_tree_recursive(nodes, child_idx, &child_prefix, child_is_last, output);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sexpr_to_dot_basic() {
        let dot = sexpr_to_dot("(relu (linear w b x))", "test").unwrap();
        assert!(dot.contains("digraph test"));
        assert!(dot.contains("ReLU"));
        assert!(dot.contains("Linear"));
    }

    #[test]
    fn test_sexpr_to_dot_constant() {
        let dot = sexpr_to_dot("(add x 1.0)", "test").unwrap();
        assert!(dot.contains("add"));
        assert!(dot.contains("1.000"));
    }

    #[test]
    fn test_node_types_count() {
        let expr: RecExpr<HypatiaLang> = "(relu (linear w b x))".parse().unwrap();
        let counts = count_node_types(&expr);
        assert_eq!(counts["ReLU"], 1);
        assert_eq!(counts["Linear"], 1);
        assert_eq!(counts["Var"], 3);
    }

    #[test]
    fn test_optimization_report() {
        let input = "(relu (linear w b x))";
        let output = "(fused_linear_relu w b x)";
        let report = build_optimization_report(input, output).unwrap();

        assert!(report.node_reduction > 0);
        assert!(!report.fusions_found.is_empty());
        assert!(report.fusions_found.iter().any(|f| f.contains("FusedLinearReLU")));
    }

    #[test]
    fn test_ascii_tree() {
        let tree = sexpr_to_ascii_tree("(relu (linear w b x))").unwrap();
        assert!(tree.contains("ReLU"));
        assert!(tree.contains("Linear"));
        assert!(tree.contains("'w'"));
    }

    #[test]
    fn test_empty_expr() {
        let result = sexpr_to_dot("x", "test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_complex_expr() {
        let expr = "(fused_attention wq bq wk bk wv bv wo bo x x)";
        let dot = sexpr_to_dot(expr, "attention").unwrap();
        assert!(dot.contains("FusedAttention"));
    }

    #[test]
    fn test_report_no_change() {
        let expr = "(relu x)";
        let report = build_optimization_report(expr, expr).unwrap();
        assert_eq!(report.node_reduction, 0);
        assert!(report.fusions_found.is_empty());
    }
}
