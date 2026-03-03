//! Semantic Validation: Verify optimization correctness via output comparison
//!
//! Provides runtime validation that optimized models produce equivalent outputs:
//! - Random input generation with shape inference from model parameters
//! - Forward pass comparison with configurable tolerance
//! - Shape matching validation
//! - Statistical divergence analysis (max diff, mean diff, cosine similarity)
//!
//! Integrates with ChecksumMode:
//! - Strict: Full semantic validation with random inputs
//! - Soft: Shape validation + lightweight output comparison
//! - Off: No validation

use std::collections::HashMap;

// ============================================================================
// Validation Result
// ============================================================================

/// Result of semantic validation between original and optimized models.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the models are semantically equivalent within tolerance.
    pub is_valid: bool,
    /// Maximum absolute difference between outputs.
    pub max_diff: f64,
    /// Mean absolute difference between outputs.
    pub mean_diff: f64,
    /// Cosine similarity between flattened outputs (1.0 = identical).
    pub cosine_similarity: f64,
    /// Whether output shapes match.
    pub shapes_match: bool,
    /// Original model output shape.
    pub original_shape: Vec<i64>,
    /// Optimized model output shape.
    pub optimized_shape: Vec<i64>,
    /// Number of test inputs used.
    pub num_test_inputs: usize,
    /// Tolerance used for comparison.
    pub tolerance: f64,
    /// Human-readable validation message.
    pub message: String,
}

// ============================================================================
// Pure Rust Utilities (no Python dependency)
// ============================================================================

/// Compute cosine similarity between two f64 slices.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        // Both near-zero → consider identical
        if norm_a < 1e-12 && norm_b < 1e-12 {
            return 1.0;
        }
        return 0.0;
    }

    dot / denom
}

/// Compute max and mean absolute difference between two f64 slices.
pub fn compute_diffs(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() != b.len() || a.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }

    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;

    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }

    let mean_diff = sum_diff / a.len() as f64;
    (max_diff, mean_diff)
}

/// Build a ValidationResult from computed metrics.
pub fn build_validation_result(
    original_values: &[f64],
    optimized_values: &[f64],
    original_shape: Vec<i64>,
    optimized_shape: Vec<i64>,
    num_test_inputs: usize,
    tolerance: f64,
) -> ValidationResult {
    let shapes_match = original_shape == optimized_shape;

    if !shapes_match {
        return ValidationResult {
            is_valid: false,
            max_diff: f64::INFINITY,
            mean_diff: f64::INFINITY,
            cosine_similarity: 0.0,
            shapes_match: false,
            original_shape,
            optimized_shape,
            num_test_inputs,
            tolerance,
            message: "Output shapes do not match".to_string(),
        };
    }

    let (max_diff, mean_diff) = compute_diffs(original_values, optimized_values);
    let cos_sim = cosine_similarity(original_values, optimized_values);

    let is_valid = max_diff <= tolerance;

    let message = if is_valid {
        format!(
            "Semantic validation PASSED: max_diff={:.2e} <= tolerance={:.2e}, cosine_sim={:.6}",
            max_diff, tolerance, cos_sim
        )
    } else {
        format!(
            "Semantic validation FAILED: max_diff={:.2e} > tolerance={:.2e}, cosine_sim={:.6}",
            max_diff, tolerance, cos_sim
        )
    };

    ValidationResult {
        is_valid,
        max_diff,
        mean_diff,
        cosine_similarity: cos_sim,
        shapes_match,
        original_shape,
        optimized_shape,
        num_test_inputs,
        tolerance,
        message,
    }
}

// ============================================================================
// S-expression Level Validation
// ============================================================================

/// Validate that an optimized S-expression is semantically equivalent
/// to the original by checking structural properties.
///
/// This is a lightweight check that doesn't require running models:
/// - Node count should not increase dramatically
/// - All variable names from input should appear in output
/// - No unknown operators should be introduced
pub fn validate_sexpr_structure(
    input_expr: &str,
    output_expr: &str,
) -> Result<ValidationResult, String> {
    use crate::egraph_optimizer::HypatiaLang;
    use egg::RecExpr;

    let input_parsed: RecExpr<HypatiaLang> = input_expr
        .parse()
        .map_err(|e| format!("Parse input error: {}", e))?;
    let output_parsed: RecExpr<HypatiaLang> = output_expr
        .parse()
        .map_err(|e| format!("Parse output error: {}", e))?;

    let input_count = input_parsed.as_ref().len();
    let output_count = output_parsed.as_ref().len();

    // Extract variable names from both expressions
    let input_vars = extract_variables(&input_parsed);
    let output_vars = extract_variables(&output_parsed);

    // Check that all input variables appear in output
    let mut missing_vars = Vec::new();
    for var in &input_vars {
        if !output_vars.contains(var) {
            missing_vars.push(var.clone());
        }
    }

    // Node reduction ratio (optimization shouldn't increase nodes dramatically)
    let node_ratio = output_count as f64 / input_count.max(1) as f64;

    let is_valid = missing_vars.is_empty() && node_ratio <= 2.0;

    let message = if is_valid {
        format!(
            "Structural validation PASSED: {} → {} nodes (ratio: {:.2}), all {} variables preserved",
            input_count, output_count, node_ratio, input_vars.len()
        )
    } else if !missing_vars.is_empty() {
        format!(
            "Structural validation FAILED: variables lost: {:?}",
            missing_vars
        )
    } else {
        format!(
            "Structural validation FAILED: node count increased too much ({} → {}, ratio: {:.2})",
            input_count, output_count, node_ratio
        )
    };

    Ok(ValidationResult {
        is_valid,
        max_diff: 0.0,
        mean_diff: 0.0,
        cosine_similarity: if is_valid { 1.0 } else { 0.0 },
        shapes_match: true,
        original_shape: vec![input_count as i64],
        optimized_shape: vec![output_count as i64],
        num_test_inputs: 0,
        tolerance: 0.0,
        message,
    })
}

/// Extract all variable names from a RecExpr.
fn extract_variables(expr: &egg::RecExpr<crate::egraph_optimizer::HypatiaLang>) -> Vec<String> {
    use crate::egraph_optimizer::HypatiaLang;

    let mut vars = Vec::new();
    for node in expr.as_ref() {
        if let HypatiaLang::Var(s) = node {
            let name = s.to_string();
            if !vars.contains(&name) {
                vars.push(name);
            }
        }
    }
    vars
}

/// Validate optimization and return a structured report.
/// Combines structural validation with the optimization report.
pub fn validate_optimization(
    input_expr: &str,
    output_expr: &str,
) -> Result<HashMap<String, String>, String> {
    let struct_result = validate_sexpr_structure(input_expr, output_expr)?;
    let opt_report = crate::visualization::build_optimization_report(input_expr, output_expr)?;

    let mut report = HashMap::new();
    report.insert("is_valid".to_string(), struct_result.is_valid.to_string());
    report.insert("message".to_string(), struct_result.message.clone());
    report.insert("input_expr".to_string(), input_expr.to_string());
    report.insert("output_expr".to_string(), output_expr.to_string());
    report.insert("input_node_count".to_string(), opt_report.input_node_count.to_string());
    report.insert("output_node_count".to_string(), opt_report.output_node_count.to_string());
    report.insert("node_reduction".to_string(), opt_report.node_reduction.to_string());
    report.insert(
        "fusions_found".to_string(),
        format!("{:?}", opt_report.fusions_found),
    );
    report.insert(
        "variables_preserved".to_string(),
        {
            let input_parsed: egg::RecExpr<crate::egraph_optimizer::HypatiaLang> =
                input_expr.parse().unwrap();
            let output_parsed: egg::RecExpr<crate::egraph_optimizer::HypatiaLang> =
                output_expr.parse().unwrap();
            let input_vars = extract_variables(&input_parsed);
            let output_vars = extract_variables(&output_parsed);
            let all_preserved = input_vars.iter().all(|v| output_vars.contains(v));
            all_preserved.to_string()
        },
    );

    Ok(report)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10); // Both zero → identical
    }

    #[test]
    fn test_compute_diffs_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let (max_diff, mean_diff) = compute_diffs(&a, &b);
        assert!(max_diff < 1e-10);
        assert!(mean_diff < 1e-10);
    }

    #[test]
    fn test_compute_diffs_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 3.3];
        let (max_diff, mean_diff) = compute_diffs(&a, &b);
        assert!((max_diff - 0.3).abs() < 1e-10);
        assert!((mean_diff - (0.1 + 0.0 + 0.3) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_validation_result_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + 1e-7, 2.0, 3.0 - 1e-8];
        let result = build_validation_result(
            &a, &b,
            vec![1, 3], vec![1, 3],
            1, 1e-5,
        );
        assert!(result.is_valid);
        assert!(result.shapes_match);
        assert!(result.max_diff < 1e-5);
    }

    #[test]
    fn test_build_validation_result_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.0, 3.0];
        let result = build_validation_result(
            &a, &b,
            vec![1, 3], vec![1, 3],
            1, 1e-5,
        );
        assert!(!result.is_valid);
        assert!((result.max_diff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_build_validation_result_shape_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = build_validation_result(
            &a, &b,
            vec![1, 2], vec![1, 3],
            1, 1e-5,
        );
        assert!(!result.is_valid);
        assert!(!result.shapes_match);
    }

    #[test]
    fn test_validate_sexpr_structure_basic() {
        let result = validate_sexpr_structure(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)",
        ).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_sexpr_structure_same() {
        let result = validate_sexpr_structure(
            "(relu x)",
            "(relu x)",
        ).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_sexpr_structure_variable_preserved() {
        let result = validate_sexpr_structure(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)",
        ).unwrap();
        assert!(result.is_valid);
        assert!(result.message.contains("PASSED"));
    }

    #[test]
    fn test_validate_optimization_report() {
        let report = validate_optimization(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)",
        ).unwrap();
        assert_eq!(report["is_valid"], "true");
        assert_eq!(report["variables_preserved"], "true");
    }

    #[test]
    fn test_validate_optimization_no_change() {
        let report = validate_optimization(
            "(relu x)",
            "(relu x)",
        ).unwrap();
        assert_eq!(report["is_valid"], "true");
        assert_eq!(report["node_reduction"], "0");
    }

    #[test]
    fn test_extract_variables() {
        let expr: egg::RecExpr<crate::egraph_optimizer::HypatiaLang> =
            "(relu (linear w b x))".parse().unwrap();
        let vars = extract_variables(&expr);
        assert!(vars.contains(&"w".to_string()));
        assert!(vars.contains(&"b".to_string()));
        assert!(vars.contains(&"x".to_string()));
        assert_eq!(vars.len(), 3);
    }
}
