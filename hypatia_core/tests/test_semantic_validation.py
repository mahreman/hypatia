"""
Semantic Validation Tests for Hypatia

Tests:
  1. Rust validate_sexpr (structural validation)
  2. Rust validate_optimization_py (combined report)
  3. Rust validate_model_outputs (model comparison)
  4. Python validate_expr
  5. Python validate_structure
  6. Python validate_models
  7. Python SemanticValidator
  8. Integration with compile pipeline
  9. Edge cases
"""

import pytest
import torch
import torch.nn as nn


# ============================================================================
# Test 1: Rust S-expression Structural Validation
# ============================================================================

class TestRustValidateSexpr:
    def test_basic_valid(self):
        """Fusion should preserve all variables."""
        from _hypatia_core import validate_sexpr

        result = validate_sexpr(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert result["is_valid"] is True
        assert result["shapes_match"] is True
        assert "PASSED" in result["message"]

    def test_same_expression(self):
        """Identical input/output should be valid."""
        from _hypatia_core import validate_sexpr

        result = validate_sexpr("(relu x)", "(relu x)")
        assert result["is_valid"] is True

    def test_single_variable(self):
        """Single variable should be valid."""
        from _hypatia_core import validate_sexpr

        result = validate_sexpr("x", "x")
        assert result["is_valid"] is True

    def test_attention_fusion(self):
        """Attention fusion should preserve variables."""
        from _hypatia_core import validate_sexpr

        result = validate_sexpr(
            "(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))",
            "(fused_attention wq bq wk bk wv bv wo bo x x)"
        )
        assert result["is_valid"] is True

    def test_parse_error(self):
        """Invalid expression should raise error."""
        from _hypatia_core import validate_sexpr

        with pytest.raises(Exception):
            validate_sexpr("(((invalid", "(relu x)")


# ============================================================================
# Test 2: Rust Optimization Validation Report
# ============================================================================

class TestRustValidateOptimization:
    def test_basic_report(self):
        """Should produce a combined validation + optimization report."""
        from _hypatia_core import validate_optimization_py

        report = validate_optimization_py(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert report["is_valid"] == "true"
        assert report["variables_preserved"] == "true"
        assert int(report["node_reduction"]) > 0

    def test_no_change(self):
        """Same expression should have zero node reduction."""
        from _hypatia_core import validate_optimization_py

        report = validate_optimization_py("(relu x)", "(relu x)")
        assert report["is_valid"] == "true"
        assert report["node_reduction"] == "0"

    def test_contains_fusions(self):
        """Report should list fusions when present."""
        from _hypatia_core import validate_optimization_py

        report = validate_optimization_py(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert "FusedLinearReLU" in report["fusions_found"]


# ============================================================================
# Test 3: Rust Model Output Validation
# ============================================================================

class TestRustValidateModelOutputs:
    def test_identical_models(self):
        """Same model should always pass."""
        from _hypatia_core import validate_model_outputs

        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        result = validate_model_outputs(model, model, [1, 16], 1e-5, 3)
        assert result["is_valid"] is True
        assert result["max_diff"] < 1e-10
        assert result["cosine_similarity"] > 0.99

    def test_different_models(self):
        """Different models should fail validation."""
        from _hypatia_core import validate_model_outputs

        model1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        model2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        # Different random weights → different outputs
        result = validate_model_outputs(model1, model2, [1, 16], 1e-5, 3)
        # Should almost certainly fail with different weights
        assert result["shapes_match"] is True
        # max_diff should be non-trivial
        assert result["max_diff"] > 1e-6

    def test_shape_mismatch(self):
        """Models with different output shapes should fail."""
        from _hypatia_core import validate_model_outputs

        model1 = nn.Sequential(nn.Linear(16, 8))
        model2 = nn.Sequential(nn.Linear(16, 4))
        result = validate_model_outputs(model1, model2, [1, 16], 1e-5, 1)
        assert result["is_valid"] is False
        assert result["shapes_match"] is False

    def test_cloned_model(self):
        """Cloned model should produce identical outputs."""
        from _hypatia_core import validate_model_outputs
        import copy

        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        clone = copy.deepcopy(model)
        result = validate_model_outputs(model, clone, [1, 32], 1e-5, 5)
        assert result["is_valid"] is True
        assert result["max_diff"] < 1e-10

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        from _hypatia_core import validate_model_outputs

        model = nn.Sequential(nn.Linear(16, 8))
        result = validate_model_outputs(model, model, [1, 16], 1e-10, 1)
        assert result["tolerance"] == pytest.approx(1e-10)
        assert result["is_valid"] is True


# ============================================================================
# Test 4: Python validate_expr
# ============================================================================

class TestPythonValidateExpr:
    def test_auto_optimize(self):
        """Should auto-optimize when output_expr is None."""
        from hypatia_core import validate_expr

        result = validate_expr("(relu (linear w b x))")
        assert result["is_valid"] == "true"
        assert result["variables_preserved"] == "true"

    def test_explicit_output(self):
        """Should accept explicit output expression."""
        from hypatia_core import validate_expr

        result = validate_expr(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert result["is_valid"] == "true"

    def test_no_change_expr(self):
        """Expression without applicable rules should still be valid."""
        from hypatia_core import validate_expr

        result = validate_expr("(add x y)")
        assert result["is_valid"] == "true"


# ============================================================================
# Test 5: Python validate_structure
# ============================================================================

class TestPythonValidateStructure:
    def test_basic_structure(self):
        """Basic structural validation."""
        from hypatia_core import validate_structure

        result = validate_structure(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert result["is_valid"] is True
        assert result["shapes_match"] is True

    def test_preserved_variables(self):
        """All variables should be preserved."""
        from hypatia_core import validate_structure

        result = validate_structure(
            "(add alpha beta)",
            "(add alpha beta)"
        )
        assert result["is_valid"] is True


# ============================================================================
# Test 6: Python validate_models
# ============================================================================

class TestPythonValidateModels:
    def test_identical_models(self):
        """Same model should pass validation."""
        from hypatia_core import validate_models

        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        result = validate_models(model, model, (1, 32))
        assert result["is_valid"] is True
        assert result["cosine_similarity"] > 0.99

    def test_with_custom_params(self):
        """Custom tolerance and samples should work."""
        from hypatia_core import validate_models
        import copy

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
        clone = copy.deepcopy(model)
        result = validate_models(clone, model, (2, 64), tolerance=1e-4, num_samples=5)
        assert result["is_valid"] is True
        assert result["num_test_inputs"] == 5


# ============================================================================
# Test 7: SemanticValidator Class
# ============================================================================

class TestSemanticValidator:
    def test_init(self):
        """Validator should initialize with defaults."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator()
        assert v.tolerance == 1e-5
        assert v.num_samples == 5
        assert v.strict is False

    def test_custom_init(self):
        """Validator should accept custom parameters."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator(tolerance=1e-3, num_samples=10, strict=True)
        assert v.tolerance == 1e-3
        assert v.num_samples == 10
        assert v.strict is True

    def test_validate_expr(self):
        """Validator should validate expressions."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator()
        result = v.validate_expr("(relu (linear w b x))")
        assert result["is_valid"] == "true"

    def test_validate_models(self):
        """Validator should validate model equivalence."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator(tolerance=1e-4, num_samples=3)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        result = v.validate_models(model, model, (1, 16))
        assert result["is_valid"] is True

    def test_full_validation(self):
        """Full validation with both expr and model."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator(tolerance=1e-4)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())

        result = v.full_validation(
            "(relu (linear w b x))",
            original=model,
            optimized=model,
            input_shape=(1, 16),
        )
        assert result["is_valid"] is True
        assert result["expr_valid"] is True
        assert result["model_valid"] is True
        assert result["cosine_similarity"] > 0.99

    def test_full_validation_expr_only(self):
        """Full validation with expr only (no models)."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator()
        result = v.full_validation("(relu (linear w b x))")
        assert result["is_valid"] is True
        assert result["expr_valid"] is True
        assert "model_valid" not in result

    def test_report(self):
        """Report should produce formatted text."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator()
        report = v.report("(relu (linear w b x))")
        assert "HYPATIA SEMANTIC VALIDATION REPORT" in report
        assert "PASSED" in report or "FAILED" in report

    def test_report_with_models(self):
        """Report with model validation should include metrics."""
        from hypatia_core import SemanticValidator

        v = SemanticValidator(tolerance=1e-4)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        report = v.report(
            "(relu (linear w b x))",
            original=model,
            optimized=model,
            input_shape=(1, 16),
        )
        assert "Max diff" in report
        assert "Cosine similarity" in report


# ============================================================================
# Test 8: Integration Tests
# ============================================================================

class TestIntegration:
    def test_optimization_pipeline(self):
        """Full pipeline: optimize → validate."""
        from _hypatia_core import optimize_ast
        from hypatia_core import validate_expr

        expr = "(relu (linear w b x))"
        optimized = optimize_ast(expr)
        result = validate_expr(expr, optimized)
        assert result["is_valid"] == "true"

    def test_complex_expression(self):
        """Complex expression should validate."""
        from _hypatia_core import optimize_ast
        from hypatia_core import validate_expr

        expr = "(relu (linear w1 b1 (relu (linear w2 b2 x))))"
        optimized = optimize_ast(expr)
        result = validate_expr(expr, optimized)
        assert result["is_valid"] == "true"
        assert result["variables_preserved"] == "true"


# ============================================================================
# Test 9: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_constant_expression(self):
        """Expression with constants should validate."""
        from hypatia_core import validate_structure

        result = validate_structure("(add x 3.14)", "(add x 3.14)")
        assert result["is_valid"] is True

    def test_large_model(self):
        """Larger model should work."""
        from hypatia_core import validate_models

        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        result = validate_models(model, model, (4, 256), tolerance=1e-5, num_samples=2)
        assert result["is_valid"] is True

    def test_single_element_output(self):
        """Model with single output element should work."""
        from hypatia_core import validate_models

        model = nn.Sequential(nn.Linear(10, 1))
        result = validate_models(model, model, (1, 10))
        assert result["is_valid"] is True

    def test_batch_input(self):
        """Batch inputs should work correctly."""
        from hypatia_core import validate_models

        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        result = validate_models(model, model, (8, 8), tolerance=1e-5, num_samples=3)
        assert result["is_valid"] is True
        assert result["shapes_match"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
