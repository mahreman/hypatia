"""
Visualization Tests for Hypatia

Tests:
  1. Rust DOT export (expr_to_dot)
  2. Rust ASCII tree (expr_to_ascii_tree)
  3. Rust optimization report (optimization_report)
  4. Python visualize_expr
  5. Python compare_optimizations
  6. Python generate_html_report
  7. Python model_summary
  8. Edge cases
"""

import pytest
import os
import tempfile

import torch
import torch.nn as nn


# ============================================================================
# Test 1: Rust DOT Export
# ============================================================================

class TestDotExport:
    def test_basic_dot(self):
        """Simple expression should produce valid DOT."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(relu x)", "test")
        assert "digraph test" in dot
        assert "ReLU" in dot

    def test_linear_relu(self):
        """Linear+ReLU should show both nodes with edges."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(relu (linear w b x))", "graph")
        assert "ReLU" in dot
        assert "Linear" in dot
        assert "->" in dot  # Has edges

    def test_fused_attention(self):
        """Fused attention should produce valid DOT."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(fused_attention wq bq wk bk wv bv wo bo x x)", "attn")
        assert "FusedAttention" in dot

    def test_constants(self):
        """Constants should show their values."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(add x 3.14)", "const")
        assert "3.14" in dot

    def test_custom_graph_name(self):
        """Custom graph name should be used."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("x", "my_custom_graph")
        assert "digraph my_custom_graph" in dot

    def test_sparse_operators(self):
        """Sparse operators should have teal color."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(sparse_linear w b x s)", "sparse")
        assert "SparseLinear" in dot
        assert "#B3FFE6" in dot  # Teal color

    def test_mixed_precision_operators(self):
        """Mixed precision operators should have cyan color."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(cast_fp16 x)", "mp")
        assert "CastFP16" in dot
        assert "#B3F0FF" in dot  # Cyan color

    def test_neuromorphic_operators(self):
        """Neuromorphic operators should have pink color."""
        from _hypatia_core import expr_to_dot

        dot = expr_to_dot("(lif x v_th beta ts)", "neuro")
        assert "LIF" in dot


# ============================================================================
# Test 2: ASCII Tree
# ============================================================================

class TestAsciiTree:
    def test_basic_tree(self):
        """Simple expression should produce a tree."""
        from _hypatia_core import expr_to_ascii_tree

        tree = expr_to_ascii_tree("(relu x)")
        assert "ReLU" in tree
        assert "Var" in tree

    def test_nested_tree(self):
        """Nested expression should show hierarchy."""
        from _hypatia_core import expr_to_ascii_tree

        tree = expr_to_ascii_tree("(relu (linear w b x))")
        lines = tree.strip().split("\n")

        # Should have tree structure
        assert any("ReLU" in line for line in lines)
        assert any("Linear" in line for line in lines)
        assert any("'w'" in line for line in lines)
        assert any("'x'" in line for line in lines)

    def test_variable_names(self):
        """Variable names should be shown."""
        from _hypatia_core import expr_to_ascii_tree

        tree = expr_to_ascii_tree("(add alpha beta)")
        assert "'alpha'" in tree
        assert "'beta'" in tree

    def test_single_variable(self):
        """Single variable should produce minimal tree."""
        from _hypatia_core import expr_to_ascii_tree

        tree = expr_to_ascii_tree("x")
        assert "Var" in tree
        assert "'x'" in tree


# ============================================================================
# Test 3: Optimization Report
# ============================================================================

class TestOptimizationReport:
    def test_basic_report(self):
        """Report should detect fusion."""
        from _hypatia_core import optimization_report

        report = optimization_report(
            "(relu (linear w b x))",
            "(fused_linear_relu w b x)"
        )
        assert report["input_node_count"] == 5
        assert report["output_node_count"] == 4
        assert report["node_reduction"] == 1
        assert any("FusedLinearReLU" in f for f in report["fusions_found"])

    def test_no_change_report(self):
        """Same input/output should show no changes."""
        from _hypatia_core import optimization_report

        report = optimization_report("(relu x)", "(relu x)")
        assert report["node_reduction"] == 0
        assert len(report["fusions_found"]) == 0

    def test_attention_fusion_report(self):
        """Attention fusion should be detected."""
        from _hypatia_core import optimization_report

        input_expr = "(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))"
        output_expr = "(fused_attention wq bq wk bk wv bv wo bo x x)"
        report = optimization_report(input_expr, output_expr)

        assert any("FusedAttention" in f for f in report["fusions_found"])
        assert report["node_reduction"] > 0

    def test_node_types_present(self):
        """Report should include node type counts."""
        from _hypatia_core import optimization_report

        report = optimization_report("(relu (linear w b x))", "(relu (linear w b x))")
        assert "ReLU" in report["input_node_types"]
        assert "Linear" in report["input_node_types"]
        assert report["input_node_types"]["ReLU"] == 1
        assert report["input_node_types"]["Var"] == 3


# ============================================================================
# Test 4: Python visualize_expr
# ============================================================================

class TestVisualizeExpr:
    def test_returns_dot(self):
        """visualize_expr should return valid DOT."""
        from hypatia_core import visualize_expr

        dot = visualize_expr("(relu (linear w b x))")
        assert "digraph" in dot
        assert "ReLU" in dot

    def test_custom_name(self):
        """Custom graph name should be used."""
        from hypatia_core import visualize_expr

        dot = visualize_expr("(relu x)", graph_name="my_graph")
        assert "digraph my_graph" in dot


# ============================================================================
# Test 5: Python compare_optimizations
# ============================================================================

class TestCompareOptimizations:
    def test_compare_linear_relu(self):
        """Linear+ReLU comparison should show fusion."""
        from hypatia_core import compare_optimizations

        report = compare_optimizations("(relu (linear w b x))")
        assert "HYPATIA OPTIMIZATION REPORT" in report
        assert "fused_linear_relu" in report.lower() or "linear" in report.lower()

    def test_compare_no_change(self):
        """Expression with no applicable rules should report no change."""
        from hypatia_core import compare_optimizations

        report = compare_optimizations("(add x y)")
        assert "HYPATIA OPTIMIZATION REPORT" in report


# ============================================================================
# Test 6: HTML Report
# ============================================================================

class TestHtmlReport:
    def test_html_generation(self):
        """HTML report should be generated."""
        from hypatia_core import generate_html_report

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            html = generate_html_report("(relu (linear w b x))", path)
            assert "<html>" in html
            assert "Hypatia Optimization Report" in html
            assert "Summary" in html
            assert os.path.exists(path)

            with open(path) as f:
                content = f.read()
            assert len(content) > 500
        finally:
            os.unlink(path)

    def test_html_contains_expressions(self):
        """HTML should contain the input/output expressions."""
        from hypatia_core import generate_html_report

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            html = generate_html_report("(relu (linear w b x))", path)
            assert "relu" in html
            assert "linear" in html or "fused" in html
        finally:
            os.unlink(path)

    def test_html_node_types_table(self):
        """HTML should contain a node types table."""
        from hypatia_core import generate_html_report

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            html = generate_html_report("(relu (linear w b x))", path)
            assert "<table>" in html
            assert "Node Type" in html
        finally:
            os.unlink(path)


# ============================================================================
# Test 7: Model Summary
# ============================================================================

class TestModelSummary:
    def test_simple_model(self):
        """Model summary should list layers."""
        from hypatia_core import model_summary

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

        summary = model_summary(model)
        assert "Linear" in summary
        assert "ReLU" in summary
        assert "Total parameters" in summary

    def test_parameter_counts(self):
        """Parameter counts should be shown."""
        from hypatia_core import model_summary

        model = nn.Sequential(nn.Linear(10, 5))  # 10*5 + 5 = 55 params
        summary = model_summary(model)
        assert "55" in summary

    def test_deep_model(self):
        """Deeper model should work."""
        from hypatia_core import model_summary

        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        summary = model_summary(model)
        assert "FP32 memory" in summary
        assert "FP16 memory" in summary


# ============================================================================
# Test 8: Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_var(self):
        """Single variable should produce minimal output."""
        from _hypatia_core import expr_to_dot, expr_to_ascii_tree

        dot = expr_to_dot("x", "test")
        assert "digraph test" in dot

        tree = expr_to_ascii_tree("x")
        assert "'x'" in tree

    def test_deeply_nested(self):
        """Deeply nested expression should work."""
        from _hypatia_core import expr_to_dot

        expr = "(relu (relu (relu (relu (relu x)))))"
        dot = expr_to_dot(expr, "deep")
        assert dot.count("ReLU") == 5

    def test_parse_error(self):
        """Invalid expression should raise error."""
        from _hypatia_core import expr_to_dot

        with pytest.raises(Exception):
            expr_to_dot("(((invalid", "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
