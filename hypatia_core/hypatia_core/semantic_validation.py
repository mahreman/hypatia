# hypatia_core/semantic_validation.py
#
# Semantic Validation tools for Hypatia compiler optimization pipeline.
# - S-expression structural validation (variable preservation, node ratio)
# - Model output equivalence testing (random inputs, tolerance comparison)
# - Optimization validation reports
# - Integration with ChecksumMode (Strict/Soft/Off)

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from _hypatia_core import (
    validate_sexpr,
    validate_optimization_py,
    validate_model_outputs,
    optimize_ast,
)


# ============================================================================
# S-expression Validation
# ============================================================================


def validate_expr(
    input_expr: str,
    output_expr: Optional[str] = None,
) -> Dict:
    """Validate that an optimization preserves semantic correctness.

    If output_expr is not provided, the optimizer is run on input_expr
    to generate the optimized version automatically.

    Args:
        input_expr: Original S-expression string
        output_expr: Optimized S-expression (None = auto-optimize)

    Returns:
        dict with validation results:
            - is_valid: bool
            - message: str
            - input_node_count, output_node_count: int
            - node_reduction: int
            - fusions_found: str representation of fusions
            - variables_preserved: bool
    """
    if output_expr is None:
        output_expr = optimize_ast(input_expr)

    return dict(validate_optimization_py(input_expr, output_expr))


def validate_structure(
    input_expr: str,
    output_expr: str,
) -> Dict:
    """Validate structural properties of an optimization.

    Checks:
    - All variables from input are preserved in output
    - Node count ratio is reasonable (< 2x)

    Args:
        input_expr: Original S-expression
        output_expr: Optimized S-expression

    Returns:
        dict with: is_valid, message, cosine_similarity, shapes_match, etc.
    """
    return dict(validate_sexpr(input_expr, output_expr))


# ============================================================================
# Model Output Validation
# ============================================================================


def validate_models(
    original: nn.Module,
    optimized: nn.Module,
    input_shape: Tuple[int, ...],
    tolerance: float = 1e-5,
    num_samples: int = 3,
) -> Dict:
    """Validate that two PyTorch models produce equivalent outputs.

    Runs random inputs through both models and compares outputs
    using max absolute difference and cosine similarity.

    Args:
        original: Original PyTorch model
        optimized: Optimized PyTorch model
        input_shape: Shape for random test inputs, e.g. (1, 64)
        tolerance: Maximum allowed absolute difference (default: 1e-5)
        num_samples: Number of random inputs to test (default: 3)

    Returns:
        dict with:
            - is_valid: bool (max_diff <= tolerance)
            - max_diff: float
            - mean_diff: float
            - cosine_similarity: float (1.0 = identical)
            - shapes_match: bool
            - message: str
    """
    return dict(validate_model_outputs(
        original, optimized,
        list(input_shape),
        tolerance,
        num_samples,
    ))


class SemanticValidator:
    """High-level semantic validation for the Hypatia optimization pipeline.

    Provides both S-expression level and model-level validation,
    with configurable tolerance and reporting.

    Example::

        validator = SemanticValidator(tolerance=1e-4)

        # Validate an S-expression optimization
        result = validator.validate_expr("(relu (linear w b x))")
        print(result["message"])

        # Validate two PyTorch models
        original = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        optimized = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        result = validator.validate_models(original, optimized, input_shape=(1, 64))
        assert result["is_valid"]
    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        num_samples: int = 5,
        strict: bool = False,
    ):
        """Initialize the semantic validator.

        Args:
            tolerance: Maximum absolute difference for output comparison
            num_samples: Number of random test inputs
            strict: If True, also validate structural properties
        """
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.strict = strict

    def validate_expr(
        self,
        input_expr: str,
        output_expr: Optional[str] = None,
    ) -> Dict:
        """Validate an S-expression optimization.

        Args:
            input_expr: Original expression
            output_expr: Optimized expression (None = auto-optimize)

        Returns:
            Validation result dict
        """
        return validate_expr(input_expr, output_expr)

    def validate_models(
        self,
        original: nn.Module,
        optimized: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Dict:
        """Validate model output equivalence.

        Args:
            original: Original model
            optimized: Optimized model
            input_shape: Input tensor shape

        Returns:
            Validation result dict
        """
        return validate_models(
            original, optimized,
            input_shape,
            tolerance=self.tolerance,
            num_samples=self.num_samples,
        )

    def full_validation(
        self,
        input_expr: str,
        original: Optional[nn.Module] = None,
        optimized: Optional[nn.Module] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict:
        """Run comprehensive validation: structural + model output.

        Args:
            input_expr: Original S-expression
            original: Original PyTorch model (optional)
            optimized: Optimized PyTorch model (optional)
            input_shape: Input tensor shape (required if models provided)

        Returns:
            Combined validation report
        """
        # Step 1: S-expression structural validation
        expr_result = self.validate_expr(input_expr)

        result = {
            "expr_valid": expr_result.get("is_valid", "true") == "true",
            "expr_message": expr_result.get("message", ""),
            "node_reduction": expr_result.get("node_reduction", "0"),
            "fusions_found": expr_result.get("fusions_found", "[]"),
            "variables_preserved": expr_result.get("variables_preserved", "true") == "true",
        }

        # Step 2: Model output validation (if models provided)
        if original is not None and optimized is not None and input_shape is not None:
            model_result = self.validate_models(original, optimized, input_shape)
            result["model_valid"] = model_result["is_valid"]
            result["max_diff"] = model_result["max_diff"]
            result["mean_diff"] = model_result["mean_diff"]
            result["cosine_similarity"] = model_result["cosine_similarity"]
            result["model_message"] = model_result["message"]
            result["is_valid"] = result["expr_valid"] and model_result["is_valid"]
        else:
            result["is_valid"] = result["expr_valid"]

        return result

    def report(
        self,
        input_expr: str,
        original: Optional[nn.Module] = None,
        optimized: Optional[nn.Module] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """Generate a human-readable validation report.

        Args:
            input_expr: Original S-expression
            original: Original PyTorch model (optional)
            optimized: Optimized PyTorch model (optional)
            input_shape: Input tensor shape (optional)

        Returns:
            Formatted report string
        """
        result = self.full_validation(input_expr, original, optimized, input_shape)

        lines = []
        lines.append("=" * 60)
        lines.append("  HYPATIA SEMANTIC VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Overall status
        status = "PASSED" if result["is_valid"] else "FAILED"
        lines.append(f"  Status: {status}")
        lines.append(f"  Tolerance: {self.tolerance}")
        lines.append(f"  Test samples: {self.num_samples}")
        lines.append("")

        # Expression validation
        lines.append("  Expression Validation:")
        lines.append(f"    Valid: {result['expr_valid']}")
        lines.append(f"    Message: {result['expr_message']}")
        lines.append(f"    Variables preserved: {result['variables_preserved']}")
        lines.append(f"    Node reduction: {result['node_reduction']}")
        lines.append(f"    Fusions: {result['fusions_found']}")
        lines.append("")

        # Model validation (if available)
        if "model_valid" in result:
            lines.append("  Model Output Validation:")
            lines.append(f"    Valid: {result['model_valid']}")
            lines.append(f"    Max diff: {result['max_diff']:.2e}")
            lines.append(f"    Mean diff: {result['mean_diff']:.2e}")
            lines.append(f"    Cosine similarity: {result['cosine_similarity']:.6f}")
            lines.append(f"    Message: {result['model_message']}")
            lines.append("")

        lines.append("=" * 60)

        report_str = "\n".join(lines)
        print(report_str)
        return report_str


__all__ = [
    "validate_expr",
    "validate_structure",
    "validate_models",
    "SemanticValidator",
]
