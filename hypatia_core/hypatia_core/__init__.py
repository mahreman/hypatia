"""
Hypatia - Hardware-Aware Symbolic Compiler for PyTorch
"""

# Import from Rust binary module (underscore prefix to avoid circular import)
try:
    from _hypatia_core import (
        compile_fx_graph,
        optimize_ast,
        parse_expr,
        is_equivalent,
        Symbol,
        PyMultiVector2D,
        PyMultiVector3D,
        HypatiaError,
        HypatiaCompileResult,
        set_log_level,
    )
except ImportError as e:
    import sys
    print(f"❌ Failed to import Rust binary: {e}", file=sys.stderr)
    print(f"   Make sure you ran: maturin develop --release", file=sys.stderr)
    raise

import torch
import warnings
import os

# Feature flag for verbose FX debugging
DEBUG_FX = os.environ.get("HYPATIA_DEBUG_FX", "0") == "1"

def hypatia_backend(gm, example_inputs):
    """Hypatia compiler backend for torch.compile()

    Args:
        gm: torch.fx.GraphModule - The compiled graph module
        example_inputs: List of example tensors

    Returns:
        Optimized GraphModule
    """
    print(f"[Hypatia] Compiling graph with {len(list(gm.graph.nodes))} nodes")

    # DEBUG: Log example_inputs order (only if HYPATIA_DEBUG_FX=1)
    if DEBUG_FX:
        print("\n[DEBUG] example_inputs types / shapes:")
        for i, t in enumerate(example_inputs):
            try:
                print(f"  [{i}] shape={tuple(t.shape)}, device={t.device}, requires_grad={t.requires_grad}")
            except Exception:
                print(f"  [{i}] non-tensor: {type(t)}")

    # Build module_info_map from GraphModule's named_modules
    module_info_map = {}
    for name, module in gm.named_modules():
        is_inference = not getattr(module, "training", False)
        has_bias = hasattr(module, "bias") and module.bias is not None
        module_type = type(module).__name__

        module_info_map[name] = {
            "type": module_type,
            "has_bias": has_bias,
            "is_inference": is_inference,
        }

    try:
        # ✅ REFACTORED: Call Rust compilation with 3 arguments:
        # (gm, example_inputs, module_info_map)
        # Parametreler artık example_inputs üzerinden çözülüyor
        result = compile_fx_graph(gm, example_inputs, module_info_map)

        # result is HypatiaCompileResult with optimized_gm and structure_changed
        if result.structure_changed:
            print("[Hypatia] ✅ Optimization successful! (graph structure changed - rewrites applied)")
        else:
            print("[Hypatia] ✅ Compilation completed (graph structure preserved)")

        return result.optimized_gm

    except Exception as e:
        warnings.warn(
            f"Hypatia compilation failed: {e}. Falling back to original model.",
            category=RuntimeWarning
        )
        print(f"[Hypatia] ❌ Compilation error: {e}")
        return gm

def register_backend():
    """Register Hypatia backend with PyTorch"""
    try:
        # Check if already registered to avoid duplicate warnings
        backends = torch._dynamo.list_backends()
        if "hypatia" in backends:
            # Already registered, skip
            return

        torch._dynamo.register_backend(name="hypatia", compiler_fn=hypatia_backend)
        print("✅ Hypatia backend registered successfully")
        print("   Usage: torch.compile(model, backend='hypatia')")
        print(f"   ✓ Backend confirmed in: {backends}")

    except Exception as e:
        warnings.warn(f"Failed to register Hypatia backend: {e}")

# Auto-register on import
register_backend()

__version__ = "1.0.0"
__all__ = [
    "hypatia_backend",
    "compile_fx_graph",
    "optimize_ast",
    "parse_expr",
    "is_equivalent",
    "Symbol",
    "PyMultiVector2D",
    "PyMultiVector3D",
    "HypatiaError",
    "HypatiaCompileResult",
    "set_log_level",
    "register_backend",
]