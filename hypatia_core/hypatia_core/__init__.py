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
        set_log_level,
    )
except ImportError as e:
    import sys
    print(f"❌ Failed to import Rust binary: {e}", file=sys.stderr)
    print(f"   Make sure you ran: maturin develop --release", file=sys.stderr)
    raise

import torch
import warnings

def hypatia_backend(gm, example_inputs):
    """Hypatia compiler backend for torch.compile()"""
    module_info_map = {}
    print(f"[Hypatia] Compiling graph with {len(list(gm.graph.nodes))} nodes")
    
    try:
        optimized_gm = compile_fx_graph(gm, example_inputs, module_info_map)
        
        if optimized_gm is gm:
            print("[Hypatia] Optimization failed or skipped, using original model")
        else:
            print("[Hypatia] ✅ Optimization successful!")
        
        return optimized_gm
        
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
        torch._dynamo.register_backend(name="hypatia", compiler_fn=hypatia_backend)
        print("✅ Hypatia backend registered successfully")
        print("   Usage: torch.compile(model, backend='hypatia')")
        
        backends = torch._dynamo.list_backends()
        if "hypatia" in backends:
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
    "set_log_level",
    "register_backend",
]