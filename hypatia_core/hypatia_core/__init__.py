"""
Hypatia - Hardware-Aware Symbolic Compiler for PyTorch
"""

# Import from Rust binary module (underscore prefix to avoid circular import)
_RUST_AVAILABLE = False
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
    _RUST_AVAILABLE = True
except ImportError as e:
    import sys
    print(f"[Hypatia] WARNING: Rust binary not available: {e}", file=sys.stderr)
    print(f"   Run: maturin develop --release", file=sys.stderr)
    print(f"   CUDA fused modules and GPU features still work without Rust core.", file=sys.stderr)
    # Provide stubs so other imports don't break
    compile_fx_graph = None
    optimize_ast = None
    parse_expr = None
    is_equivalent = None
    Symbol = None
    PyMultiVector2D = None
    PyMultiVector3D = None
    HypatiaError = Exception
    HypatiaCompileResult = None
    set_log_level = lambda *a, **kw: None

import torch
import warnings
import os

# Import fused modules
from .fused_modules import HypatiaFusedLinearReLU, create_fused_linear_relu_from_tensors

# Import direct model optimizer
from .optimizer import optimize, count_optimizations, HypatiaTrainer

# Import Rust-native forward pass (bypasses PyTorch dispatch)
from .native_model import NativeModel, NativeTrainer, QuantizedModel, QuantizedTrainer, TransformerModel, GpuTransformerModel

# Import Neuromorphic computing (ANN→SNN, LIF neurons)
from .neuromorphic_model import NeuromorphicModel, compile_neuromorphic

# Import GPU-aware fused modules
from .fused_modules import FusedGeluMLP, FusedAttention, FusedLayerNorm, FusedTransformerBlock

# Import Sparse Tensor IR
from .sparse import SparseLinear, sparsify_model, model_sparsity_report

# Import Mixed Precision
from .mixed_precision import MixedPrecisionLinear, convert_to_mixed_precision, model_precision_report

# Import Visualization tools
from .visualization import (
    visualize_expr, visualize_optimization, print_expr_tree,
    compare_optimizations, generate_html_report, model_summary,
)

# Import Semantic Validation
from .semantic_validation import (
    validate_expr, validate_structure, validate_models, SemanticValidator,
)

# Import Profiler (FLOPs estimation, hardware detection, benchmarking)
from .profiler import (
    estimate_flops, detect_hardware, profile_model, compare_flops,
    benchmark_inference, roofline_analysis,
    ModelProfile, LayerProfile, HardwareInfo, FlopsComparison,
)

# Import Auto-tuner
from .autotuner import auto_tune, quick_tune, benchmark_tune, TuneConfig

# Import Benchmark Dashboard
from .dashboard import BenchmarkDashboard, generate_benchmark_dashboard

# Feature flag for verbose FX debugging
DEBUG_FX = os.environ.get("HYPATIA_DEBUG_FX", "0") == "1"

# E-graph + torch.compile chain control
# Set HYPATIA_CHAIN_COMPILE=0 to disable torch.compile after e-graph
CHAIN_COMPILE = os.environ.get("HYPATIA_CHAIN_COMPILE", "1") != "0"

def hypatia_backend(gm, example_inputs):
    """Hypatia compiler backend for torch.compile()

    Pipeline:
      1. E-graph structural optimization (fusion rewrites via Rust)
      2. torch.compile kernel optimization (Triton kernel fusion, if GPU)

    This chains symbolic optimization (e-graph) with kernel-level optimization
    (torch.compile/Triton), getting benefits from both levels.

    Args:
        gm: torch.fx.GraphModule - The compiled graph module
        example_inputs: List of example tensors

    Returns:
        Optimized GraphModule (possibly torch.compile'd for GPU kernel fusion)
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
        # Phase 1: E-graph structural optimization (Rust)
        result = compile_fx_graph(gm, example_inputs, module_info_map)

        if result.structure_changed:
            print("[Hypatia] Phase 1: E-graph structural optimization applied (rewrites found)")
            optimized_gm = result.optimized_gm

            # Quick correctness check: verify E-graph output matches original
            # E-graph rewrites can produce incorrect results for complex
            # multi-layer architectures (e.g., pre-norm Transformers with
            # LayerNorm dependencies across layers).
            try:
                with torch.no_grad():
                    y_orig = gm(*example_inputs)
                    y_opt = optimized_gm(*example_inputs)

                # Flatten and compute cosine similarity
                y_orig_flat = y_orig.flatten().float() if isinstance(y_orig, torch.Tensor) else y_orig[0].flatten().float()
                y_opt_flat = y_opt.flatten().float() if isinstance(y_opt, torch.Tensor) else y_opt[0].flatten().float()

                cos_sim = torch.nn.functional.cosine_similarity(
                    y_orig_flat, y_opt_flat, dim=0
                ).item()

                if cos_sim < 0.99:
                    print(f"[Hypatia] WARNING: E-graph output diverged (cosine_sim={cos_sim:.4f})")
                    print(f"[Hypatia] Falling back to original graph for correctness")
                    optimized_gm = gm
                elif DEBUG_FX:
                    print(f"[Hypatia] Phase 1 correctness check passed (cosine_sim={cos_sim:.6f})")
            except Exception as e:
                if DEBUG_FX:
                    print(f"[Hypatia] Phase 1 correctness check skipped: {e}")
        else:
            # IMPORTANT: When no rewrites are found, use the ORIGINAL graph.
            # The Rust FX bridge round-trip (FX → S-expr → E-graph → S-expr → FX)
            # can introduce subtle reconstruction errors (parameter ordering,
            # node naming) that cause numerical divergence in deep models.
            print("[Hypatia] Phase 1: E-graph analysis complete (graph preserved)")
            optimized_gm = gm

        # Phase 2: torch.compile kernel optimization (Triton, GPU only)
        # This wraps the e-graph-optimized graph with torch.compile for
        # additional kernel-level fusion (epilogue fusion, memory coalescing)
        if CHAIN_COMPILE and _is_gpu_model(optimized_gm, example_inputs):
            try:
                compiled_gm = torch.compile(
                    optimized_gm,
                    backend="inductor",
                    mode="max-autotune",
                    dynamic=False,
                )
                # Warm up the compiled model to trigger Triton compilation
                with torch.no_grad():
                    compiled_gm(*example_inputs)
                print("[Hypatia] Phase 2: torch.compile kernel fusion applied (Triton)")
                return compiled_gm
            except Exception as e:
                if DEBUG_FX:
                    print(f"[Hypatia] Phase 2 skipped (torch.compile error: {e})")

        return optimized_gm

    except Exception as e:
        node_count = len(list(gm.graph.nodes))
        device_info = "CPU"
        for inp in example_inputs:
            if hasattr(inp, 'device'):
                device_info = str(inp.device)
                break

        error_msg = (
            f"Hypatia compilation failed on {node_count}-node graph ({device_info}): {e}\n"
            f"  Falling back to original (unoptimized) model.\n"
            f"  Troubleshooting:\n"
            f"    - Set HYPATIA_DEBUG_FX=1 for detailed node-level tracing\n"
            f"    - Check model compatibility: torch.fx.symbolic_trace(model)\n"
            f"    - Verify Rust core: python -c \"from _hypatia_core import compile_fx_graph\""
        )
        warnings.warn(error_msg, category=RuntimeWarning)
        if DEBUG_FX:
            import traceback
            traceback.print_exc()
        return gm


def _is_gpu_model(gm, example_inputs):
    """Check if model/inputs are on GPU (worth applying torch.compile)."""
    # Check example inputs
    for inp in example_inputs:
        if hasattr(inp, 'is_cuda') and inp.is_cuda:
            return True
    # Check model parameters
    try:
        for p in gm.parameters():
            if p.is_cuda:
                return True
    except Exception:
        pass
    return False

# Global flag to track backend registration
_HYPATIA_BACKEND_REGISTERED = False

def register_backend():
    """Register Hypatia backend with PyTorch

    This function is idempotent - calling it multiple times is safe.
    It will only register the backend once, even across multiple imports.
    """
    global _HYPATIA_BACKEND_REGISTERED

    # Quick return if already registered
    if _HYPATIA_BACKEND_REGISTERED:
        return

    try:
        # Double-check with torch._dynamo
        backends = torch._dynamo.list_backends()
        if "hypatia" in backends:
            _HYPATIA_BACKEND_REGISTERED = True
            return

        torch._dynamo.register_backend(name="hypatia", compiler_fn=hypatia_backend)
        _HYPATIA_BACKEND_REGISTERED = True

        print("[Hypatia] Backend registered successfully")
        print("   Usage: torch.compile(model, backend='hypatia')")
        print(f"   Backend confirmed in: {torch._dynamo.list_backends()}")

    except Exception as e:
        warnings.warn(f"Failed to register Hypatia backend: {e}", category=RuntimeWarning)
        raise

# Auto-register on import (only if Rust core is available)
if _RUST_AVAILABLE:
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
    # Fused modules
    "HypatiaFusedLinearReLU",
    "create_fused_linear_relu_from_tensors",
    # Direct model optimizer
    "optimize",
    "count_optimizations",
    # Rust-native forward pass
    "NativeModel",
    "NativeTrainer",
    "QuantizedModel",
    "QuantizedTrainer",
    "TransformerModel",
    "GpuTransformerModel",
    "native_forward",
    "native_train_step",
    # Neuromorphic computing
    "NeuromorphicModel",
    "compile_neuromorphic",
    # GPU-aware fused modules
    "FusedGeluMLP",
    "FusedAttention",
    "FusedLayerNorm",
    "FusedTransformerBlock",
    # Sparse Tensor IR
    "SparseLinear",
    "sparsify_model",
    "model_sparsity_report",
    # Mixed Precision
    "MixedPrecisionLinear",
    "convert_to_mixed_precision",
    "model_precision_report",
    # Visualization
    "visualize_expr",
    "visualize_optimization",
    "print_expr_tree",
    "compare_optimizations",
    "generate_html_report",
    "model_summary",
    # Semantic Validation
    "validate_expr",
    "validate_structure",
    "validate_models",
    "SemanticValidator",
    # Profiler
    "estimate_flops",
    "detect_hardware",
    "profile_model",
    "compare_flops",
    "benchmark_inference",
    "roofline_analysis",
    "ModelProfile",
    "LayerProfile",
    "HardwareInfo",
    "FlopsComparison",
    # Auto-tuner
    "auto_tune",
    "quick_tune",
    "benchmark_tune",
    "TuneConfig",
    # Benchmark Dashboard
    "BenchmarkDashboard",
    "generate_benchmark_dashboard",
]