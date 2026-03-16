# hypatia_core/fused_modules.py
#
# GPU-aware fused modules with JIT-compiled CUDA kernels + CPU fallback.
# Supports: FusedLinearReLU, FusedGeluMLP, FusedAttention, FusedLayerNorm

from __future__ import annotations

import os
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.cpp_extension import load as _load_ext
except ImportError:
    _load_ext = None

# -----------------------------------------------------------------------------
# C++/CUDA extension loading (lazy, with CPU fallback)
# -----------------------------------------------------------------------------

_CSRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")

# Extension cache: name -> (module_or_None, loaded_flag)
_EXT_CACHE: Dict[str, object] = {}
_EXT_LOADED: Dict[str, bool] = {}

# Legacy aliases
_FUSED_LINEAR_RELU_EXT = None
_HAS_CUDA_KERNEL = False


def _load_cuda_ext(name: str, cpp_file: str, cu_file: str) -> Optional[object]:
    """Load a CUDA extension by name. Returns module or None on failure."""
    if name in _EXT_CACHE:
        return _EXT_CACHE[name]

    if _load_ext is None:
        _EXT_CACHE[name] = None
        _EXT_LOADED[name] = False
        return None

    src_cpp = os.path.join(_CSRC_DIR, cpp_file)
    src_cu = os.path.join(_CSRC_DIR, cu_file)

    if not (os.path.exists(src_cpp) and os.path.exists(src_cu)):
        _EXT_CACHE[name] = None
        _EXT_LOADED[name] = False
        return None

    try:
        ext = _load_ext(
            name=f"hypatia_{name}",
            sources=[src_cpp, src_cu],
            verbose=False,
        )
        _EXT_CACHE[name] = ext
        _EXT_LOADED[name] = True
        print(f"[Hypatia] CUDA extension '{name}' loaded successfully")
        return ext
    except Exception as exc:
        print(f"[Hypatia] Warning: failed to build {name} CUDA extension: {exc}")
        _EXT_CACHE[name] = None
        _EXT_LOADED[name] = False
        return None


def _get_ext(name: str) -> Optional[object]:
    """Get a cached extension, or None if not available."""
    return _EXT_CACHE.get(name)


def _has_ext(name: str) -> bool:
    """Check if a CUDA extension is available."""
    return _EXT_LOADED.get(name, False)


def _load_fused_linear_relu_ext() -> None:
    """Legacy loader for backward compatibility."""
    global _FUSED_LINEAR_RELU_EXT, _HAS_CUDA_KERNEL
    ext = _load_cuda_ext("fused_linear_relu", "fused_linear_relu.cpp", "fused_linear_relu_kernel.cu")
    _FUSED_LINEAR_RELU_EXT = ext
    _HAS_CUDA_KERNEL = ext is not None


# -----------------------------------------------------------------------------
# Autograd.Function
# -----------------------------------------------------------------------------

class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor]) -> torch.Tensor:
        """
        If CUDA kernel is available:
          y = ReLU(input @ weight.T + bias) computed in a single kernel.
        Otherwise:
          Falls back to F.linear + F.relu.
        """
        use_cuda_kernel = (
            input.is_cuda
            and weight.is_cuda
            and (bias is None or bias.is_cuda)
            and _HAS_CUDA_KERNEL
        )

        if use_cuda_kernel:
            # C++/CUDA kernel call
            y = _FUSED_LINEAR_RELU_EXT.forward(input, weight, bias)
        else:
            # Graph-level fusion fallback (no kernel fusion)
            y = F.relu(F.linear(input, weight, bias))

        ctx.save_for_backward(input, weight, bias if bias is not None else None, y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward'ı PyTorch op'larıyla yapıyoruz.
        Bu, kernel-level fused backward değil ama:
          - correctness için yeterli
          - training'de güvenli
        """
        input, weight, bias, output = ctx.saved_tensors

        # grad_output shape: [N, out_features]
        # output: [N, out_features]  (ReLU sonrası)
        grad_input = grad_weight = grad_bias = None

        # dL/dz = dL/dy * 1_{y>0}, z = x @ W^T + b, y = ReLU(z)
        grad_z = grad_output
        if output is not None:
            grad_z = grad_output * (output > 0).to(grad_output.dtype)

        if ctx.needs_input_grad[0]:
            # grad_input = grad_z @ W
            grad_input = grad_z.matmul(weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_z^T @ input
            grad_weight = grad_z.t().matmul(input)
        if ctx.needs_input_grad[2] and bias is not None:
            # grad_bias = sum_n grad_z[n, :]
            grad_bias = grad_z.sum(dim=0)

        return grad_input, grad_weight, grad_bias


# -----------------------------------------------------------------------------
# nn.Module wrapper (Hypatia FX e-graph'in ürettiği modül)
# -----------------------------------------------------------------------------

class FusedLinearReLU(nn.Module):
    """
    Hypatia'nın e-graph'ı tarafından kullanılan fused Linear+ReLU modülü.

    - State dict uyumluluğu için içerde bir nn.Linear tutuyoruz (self.fc).
    - CUDA'da mümkünse C++/CUDA kernel'i, aksi halde F.linear+ReLU kullanıyoruz.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)

        # Extension'ı lazy-load
        _load_fused_linear_relu_ext()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamo tracing: use pure PyTorch ops to avoid graph breaks
        if _is_dynamo_tracing():
            return F.relu(self.fc(x))
        if x.is_cuda and _HAS_CUDA_KERNEL:
            return FusedLinearReLUFunction.apply(x, self.fc.weight, self.fc.bias)
        else:
            # CPU veya extension yoksa: normal path
            return F.relu(self.fc(x))


# Backward compatibility alias
HypatiaFusedLinearReLU = FusedLinearReLU


# -----------------------------------------------------------------------------
# Helper function for FX graph reconstruction
# -----------------------------------------------------------------------------

def create_fused_linear_relu_from_tensors(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> FusedLinearReLU:
    """
    Create a FusedLinearReLU module from existing weight and bias tensors.

    This helper is called from Rust (fx_bridge::reconstruct_fused_linear_relu)
    during graph reconstruction.
    """
    if not torch.is_tensor(weight):
        raise TypeError(f"weight must be a Tensor, got {type(weight)}")

    device = weight.device
    dtype = weight.dtype

    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D (out_features, in_features), got shape={tuple(weight.shape)}"
        )

    out_features, in_features = weight.shape
    use_bias = bias is not None and torch.is_tensor(bias)

    # Create module on correct device & dtype
    fused = FusedLinearReLU(
        in_features=in_features,
        out_features=out_features,
        bias=use_bias,
        device=device,
        dtype=dtype,
    )

    # Copy parameters
    with torch.no_grad():
        fused.fc.weight.copy_(weight)
        if use_bias:
            fused.fc.bias.copy_(bias.to(device=device, dtype=dtype))

    return fused


# -----------------------------------------------------------------------------
# FusedMLP: Multi-layer fusion using FusedLinearReLU
# -----------------------------------------------------------------------------

class FusedMLP(nn.Module):
    """
    2-layer MLP with fused Linear+ReLU for first layer.

    This module is created by the e-graph optimizer when it detects:
        linear(w2, b2, relu(linear(w1, b1, x)))

    The e-graph rewrites this to:
        (fused-mlp w1 b1 w2 b2 x)

    Architecture:
        x -> FusedLinearReLU(w1, b1) -> Linear(w2, b2) -> output
    """

    def __init__(self,
                 weight1: torch.Tensor,
                 bias1: Optional[torch.Tensor],
                 weight2: torch.Tensor,
                 bias2: Optional[torch.Tensor],
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Args:
            weight1: Weight tensor for first layer [hidden, in_features]
            bias1: Bias tensor for first layer [hidden] or None
            weight2: Weight tensor for second layer [out_features, hidden]
            bias2: Bias tensor for second layer [out_features] or None
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        _load_fused_linear_relu_ext()

        # Layer 1: FusedLinearReLU
        self.layer1 = create_fused_linear_relu_from_tensors(weight1, bias1)

        # Layer 2: Linear (no activation)
        out_features, hidden_size = weight2.shape
        self.layer2 = nn.Linear(hidden_size, out_features, bias=(bias2 is not None),
                               device=device, dtype=dtype)

        with torch.no_grad():
            self.layer2.weight.copy_(weight2)
            if bias2 is not None:
                self.layer2.bias.copy_(bias2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def create_fused_mlp_from_tensors(
    weight1: torch.Tensor,
    bias1: Optional[torch.Tensor],
    weight2: torch.Tensor,
    bias2: Optional[torch.Tensor],
) -> FusedMLP:
    """
    Create a FusedMLP module from weight and bias tensors.

    This helper is called from Rust (fx_bridge) during graph reconstruction.

    Args:
        weight1: Weight for first layer (FusedLinearReLU)
        bias1: Bias for first layer or None
        weight2: Weight for second layer (Linear)
        bias2: Bias for second layer or None

    Returns:
        FusedMLP module with parameters copied from tensors
    """
    device = weight1.device
    dtype = weight1.dtype

    return FusedMLP(weight1, bias1, weight2, bias2, device=device, dtype=dtype)


# =============================================================================
# FusedGeluMLP: GPU-aware Linear -> GELU -> Linear fusion
# =============================================================================

class FusedGeluMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w1, b1, w2, b2):
        ext = _get_ext("fused_gelu_mlp")
        use_cuda = (
            ext is not None
            and input.is_cuda
            and w1.is_cuda
            and w2.is_cuda
        )

        if use_cuda:
            y = ext.forward(input, w1, b1, w2, b2)
        else:
            # CPU fallback
            h = F.linear(input, w1, b1)
            h = F.gelu(h)
            y = F.linear(h, w2, b2)

        ctx.save_for_backward(input, w1, b1 if b1 is not None else torch.tensor([]),
                              w2, b2 if b2 is not None else torch.tensor([]))
        ctx.had_b1 = b1 is not None
        ctx.had_b2 = b2 is not None
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, w1, b1, w2, b2 = ctx.saved_tensors
        if not ctx.had_b1:
            b1 = None
        if not ctx.had_b2:
            b2 = None

        # Forward recomputation for GELU gradient
        h = F.linear(input, w1, b1)
        # GELU derivative: gelu'(x) ~= 0.5 * (1 + tanh(s)) + 0.5 * x * (1 - tanh(s)^2) * s'
        # Use PyTorch autograd for correctness
        h.requires_grad_(True)
        h_gelu = F.gelu(h)
        output = F.linear(h_gelu, w2, b2)

        output.backward(grad_output)
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.autograd.grad(output, input, grad_output, retain_graph=True)[0] if input.requires_grad else None

        # Simplified: return None for weight/bias gradients (inference-focused)
        return grad_input, None, None, None, None


class FusedGeluMLP(nn.Module):
    """GPU-aware fused Linear -> GELU -> Linear.

    Uses custom CUDA kernel on GPU (cuBLAS GEMM + in-place GELU),
    falls back to PyTorch ops on CPU.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory = {'device': device, 'dtype': dtype}
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory)
        # Lazy-load extension
        _load_cuda_ext("fused_gelu_mlp", "fused_gelu_mlp.cpp", "fused_gelu_mlp_kernel.cu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamo tracing: use pure PyTorch ops to avoid graph breaks
        if _is_dynamo_tracing():
            h = F.gelu(self.fc1(x))
            return self.fc2(h)
        if x.is_cuda and _has_ext("fused_gelu_mlp"):
            return FusedGeluMLPFunction.apply(
                x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)
        # CPU fallback
        h = F.gelu(self.fc1(x))
        return self.fc2(h)


# =============================================================================
# FusedAttention: GPU-aware multi-head causal self-attention
# =============================================================================

class FusedAttention(nn.Module):
    """GPU-aware fused multi-head causal self-attention.

    Fuses Q/K/V projection -> reshape -> scaled dot-product -> causal mask
    -> softmax -> V aggregation -> output projection into a single CUDA call.

    On CPU, uses standard PyTorch ops.
    """

    def __init__(self, hidden: int, n_heads: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        factory = {'device': device, 'dtype': dtype}

        self.q_proj = nn.Linear(hidden, hidden, bias=bias, **factory)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias, **factory)
        self.v_proj = nn.Linear(hidden, hidden, bias=bias, **factory)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias, **factory)

        _load_cuda_ext("fused_attention", "fused_attention.cpp", "fused_attention_kernel.cu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamo tracing: use pure PyTorch ops to avoid graph breaks
        if _is_dynamo_tracing():
            return self._forward_cpu(x)
        if x.is_cuda and _has_ext("fused_attention"):
            ext = _get_ext("fused_attention")
            return ext.forward(
                x,
                self.q_proj.weight, self.q_proj.bias,
                self.k_proj.weight, self.k_proj.bias,
                self.v_proj.weight, self.v_proj.bias,
                self.o_proj.weight, self.o_proj.bias,
                self.n_heads)

        # CPU fallback
        return self._forward_cpu(x)

    def _forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)  # batch*seq_len
        q = self.q_proj(x).view(B, self.n_heads, self.head_dim).permute(1, 0, 2)
        k = self.k_proj(x).view(B, self.n_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(x).view(B, self.n_heads, self.head_dim).permute(1, 0, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale

        # Causal mask
        mask = torch.ones(B, B, device=x.device, dtype=x.dtype).triu(1) * (-1e9)
        scores = scores + mask.unsqueeze(0)
        scores = F.softmax(scores, dim=-1)

        attn_out = torch.bmm(scores, v)
        attn_out = attn_out.permute(1, 0, 2).contiguous().view(B, self.hidden)
        return self.o_proj(attn_out)

    @classmethod
    def from_gpt2_attn(cls, attn_module, n_heads: int) -> 'FusedAttention':
        """Create from GPT-2 style attention (c_attn + c_proj)."""
        from .native_model import _get_linear_weight_bias
        c_attn = attn_module.c_attn
        c_proj = attn_module.c_proj

        weight, bias = _get_linear_weight_bias(c_attn)
        hidden = weight.shape[1]

        fused = cls(hidden, n_heads, bias=(bias is not None),
                    device=weight.device, dtype=weight.dtype)

        with torch.no_grad():
            fused.q_proj.weight.copy_(weight[:hidden])
            fused.k_proj.weight.copy_(weight[hidden:2*hidden])
            fused.v_proj.weight.copy_(weight[2*hidden:])
            if bias is not None:
                fused.q_proj.bias.copy_(bias[:hidden])
                fused.k_proj.bias.copy_(bias[hidden:2*hidden])
                fused.v_proj.bias.copy_(bias[2*hidden:])

            wo, bo = _get_linear_weight_bias(c_proj)
            fused.o_proj.weight.copy_(wo)
            if bo is not None:
                fused.o_proj.bias.copy_(bo)

        return fused


# =============================================================================
# FusedLayerNorm: GPU-aware LayerNorm with warp-reduction
# =============================================================================

class FusedLayerNorm(nn.Module):
    """GPU-aware fused LayerNorm.

    Single kernel: mean + variance reduction + normalize + affine transform.
    Uses warp shuffle reduction for minimal shared memory usage.

    On CPU, delegates to PyTorch's optimized LayerNorm.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory = {'device': device, 'dtype': dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, **factory))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, **factory))

        _load_cuda_ext("fused_layernorm", "fused_layernorm.cpp", "fused_layernorm_kernel.cu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamo tracing: use pure PyTorch ops to avoid graph breaks
        if _is_dynamo_tracing():
            return F.layer_norm(x, [self.normalized_shape], self.weight, self.bias, self.eps)
        if x.is_cuda and _has_ext("fused_layernorm"):
            ext = _get_ext("fused_layernorm")
            return ext.forward(x, self.weight, self.bias, self.eps)
        # CPU fallback
        return F.layer_norm(x, [self.normalized_shape], self.weight, self.bias, self.eps)

    @classmethod
    def from_torch_layernorm(cls, ln: nn.LayerNorm) -> 'FusedLayerNorm':
        """Create from an existing PyTorch LayerNorm."""
        normalized_shape = ln.normalized_shape[0] if isinstance(ln.normalized_shape, (list, tuple)) else ln.normalized_shape
        fused = cls(normalized_shape, eps=ln.eps,
                    device=ln.weight.device, dtype=ln.weight.dtype)
        with torch.no_grad():
            fused.weight.copy_(ln.weight)
            fused.bias.copy_(ln.bias)
        return fused


# =============================================================================
# GPU TransformerBlock: Full fused transformer block
# =============================================================================

class FusedTransformerBlock(nn.Module):
    """GPU-aware fused transformer block.

    Combines FusedLayerNorm + FusedAttention + FusedGeluMLP with residual connections.
    Automatically dispatches to CUDA kernels on GPU, PyTorch ops on CPU.
    """

    def __init__(self, hidden: int, n_heads: int, mlp_hidden: int,
                 bias: bool = True, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.ln_1 = FusedLayerNorm(hidden, eps=eps, device=device, dtype=dtype)
        self.attn = FusedAttention(hidden, n_heads, bias=bias, device=device, dtype=dtype)
        self.ln_2 = FusedLayerNorm(hidden, eps=eps, device=device, dtype=dtype)
        self.mlp = FusedGeluMLP(hidden, mlp_hidden, hidden, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln_1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# Auto-dispatch functions for torch.compile Phase 3 reconstruction
# =============================================================================
# These are the target functions used by fx_bridge.rs Phase 3.
# They auto-detect device and dispatch to CUDA kernels or Rust native ops.

def _is_dynamo_tracing():
    """Check if torch._dynamo is currently tracing (e.g. during Phase 2 torch.compile).

    When Dynamo is tracing, we must use pure PyTorch ops so the graph
    can be captured without graph breaks.  Custom pybind11 / PyO3 calls
    are opaque to Dynamo and cause silent partial compilation.
    """
    try:
        return torch.compiler.is_compiling()  # PyTorch 2.1+
    except AttributeError:
        pass
    try:
        return torch._dynamo.is_compiling()  # PyTorch 2.0
    except AttributeError:
        return False


def dispatch_fused_gelu_mlp(input, w1, b1, w2, b2):
    """Auto-dispatch fused GELU MLP: GPU -> CUDA kernel, CPU -> Rust native."""
    # When Dynamo is tracing (Phase 2), use pure PyTorch ops to avoid graph breaks.
    # Inductor can still fuse Linear+GELU+Linear into efficient Triton kernels.
    if _is_dynamo_tracing():
        h = F.linear(input, w1, b1)
        h = F.gelu(h)
        return F.linear(h, w2, b2)
    if input.is_cuda and _has_ext("fused_gelu_mlp"):
        ext = _get_ext("fused_gelu_mlp")
        return ext.forward(input, w1, b1, w2, b2)
    # CPU: try Rust native kernel (faster than PyTorch for small/medium)
    try:
        from _hypatia_core import fused_gelu_mlp_forward
        return fused_gelu_mlp_forward(input, w1, b1, w2, b2)
    except (ImportError, RuntimeError):
        pass
    # Final fallback: PyTorch ops
    h = F.linear(input, w1, b1)
    h = F.gelu(h)
    return F.linear(h, w2, b2)


def dispatch_fused_linear_relu(input, weight, bias):
    """Auto-dispatch fused Linear+ReLU: GPU -> CUDA kernel, CPU -> Rust native."""
    # When Dynamo is tracing (Phase 2), use pure PyTorch ops to avoid graph breaks.
    if _is_dynamo_tracing():
        return F.relu(F.linear(input, weight, bias))
    if input.is_cuda and _has_ext("fused_linear_relu"):
        ext = _get_ext("fused_linear_relu")
        return ext.forward(input, weight, bias)
    # CPU: try Rust native kernel
    try:
        from _hypatia_core import fused_linear_relu_forward
        return fused_linear_relu_forward(input, weight, bias)
    except (ImportError, RuntimeError):
        pass
    # Final fallback
    return F.relu(F.linear(input, weight, bias))


def dispatch_fused_attention(input, wq, bq, wk, bk, wv, bv, wo, bo, n_heads):
    """Auto-dispatch fused multi-head self-attention.

    GPU -> CUDA fused attention kernel
    CPU -> Rust native multi_head_attention (strided GEMM)
    Fallback -> PyTorch ops

    Args:
        input: [batch*seq_len, hidden] tensor
        wq, wk, wv, wo: [hidden, hidden] weight tensors
        bq, bk, bv, bo: [hidden] bias tensors or None
        n_heads: number of attention heads

    Returns:
        [batch*seq_len, hidden] output tensor
    """
    # When Dynamo is tracing (Phase 2), use pure PyTorch ops to avoid graph breaks.
    if _is_dynamo_tracing():
        total_rows = input.size(0)
        hidden = input.size(1)
        head_dim = hidden // n_heads
        q = F.linear(input, wq, bq)
        k = F.linear(input, wk, bk)
        v = F.linear(input, wv, bv)
        q = q.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
        k = k.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
        v = v.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        mask = torch.ones(total_rows, total_rows, device=input.device, dtype=input.dtype).triu(1) * (-1e9)
        scores = scores + mask.unsqueeze(0)
        scores = F.softmax(scores, dim=-1)
        attn_out = torch.bmm(scores, v)
        attn_out = attn_out.permute(1, 0, 2).contiguous().view(total_rows, hidden)
        return F.linear(attn_out, wo, bo)

    # GPU: try CUDA fused attention kernel
    if input.is_cuda and _has_ext("fused_attention"):
        ext = _get_ext("fused_attention")
        return ext.forward(input, wq, bq, wk, bk, wv, bv, wo, bo, n_heads)

    # CPU: try Rust native kernel (optimized strided GEMM)
    if not input.is_cuda:
        try:
            from _hypatia_core import fused_attention_forward
            import numpy as np

            x_np = input.detach().cpu().numpy()
            total_rows, hidden = x_np.shape

            # Infer batch and seq_len from input shape
            # Default: batch=1, seq_len=total_rows
            batch = 1
            seq_len = total_rows

            wq_np = wq.detach().cpu().numpy()
            wk_np = wk.detach().cpu().numpy()
            wv_np = wv.detach().cpu().numpy()
            wo_np = wo.detach().cpu().numpy()

            bq_np = bq.detach().cpu().numpy() if bq is not None else None
            bk_np = bk.detach().cpu().numpy() if bk is not None else None
            bv_np = bv.detach().cpu().numpy() if bv is not None else None
            bo_np = bo.detach().cpu().numpy() if bo is not None else None

            result = fused_attention_forward(
                x_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np, wo_np, bo_np,
                batch, seq_len, n_heads,
            )
            return torch.from_numpy(np.array(result)).to(input.device, input.dtype)
        except (ImportError, RuntimeError):
            pass

    # Final fallback: PyTorch ops
    total_rows = input.size(0)
    hidden = input.size(1)
    head_dim = hidden // n_heads

    q = F.linear(input, wq, bq)  # [total_rows, hidden]
    k = F.linear(input, wk, bk)
    v = F.linear(input, wv, bv)

    # Reshape to [batch*n_heads, seq_len, head_dim] for bmm
    # Assume batch=1 for simplicity
    q = q.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
    k = k.view(total_rows, n_heads, head_dim).permute(1, 0, 2)
    v = v.view(total_rows, n_heads, head_dim).permute(1, 0, 2)

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale

    # Causal mask
    mask = torch.ones(total_rows, total_rows, device=input.device, dtype=input.dtype).triu(1) * (-1e9)
    scores = scores + mask.unsqueeze(0)
    scores = F.softmax(scores, dim=-1)

    attn_out = torch.bmm(scores, v)
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(total_rows, hidden)
    return F.linear(attn_out, wo, bo)


__all__ = [
    "FusedLinearReLU", "HypatiaFusedLinearReLU", "FusedLinearReLUFunction",
    "create_fused_linear_relu_from_tensors", "FusedMLP", "create_fused_mlp_from_tensors",
    "FusedGeluMLP", "FusedGeluMLPFunction",
    "FusedAttention", "FusedLayerNorm", "FusedTransformerBlock",
    "dispatch_fused_gelu_mlp", "dispatch_fused_linear_relu", "dispatch_fused_attention",
    "CUDA_EXTENSION_AVAILABLE",
]

# Eager load CUDA extensions when module is imported
_load_fused_linear_relu_ext()

# Legacy alias for backward compatibility with tests
CUDA_EXTENSION_AVAILABLE = _HAS_CUDA_KERNEL
