# hypatia_core/profiler.py
#
# Shape-aware FLOPs estimation, hardware detection, and model profiling.
#
# Provides:
#   - estimate_flops(model, input_shape)  — per-layer and total FLOP counts
#   - detect_hardware()                   — GPU/CPU capability detection
#   - profile_model(model, input_shape)   — combined profiling report
#   - compare_flops(original, optimized)  — FLOPs savings from optimization

from __future__ import annotations

import time
import platform
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    # CPU info
    cpu_name: str = "Unknown"
    cpu_cores: int = 1
    cpu_threads: int = 1

    # GPU info
    has_cuda: bool = False
    gpu_name: str = ""
    gpu_compute_capability: Tuple[int, int] = (0, 0)
    gpu_memory_gb: float = 0.0
    gpu_sm_count: int = 0
    gpu_clock_mhz: int = 0

    # Precision support
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_int8: bool = False

    # Tensor cores
    has_tensor_cores: bool = False
    tensor_core_gen: str = ""  # "Volta", "Turing", "Ampere", "Ada Lovelace", "Hopper"

    # Memory bandwidth (theoretical peak, GB/s)
    memory_bandwidth_gbs: float = 0.0

    # Peak TFLOPS (theoretical)
    peak_tflops_fp32: float = 0.0
    peak_tflops_fp16: float = 0.0

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = []
        lines.append(f"CPU: {self.cpu_name} ({self.cpu_cores}C/{self.cpu_threads}T)")

        if self.has_cuda:
            lines.append(f"GPU: {self.gpu_name}")
            lines.append(f"  Compute: SM {self.gpu_compute_capability[0]}.{self.gpu_compute_capability[1]}")
            lines.append(f"  VRAM: {self.gpu_memory_gb:.1f} GB")
            lines.append(f"  SMs: {self.gpu_sm_count}, Clock: {self.gpu_clock_mhz} MHz")
            if self.memory_bandwidth_gbs > 0:
                lines.append(f"  Bandwidth: {self.memory_bandwidth_gbs:.0f} GB/s")
            if self.peak_tflops_fp32 > 0:
                lines.append(f"  Peak: {self.peak_tflops_fp32:.1f} TFLOPS (FP32), {self.peak_tflops_fp16:.1f} TFLOPS (FP16)")
            precision = []
            if self.supports_fp16: precision.append("FP16")
            if self.supports_bf16: precision.append("BF16")
            if self.supports_tf32: precision.append("TF32")
            if self.supports_int8: precision.append("INT8")
            if precision:
                lines.append(f"  Precision: {', '.join(precision)}")
            if self.has_tensor_cores:
                lines.append(f"  Tensor Cores: {self.tensor_core_gen}")
        else:
            lines.append("GPU: Not available (CPU-only)")

        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """Detect available hardware capabilities.

    Returns:
        HardwareInfo with CPU and GPU details, precision support,
        and theoretical peak performance numbers.

    Examples:
        hw = detect_hardware()
        print(hw.summary())
        if hw.supports_bf16:
            model = model.to(torch.bfloat16)
    """
    info = HardwareInfo()

    # CPU detection
    info.cpu_name = platform.processor() or platform.machine()
    try:
        import os
        info.cpu_cores = os.cpu_count() or 1
        info.cpu_threads = info.cpu_cores  # Approximate; actual HT detection is OS-specific
    except Exception:
        pass

    # GPU detection
    if torch.cuda.is_available():
        info.has_cuda = True
        try:
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.gpu_compute_capability = (props.major, props.minor)
            info.gpu_memory_gb = props.total_memory / (1024 ** 3)
            info.gpu_sm_count = props.multi_processor_count

            cc = (props.major, props.minor)

            # Precision support based on compute capability
            info.supports_fp16 = cc >= (5, 3)   # Maxwell+
            info.supports_int8 = cc >= (6, 1)    # Pascal+
            info.supports_tf32 = cc >= (8, 0)    # Ampere+
            info.supports_bf16 = cc >= (8, 0)    # Ampere+

            # Try to get GPU clock speed via nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=clocks.max.sm', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.gpu_clock_mhz = int(result.stdout.strip().split('\n')[0])
            except Exception:
                # Estimate from known GPUs
                _known_clocks = {
                    "RTX 4090": 2520, "RTX 4080": 2505, "RTX 4070": 2475,
                    "RTX 4070 Laptop": 2175, "RTX 4060": 2370,
                    "RTX 3090": 1695, "RTX 3080": 1710, "RTX 3070": 1725,
                    "A100": 1410, "H100": 1830, "V100": 1380,
                }
                for key, clk in _known_clocks.items():
                    if key.lower() in info.gpu_name.lower():
                        info.gpu_clock_mhz = clk
                        break

            # Tensor core generation
            if cc >= (9, 0):
                info.has_tensor_cores = True
                info.tensor_core_gen = "Hopper"
            elif cc >= (8, 9):
                info.has_tensor_cores = True
                info.tensor_core_gen = "Ada Lovelace"
            elif cc >= (8, 0):
                info.has_tensor_cores = True
                info.tensor_core_gen = "Ampere"
            elif cc >= (7, 5):
                info.has_tensor_cores = True
                info.tensor_core_gen = "Turing"
            elif cc >= (7, 0):
                info.has_tensor_cores = True
                info.tensor_core_gen = "Volta"

            # Estimate peak TFLOPS
            # FP32: 2 FLOPs/clock/CUDA_core * cores_per_SM * SM_count * clock_GHz
            # Approximate: each SM has ~64-128 CUDA cores depending on arch
            _cuda_cores_per_sm = {
                (7, 0): 64, (7, 5): 64,   # Volta, Turing
                (8, 0): 64, (8, 6): 128, (8, 9): 128,  # Ampere, Ada
                (9, 0): 128,  # Hopper
            }
            cores_per_sm = _cuda_cores_per_sm.get(cc, _cuda_cores_per_sm.get((cc[0], 0), 64))
            total_cores = info.gpu_sm_count * cores_per_sm
            clock_ghz = info.gpu_clock_mhz / 1000.0 if info.gpu_clock_mhz > 0 else 1.5
            info.peak_tflops_fp32 = (2.0 * total_cores * clock_ghz) / 1000.0
            # FP16 with tensor cores is ~2x FP32
            info.peak_tflops_fp16 = info.peak_tflops_fp32 * (2.0 if info.has_tensor_cores else 1.0)

            # Memory bandwidth estimate (very rough based on memory type)
            # For common GPUs:
            _known_bandwidths = {
                "RTX 4090": 1008, "RTX 4080": 717, "RTX 4070": 504,
                "RTX 4070 Laptop": 256, "RTX 4060": 272,
                "RTX 3090": 936, "RTX 3080": 760, "RTX 3070": 448,
                "A100": 2039, "H100": 3350, "V100": 900,
            }
            for key, bw in _known_bandwidths.items():
                if key.lower() in info.gpu_name.lower():
                    info.memory_bandwidth_gbs = bw
                    break

        except Exception:
            pass

    return info


# ============================================================================
# FLOPs ESTIMATION
# ============================================================================

@dataclass
class LayerProfile:
    """Per-layer FLOPs and memory profile."""
    name: str
    module_type: str
    flops: int = 0
    mac: int = 0   # Multiply-Accumulate operations
    params: int = 0
    input_shape: Tuple = ()
    output_shape: Tuple = ()
    # Memory (bytes)
    weight_memory: int = 0
    activation_memory: int = 0

    @property
    def flops_str(self) -> str:
        """Human-readable FLOPs string (e.g., '2.4 GFLOPs')."""
        if self.flops >= 1e12:
            return f"{self.flops / 1e12:.2f} TFLOPs"
        elif self.flops >= 1e9:
            return f"{self.flops / 1e9:.2f} GFLOPs"
        elif self.flops >= 1e6:
            return f"{self.flops / 1e6:.2f} MFLOPs"
        elif self.flops >= 1e3:
            return f"{self.flops / 1e3:.2f} KFLOPs"
        return f"{self.flops} FLOPs"


@dataclass
class ModelProfile:
    """Complete model FLOPs and memory profile."""
    layers: List[LayerProfile] = field(default_factory=list)
    total_flops: int = 0
    total_mac: int = 0
    total_params: int = 0
    total_weight_memory: int = 0
    total_activation_memory: int = 0
    # Timing (if measured)
    inference_ms: float = 0.0
    hardware: Optional[HardwareInfo] = None

    @property
    def total_flops_str(self) -> str:
        if self.total_flops >= 1e12:
            return f"{self.total_flops / 1e12:.2f} TFLOPs"
        elif self.total_flops >= 1e9:
            return f"{self.total_flops / 1e9:.2f} GFLOPs"
        elif self.total_flops >= 1e6:
            return f"{self.total_flops / 1e6:.2f} MFLOPs"
        return f"{self.total_flops} FLOPs"

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs / Bytes transferred (higher = more compute-bound)."""
        total_bytes = self.total_weight_memory + self.total_activation_memory
        if total_bytes == 0:
            return 0.0
        return self.total_flops / total_bytes

    @property
    def compute_utilization(self) -> float:
        """Actual vs theoretical peak performance (0.0-1.0)."""
        if self.hardware is None or self.inference_ms <= 0:
            return 0.0
        peak = self.hardware.peak_tflops_fp32 * 1e12  # FLOPs/sec
        actual = self.total_flops / (self.inference_ms / 1000.0)  # FLOPs/sec
        return min(actual / peak, 1.0) if peak > 0 else 0.0

    def summary(self) -> str:
        """Human-readable model profile summary."""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  Hypatia Model Profile")
        lines.append(f"{'='*70}")
        lines.append(f"  Total FLOPs:       {self.total_flops_str}")
        lines.append(f"  Total MACs:        {self.total_mac:,}")
        lines.append(f"  Total Parameters:  {self.total_params:,}")
        lines.append(f"  Weight Memory:     {self.total_weight_memory / 1024 / 1024:.2f} MB")
        lines.append(f"  Activation Memory: {self.total_activation_memory / 1024 / 1024:.2f} MB")
        lines.append(f"  Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/byte")

        if self.inference_ms > 0:
            lines.append(f"  Inference Time:    {self.inference_ms:.2f} ms")
            throughput = self.total_flops / (self.inference_ms / 1000.0)
            if throughput >= 1e12:
                lines.append(f"  Throughput:        {throughput / 1e12:.2f} TFLOPS")
            else:
                lines.append(f"  Throughput:        {throughput / 1e9:.2f} GFLOPS")
            if self.compute_utilization > 0:
                lines.append(f"  GPU Utilization:   {self.compute_utilization * 100:.1f}%")

        lines.append(f"{'='*70}")

        # Per-layer breakdown
        if self.layers:
            lines.append(f"  {'Layer':<30} {'Type':<20} {'FLOPs':<15} {'Params':<12} {'%FLOPs':<8}")
            lines.append(f"  {'-'*85}")
            for lp in self.layers:
                if lp.flops == 0:
                    continue
                pct = (lp.flops / self.total_flops * 100) if self.total_flops > 0 else 0
                lines.append(
                    f"  {lp.name:<30} {lp.module_type:<20} {lp.flops_str:<15} {lp.params:<12,} {pct:>5.1f}%"
                )
            lines.append(f"{'='*70}")

        return "\n".join(lines)


def _count_linear_flops(m: nn.Linear, input_shape: Tuple) -> Tuple[int, int]:
    """Count FLOPs for Linear layer: output = input @ weight.T + bias.

    FLOPs = 2 * in_features * out_features * batch_elements
    (multiply + accumulate for each output element)
    """
    # batch_elements = product of all dims except last
    batch_elements = 1
    for d in input_shape[:-1]:
        batch_elements *= d

    mac = batch_elements * m.in_features * m.out_features
    flops = 2 * mac
    if m.bias is not None:
        flops += batch_elements * m.out_features  # bias add
    return flops, mac


def _count_conv2d_flops(m: nn.Conv2d, input_shape: Tuple) -> Tuple[int, int]:
    """Count FLOPs for Conv2d layer.

    FLOPs = 2 * batch * out_channels * out_h * out_w * in_channels/groups * kH * kW
    """
    batch = input_shape[0] if len(input_shape) >= 4 else 1
    in_h = input_shape[2] if len(input_shape) >= 4 else input_shape[1] if len(input_shape) >= 3 else 1
    in_w = input_shape[3] if len(input_shape) >= 4 else input_shape[2] if len(input_shape) >= 3 else 1

    # Output spatial dimensions
    pad_h = m.padding[0] if isinstance(m.padding, tuple) else m.padding
    pad_w = m.padding[1] if isinstance(m.padding, tuple) else m.padding
    stride_h = m.stride[0] if isinstance(m.stride, tuple) else m.stride
    stride_w = m.stride[1] if isinstance(m.stride, tuple) else m.stride
    dil_h = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
    dil_w = m.dilation[1] if isinstance(m.dilation, tuple) else m.dilation

    kH, kW = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    out_h = (in_h + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    mac = batch * m.out_channels * out_h * out_w * (m.in_channels // m.groups) * kH * kW
    flops = 2 * mac
    if m.bias is not None:
        flops += batch * m.out_channels * out_h * out_w
    return flops, mac


def _count_batchnorm_flops(m: nn.Module, input_shape: Tuple) -> Tuple[int, int]:
    """Count FLOPs for BatchNorm: normalize + scale + shift = 4 ops per element."""
    elements = 1
    for d in input_shape:
        elements *= d
    flops = 4 * elements  # mean, var, normalize, affine
    return flops, 0


def _count_layernorm_flops(m: nn.Module, input_shape: Tuple) -> Tuple[int, int]:
    """Count FLOPs for LayerNorm: 5 ops per element (mean, var, normalize, scale, shift)."""
    elements = 1
    for d in input_shape:
        elements *= d
    flops = 5 * elements
    return flops, 0


def _count_attention_flops(input_shape: Tuple, n_heads: int = 1) -> Tuple[int, int]:
    """Count FLOPs for self-attention: Q@K^T + softmax + attn@V.

    For sequence length S and head dim D:
      Q@K^T: 2*B*H*S*S*D
      softmax: 5*B*H*S*S (exp + sum + div + max + sub)
      attn@V: 2*B*H*S*S*D
    """
    if len(input_shape) >= 3:
        batch, seq_len, d_model = input_shape[0], input_shape[1], input_shape[2]
    elif len(input_shape) == 2:
        batch, seq_len, d_model = 1, input_shape[0], input_shape[1]
    else:
        return 0, 0

    head_dim = d_model // max(n_heads, 1)
    # QK^T
    qk_flops = 2 * batch * n_heads * seq_len * seq_len * head_dim
    # softmax
    softmax_flops = 5 * batch * n_heads * seq_len * seq_len
    # attn @ V
    av_flops = 2 * batch * n_heads * seq_len * seq_len * head_dim
    total = qk_flops + softmax_flops + av_flops
    mac = qk_flops // 2 + av_flops // 2
    return total, mac


def _count_elementwise_flops(input_shape: Tuple, ops_per_element: int = 1) -> Tuple[int, int]:
    """Count FLOPs for elementwise ops (ReLU=1, GELU=8, SiLU=4, Mish=6)."""
    elements = 1
    for d in input_shape:
        elements *= d
    return elements * ops_per_element, 0


# Elementwise op costs (FLOPs per element)
_ACTIVATION_FLOPS = {
    nn.ReLU: 1,
    nn.LeakyReLU: 2,
    nn.ELU: 3,
    nn.GELU: 8,    # erf approximation
    nn.SiLU: 4,    # x * sigmoid(x)
    nn.Sigmoid: 4,  # exp + div
    nn.Tanh: 5,    # exp ops
    nn.Softmax: 5,  # exp + sum + div per element
    nn.Mish: 6,    # x * tanh(softplus(x))
}


def estimate_flops(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    n_heads: int = 1,
) -> ModelProfile:
    """Estimate FLOPs for each layer of a PyTorch model.

    Uses shape propagation to compute accurate per-layer FLOPs based on
    actual tensor dimensions, not fixed constants.

    Args:
        model: PyTorch model (nn.Module)
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224) for images,
                     (1, 128, 768) for sequences)
        n_heads: Number of attention heads (for transformer models)

    Returns:
        ModelProfile with per-layer and total FLOPs, MACs, memory estimates.

    Examples:
        profile = estimate_flops(model, (1, 3, 224, 224))
        print(profile.summary())
        print(f"Total: {profile.total_flops_str}")
    """
    profile = ModelProfile()
    profile.hardware = detect_hardware()

    # Use hooks to capture input/output shapes during forward pass
    hooks = []
    layer_data = []

    def _make_hook(name, module):
        def hook(m, inp, out):
            inp_shape = tuple(inp[0].shape) if isinstance(inp, tuple) and len(inp) > 0 and hasattr(inp[0], 'shape') else ()
            out_shape = tuple(out.shape) if hasattr(out, 'shape') else ()

            flops, mac = 0, 0
            module_type = type(m).__name__

            if isinstance(m, nn.Linear):
                flops, mac = _count_linear_flops(m, inp_shape)
            elif isinstance(m, nn.Conv2d):
                flops, mac = _count_conv2d_flops(m, inp_shape)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                flops, _ = _count_batchnorm_flops(m, inp_shape)
            elif isinstance(m, nn.LayerNorm):
                flops, _ = _count_layernorm_flops(m, inp_shape)
            elif isinstance(m, nn.MultiheadAttention):
                flops, mac = _count_attention_flops(inp_shape, n_heads=m.num_heads)
                module_type = f"MHA(h={m.num_heads})"
            elif type(m) in _ACTIVATION_FLOPS:
                flops, _ = _count_elementwise_flops(inp_shape, _ACTIVATION_FLOPS[type(m)])

            # Parameter count
            params = sum(p.numel() for p in m.parameters(recurse=False))
            weight_mem = sum(p.numel() * p.element_size() for p in m.parameters(recurse=False))

            # Activation memory (output tensor)
            act_mem = 0
            if hasattr(out, 'numel') and hasattr(out, 'element_size'):
                act_mem = out.numel() * out.element_size()

            lp = LayerProfile(
                name=name,
                module_type=module_type,
                flops=flops,
                mac=mac,
                params=params,
                input_shape=inp_shape,
                output_shape=out_shape,
                weight_memory=weight_mem,
                activation_memory=act_mem,
            )
            layer_data.append(lp)
        return hook

    # Register hooks on leaf modules (no children)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(_make_hook(name, module))
            hooks.append(h)

    # Run forward pass with dummy input
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    dtype = next(model.parameters()).dtype if len(list(model.parameters())) > 0 else torch.float32
    dummy = torch.randn(*input_shape, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            # If forward fails, still return partial results
            pass

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate results
    profile.layers = layer_data
    profile.total_flops = sum(lp.flops for lp in layer_data)
    profile.total_mac = sum(lp.mac for lp in layer_data)
    profile.total_params = sum(p.numel() for p in model.parameters())
    profile.total_weight_memory = sum(lp.weight_memory for lp in layer_data)
    profile.total_activation_memory = sum(lp.activation_memory for lp in layer_data)

    return profile


# ============================================================================
# INFERENCE TIMING
# ============================================================================

def benchmark_inference(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Measure average inference time in milliseconds.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        warmup: Number of warmup iterations (not measured)
        runs: Number of timed iterations

    Returns:
        Average inference time in milliseconds.

    Examples:
        ms = benchmark_inference(model, (1, 768))
        print(f"Inference: {ms:.2f} ms")
    """
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    dtype = next(model.parameters()).dtype if len(list(model.parameters())) > 0 else torch.float32
    dummy = torch.randn(*input_shape, device=device, dtype=dtype)

    model.eval()
    use_cuda = device.type == 'cuda'

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            for _ in range(runs):
                model(dummy)
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                model(dummy)
        total_ms = (time.perf_counter() - start) * 1000.0

    return total_ms / runs


# ============================================================================
# FULL PROFILING
# ============================================================================

def profile_model(
    model: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    n_heads: int = 1,
    benchmark: bool = True,
    warmup: int = 5,
    runs: int = 20,
) -> ModelProfile:
    """Complete model profiling: FLOPs + timing + hardware info.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        n_heads: Number of attention heads (for transformer models)
        benchmark: If True, measure actual inference time
        warmup: Warmup iterations for benchmark
        runs: Timed iterations for benchmark

    Returns:
        ModelProfile with FLOPs, timing, hardware info, and per-layer breakdown.

    Examples:
        profile = profile_model(model, (1, 3, 224, 224))
        print(profile.summary())
    """
    profile = estimate_flops(model, input_shape, n_heads=n_heads)

    if benchmark:
        profile.inference_ms = benchmark_inference(model, input_shape, warmup=warmup, runs=runs)

    return profile


# ============================================================================
# OPTIMIZATION COMPARISON
# ============================================================================

@dataclass
class FlopsComparison:
    """FLOPs comparison between original and optimized models."""
    original_flops: int = 0
    optimized_flops: int = 0
    original_params: int = 0
    optimized_params: int = 0
    original_ms: float = 0.0
    optimized_ms: float = 0.0
    original_memory_mb: float = 0.0
    optimized_memory_mb: float = 0.0

    @property
    def flops_reduction(self) -> float:
        """FLOPs reduction ratio (0.0-1.0). E.g., 0.3 = 30% fewer FLOPs."""
        if self.original_flops == 0:
            return 0.0
        return 1.0 - (self.optimized_flops / self.original_flops)

    @property
    def speedup(self) -> float:
        """Wall-clock speedup ratio. E.g., 2.0 = 2x faster."""
        if self.optimized_ms <= 0:
            return 0.0
        return self.original_ms / self.optimized_ms

    @property
    def memory_savings(self) -> float:
        """Memory savings ratio (0.0-1.0). E.g., 0.5 = 50% less memory."""
        if self.original_memory_mb == 0:
            return 0.0
        return 1.0 - (self.optimized_memory_mb / self.original_memory_mb)

    def summary(self) -> str:
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  Hypatia Optimization Comparison")
        lines.append(f"{'='*60}")

        def _fmt_flops(f):
            if f >= 1e9: return f"{f/1e9:.2f} GFLOPs"
            if f >= 1e6: return f"{f/1e6:.2f} MFLOPs"
            return f"{f:,} FLOPs"

        lines.append(f"  {'Metric':<25} {'Original':<20} {'Optimized':<20}")
        lines.append(f"  {'-'*65}")
        lines.append(f"  {'FLOPs':<25} {_fmt_flops(self.original_flops):<20} {_fmt_flops(self.optimized_flops):<20}")
        lines.append(f"  {'Parameters':<25} {self.original_params:<20,} {self.optimized_params:<20,}")

        if self.original_ms > 0:
            lines.append(f"  {'Inference (ms)':<25} {self.original_ms:<20.2f} {self.optimized_ms:<20.2f}")

        if self.original_memory_mb > 0:
            lines.append(f"  {'Memory (MB)':<25} {self.original_memory_mb:<20.2f} {self.optimized_memory_mb:<20.2f}")

        lines.append(f"  {'-'*65}")
        if self.flops_reduction > 0:
            lines.append(f"  FLOPs Reduction: {self.flops_reduction * 100:.1f}%")
        if self.speedup > 0:
            lines.append(f"  Speedup: {self.speedup:.2f}x")
        if self.memory_savings > 0:
            lines.append(f"  Memory Savings: {self.memory_savings * 100:.1f}%")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def compare_flops(
    original: nn.Module,
    optimized: nn.Module,
    input_shape: Union[Tuple, torch.Size],
    n_heads: int = 1,
    benchmark: bool = True,
) -> FlopsComparison:
    """Compare FLOPs and performance between original and optimized models.

    Args:
        original: Original PyTorch model
        optimized: Optimized model (after hypatia.optimize())
        input_shape: Input tensor shape
        n_heads: Number of attention heads
        benchmark: If True, measure actual inference times

    Returns:
        FlopsComparison with FLOPs, timing, and memory comparison.

    Examples:
        optimized = hypatia.optimize(model)
        cmp = compare_flops(model, optimized, (1, 768))
        print(cmp.summary())
    """
    orig_profile = profile_model(original, input_shape, n_heads=n_heads, benchmark=benchmark)
    opt_profile = profile_model(optimized, input_shape, n_heads=n_heads, benchmark=benchmark)

    return FlopsComparison(
        original_flops=orig_profile.total_flops,
        optimized_flops=opt_profile.total_flops,
        original_params=orig_profile.total_params,
        optimized_params=opt_profile.total_params,
        original_ms=orig_profile.inference_ms,
        optimized_ms=opt_profile.inference_ms,
        original_memory_mb=(orig_profile.total_weight_memory + orig_profile.total_activation_memory) / 1024 / 1024,
        optimized_memory_mb=(opt_profile.total_weight_memory + opt_profile.total_activation_memory) / 1024 / 1024,
    )


# ============================================================================
# ROOFLINE MODEL
# ============================================================================

def roofline_analysis(profile: ModelProfile) -> Dict[str, Union[str, float]]:
    """Determine if model is compute-bound or memory-bound using roofline model.

    The roofline model compares arithmetic intensity (FLOPs/byte) against
    the hardware's compute-to-bandwidth ratio to determine the bottleneck.

    Args:
        profile: ModelProfile from profile_model() or estimate_flops()

    Returns:
        Dict with 'bottleneck' ('compute' or 'memory'), 'arithmetic_intensity',
        'ridge_point', and 'recommendation'.
    """
    hw = profile.hardware or detect_hardware()

    ai = profile.arithmetic_intensity
    peak_flops = hw.peak_tflops_fp32 * 1e12 if hw.peak_tflops_fp32 > 0 else 1e12
    bandwidth = hw.memory_bandwidth_gbs * 1e9 if hw.memory_bandwidth_gbs > 0 else 100e9

    # Ridge point: where compute ceiling meets memory ceiling
    ridge_point = peak_flops / bandwidth if bandwidth > 0 else float('inf')

    if ai < ridge_point:
        bottleneck = "memory"
        recommendation = (
            "Model is memory-bandwidth bound. Optimizations:\n"
            "  - Use quantization (INT8/INT4) to reduce data movement\n"
            "  - Use operator fusion to reduce intermediate memory reads/writes\n"
            "  - Consider mixed precision (FP16/BF16) for 2x bandwidth savings"
        )
    else:
        bottleneck = "compute"
        recommendation = (
            "Model is compute-bound. Optimizations:\n"
            "  - Use Tensor Cores (FP16/BF16) for ~2x compute throughput\n"
            "  - Reduce model FLOPs (pruning, distillation)\n"
            "  - Use flash attention for O(N) memory attention"
        )

    return {
        "bottleneck": bottleneck,
        "arithmetic_intensity": ai,
        "ridge_point": ridge_point,
        "peak_compute_tflops": hw.peak_tflops_fp32,
        "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
        "recommendation": recommendation,
    }
