"""
CUDA Extension Build and Integration Test

Verifies:
1. CUDA extension can be imported
2. Forward/backward functions work correctly
3. Autograd integration is correct
4. Fallback logic works when CUDA unavailable
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn


def _get_cuda_ext():
    """Try to load the CUDA extension, return (available, ext_module)."""
    try:
        from hypatia_core.fused_modules import CUDA_EXTENSION_AVAILABLE
        if CUDA_EXTENSION_AVAILABLE:
            try:
                import hypatia_core._linear_relu_cuda as cuda_ext
                return True, cuda_ext
            except ImportError:
                return False, None
        return False, None
    except ImportError:
        return False, None


def test_import():
    """Test if CUDA extension can be imported"""
    from hypatia_core.fused_modules import (
        CUDA_EXTENSION_AVAILABLE,
        FusedLinearReLUFunction,
        HypatiaFusedLinearReLU,
    )
    # Just verify the imports succeed - CUDA may or may not be available
    assert FusedLinearReLUFunction is not None
    assert HypatiaFusedLinearReLU is not None


def test_cuda_kernel_direct():
    """Test CUDA kernel functions directly"""
    cuda_available, cuda_ext = _get_cuda_ext()

    if not cuda_available or cuda_ext is None:
        pytest.skip("CUDA extension not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    # Create test tensors
    batch, in_features, out_features = 32, 128, 64
    x = torch.randn(batch, in_features, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_features, device='cuda', dtype=torch.float32)

    # Test forward
    output = cuda_ext.forward(x, weight, bias)
    assert output.shape == (batch, out_features), f"Wrong output shape: {output.shape}"
    assert (output >= 0).all(), "ReLU not applied correctly (found negative values)"

    # Test backward
    grad_out = torch.randn_like(output)
    grad_x, grad_w, grad_b = cuda_ext.backward(grad_out, x, weight, bias, output)

    assert grad_x.shape == x.shape, "Wrong grad_x shape"
    assert grad_w.shape == weight.shape, "Wrong grad_w shape"
    assert grad_b.shape == bias.shape, "Wrong grad_b shape"


def test_autograd_function():
    """Test FusedLinearReLUFunction autograd integration"""
    from hypatia_core.fused_modules import (
        CUDA_EXTENSION_AVAILABLE,
        FusedLinearReLUFunction,
    )

    if not CUDA_EXTENSION_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create test tensors with grad
    batch, in_features, out_features = 16, 64, 32
    x = torch.randn(batch, in_features, device='cuda', requires_grad=True)
    weight = torch.randn(out_features, in_features, device='cuda', requires_grad=True)
    bias = torch.randn(out_features, device='cuda', requires_grad=True)

    # Forward
    output = FusedLinearReLUFunction.apply(x, weight, bias)
    assert output.shape == (batch, out_features)

    # Backward
    loss = output.sum()
    loss.backward()

    assert torch.isfinite(x.grad).all(), "x.grad contains inf/nan"
    assert torch.isfinite(weight.grad).all(), "weight.grad contains inf/nan"
    assert torch.isfinite(bias.grad).all(), "bias.grad contains inf/nan"


def test_module_integration():
    """Test HypatiaFusedLinearReLU module"""
    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    module = HypatiaFusedLinearReLU(128, 64, bias=True, device=device)
    x = torch.randn(32, 128, device=device)
    output = module(x)

    assert output.shape == (32, 64), f"Wrong output shape"
    assert (output >= 0).all(), "ReLU not applied"

    loss = output.sum()
    loss.backward()

    assert module.fc.weight.grad is not None, "weight.grad is None"
    assert module.fc.bias.grad is not None, "bias.grad is None"


def test_fallback_logic():
    """Test fallback to PyTorch when CUDA unavailable"""
    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    # Test on CPU (should always use fallback)
    module_cpu = HypatiaFusedLinearReLU(64, 32, device='cpu')
    x_cpu = torch.randn(16, 64, device='cpu')
    output_cpu = module_cpu(x_cpu)

    assert output_cpu.shape == (16, 32)
    assert (output_cpu >= 0).all(), "ReLU not applied on CPU fallback"


def test_correctness():
    """Test numerical correctness against baseline"""
    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create baseline model
    baseline = nn.Sequential(
        nn.Linear(256, 128, device=device),
        nn.ReLU()
    )

    # Create fused model with same weights
    fused = HypatiaFusedLinearReLU(256, 128, device=device)
    fused.fc.weight.data.copy_(baseline[0].weight.data)
    fused.fc.bias.data.copy_(baseline[0].bias.data)

    # Test input
    x = torch.randn(64, 256, device=device)

    # Forward
    output_baseline = baseline(x)
    output_fused = fused(x)

    max_diff = (output_baseline - output_fused).abs().max().item()
    assert max_diff < 1e-5, f"Forward outputs differ too much: {max_diff}"

    # Backward
    x_base = x.detach().clone().requires_grad_(True)
    x_fused = x.detach().clone().requires_grad_(True)

    y_base = baseline(x_base)
    y_fused = fused(x_fused)

    grad_out = torch.randn_like(y_base)

    y_base.backward(grad_out)
    y_fused.backward(grad_out)

    grad_diff = (x_base.grad - x_fused.grad).abs().max().item()
    assert grad_diff < 1e-5, f"Backward gradients differ too much: {grad_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
