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

import torch
import torch.nn as nn


def test_import():
    """Test if CUDA extension can be imported"""
    print("=" * 80)
    print("Test 1: Import Test")
    print("=" * 80)

    try:
        from hypatia_core.fused_modules import (
            CUDA_EXTENSION_AVAILABLE,
            FusedLinearReLUFunction,
            HypatiaFusedLinearReLU,
        )
        print(f"✅ Successfully imported fused_modules")
        print(f"   CUDA_EXTENSION_AVAILABLE: {CUDA_EXTENSION_AVAILABLE}")

        if CUDA_EXTENSION_AVAILABLE:
            import hypatia_core._linear_relu_cuda as cuda_ext
            print(f"✅ Successfully imported CUDA extension")
            print(f"   Available functions: {[f for f in dir(cuda_ext) if not f.startswith('_')]}")
            return True, cuda_ext
        else:
            print(f"⚠️  CUDA extension not available (fallback mode)")
            return False, None

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_cuda_kernel_direct(cuda_ext):
    """Test CUDA kernel functions directly"""
    print("\n" + "=" * 80)
    print("Test 2: Direct CUDA Kernel Test")
    print("=" * 80)

    if cuda_ext is None:
        print("⚠️  Skipped (CUDA extension not available)")
        return False

    if not torch.cuda.is_available():
        print("⚠️  Skipped (CUDA device not available)")
        return False

    try:
        # Create test tensors
        batch, in_features, out_features = 32, 128, 64
        x = torch.randn(batch, in_features, device='cuda', dtype=torch.float32)
        weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
        bias = torch.randn(out_features, device='cuda', dtype=torch.float32)

        print(f"Input shapes:")
        print(f"  x: {x.shape}")
        print(f"  weight: {weight.shape}")
        print(f"  bias: {bias.shape}")

        # Test forward
        output = cuda_ext.forward(x, weight, bias)
        print(f"\n✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")

        # Verify output shape
        assert output.shape == (batch, out_features), f"Wrong output shape: {output.shape}"

        # Verify ReLU applied (no negative values)
        assert (output >= 0).all(), "ReLU not applied correctly (found negative values)"
        print(f"✅ ReLU verification passed (no negative values)")

        # Test backward
        grad_out = torch.randn_like(output)
        grad_x, grad_w, grad_b = cuda_ext.backward(grad_out, x, weight, bias, output)

        print(f"\n✅ Backward pass successful")
        print(f"   grad_x shape: {grad_x.shape}")
        print(f"   grad_w shape: {grad_w.shape}")
        print(f"   grad_b shape: {grad_b.shape}")

        # Verify gradient shapes
        assert grad_x.shape == x.shape, f"Wrong grad_x shape"
        assert grad_w.shape == weight.shape, f"Wrong grad_w shape"
        assert grad_b.shape == bias.shape, f"Wrong grad_b shape"

        print(f"✅ All gradient shapes correct")

        return True

    except Exception as e:
        print(f"❌ CUDA kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autograd_function():
    """Test FusedLinearReLUFunction autograd integration"""
    print("\n" + "=" * 80)
    print("Test 3: Autograd Function Test")
    print("=" * 80)

    from hypatia_core.fused_modules import (
        CUDA_EXTENSION_AVAILABLE,
        FusedLinearReLUFunction,
    )

    if not CUDA_EXTENSION_AVAILABLE or not torch.cuda.is_available():
        print("⚠️  Skipped (CUDA not available)")
        return False

    try:
        # Create test tensors with grad
        batch, in_features, out_features = 16, 64, 32
        x = torch.randn(batch, in_features, device='cuda', requires_grad=True)
        weight = torch.randn(out_features, in_features, device='cuda', requires_grad=True)
        bias = torch.randn(out_features, device='cuda', requires_grad=True)

        # Forward
        output = FusedLinearReLUFunction.apply(x, weight, bias)
        print(f"✅ Forward pass successful: {output.shape}")

        # Backward
        loss = output.sum()
        loss.backward()

        print(f"✅ Backward pass successful")
        print(f"   x.grad: {x.grad.shape}, finite: {torch.isfinite(x.grad).all()}")
        print(f"   weight.grad: {weight.grad.shape}, finite: {torch.isfinite(weight.grad).all()}")
        print(f"   bias.grad: {bias.grad.shape}, finite: {torch.isfinite(bias.grad).all()}")

        # Verify all gradients are finite
        assert torch.isfinite(x.grad).all(), "x.grad contains inf/nan"
        assert torch.isfinite(weight.grad).all(), "weight.grad contains inf/nan"
        assert torch.isfinite(bias.grad).all(), "bias.grad contains inf/nan"

        print(f"✅ All gradients are finite")

        return True

    except Exception as e:
        print(f"❌ Autograd function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_integration():
    """Test HypatiaFusedLinearReLU module"""
    print("\n" + "=" * 80)
    print("Test 4: Module Integration Test")
    print("=" * 80)

    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    try:
        # Create module
        module = HypatiaFusedLinearReLU(128, 64, bias=True, device=device)
        print(f"✅ Module created: {module}")

        # Test forward
        x = torch.randn(32, 128, device=device)
        output = module(x)

        print(f"✅ Forward pass successful: {output.shape}")
        assert output.shape == (32, 64), f"Wrong output shape"
        assert (output >= 0).all(), "ReLU not applied"

        # Test backward
        loss = output.sum()
        loss.backward()

        print(f"✅ Backward pass successful")
        assert module.weight.grad is not None, "weight.grad is None"
        assert module.bias.grad is not None, "bias.grad is None"

        print(f"✅ Module integration test passed")

        return True

    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_logic():
    """Test fallback to PyTorch when CUDA unavailable"""
    print("\n" + "=" * 80)
    print("Test 5: Fallback Logic Test")
    print("=" * 80)

    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    try:
        # Test on CPU (should always use fallback)
        module_cpu = HypatiaFusedLinearReLU(64, 32, device='cpu')
        x_cpu = torch.randn(16, 64, device='cpu')
        output_cpu = module_cpu(x_cpu)

        print(f"✅ CPU fallback working: {output_cpu.shape}")

        # Test with unsupported dtype (should use fallback even on CUDA)
        if torch.cuda.is_available():
            module_cuda = HypatiaFusedLinearReLU(64, 32, device='cuda', dtype=torch.float64)
            x_fp64 = torch.randn(16, 64, device='cuda', dtype=torch.float64)
            output_fp64 = module_cuda(x_fp64)

            print(f"✅ FP64 fallback working: {output_fp64.shape}")

        print(f"✅ Fallback logic test passed")

        return True

    except Exception as e:
        print(f"❌ Fallback logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correctness():
    """Test numerical correctness against baseline"""
    print("\n" + "=" * 80)
    print("Test 6: Numerical Correctness Test")
    print("=" * 80)

    from hypatia_core.fused_modules import HypatiaFusedLinearReLU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Create baseline model
        baseline = nn.Sequential(
            nn.Linear(256, 128, device=device),
            nn.ReLU()
        )

        # Create fused model with same weights
        fused = HypatiaFusedLinearReLU(256, 128, device=device)
        fused.weight.data.copy_(baseline[0].weight.data)
        fused.bias.data.copy_(baseline[0].bias.data)

        # Test input
        x = torch.randn(64, 256, device=device)

        # Forward
        output_baseline = baseline(x)
        output_fused = fused(x)

        max_diff = (output_baseline - output_fused).abs().max().item()
        print(f"Max forward difference: {max_diff:.2e}")

        assert max_diff < 1e-5, f"Forward outputs differ too much: {max_diff}"
        print(f"✅ Forward pass matches baseline (tol=1e-5)")

        # Backward
        x_base = x.detach().clone().requires_grad_(True)
        x_fused = x.detach().clone().requires_grad_(True)

        y_base = baseline(x_base)
        y_fused = fused(x_fused)

        grad_out = torch.randn_like(y_base)

        y_base.backward(grad_out)
        y_fused.backward(grad_out)

        grad_diff = (x_base.grad - x_fused.grad).abs().max().item()
        print(f"Max backward difference: {grad_diff:.2e}")

        assert grad_diff < 1e-5, f"Backward gradients differ too much: {grad_diff}"
        print(f"✅ Backward pass matches baseline (tol=1e-5)")

        print(f"✅ Numerical correctness test passed")

        return True

    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("CUDA Extension Build and Integration Test Suite")
    print("=" * 80)
    print()

    results = []

    # Test 1: Import
    cuda_available, cuda_ext = test_import()
    results.append(("Import Test", cuda_available or cuda_ext is not None))

    # Test 2: Direct CUDA kernel
    results.append(("CUDA Kernel Direct Test", test_cuda_kernel_direct(cuda_ext)))

    # Test 3: Autograd function
    results.append(("Autograd Function Test", test_autograd_function()))

    # Test 4: Module integration
    results.append(("Module Integration Test", test_module_integration()))

    # Test 5: Fallback logic
    results.append(("Fallback Logic Test", test_fallback_logic()))

    # Test 6: Numerical correctness
    results.append(("Numerical Correctness Test", test_correctness()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
