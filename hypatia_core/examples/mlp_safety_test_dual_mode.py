"""
MLP Safety/Correctness Test - Dual Mode

Tests Hypatia compilation in TWO modes:

1. STRICT Mode (Production):
   - Checksum validation enabled
   - Falls back to _orig_mod if mismatch detected
   - Safety guaranteed, may miss fusion opportunities

2. OFF Mode (Testing):
   - Checksum validation disabled
   - Fusion allowed to proceed
   - Numerical diff verification (eager vs fused)
   - Validates end-to-end correctness of fusion pipeline

This allows us to:
- Verify fusion correctness without checksum blocking
- Validate CUDA kernel + E-graph + FX reconstruction together
- Ensure numerical accuracy of full optimized path
"""

import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend


class MLP(nn.Module):
    """Test MLP with multiple Linear+ReLU patterns"""
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2 = self.act(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3, (x1, x2, x3)  # Return intermediates for debugging


def dump_weight_norms(model, prefix):
    """Debug helper: print weight norms for each layer"""
    print(f"\n[{prefix}] weight norms:")
    for name, param in model.named_parameters():
        norm_val = param.norm().item() if hasattr(param, 'norm') else 0.0
        print(f"  {name:30s}  ||param|| = {norm_val:.6f}")


def run_test_mode(mode_name, checksum_mode, device):
    """
    Run test in specified mode

    Args:
        mode_name: "STRICT" or "OFF"
        checksum_mode: "strict" or "off"
        device: torch device
    """
    print("\n" + "=" * 80)
    print(f"Testing in {mode_name} Mode")
    print("=" * 80)
    print(f"Checksum Mode: {checksum_mode}")
    print()

    # Set environment
    os.environ["HYPATIA_CHECKSUM_MODE"] = checksum_mode
    os.environ["HYPATIA_ENABLE_LINRELU_FUSION"] = "1"
    os.environ["HYPATIA_DEBUG_FX"] = "0"  # Clean output

    # Create fresh model for this mode
    model = MLP().to(device)
    model.eval()
    x = torch.randn(32, 784, device=device)

    # Step 1: Eager (reference)
    print("Step 1: Running eager (reference) model...")
    with torch.no_grad():
        y_ref, (ref_x1, ref_x2, ref_x3) = model(x)

    if torch.isnan(y_ref).any():
        print("❌ ERROR: Reference output contains NaN!")
        return None

    print(f"  Reference output shape: {y_ref.shape}")
    print(f"  Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Step 2: Compile
    print("\nStep 2: Compiling with Hypatia backend...")
    opt = torch.compile(model, backend="hypatia")

    # Step 3: Run compiled
    print("\nStep 3: Running compiled model...")
    with torch.no_grad():
        y_opt, (opt_x1, opt_x2, opt_x3) = opt(x)

    if torch.isnan(y_opt).any():
        print("❌ ERROR: Optimized output contains NaN!")
        return None

    print(f"  Optimized output shape: {y_opt.shape}")
    print(f"  Optimized output range: [{y_opt.min():.4f}, {y_opt.max():.4f}]")

    # Step 4: Compare outputs
    print("\nStep 4: Numerical Comparison")
    print("-" * 80)

    abs_diff = (y_ref - y_opt).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"  Final output:")
    print(f"    Max absolute difference:  {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")

    # Layer-by-layer diff
    print(f"\n  Layer-by-layer differences:")
    layer1_diff = (ref_x1 - opt_x1).abs().max().item()
    layer2_diff = (ref_x2 - opt_x2).abs().max().item()
    layer3_diff = (ref_x3 - opt_x3).abs().max().item()
    print(f"    After fc1+relu: {layer1_diff:.6e}")
    print(f"    After fc2+relu: {layer2_diff:.6e}")
    print(f"    After fc3:      {layer3_diff:.6e}")

    # Validate
    TOLERANCE = 1e-5
    passed = max_diff < TOLERANCE

    print("\nStep 5: Validation")
    print("-" * 80)
    if passed:
        print(f"✅ PASS: Max difference ({max_diff:.6e}) < tolerance ({TOLERANCE})")
    else:
        print(f"❌ FAIL: Max difference ({max_diff:.6e}) >= tolerance ({TOLERANCE})")

    # Mode-specific interpretation
    print("\nInterpretation:")
    if mode_name == "STRICT":
        if passed:
            print("  ✅ Checksum validation allows fusion (or fallback works correctly)")
        else:
            print("  ⚠️  Possible checksum issue or numerical instability")
    else:  # OFF mode
        if passed:
            print("  ✅ Fusion pipeline produces numerically correct results")
            print("     (CUDA kernel + E-graph + FX reconstruction all working)")
        else:
            print("  ❌ Fusion pipeline has numerical errors")
            print("     Check: CUDA kernel correctness, parameter binding, graph reconstruction")

    return {
        'mode': mode_name,
        'checksum': checksum_mode,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'layer_diffs': [layer1_diff, layer2_diff, layer3_diff],
        'passed': passed,
    }


def main():
    print("=" * 80)
    print("MLP Safety Test - Dual Mode (STRICT + OFF)")
    print("=" * 80)
    print()
    print("This test verifies Hypatia fusion in two modes:")
    print("  1. STRICT: Production mode with checksum validation")
    print("  2. OFF:    Test mode for end-to-end fusion correctness")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    results = []

    # Test 1: STRICT mode (production)
    strict_result = run_test_mode("STRICT", "strict", device)
    if strict_result:
        results.append(strict_result)

    # Test 2: OFF mode (testing - allows fusion)
    off_result = run_test_mode("OFF", "off", device)
    if off_result:
        results.append(off_result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results:
        print("❌ No tests completed successfully")
        return 1

    print(f"\n{'Mode':<10} {'Checksum':<10} {'Max Diff':<15} {'Mean Diff':<15} {'Status':<10}")
    print("-" * 80)
    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"{r['mode']:<10} {r['checksum']:<10} {r['max_diff']:<15.6e} "
              f"{r['mean_diff']:<15.6e} {status:<10}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if len(results) == 2:
        strict_pass = results[0]['passed']
        off_pass = results[1]['passed']

        if strict_pass and off_pass:
            print("✅ EXCELLENT: Both modes pass")
            print("   - Production (STRICT) is safe")
            print("   - Fusion (OFF) is numerically correct")
            print("   - Full pipeline validated")

        elif not strict_pass and off_pass:
            print("⚠️  INTERESTING: STRICT fails but OFF passes")
            print("   - Fusion itself is correct")
            print("   - But checksum validation blocks it in STRICT mode")
            print("   - May need to adjust checksum logic or fusion rules")

        elif strict_pass and not off_pass:
            print("⚠️  CONCERNING: STRICT passes but OFF fails")
            print("   - STRICT likely fell back to _orig_mod (safe)")
            print("   - But fusion pipeline has numerical errors")
            print("   - Need to fix fusion implementation")

        else:
            print("❌ CRITICAL: Both modes fail")
            print("   - Numerical instability detected")
            print("   - Need to investigate model or compilation")

        # Diff comparison
        diff_change = results[1]['max_diff'] - results[0]['max_diff']
        print(f"\nNumerical difference delta (OFF - STRICT): {diff_change:.6e}")
        if abs(diff_change) < 1e-7:
            print("  → Negligible difference (likely same code path)")
        elif diff_change < 0:
            print("  → OFF mode more accurate (unexpected)")
        else:
            print("  → OFF mode less accurate (fusion introduces small error)")

    print("\n" + "=" * 80)

    # Return code
    all_pass = all(r['passed'] for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
