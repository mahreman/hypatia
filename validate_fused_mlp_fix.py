#!/usr/bin/env python3
"""
Validation script for FusedMLP implementation fix.

This script validates the code structure without running tests,
useful when PyTorch is not available.
"""

import os
import sys
import re


def check_file_exists(filepath):
    """Check if file exists and return status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} File exists: {filepath}")
    return exists


def check_pattern_in_file(filepath, pattern, description):
    """Check if pattern exists in file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            found = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            status = "✅" if found else "❌"
            print(f"{status} {description}")
            return found is not None
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False


def validate_python_module():
    """Validate fused_modules.py has correct implementation."""
    print("\n=== Validating hypatia_core/fused_modules.py ===")

    filepath = "hypatia_core/hypatia_core/fused_modules.py"

    checks = [
        (r"class FusedMLP\(nn\.Module\):", "FusedMLP class defined"),
        (r"self\.layer1 = create_fused_linear_relu_from_tensors\(weight1, bias1\)",
         "Layer 1 uses create_fused_linear_relu_from_tensors"),
        (r"self\.layer2 = nn\.Linear\(", "Layer 2 is nn.Linear"),
        (r"def create_fused_mlp_from_tensors\(", "create_fused_mlp_from_tensors helper exists"),
        (r'"FusedMLP".*"create_fused_mlp_from_tensors"', "Both exported in __all__"),
        (r"x = self\.layer1\(x\)\s+x = self\.layer2\(x\)", "Forward pass uses both layers"),
    ]

    all_passed = check_file_exists(filepath)

    for pattern, description in checks:
        all_passed = check_pattern_in_file(filepath, pattern, description) and all_passed

    return all_passed


def validate_rust_reconstruction():
    """Validate fx_bridge.rs has correct reconstruction logic."""
    print("\n=== Validating hypatia_core/src/fx_bridge.rs ===")

    filepath = "hypatia_core/src/fx_bridge.rs"

    checks = [
        (r"fn reconstruct_fused_mlp\(",
         "reconstruct_fused_mlp helper method exists"),
        (r"w1_id: Id, b1_id: Id,\s+w2_id: Id, b2_id: Id,\s+x_id: Id",
         "Helper has correct parameters (w1, b1, w2, b2, x)"),
        (r'hypatia_core\.getattr\("create_fused_mlp_from_tensors"\)',
         "Calls Python helper create_fused_mlp_from_tensors"),
        (r"HypatiaLang::FusedMLP\(ids\) => \{[^}]*self\.reconstruct_fused_mlp",
         "FusedMLP case uses reconstruct_fused_mlp helper"),
        (r'log::info!\("✅ Created fused MLP module:',
         "Logs successful FusedMLP creation"),
    ]

    all_passed = check_file_exists(filepath)

    for pattern, description in checks:
        all_passed = check_pattern_in_file(filepath, pattern, description) and all_passed

    return all_passed


def validate_no_old_implementation():
    """Validate old broken implementation is removed."""
    print("\n=== Validating old implementation is removed ===")

    filepath = "hypatia_core/src/fx_bridge.rs"

    # These patterns should NOT exist anymore
    bad_patterns = [
        (r'nn\.getattr\("Sequential"\)\?\.call1\(\(linear1, relu, linear2\)\)',
         "Old Sequential(linear1, relu, linear2) removed"),
        (r"let sequential = nn\.getattr",
         "No manual sequential creation in FusedMLP case"),
    ]

    all_passed = True

    with open(filepath, 'r') as f:
        content = f.read()

        # Find the FusedMLP case
        mlp_case = re.search(
            r'HypatiaLang::FusedMLP\(ids\) => \{(.*?)\},',
            content,
            re.MULTILINE | re.DOTALL
        )

        if mlp_case:
            mlp_code = mlp_case.group(1)

            # Check that old patterns are NOT in the FusedMLP case
            for pattern, description in bad_patterns:
                found = re.search(pattern, mlp_code)
                status = "✅" if not found else "❌"
                print(f"{status} {description}")
                all_passed = all_passed and (not found)
        else:
            print("❌ Could not find FusedMLP case")
            all_passed = False

    return all_passed


def check_build_status():
    """Check if Rust build succeeded."""
    print("\n=== Checking build status ===")

    release_lib = "hypatia_core/target/release/libhypatia_core.so"
    exists = os.path.exists(release_lib)

    if exists:
        print(f"✅ Release build exists: {release_lib}")
        # Check modification time
        mtime = os.path.getmtime(release_lib)
        import time
        age_seconds = time.time() - mtime
        age_minutes = age_seconds / 60

        if age_minutes < 10:
            print(f"✅ Build is recent (%.1f minutes old)" % age_minutes)
        else:
            print(f"⚠️  Build is %.1f minutes old (consider rebuilding)" % age_minutes)
        return True
    else:
        print(f"❌ Release build not found: {release_lib}")
        print("   Run: cd hypatia_core && cargo build --release")
        return False


def main():
    """Run all validations."""
    print("=" * 70)
    print("FusedMLP Implementation Fix - Validation")
    print("=" * 70)

    os.chdir("/home/user/hypatia")

    results = []

    results.append(("Python module", validate_python_module()))
    results.append(("Rust reconstruction", validate_rust_reconstruction()))
    results.append(("Old code removed", validate_no_old_implementation()))
    results.append(("Build status", check_build_status()))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        all_passed = all_passed and passed

    print("=" * 70)

    if all_passed:
        print("✅ All validations passed!")
        print("\nNext steps:")
        print("1. Run tests in PyTorch environment:")
        print("   cd hypatia_core/examples && python mlp_safety_test_dual_mode.py")
        print("2. Verify numerical accuracy improves to < 1e-5")
        print("3. Check performance benchmarks")
        return 0
    else:
        print("❌ Some validations failed!")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
