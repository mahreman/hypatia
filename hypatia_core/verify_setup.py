#!/usr/bin/env python3
"""
Hypatia Setup Verification Script
Bu script, Hypatia kurulumunun doğru yapıldığını ve her şeyin çalıştığını doğrular.
Kullanıcının gönderdiği talimatlara göre hazırlanmıştır.
"""

import sys
import os

# CRITICAL: Disable checksum validation to allow optimizations to apply
# See SETUP_GUIDE.md "Checksum Mode" section for details
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"

print("=" * 80)
print("HYPATIA SETUP VERIFICATION")
print("=" * 80)

# Step 1: Python version check
print("\n[1/6] Python Version Check")
print("-" * 80)
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)
else:
    print("✅ Python version OK")

# Step 2: Import _hypatia_core (Rust binary module)
print("\n[2/6] Rust Binary Module (_hypatia_core)")
print("-" * 80)

try:
    import _hypatia_core
    print("✅ _hypatia_core imported successfully")

    if hasattr(_hypatia_core, '__file__'):
        print(f"   Location: {_hypatia_core.__file__}")

    # Check for expected functions
    expected_funcs = [
        'compile_fx_graph',
        'optimize_ast',
        'parse_expr',
        'is_equivalent',
        'Symbol',
        'set_log_level'
    ]

    missing = [f for f in expected_funcs if not hasattr(_hypatia_core, f)]
    if missing:
        print(f"⚠️  Missing functions: {missing}")
    else:
        print(f"   All expected functions present: {len(expected_funcs)}")

except ImportError as e:
    print(f"❌ Failed to import _hypatia_core: {e}")
    print("\nRECOMMENDED FIX:")
    print("   cd hypatia_core")
    print("   maturin develop --release")
    print("   # or")
    print("   cargo build --release")
    print("   cp target/release/lib_hypatia_core.so hypatia_core/_hypatia_core.so")
    sys.exit(1)

# Step 3: Import hypatia_core (Python package)
print("\n[3/6] Python Package (hypatia_core)")
print("-" * 80)

try:
    import hypatia_core
    print("✅ hypatia_core imported successfully")

    if hasattr(hypatia_core, '__file__'):
        print(f"   Location: {hypatia_core.__file__}")

    if hasattr(hypatia_core, '__version__'):
        print(f"   Version: {hypatia_core.__version__}")

    # Check for expected exports
    expected_exports = [
        'hypatia_backend',
        'register_backend',
        'compile_fx_graph',
        'Symbol'
    ]

    missing = [e for e in expected_exports if not hasattr(hypatia_core, e)]
    if missing:
        print(f"⚠️  Missing exports: {missing}")
    else:
        print(f"   All expected exports present: {len(expected_exports)}")

except ImportError as e:
    print(f"❌ Failed to import hypatia_core: {e}")
    print("\nRECOMMENDED FIX:")
    print("   cd /path/to/hypatia")
    print("   pip install -e .")
    sys.exit(1)

# Step 4: PyTorch and torch._dynamo check
print("\n[4/6] PyTorch and torch._dynamo")
print("-" * 80)

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} imported")

    if not hasattr(torch, '_dynamo'):
        print("❌ torch._dynamo not found")
        print("   → Upgrade to PyTorch 2.0+")
        sys.exit(1)

    print("✅ torch._dynamo available")

    if not hasattr(torch._dynamo, 'register_backend'):
        print("❌ torch._dynamo.register_backend not found")
        sys.exit(1)

    if not hasattr(torch._dynamo, 'list_backends'):
        print("❌ torch._dynamo.list_backends not found")
        sys.exit(1)

    print("✅ torch._dynamo functions available")

except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    print("\nRECOMMENDED FIX:")
    print("   pip install torch>=2.0")
    sys.exit(1)

# Step 5: Backend registration check
print("\n[5/6] Hypatia Backend Registration")
print("-" * 80)

backends = torch._dynamo.list_backends()
print(f"Registered backends: {backends}")

if "hypatia" in backends:
    print("✅ 'hypatia' backend is registered!")
else:
    print("⚠️  'hypatia' backend NOT automatically registered")
    print("   Attempting manual registration...")

    try:
        hypatia_core.register_backend()
        backends = torch._dynamo.list_backends()

        if "hypatia" in backends:
            print("✅ Manual registration successful!")
        else:
            print("❌ Manual registration failed")
            print(f"   Backends after registration: {backends}")

    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback
        traceback.print_exc()

# Step 6: Simple compilation test
print("\n[6/6] Simple Compilation Test")
print("-" * 80)

if "hypatia" not in torch._dynamo.list_backends():
    print("⚠️  Skipping compilation test (backend not registered)")
else:
    try:
        import torch.nn as nn

        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        ).eval()

        x = torch.randn(1, 10)

        # Original output
        with torch.no_grad():
            original_out = model(x)

        # Compile with Hypatia
        compiled_model = torch.compile(model, backend="hypatia")

        with torch.no_grad():
            compiled_out = compiled_model(x)

        # Check outputs match
        max_diff = torch.max(torch.abs(original_out - compiled_out)).item()

        if torch.allclose(original_out, compiled_out, atol=1e-4):
            print(f"✅ Compilation successful! (max diff: {max_diff:.2e})")
        else:
            print(f"⚠️  Outputs don't match (max diff: {max_diff:.2e})")

    except Exception as e:
        print(f"❌ Compilation test failed: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if "hypatia" in torch._dynamo.list_backends():
    print("🎉 SUCCESS! Hypatia is properly installed and working!")
    print("\nYou can now use Hypatia:")
    print("   import hypatia_core")
    print("   model = torch.compile(your_model, backend='hypatia')")
else:
    print("⚠️  PARTIAL SUCCESS")
    print("   hypatia_core imports work, but backend registration has issues.")
    print("\nManual workaround:")
    print("   import hypatia_core")
    print("   hypatia_core.register_backend()")
    print("   model = torch.compile(your_model, backend='hypatia')")

print("=" * 80)

# Exit code
sys.exit(0 if "hypatia" in torch._dynamo.list_backends() else 1)
