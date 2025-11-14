#!/usr/bin/env python3
"""
HYPATIA FX GRAPH COMPILATION - PHASE 1 QUICK TEST
==================================================
Phase 1 identity pass-through'u test eder.

Kullanım:
    python test_fx_phase1.py
"""

import sys
import torch
import torch.nn as nn

print("=" * 80)
print("HYPATIA FX GRAPH COMPILATION - PHASE 1 TEST")
print("=" * 80)

# Step 1: Hypatia import
print("\n[1/5] Importing hypatia_core...")
try:
    import hypatia_core
    print("✅ hypatia_core imported successfully")
except ImportError as e:
    print(f"❌ Failed to import hypatia_core: {e}")
    print("\n💡 Build Hypatia first:")
    print("   cd ~/hypatia/hypatia_core")
    print("   maturin develop --release")
    sys.exit(1)

# Step 2: Check compile_fx_graph
print("\n[2/5] Checking compile_fx_graph availability...")
if hasattr(hypatia_core, 'compile_fx_graph'):
    print("✅ compile_fx_graph function is available")
else:
    print("❌ compile_fx_graph not found!")
    print("\n💡 Make sure you copied the updated files:")
    print("   - fx_bridge.rs")
    print("   - python_bindings.rs (updated)")
    print("   - lib.rs (updated)")
    sys.exit(1)

# Step 3: Create test model
print("\n[3/5] Creating test model...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()
example_input = torch.randn(2, 10)

print("✅ Model created:")
print(f"   - Input shape: {example_input.shape}")
print(f"   - Model: Linear(10,20) → ReLU → Linear(20,5)")

# Step 4: FX Trace
print("\n[4/5] Creating FX GraphModule...")
try:
    traced = torch.fx.symbolic_trace(model)
    print("✅ FX GraphModule created")
    print(f"\n   Graph structure:")
    print("   " + "\n   ".join(str(traced.graph).split('\n')))
except Exception as e:
    print(f"❌ FX trace failed: {e}")
    sys.exit(1)

# Step 5: compile_fx_graph
print("\n[5/5] Testing compile_fx_graph (Phase 1)...")
print("   (Check stderr for debug output)")

try:
    optimized_gm = hypatia_core.compile_fx_graph(traced)
    print("✅ compile_fx_graph returned successfully")
except Exception as e:
    print(f"❌ compile_fx_graph failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Numerical validation
print("\n[Validation] Comparing outputs...")
with torch.no_grad():
    out_original = traced(example_input)
    out_optimized = optimized_gm(example_input)
    
    max_diff = torch.max(torch.abs(out_original - out_optimized)).item()
    
    print(f"   Original output shape: {out_original.shape}")
    print(f"   Optimized output shape: {out_optimized.shape}")
    print(f"   Max absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✅ Numerical validation PASSED (identity)")
    elif max_diff < 1e-6:
        print("⚠️  Small numerical difference (acceptable for Phase 1)")
    else:
        print(f"❌ FAILED: Difference too large! ({max_diff})")
        sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("PHASE 1 TEST SUMMARY")
print("=" * 80)
print("✅ hypatia_core import: OK")
print("✅ compile_fx_graph available: OK")
print("✅ FX trace: OK")
print("✅ compile_fx_graph execution: OK")
print("✅ Numerical validation: OK")
print("\n🎉 PHASE 1 TEST PASSED!")
print("\nNext steps:")
print("1. Run benchmark_harness_v2.py to see it in action")
print("2. Implement Phase 2 (real FX parsing)")
print("3. Add optimization rules")
print("=" * 80)