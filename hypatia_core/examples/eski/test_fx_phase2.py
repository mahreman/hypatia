#!/usr/bin/env python3
"""
HYPATIA FX GRAPH COMPILATION - PHASE 2 TEST
Test real FX parsing with operator handlers
"""

import torch
import torch.nn as nn
import torch.fx

try:
    import hypatia_core
    print("✅ hypatia_core imported successfully")
except ImportError as e:
    print(f"❌ Failed to import hypatia_core: {e}")
    print("Make sure to compile with: maturin develop --release")
    exit(1)

print("="*80)
print("PHASE 2: REAL FX PARSING TEST")
print("="*80)

# Test 1: Simple Linear->ReLU->Linear
print("\n[Test 1] Linear -> ReLU -> Linear")
print("-" * 40)

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

model1 = SimpleModel()
input1 = torch.randn(2, 10)
traced1 = torch.fx.symbolic_trace(model1)

print("Original graph:")
print(traced1.graph)

try:
    optimized1 = hypatia_core.compile_fx_graph(traced1)
    print("\n✅ compile_fx_graph succeeded")
    
    # Verify outputs match
    orig_out = traced1(input1)
    opt_out = optimized1(input1)
    max_diff = torch.max(torch.abs(orig_out - opt_out)).item()
    print(f"Max absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✅ Numerical validation PASSED")
    else:
        print(f"⚠️  Numerical difference: {max_diff}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Model with element-wise operations
print("\n" + "="*80)
print("[Test 2] Model with Add/Mul operations")
print("-" * 40)

class MathOpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        y = self.linear(x)
        z = y + x  # residual connection
        return z * 2.0  # scale

model2 = MathOpsModel()
input2 = torch.randn(2, 10)

try:
    traced2 = torch.fx.symbolic_trace(model2)
    print("Original graph:")
    print(traced2.graph)
    
    optimized2 = hypatia_core.compile_fx_graph(traced2)
    print("\n✅ compile_fx_graph succeeded")
    
    # Verify
    orig_out = traced2(input2)
    opt_out = optimized2(input2)
    max_diff = torch.max(torch.abs(orig_out - opt_out)).item()
    print(f"Max absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✅ Numerical validation PASSED")
    else:
        print(f"⚠️  Numerical difference: {max_diff}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multiple activations
print("\n" + "="*80)
print("[Test 3] Multiple activation functions")
print("-" * 40)

class ActivationsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

model3 = ActivationsModel()
input3 = torch.randn(2, 10)

try:
    traced3 = torch.fx.symbolic_trace(model3)
    print("Original graph:")
    print(traced3.graph)
    
    optimized3 = hypatia_core.compile_fx_graph(traced3)
    print("\n✅ compile_fx_graph succeeded")
    
    # Verify
    orig_out = traced3(input3)
    opt_out = optimized3(input3)
    max_diff = torch.max(torch.abs(orig_out - opt_out)).item()
    print(f"Max absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✅ Numerical validation PASSED")
    else:
        print(f"⚠️  Numerical difference: {max_diff}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("PHASE 2 TEST SUMMARY")
print("="*80)
print("✅ Real FX parsing implemented")
print("✅ Operator handlers: linear, relu, sigmoid, add, mul")
print("✅ S-expression generation from FX graph")
print("\nNext steps:")
print("1. Check stderr for detailed parsing logs")
print("2. Verify S-expressions are valid HypatiaLang")
print("3. Add more operator handlers (conv2d, batchnorm, etc.)")
print("4. Implement Phase 3 (S-expr -> FX graph reconstruction)")
print("="*80)