# Safety Testing Guide - Checksum Modes & Fusion Verification

## Overview

Hypatia uses a **dual-mode safety system** to ensure numerical correctness while enabling aggressive optimizations:

1. **STRICT Mode** (Production): Checksum validation prevents incorrect optimizations
2. **OFF Mode** (Testing): Allows fusion to proceed for end-to-end validation

## Checksum Modes

### STRICT Mode (Default for Production)

**Purpose:** Safety-first approach

**Behavior:**
```python
os.environ["HYPATIA_CHECKSUM_MODE"] = "strict"

# Compilation flow:
1. E-graph optimizer applies fusion rules
2. FX graph reconstructor rebuilds PyTorch graph
3. Checksum validator compares eager vs optimized outputs
4. If mismatch detected → fallback to _orig_mod (safe)
5. If match → use optimized graph
```

**Pros:**
- ✅ Guaranteed numerical safety
- ✅ Automatic fallback on any issue
- ✅ Safe for production

**Cons:**
- ⚠️ May block valid optimizations if checksum is too strict
- ⚠️ Fusion might not activate even when correct

### OFF Mode (Testing Only)

**Purpose:** Validate fusion correctness without checksum blocking

**Behavior:**
```python
os.environ["HYPATIA_CHECKSUM_MODE"] = "off"

# Compilation flow:
1. E-graph optimizer applies fusion rules
2. FX graph reconstructor rebuilds PyTorch graph
3. No checksum validation → use optimized graph
4. Numerical validation done manually in tests
```

**Pros:**
- ✅ Fusion always proceeds (no blocking)
- ✅ Can validate full optimized path
- ✅ Tests CUDA kernel + E-graph + FX reconstruction together

**Cons:**
- ⚠️ No automatic safety net
- ⚠️ Must manually verify numerical correctness
- ❌ Never use in production!

## Test Suite Structure

### 1. Basic Safety Test (STRICT)

**File:** `examples/mlp_safety_test.py`

**Tests:**
- Checksum validation in STRICT mode
- Numerical correctness (eager vs compiled)
- Layer-by-layer differences
- Weight norm preservation

**Usage:**
```bash
cd hypatia_core/examples
python mlp_safety_test.py
```

**Expected Output (STRICT mode):**
```
Checksum Mode: strict
✅ PASS: Max difference (1.234e-06) < tolerance (1e-05)
```

### 2. Dual-Mode Safety Test

**File:** `examples/mlp_safety_test_dual_mode.py`

**Tests both modes side-by-side:**
- STRICT: Production safety
- OFF: Fusion correctness

**Usage:**
```bash
cd hypatia_core/examples
python mlp_safety_test_dual_mode.py
```

**Expected Output:**
```
Testing in STRICT Mode
✅ PASS: Max difference < tolerance

Testing in OFF Mode
✅ PASS: Max difference < tolerance

ANALYSIS
✅ EXCELLENT: Both modes pass
   - Production (STRICT) is safe
   - Fusion (OFF) is numerically correct
   - Full pipeline validated
```

### 3. CUDA Extension Tests

**File:** `tests/test_cuda_extension.py`

**Tests CUDA kernel directly:**
- Forward/backward kernel correctness
- Autograd integration
- Fallback logic
- Numerical accuracy

**Usage:**
```bash
cd hypatia_core/tests
python test_cuda_extension.py
```

## Test Scenarios & Interpretation

### Scenario 1: Both STRICT and OFF Pass ✅

```
STRICT: ✅ PASS (max_diff: 1.2e-06)
OFF:    ✅ PASS (max_diff: 1.3e-06)
```

**Interpretation:**
- ✅ Production is safe
- ✅ Fusion is numerically correct
- ✅ Checksum allows fusion (or fallback works)
- **Action:** Ship it! 🚀

### Scenario 2: STRICT Fails, OFF Passes ⚠️

```
STRICT: ❌ FAIL (max_diff: 2.5e-04)
OFF:    ✅ PASS (max_diff: 1.3e-06)
```

**Interpretation:**
- ✅ Fusion itself is correct
- ⚠️ But checksum validation blocks it in STRICT
- Possible causes:
  - Checksum tolerance too strict
  - Numerical instability in checksum computation
  - FX graph reconstruction issue

**Action:**
1. Check checksum implementation
2. Verify fusion rules don't introduce numerical drift
3. Consider adjusting checksum tolerance
4. Validate that OFF mode truly tests the fused path

### Scenario 3: STRICT Passes, OFF Fails ⚠️

```
STRICT: ✅ PASS (max_diff: 1.2e-06)
OFF:    ❌ FAIL (max_diff: 8.7e-03)
```

**Interpretation:**
- ✅ STRICT likely fell back to _orig_mod (safe)
- ❌ Fusion pipeline has numerical errors
- The fusion implementation is broken!

**Action:**
1. Check CUDA kernel correctness
2. Verify parameter binding in FX reconstruction
3. Test E-graph rewrite rules
4. Validate autograd gradients

### Scenario 4: Both Fail ❌

```
STRICT: ❌ FAIL (max_diff: 1.2e-03)
OFF:    ❌ FAIL (max_diff: 1.3e-03)
```

**Interpretation:**
- ❌ Model or compilation has fundamental issues
- Not fusion-specific

**Action:**
1. Check model definition
2. Verify input data quality
3. Test with simpler model
4. Check for NaN/inf in weights

## Validation Workflow

### Development Workflow

```bash
# 1. Write fusion code (CUDA kernel, E-graph rules, etc.)

# 2. Test CUDA kernel directly
cd hypatia_core/tests
python test_cuda_extension.py

# 3. Test with OFF mode (no checksum blocking)
cd ../examples
python mlp_safety_test_dual_mode.py
# → Should see OFF mode pass

# 4. Test with STRICT mode (production safety)
# Same script tests both
# → Should see STRICT mode pass

# 5. If STRICT fails but OFF passes:
#    - Adjust checksum logic
#    - OR: Accept that fusion won't activate in STRICT
#    - OR: Fix fusion to match eager exactly

# 6. Benchmark performance
python mlp_multiconfig_benchmark.py
```

### Integration Testing

```bash
# Full test suite
cd hypatia_core

# 1. Unit tests
python tests/test_cuda_extension.py
python tests/test_backend_registration.py
python tests/mlp_safety_test.py

# 2. Safety tests
python examples/mlp_safety_test.py          # STRICT only
python examples/mlp_safety_test_dual_mode.py  # STRICT + OFF

# 3. Performance tests
python examples/mlp_multiconfig_benchmark.py
python benchmarks/mlp_fusion_benchmark.py

# 4. Large-scale test
python examples/mlp_perf_test.py
```

## Environment Variables

### Core Settings

```bash
# Checksum mode
export HYPATIA_CHECKSUM_MODE="strict"  # or "off"

# Fusion control
export HYPATIA_ENABLE_LINRELU_FUSION="1"  # Enable Linear+ReLU fusion

# Debug output
export HYPATIA_DEBUG_FX="0"  # Clean output
export HYPATIA_DEBUG_FX="1"  # Verbose debugging
```

### Test Configurations

```bash
# Production testing (default)
export HYPATIA_CHECKSUM_MODE="strict"
export HYPATIA_ENABLE_LINRELU_FUSION="1"
python examples/mlp_safety_test.py

# Fusion validation (testing only)
export HYPATIA_CHECKSUM_MODE="off"
export HYPATIA_ENABLE_LINRELU_FUSION="1"
python examples/mlp_safety_test_dual_mode.py

# Disable fusion (baseline)
export HYPATIA_ENABLE_LINRELU_FUSION="0"
python examples/mlp_multiconfig_benchmark.py
```

## Numerical Tolerances

### Standard Tolerances

```python
# Forward pass
FORWARD_ATOL = 1e-5  # Absolute tolerance
FORWARD_RTOL = 1e-5  # Relative tolerance

# Backward pass (gradients)
BACKWARD_ATOL = 1e-5
BACKWARD_RTOL = 1e-5

# Checksum validation (may be stricter)
CHECKSUM_ATOL = 1e-6
```

### Why Different Tolerances?

- **Forward:** Accumulation of floating-point errors in GEMM
- **Backward:** Additional errors from chain rule
- **Checksum:** Needs to be tight enough to catch real errors, loose enough to allow valid optimizations

## Debugging Failed Tests

### Step 1: Isolate the Issue

```bash
# Test each component separately

# 1. CUDA kernel alone
python tests/test_cuda_extension.py
# → If fails: CUDA kernel bug

# 2. E-graph optimizer alone
python -c "
from hypatia_core.src.egraph_optimizer import optimize_to_ast
result = optimize_to_ast('(relu (linear w b x))')
print(result)
"
# → If wrong: E-graph rewrite issue

# 3. Full pipeline
python examples/mlp_safety_test_dual_mode.py
# → If OFF fails: Integration issue
# → If STRICT fails but OFF passes: Checksum issue
```

### Step 2: Check Intermediate Values

```python
# Add debug prints in mlp_safety_test_dual_mode.py

# After eager:
print(f"Eager output stats: min={y_ref.min()}, max={y_ref.max()}, mean={y_ref.mean()}")

# After optimized:
print(f"Optimized output stats: min={y_opt.min()}, max={y_opt.max()}, mean={y_opt.mean()}")

# Per-layer:
for i, (ref, opt) in enumerate([(ref_x1, opt_x1), (ref_x2, opt_x2)]):
    diff = (ref - opt).abs().max()
    print(f"Layer {i+1} max diff: {diff:.6e}")
```

### Step 3: Profile Execution

```bash
# Check if fusion actually activated
export HYPATIA_DEBUG_FX="1"
python examples/mlp_safety_test_dual_mode.py 2>&1 | grep -i "fused"

# Expected in OFF mode:
# "Applying fusion rule: linear-relu-fusion"
# "Reconstructed FusedLinearReLU module"

# In STRICT mode (if checksum blocks):
# "Checksum mismatch, using _orig_mod"
```

## Best Practices

### Development

1. ✅ Always test with `test_cuda_extension.py` first
2. ✅ Use OFF mode to validate fusion correctness
3. ✅ Use STRICT mode to test production safety
4. ✅ Compare STRICT vs OFF results in dual-mode test
5. ✅ Document any numerical drift

### Production

1. ✅ Always use STRICT mode
2. ✅ Set `HYPATIA_DEBUG_FX="0"` for clean logs
3. ✅ Monitor for checksum mismatches
4. ❌ Never use OFF mode

### Testing

1. ✅ Run full test suite before commits
2. ✅ Test on both CPU and CUDA
3. ✅ Test multiple batch sizes
4. ✅ Validate gradients in backward pass
5. ✅ Benchmark before/after changes

## FAQ

**Q: Why does STRICT mode give different results than OFF?**
A: STRICT may fall back to `_orig_mod` if checksum fails. OFF always uses optimized path.

**Q: Should I adjust tolerances if tests fail?**
A: Only if you're sure the numerical difference is acceptable. Better to fix the root cause.

**Q: Can I use OFF mode in production?**
A: ❌ NO! OFF mode has no safety net. Only for testing.

**Q: What if fusion passes tests but is slower?**
A: Check problem size. Fusion helps in compute-bound regime (large batch, large hidden). Small models may be overhead-bound.

**Q: How do I know if CUDA kernel is actually being used?**
A: Check `test_cuda_extension.py` output for "CUDA extension available". Also check device of tensors.

## References

- Test files: `tests/test_cuda_extension.py`, `examples/mlp_safety_test*.py`
- Benchmark: `examples/mlp_multiconfig_benchmark.py`
- Implementation: `hypatia_core/fused_modules.py`, `hypatia_core/fused_kernels/`
- E-graph rules: `hypatia_core/src/egraph_optimizer.rs`
