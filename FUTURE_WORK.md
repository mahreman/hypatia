# Future Work - Hypatia Compiler

## High Priority TODOs

### 1. ✅ DONE: Fix ReLU(3) Reconstruction Error
**Status:** Fixed in current commit

**Problem:**
- Error: `Expected Var or Constant, found ReLU(3)`
- Occurred when debug logging tried to extract variable name from nested expressions

**Solution:**
- Made debug logging safe for nested expressions (e.g., `ReLU(...)`)
- Added fallback to debug representation when node is not a simple Var/Constant

### 2. Implement Real Rewrite + Reconstruction Flow
**Status:** TODO

**Background:**
Currently, E-graph saturation produces no rewrites → `structure_changed: false` → fallback to original GraphModule. No actual fusion is happening yet.

**Required Work:**

#### 2.1 Define Fusion Node Types
Add new node types to `HypatiaLang` enum:
```rust
// In egraph_optimizer.rs
pub enum HypatiaLang {
    // ... existing nodes ...

    // New fused nodes
    FusedLinearReLU([Id; 3]),      // (fused-linear-relu w b x)
    FusedConvReLU([Id; 7]),        // (fused-conv-relu w b x s p d g)
    // ... more fusion patterns
}
```

#### 2.2 Add E-graph Rewrite Rules
```rust
// Example fusion rule
rewrite!("fuse-linear-relu";
    "(relu (linear ?w ?b ?x))" =>
    "(fused-linear-relu ?w ?b ?x)"
)
```

#### 2.3 Implement Reconstruction Handlers
In `fx_bridge.rs::reconstruct_node()`:
```rust
HypatiaLang::FusedLinearReLU([w_id, b_id, x_id]) => {
    // Option A: Custom HypatiaLinearReLU module
    self.build_fused_linear_relu_module(w_id, b_id, x_id, expr)

    // Option B: nn.Sequential but as single FX node
    self.build_sequential_module(&["Linear", "ReLU"], ...)
}
```

**When to tackle:** After parameter binding is fully verified and safety tests pass.

### 3. Refine Checksum Validation for Structural Changes
**Status:** TODO

**Current Behavior:**
- `structure_changed: false` → Compare parameter checksums (strict mode)
- `structure_changed: true` → ??? (undefined behavior in strict mode)

**Required Work:**

#### 3.1 Define Validation Strategy
When fusion rewrites are active:
1. **Option A: Semantic Validation**
   - Run random test inputs through both models
   - Compare outputs with tolerance (e.g., max diff < 1e-5)
   - Accept if semantically equivalent

2. **Option B: Checksum Mode Hierarchy**
   - `HYPATIA_CHECKSUM_MODE=strict` → Disable fusion, only safe transforms
   - `HYPATIA_CHECKSUM_MODE=soft` → Enable fusion, semantic validation
   - `HYPATIA_CHECKSUM_MODE=off` → Enable all optimizations, no validation

#### 3.2 Implementation
In `compile_fx_graph()`:
```rust
if result.structure_changed {
    match checksum_mode {
        ChecksumMode::Strict => {
            // Run semantic validation with random inputs
            validate_semantic_equivalence(original_gm, optimized_gm)?
        }
        ChecksumMode::Soft => {
            // Lighter validation
            validate_output_shapes(original_gm, optimized_gm)?
        }
        ChecksumMode::Off => {
            // Trust the optimizer
        }
    }
}
```

### 4. ✅ DONE: Feature Flag for Debug Logging
**Status:** Implemented in current commit

**Environment Variable:**
- `HYPATIA_DEBUG_FX=1` → Enable verbose FX graph and reconstruction logs
- Default (not set or 0) → High-level summary only

**What was added:**
- Python side: Conditional logging in `hypatia_backend()`
- Rust side: `is_debug_fx_enabled()` helper function
- Wraps verbose debug messages (placeholder mapping, linear args, etc.)

## Medium Priority

### 5. Expand Test Coverage
- Add tests for each fusion pattern (when implemented)
- Test CUDA vs CPU consistency
- Test mixed precision (FP16, BF16)
- Add regression tests for parameter binding

### 6. Performance Benchmarking
- Measure actual speedup from fusion (when enabled)
- Profile compilation overhead
- Compare against torch.compile eager mode

### 7. Better Error Messages
- When reconstruction fails, show which pattern failed
- Suggest fixes (e.g., "Missing parameter X in state_dict")
- Point to documentation

## Low Priority / Nice to Have

### 8. Support More PyTorch Ops
- Attention modules (scaled_dot_product_attention)
- More activation functions (Swish, Mish, etc.)
- GroupNorm, InstanceNorm
- Einsum operations

### 9. Visualization Tools
- Generate GraphViz visualization of E-graph
- Show before/after FX graphs for fusion
- Display parameter flow diagrams

### 10. Documentation
- User guide for enabling optimizations
- Developer guide for adding new fusion patterns
- Troubleshooting common issues

---

## Current Status Summary

✅ **Working:**
- Parameter binding via `example_inputs`
- Device handling (CUDA tensors)
- Checksum validation (strict mode)
- Debug logging (with feature flag)
- Safety tests passing (single-layer models)

⚠️ **In Progress:**
- Multi-layer parameter binding verification (0.30 diff investigation)

❌ **Not Yet Implemented:**
- Real fusion rewrites (all structure_changed = false currently)
- Semantic validation for fused graphs
- Performance optimization beyond graph structure preservation

---

## How to Use Current Debug Features

### 1. Enable Verbose Logging
```bash
export HYPATIA_DEBUG_FX=1
python examples/mlp_safety_test.py
```

**Output includes:**
- `example_inputs` order with shapes and devices
- Placeholder → tensor mapping
- Linear node argument binding
- Reconstruction details

### 2. Enable Strict Checksum Validation
```bash
export HYPATIA_CHECKSUM_MODE=strict
python examples/mlp_safety_test.py
```

**Behavior:**
- Compares parameter checksums after compilation
- Falls back to original model if mismatch detected
- Logs detailed parameter information

### 3. Combined Debug Mode
```bash
export HYPATIA_DEBUG_FX=1
export HYPATIA_CHECKSUM_MODE=strict
python examples/mlp_safety_test.py 2>&1 | tee debug.log
```

**Best for:**
- Investigating parameter binding issues
- Debugging reconstruction failures
- Understanding FX graph transformation flow
