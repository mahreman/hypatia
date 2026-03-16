# Adding New Fusion Patterns to Hypatia

This guide explains how to add a new operator or fusion pattern to the Hypatia e-graph compiler. We'll use the Mish activation fusion as a concrete example.

## Architecture Overview

Adding a new fusion pattern requires changes in 4 files:

```
src/egraph_optimizer.rs    — Define operator + rewrite rules + cost model
src/fx_bridge.rs           — FX graph ↔ e-graph conversion + reconstruction
src/visualization.rs       — Display name for visualization
hypatia_core/fused_modules.py — Python dispatch function + GPU kernel
```

## Step 1: Define the Operator (egraph_optimizer.rs)

### 1.1 Add to `define_language!`

```rust
define_language! {
    pub enum HypatiaLang {
        // ... existing operators ...

        // New: Mish activation
        "mish" = Mish(Id),
        // New: Fused Mish MLP (linear -> mish -> linear)
        "fused_mish_mlp" = FusedMishMLP([Id; 5]),
    }
}
```

**Rules:**
- Unary operators: `Op(Id)` — takes 1 child
- Binary operators: `Op([Id; 2])` — takes 2 children
- N-ary: `Op([Id; N])` — e.g., Linear takes `[Id; 3]` for (weight, bias, input)
- String must be valid S-expression token (lowercase, underscores OK)

### 1.2 Add Cost Model Entry

```rust
// In HardwareAwareCost::cost()
HypatiaLang::Mish(_) => (12.0, 0.0, 0.0),
// (flop_cost, memory_access_cost, stability_penalty)

HypatiaLang::FusedMishMLP(_) => (200.0, 10.0, 0.0),
// Fused ops should have LOWER cost than the sum of their components
// This incentivizes the e-graph to prefer the fused version
```

**Cost formula:** `total = alpha * flops + beta * memory + stability`
- `alpha = 1.0`, `beta = 2.0` (memory 2x more expensive)
- Fused ops MUST cost less than their decomposed form

### 1.3 Add Rewrite Rules

```rust
fn optimization_rules() -> Vec<Rewrite<HypatiaLang, ()>> {
    vec![
        // Pattern recognition: decomposed mish -> mish op
        rewrite!("mish-decomposed";
            "(mul ?x (tanh (log (add 1.0 (exp ?x)))))"
            => "(mish ?x)"),

        // Fusion: linear -> mish -> linear -> fused_mish_mlp
        rewrite!("mish-mlp-fusion";
            "(linear ?w2 ?b2 (mish (linear ?w1 ?b1 ?x)))"
            => "(fused_mish_mlp ?w1 ?b1 ?w2 ?b2 ?x)"),
    ]
}
```

**Pattern variables:** `?x`, `?w1`, `?b1`, etc. match any sub-expression.

### 1.4 Add `build_symbol` Entry

```rust
// In build_symbol() — converts e-graph node back to Symbol enum
HypatiaLang::Mish(id) => Symbol::SiLU(Box::new(build_symbol(*id, rec_expr))),
HypatiaLang::FusedMishMLP([_w1, _b1, _w2, _b2, x]) => build_symbol(*x, rec_expr),
```

### 1.5 Add Unit Tests

```rust
#[test]
fn test_mish_direct_parse() {
    let result = optimize_ast("(mish x)");
    assert!(result.contains("mish"), "got: {}", result);
}

#[test]
fn test_mish_mlp_fusion() {
    let pattern = "(linear w2 b2 (mish (linear w1 b1 x)))";
    let result = optimize_ast(pattern);
    assert!(result.contains("fused_mish_mlp"), "got: {}", result);
}
```

Run: `cargo test`

## Step 2: FX Bridge (fx_bridge.rs)

### 2.1 Mark as Supported

```rust
// In is_supported_node()
HypatiaLang::Mish(_) | HypatiaLang::FusedMishMLP(_) |
```

### 2.2 Add Reconstruction

```rust
// In reconstruct_node() match arm
HypatiaLang::FusedMishMLP([w1, b1, w2, b2, x]) => {
    self.reconstruct_fused_mish_mlp(py, &optimized_gm, w1, b1, w2, b2, x, rec_expr)
}
```

### 2.3 Implement Reconstruction Method

```rust
fn reconstruct_fused_mish_mlp(
    &self, py: Python, gm: &Bound<PyAny>,
    w1: &Id, b1: &Id, w2: &Id, b2: &Id, x: &Id,
    rec_expr: &RecExpr<HypatiaLang>,
) -> PyResult<()> {
    // 1. Get weight/bias tensors from original graph
    // 2. Import dispatch function from Python
    // 3. Replace graph nodes with fused call
    let fused_modules = py.import("hypatia_core.fused_modules")?;
    let dispatch_fn = fused_modules.getattr("dispatch_fused_mish_mlp")?;
    // ... graph manipulation ...
}
```

## Step 3: Visualization (visualization.rs)

```rust
// In node_display_name()
HypatiaLang::Mish(_) => "Mish",
HypatiaLang::FusedMishMLP(_) => "FusedMishMLP",
```

## Step 4: Python Dispatch (fused_modules.py)

### 4.1 Add Dispatch Function

```python
def dispatch_fused_mish_mlp(input, w1, b1, w2, b2):
    """Fused Mish MLP: linear -> mish -> linear.

    4-tier dispatch:
      1. Custom CUDA extension (if available)
      2. torch.compile + Triton kernel (GPU)
      3. Rust native kernel (CPU)
      4. Eager PyTorch fallback
    """
    # Tier 2: torch.compile (GPU)
    if input.is_cuda:
        compiled = _get_compiled("mish_mlp")
        if compiled is not None:
            return compiled(input, w1, b1, w2, b2)

    # Tier 4: Eager fallback
    h = F.linear(input, w1, b1)
    h = F.mish(h)
    return F.linear(h, w2, b2)
```

### 4.2 Add Compiled Kernel

```python
# In _make_compiled_kernels()
def _mish_mlp(x, w1, b1, w2, b2):
    h = F.linear(x, w1, b1)
    h = F.mish(h)
    return F.linear(h, w2, b2)

kernels["mish_mlp"] = torch.compile(
    _mish_mlp, mode="max-autotune", dynamic=False
)
```

## Step 5: Build and Test

```bash
# 1. Run Rust tests
cargo test

# 2. Rebuild Python binding
maturin develop --release

# 3. Run Python tests
python -m pytest tests/ -v
```

## Checklist for New Fusion Patterns

- [ ] `define_language!` — operator variant with correct arity
- [ ] Cost model entry — fused cost < sum of components
- [ ] Rewrite rule(s) — LHS pattern -> RHS fused op
- [ ] `build_symbol` — conversion back to Symbol enum
- [ ] `is_supported_node` — mark as supported in FX bridge
- [ ] Reconstruction method — graph manipulation in fx_bridge.rs
- [ ] Visualization name — display string
- [ ] Python dispatch — 4-tier dispatch function
- [ ] Compiled kernel — torch.compile kernel for GPU tier
- [ ] Rust unit tests — parse + fusion rule tests
- [ ] Python integration test — end-to-end correctness

## E-graph Concepts

**Equality Saturation:** Instead of applying rewrite rules in a fixed order,
the e-graph explores ALL possible rewrites simultaneously and picks the
lowest-cost equivalent expression.

**Cost Function:** The `HardwareAwareCost` considers FLOPs, memory accesses,
and numerical stability. Fused operators have lower costs because:
- Fewer memory round-trips (intermediate tensors eliminated)
- Fewer kernel launches (one fused kernel vs multiple)
- Better cache utilization

**Pattern Matching:** The `rewrite!` macro matches S-expression patterns.
Variables (`?x`) match any sub-expression, and the same variable in LHS and
RHS ensures the matched expression is reused.
