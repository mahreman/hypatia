#!/usr/bin/env python3
"""
Hypatia Comprehensive Test Suite - Final State
==============================================
Tests ALL features with correctness verification and performance benchmarks.

Sections:
1. NativeModel (small MLP) - correctness + speed
2. QuantizedModel (large MLP) - correctness + speed + compression
3. TransformerModel (real GPT-2) - correctness + speed
4. QAT Training (QuantizedTrainer) - convergence
5. NativeTrainer - convergence
6. torch.compile backend - batch=1, batch>1
7. Geometric Algebra operations - rotation, product, normalize
8. GPU backend info
9. hypatia.optimize() auto-selection
10. Summary table

Usage:
    cd hypatia_core && PYTHONPATH="$(pwd):$PYTHONPATH" python tests/comprehensive_test.py
"""

import sys
import os
import time
import math
import traceback
import numpy as np

# Ensure path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

# Suppress registration message for cleaner output
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    import hypatia_core
    from hypatia_core import (
        NativeModel, NativeTrainer, QuantizedModel, QuantizedTrainer,
        TransformerModel, optimize, count_optimizations,
    )
    from _hypatia_core import (
        ga_batch_rotate_2d, ga_batch_rotate_3d,
        ga2d_product_layer, ga3d_product_layer,
        ga2d_normalize, ga3d_normalize,
        gpu_info,
    )
finally:
    sys.stderr = _stderr


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

RESULTS = []

def benchmark(fn, warmup=5, iters=50, label=""):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]
    return median

def record(section, test_name, status, details=""):
    """Record a test result."""
    icon = "PASS" if status else "FAIL"
    RESULTS.append((section, test_name, status, details))
    print(f"  [{icon}] {test_name}: {details}")


def section_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════
# 1. NativeModel - Small MLP
# ═══════════════════════════════════════════════════════════════════
def test_native_model():
    section_header("1. NativeModel (Small MLP, f32 Rust-native)")

    # Build a simple MLP
    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 10),
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: 256->512->512->10  ({n_params:,} params)")

    # Create NativeModel
    try:
        native = NativeModel(model)
        record("NativeModel", "Construction", True, repr(native))
    except Exception as e:
        record("NativeModel", "Construction", False, str(e))
        return

    # Correctness - batch=1
    x1 = torch.randn(1, 256)
    with torch.no_grad():
        pt_out = model(x1)
    native_out = native(x1)
    rmse1 = torch.sqrt(torch.mean((pt_out - native_out) ** 2)).item()
    max_diff1 = torch.max(torch.abs(pt_out - native_out)).item()
    ok1 = rmse1 < 1e-4
    record("NativeModel", "Correctness batch=1", ok1,
           f"RMSE={rmse1:.8f}, MaxDiff={max_diff1:.8f}")

    # Correctness - batch=64
    x64 = torch.randn(64, 256)
    with torch.no_grad():
        pt_out64 = model(x64)
    native_out64 = native(x64)
    rmse64 = torch.sqrt(torch.mean((pt_out64 - native_out64) ** 2)).item()
    max_diff64 = torch.max(torch.abs(pt_out64 - native_out64)).item()
    ok64 = rmse64 < 1e-4
    record("NativeModel", "Correctness batch=64", ok64,
           f"RMSE={rmse64:.8f}, MaxDiff={max_diff64:.8f}")

    # Performance benchmark
    for batch_size in [1, 16, 64, 128]:
        x = torch.randn(batch_size, 256)
        pt_ms = benchmark(lambda: model(x), label=f"PyTorch b={batch_size}")
        native_ms = benchmark(lambda: native(x), label=f"Native b={batch_size}")
        speedup = pt_ms / native_ms if native_ms > 0 else 0
        ok = True  # performance not a pass/fail
        record("NativeModel", f"Speed batch={batch_size}", ok,
               f"PyTorch={pt_ms:.2f}ms, Hypatia={native_ms:.2f}ms, {speedup:.2f}x")


# ═══════════════════════════════════════════════════════════════════
# 2. QuantizedModel - Large MLP (LLaMA-7B FFN dims)
# ═══════════════════════════════════════════════════════════════════
def test_quantized_model():
    section_header("2. QuantizedModel (Large MLP, INT4 SIMD)")

    # LLaMA-7B FFN dimensions: 4096 -> 11008 -> 4096
    model_large = nn.Sequential(
        nn.Linear(4096, 11008), nn.ReLU(),
        nn.Linear(11008, 4096),
    )
    model_large.eval()

    n_params = sum(p.numel() for p in model_large.parameters())
    print(f"  Model: 4096->11008->4096 (LLaMA-7B FFN, {n_params/1e6:.1f}M params)")

    # Create QuantizedModel
    try:
        qmodel = QuantizedModel(model_large, group_size=128)
        record("QuantizedModel", "Construction", True, repr(qmodel))
    except Exception as e:
        record("QuantizedModel", "Construction", False, str(e))
        return

    # Memory stats
    record("QuantizedModel", "Compression", True,
           f"{qmodel.compression_ratio:.1f}x compression, {qmodel.memory_saved_mb:.0f}MB saved")

    # Correctness (INT4 has ~0.02 RMSE, that's expected)
    x1 = torch.randn(1, 4096)
    with torch.no_grad():
        pt_out = model_large(x1)
    q_out = qmodel(x1)
    rmse = torch.sqrt(torch.mean((pt_out - q_out) ** 2)).item()
    max_diff = torch.max(torch.abs(pt_out - q_out)).item()
    # INT4 relative RMSE ~10-15%
    pt_norm = torch.sqrt(torch.mean(pt_out ** 2)).item()
    rel_rmse = rmse / pt_norm if pt_norm > 0 else float('inf')
    ok_corr = rel_rmse < 0.25  # 25% tolerance for INT4
    record("QuantizedModel", "Correctness batch=1", ok_corr,
           f"RMSE={rmse:.4f}, RelRMSE={rel_rmse*100:.1f}%, MaxDiff={max_diff:.4f}")

    # Performance benchmark
    for batch_size in [1, 4]:
        x = torch.randn(batch_size, 4096)
        with torch.no_grad():
            pt_ms = benchmark(lambda: model_large(x), warmup=3, iters=20,
                            label=f"PyTorch b={batch_size}")
        q_ms = benchmark(lambda: qmodel(x), warmup=3, iters=20,
                        label=f"Quantized b={batch_size}")
        speedup = pt_ms / q_ms if q_ms > 0 else 0
        record("QuantizedModel", f"Speed batch={batch_size}", True,
               f"PyTorch={pt_ms:.1f}ms, Hypatia={q_ms:.1f}ms, {speedup:.1f}x")

    # Medium model: GPT-2 XL FFN dims
    model_med = nn.Sequential(
        nn.Linear(1600, 6400), nn.ReLU(),
        nn.Linear(6400, 1600),
    )
    model_med.eval()
    n_params_med = sum(p.numel() for p in model_med.parameters())
    print(f"\n  Medium model: 1600->6400->1600 (GPT-2 XL FFN, {n_params_med/1e6:.1f}M params)")

    try:
        qmodel_med = QuantizedModel(model_med, group_size=128)
        x = torch.randn(1, 1600)
        with torch.no_grad():
            pt_ms = benchmark(lambda: model_med(x), warmup=3, iters=20)
        q_ms = benchmark(lambda: qmodel_med(x), warmup=3, iters=20)
        speedup = pt_ms / q_ms if q_ms > 0 else 0
        record("QuantizedModel", "Speed GPT2-XL FFN batch=1", True,
               f"PyTorch={pt_ms:.1f}ms, Hypatia={q_ms:.1f}ms, {speedup:.1f}x")
    except Exception as e:
        record("QuantizedModel", "Speed GPT2-XL FFN", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 3. TransformerModel - Real GPT-2
# ═══════════════════════════════════════════════════════════════════
def test_transformer_model():
    section_header("3. TransformerModel (Real GPT-2)")

    try:
        from transformers import GPT2Model, GPT2Config
    except ImportError:
        record("TransformerModel", "GPT-2 Load", False, "transformers not installed")
        return

    # GPT-2 small (124M)
    print("  Loading GPT-2 small (12 blocks, 768 hidden, 12 heads)...")
    config = GPT2Config(
        n_layer=12, n_head=12, n_embd=768,
        vocab_size=50257, n_positions=1024,
    )
    gpt2 = GPT2Model(config).eval()
    n_params = sum(p.numel() for p in gpt2.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")

    # Create TransformerModel
    try:
        tmodel = TransformerModel(gpt2, n_heads=12)
        record("TransformerModel", "Construction", True, repr(tmodel))
    except Exception as e:
        record("TransformerModel", "Construction", False, str(e))
        return

    # Correctness: compare hidden states
    x = torch.randn(1, 8, 768)  # [batch=1, seq_len=8, hidden=768]
    with torch.no_grad():
        # GPT-2 forward through transformer blocks (skip embedding)
        hidden = x
        for block in gpt2.h:
            hidden = block(hidden)[0]
        hidden = gpt2.ln_f(hidden)
        pt_out = hidden

    hyp_out = tmodel(x)
    rmse = torch.sqrt(torch.mean((pt_out - hyp_out) ** 2)).item()
    max_diff = torch.max(torch.abs(pt_out - hyp_out)).item()
    ok = rmse < 0.01  # Allow more tolerance for accumulated float ops
    record("TransformerModel", "Correctness vs PyTorch", ok,
           f"RMSE={rmse:.6f}, MaxDiff={max_diff:.6f}")

    # Performance: batch=1, seq_len=8
    def pt_forward():
        h = x
        for block in gpt2.h:
            h = block(h)[0]
        return gpt2.ln_f(h)

    pt_ms = benchmark(pt_forward, warmup=3, iters=10)
    hyp_ms = benchmark(lambda: tmodel(x), warmup=3, iters=10)
    speedup = pt_ms / hyp_ms if hyp_ms > 0 else 0
    record("TransformerModel", "Speed batch=1,seq=8", True,
           f"PyTorch={pt_ms:.1f}ms, Hypatia={hyp_ms:.1f}ms, {speedup:.2f}x")

    # Performance: batch=1, seq_len=32
    x32 = torch.randn(1, 32, 768)
    def pt_forward32():
        h = x32
        for block in gpt2.h:
            h = block(h)[0]
        return gpt2.ln_f(h)

    pt_ms32 = benchmark(pt_forward32, warmup=2, iters=5)
    hyp_ms32 = benchmark(lambda: tmodel(x32), warmup=2, iters=5)
    speedup32 = pt_ms32 / hyp_ms32 if hyp_ms32 > 0 else 0
    record("TransformerModel", "Speed batch=1,seq=32", True,
           f"PyTorch={pt_ms32:.1f}ms, Hypatia={hyp_ms32:.1f}ms, {speedup32:.2f}x")

    # GPT-2 with fewer layers for quick test
    print("\n  Testing GPT-2 (4 blocks, lighter)...")
    config4 = GPT2Config(n_layer=4, n_head=12, n_embd=768,
                         vocab_size=50257, n_positions=1024)
    gpt2_4 = GPT2Model(config4).eval()
    try:
        tmodel4 = TransformerModel(gpt2_4, n_heads=12)
        x4 = torch.randn(1, 16, 768)
        with torch.no_grad():
            h = x4
            for block in gpt2_4.h:
                h = block(h)[0]
            pt_out4 = gpt2_4.ln_f(h)
        hyp_out4 = tmodel4(x4)
        rmse4 = torch.sqrt(torch.mean((pt_out4 - hyp_out4) ** 2)).item()
        record("TransformerModel", "Correctness 4-block GPT2", rmse4 < 0.01,
               f"RMSE={rmse4:.6f}")
    except Exception as e:
        record("TransformerModel", "Correctness 4-block GPT2", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 4. QAT Training (QuantizedTrainer)
# ═══════════════════════════════════════════════════════════════════
def test_qat_training():
    section_header("4. QAT Training (QuantizedTrainer)")

    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )

    try:
        trainer = QuantizedTrainer(model, lr=0.01, group_size=32)
        record("QAT", "Construction", True, repr(trainer))
    except Exception as e:
        record("QAT", "Construction", False, str(e))
        return

    # Training loop
    torch.manual_seed(42)
    losses = []
    for step in range(100):
        x = torch.randn(32, 64)
        y = torch.randn(32, 10)
        loss = trainer.step(x, y)
        losses.append(loss)

    # Check convergence
    first_10 = np.mean(losses[:10])
    last_10 = np.mean(losses[-10:])
    improved = last_10 < first_10
    record("QAT", "Convergence (100 steps)", improved,
           f"First10={first_10:.4f} -> Last10={last_10:.4f} ({(1-last_10/first_10)*100:.1f}% reduction)")

    # Convert to QuantizedModel
    try:
        qmodel = trainer.to_quantized_model()
        x_test = torch.randn(1, 64)
        out = qmodel(x_test)
        record("QAT", "to_quantized_model()", True,
               f"Output shape={tuple(out.shape)}, {repr(qmodel)}")
    except Exception as e:
        record("QAT", "to_quantized_model()", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 5. NativeTrainer
# ═══════════════════════════════════════════════════════════════════
def test_native_trainer():
    section_header("5. NativeTrainer (Rust-native SGD)")

    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 5),
    )

    try:
        trainer = NativeTrainer(model, lr=0.01)
        record("NativeTrainer", "Construction", True, "OK")
    except Exception as e:
        record("NativeTrainer", "Construction", False, str(e))
        return

    # Training loop
    torch.manual_seed(123)
    losses = []
    for step in range(100):
        x = torch.randn(16, 32)
        y = torch.randn(16, 5)
        loss = trainer.step(x, y)
        losses.append(loss)

    first_10 = np.mean(losses[:10])
    last_10 = np.mean(losses[-10:])
    improved = last_10 < first_10
    record("NativeTrainer", "Convergence (100 steps)", improved,
           f"First10={first_10:.4f} -> Last10={last_10:.4f} ({(1-last_10/first_10)*100:.1f}% reduction)")

    # Speed comparison vs PyTorch
    model2 = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 5),
    )
    pt_optim = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(16, 32)
    y = torch.randn(16, 5)

    def pt_step():
        pt_optim.zero_grad()
        out = model2(x)
        loss = loss_fn(out, y)
        loss.backward()
        pt_optim.step()
        return loss.item()

    pt_ms = benchmark(pt_step, warmup=5, iters=50)
    native_ms = benchmark(lambda: trainer.step(x, y), warmup=5, iters=50)
    speedup = pt_ms / native_ms if native_ms > 0 else 0
    record("NativeTrainer", "Speed vs PyTorch", True,
           f"PyTorch={pt_ms:.2f}ms, Hypatia={native_ms:.2f}ms, {speedup:.2f}x")


# ═══════════════════════════════════════════════════════════════════
# 6. torch.compile backend
# ═══════════════════════════════════════════════════════════════════
def test_torch_compile():
    section_header("6. torch.compile Backend")

    # Check registration
    backends = torch._dynamo.list_backends()
    registered = "hypatia" in backends
    record("torch.compile", "Backend registered", registered,
           f"'hypatia' in backends: {registered}")

    if not registered:
        return

    model = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).eval()

    try:
        compiled = torch.compile(model, backend="hypatia")
        record("torch.compile", "Compilation", True, "torch.compile() succeeded")
    except Exception as e:
        record("torch.compile", "Compilation", False, str(e))
        return

    # Test different batch sizes
    for batch_size in [1, 16, 64]:
        try:
            torch._dynamo.reset()
            compiled = torch.compile(model, backend="hypatia")
            x = torch.randn(batch_size, 128)
            with torch.no_grad():
                pt_out = model(x)
                comp_out = compiled(x)
            rmse = torch.sqrt(torch.mean((pt_out - comp_out) ** 2)).item()
            ok = rmse < 1e-4
            record("torch.compile", f"Correctness batch={batch_size}", ok,
                   f"RMSE={rmse:.8f}")
        except Exception as e:
            record("torch.compile", f"Correctness batch={batch_size}", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 7. Geometric Algebra Operations
# ═══════════════════════════════════════════════════════════════════
def test_geometric_algebra():
    section_header("7. Geometric Algebra Operations")

    # --- 2D Rotation ---
    # Rotate [1,0] by pi/2 -> should get [0,1]
    input_2d = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    theta = float(np.pi / 2)
    try:
        result_2d = ga_batch_rotate_2d(input_2d, theta)
        expected_0 = [0.0, 1.0]   # [1,0] rotated 90° -> [0,1]
        expected_1 = [-1.0, 0.0]  # [0,1] rotated 90° -> [-1,0]
        err0 = np.linalg.norm(result_2d[0] - expected_0)
        err1 = np.linalg.norm(result_2d[1] - expected_1)
        ok = err0 < 1e-5 and err1 < 1e-5
        record("GA", "2D Rotation (pi/2)", ok,
               f"[1,0]->[{result_2d[0][0]:.4f},{result_2d[0][1]:.4f}], "
               f"[0,1]->[{result_2d[1][0]:.4f},{result_2d[1][1]:.4f}]")
    except Exception as e:
        record("GA", "2D Rotation", False, str(e))

    # --- 3D Rotation ---
    # Rotate [1,0,0] around z-axis by pi/2 -> should get [0,1,0]
    input_3d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    try:
        result_3d = ga_batch_rotate_3d(input_3d, axis_z, theta)
        expected_3d = [0.0, 1.0, 0.0]
        err = np.linalg.norm(result_3d[0] - expected_3d)
        ok = err < 1e-5
        record("GA", "3D Rotation (z-axis, pi/2)", ok,
               f"[1,0,0]->[{result_3d[0][0]:.4f},{result_3d[0][1]:.4f},{result_3d[0][2]:.4f}]")
    except Exception as e:
        record("GA", "3D Rotation", False, str(e))

    # --- 2D Geometric Product Layer ---
    # Identity: e0=1 * any = any
    batch_in = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # scalar 1
    weights = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)   # e1
    try:
        result = ga2d_product_layer(batch_in, weights)
        # 1 * e1 = e1: should get [0, 1, 0, 0]
        expected = [0.0, 1.0, 0.0, 0.0]
        err = np.linalg.norm(result[0] - expected)
        ok = err < 1e-5
        record("GA", "2D Product (1 * e1 = e1)", ok,
               f"Result=[{result[0][0]:.4f},{result[0][1]:.4f},{result[0][2]:.4f},{result[0][3]:.4f}]")
    except Exception as e:
        record("GA", "2D Product Layer", False, str(e))

    # --- 3D Geometric Product Layer ---
    batch_3d = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)  # scalar 1
    weights_3d = np.array([[0.0, 0, 0, 1.0, 0, 0, 0, 0]], dtype=np.float32)  # e3
    try:
        result_3d = ga3d_product_layer(batch_3d, weights_3d)
        # 1 * e3 = e3: [0,0,0,1,0,0,0,0]
        expected_3d = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        err = np.linalg.norm(result_3d[0] - expected_3d)
        ok = err < 1e-5
        record("GA", "3D Product (1 * e3 = e3)", ok,
               f"Result={[f'{v:.2f}' for v in result_3d[0]]}")
    except Exception as e:
        record("GA", "3D Product Layer", False, str(e))

    # --- 2D Normalize ---
    batch_norm = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    try:
        normed = ga2d_normalize(batch_norm)
        mag0 = np.linalg.norm(normed[0])
        mag1 = np.linalg.norm(normed[1])
        ok = abs(mag0 - 1.0) < 1e-5 and abs(mag1 - 1.0) < 1e-5
        record("GA", "2D Normalize (unit magnitude)", ok,
               f"|v0|={mag0:.6f}, |v1|={mag1:.6f}")
    except Exception as e:
        record("GA", "2D Normalize", False, str(e))

    # --- 3D Normalize ---
    batch_3d_norm = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    try:
        normed_3d = ga3d_normalize(batch_3d_norm)
        mag = np.linalg.norm(normed_3d[0])
        ok = abs(mag - 1.0) < 1e-5
        record("GA", "3D Normalize (unit magnitude)", ok, f"|v|={mag:.6f}")
    except Exception as e:
        record("GA", "3D Normalize", False, str(e))

    # --- Performance: batch rotation ---
    big_batch = np.random.randn(10000, 2).astype(np.float32)
    try:
        t0 = time.perf_counter()
        for _ in range(100):
            ga_batch_rotate_2d(big_batch, 0.5)
        ms = (time.perf_counter() - t0) / 100 * 1000
        record("GA", "2D Rotation perf (10K vectors)", True,
               f"{ms:.2f}ms per batch")
    except Exception as e:
        record("GA", "2D Rotation perf", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 8. GPU Backend Info
# ═══════════════════════════════════════════════════════════════════
def test_gpu_info():
    section_header("8. GPU Backend Info")

    try:
        info = gpu_info()
        record("GPU", "gpu_info()", True, info)
    except Exception as e:
        record("GPU", "gpu_info()", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 9. hypatia.optimize() Auto-Selection
# ═══════════════════════════════════════════════════════════════════
def test_auto_optimize():
    section_header("9. hypatia.optimize() Auto-Selection")

    # Small MLP -> should pick NativeModel
    small = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    try:
        result = optimize(small, mode='auto', quantize=False)
        is_native = isinstance(result, NativeModel)
        record("optimize()", "Small MLP -> NativeModel", is_native,
               f"Got {type(result).__name__}")
    except Exception as e:
        record("optimize()", "Small MLP", False, str(e))

    # Large MLP -> should pick QuantizedModel
    large = nn.Sequential(
        nn.Linear(4096, 8192), nn.ReLU(),
        nn.Linear(8192, 4096),
    )
    try:
        result = optimize(large, mode='auto')
        is_quant = isinstance(result, QuantizedModel)
        record("optimize()", "Large MLP -> QuantizedModel", is_quant,
               f"Got {type(result).__name__}")
    except Exception as e:
        record("optimize()", "Large MLP", False, str(e))

    # Transformer -> should pick TransformerModel
    try:
        from transformers import GPT2Model, GPT2Config
        config = GPT2Config(n_layer=2, n_head=12, n_embd=768,
                           vocab_size=50257, n_positions=1024)
        gpt2_tiny = GPT2Model(config).eval()
        result = optimize(gpt2_tiny, mode='auto')
        is_transformer = isinstance(result, TransformerModel)
        record("optimize()", "GPT-2 -> TransformerModel", is_transformer,
               f"Got {type(result).__name__}")
    except ImportError:
        record("optimize()", "GPT-2 -> TransformerModel", False, "transformers not installed")
    except Exception as e:
        record("optimize()", "GPT-2 -> TransformerModel", False, str(e))

    # Explicit modes
    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    try:
        result = optimize(model, mode='native')
        ok = isinstance(result, NativeModel)
        record("optimize()", "mode='native'", ok, f"Got {type(result).__name__}")
    except Exception as e:
        record("optimize()", "mode='native'", False, str(e))

    try:
        result = optimize(model, mode='fusion')
        # Fusion returns regular model
        record("optimize()", "mode='fusion'", True, f"Got {type(result).__name__}")
    except Exception as e:
        record("optimize()", "mode='fusion'", False, str(e))

    # count_optimizations
    try:
        model_count = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
        stats = count_optimizations(model_count)
        record("optimize()", "count_optimizations()", True,
               f"total_linear={stats.get('total_linear', 0)}, "
               f"total_params={stats.get('total_params', 0)}")
    except Exception as e:
        record("optimize()", "count_optimizations()", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 10. Summary
# ═══════════════════════════════════════════════════════════════════
def print_summary():
    section_header("SUMMARY")

    total = len(RESULTS)
    passed = sum(1 for _, _, s, _ in RESULTS if s)
    failed = sum(1 for _, _, s, _ in RESULTS if not s)

    print(f"\n  Total: {total} tests | Passed: {passed} | Failed: {failed}")
    print(f"  Pass rate: {passed/total*100:.0f}%\n")

    if failed > 0:
        print("  FAILED TESTS:")
        for section, name, status, details in RESULTS:
            if not status:
                print(f"    [{section}] {name}: {details}")
        print()

    # Performance summary
    print("  PERFORMANCE SUMMARY:")
    print(f"  {'Section':<25} {'Test':<30} {'Details'}")
    print(f"  {'-'*25} {'-'*30} {'-'*40}")
    for section, name, _, details in RESULTS:
        if "Speed" in name or "perf" in name.lower():
            print(f"  {section:<25} {name:<30} {details}")

    print(f"\n{'='*70}")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TEST(S) FAILED")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  HYPATIA COMPREHENSIVE TEST SUITE")
    print(f"  Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    print(f"  Platform: {sys.platform}")
    print("=" * 70)

    test_functions = [
        test_native_model,
        test_quantized_model,
        test_transformer_model,
        test_qat_training,
        test_native_trainer,
        test_torch_compile,
        test_geometric_algebra,
        test_gpu_info,
        test_auto_optimize,
    ]

    for test_fn in test_functions:
        try:
            test_fn()
        except Exception as e:
            section_header(f"ERROR in {test_fn.__name__}")
            traceback.print_exc()
            record("ERROR", test_fn.__name__, False, str(e))

    print_summary()
