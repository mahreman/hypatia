import torch
import torch.nn as nn
import time

from hypatia_core.fused_modules import (
    HypatiaFusedLinearReLU,
    create_fused_linear_relu_from_tensors,
)


def make_baseline(in_features: int, out_features: int, device: str):
    layer = nn.Linear(in_features, out_features, bias=True, device=device)
    act = nn.ReLU()
    model = nn.Sequential(layer, act).to(device)
    return model


def make_fused_from_baseline(baseline: nn.Sequential, device: str):
    linear = baseline[0]
    fused = create_fused_linear_relu_from_tensors(
        weight=linear.weight,
        bias=linear.bias,
    )
    fused.to(device)
    return fused


def check_forward_backward(device: str = "cpu", atol: float = 1e-6):
    print(f"\n=== Forward/Backward correctness on {device} ===")

    torch.manual_seed(0)
    in_features = 784
    out_features = 256
    batch = 32

    baseline = make_baseline(in_features, out_features, device=device)
    fused = make_fused_from_baseline(baseline, device=device)

    x = torch.randn(batch, in_features, device=device, requires_grad=True)
    x_fused = x.detach().clone().requires_grad_(True)

    # forward
    y_base = baseline(x)
    y_fused = fused(x_fused)

    max_diff = (y_base - y_fused).abs().max().item()
    print(f"  [forward] max |y_base - y_fused| = {max_diff:.3e}")
    assert max_diff < atol, "Forward outputs differ!"

    # backward
    grad_out = torch.randn_like(y_base)
    y_base.backward(grad_out)
    y_fused.backward(grad_out)

    gx_diff = (x.grad - x_fused.grad).abs().max().item()
    print(f"  [backward] max |∂L/∂x_base - ∂L/∂x_fused| = {gx_diff:.3e}")
    assert gx_diff < atol, "Input gradients differ!"

    w_base = baseline[0].weight.grad
    w_fused = fused.weight.grad
    gw_diff = (w_base - w_fused).abs().max().item()
    print(f"  [backward] max |∂L/∂W_base - ∂L/∂W_fused| = {gw_diff:.3e}")
    assert gw_diff < atol, "Weight gradients differ!"

    print("  ✅ Forward & backward match.")


def microbench(device: str = "cuda", iters: int = 1000):
    if device == "cuda" and not torch.cuda.is_available():
        print("\n[WARN] CUDA not available, skipping CUDA microbench.")
        return

    print(f"\n=== Microbenchmark on {device} ({iters} iters) ===")

    torch.manual_seed(0)
    in_features = 784
    out_features = 256
    batch = 256

    baseline = make_baseline(in_features, out_features, device=device)
    fused = make_fused_from_baseline(baseline, device=device)

    x = torch.randn(batch, in_features, device=device)

    def bench(fn, x, iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            y = fn(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        return (t1 - t0) * 1000.0 / iters  # ms/iter

    baseline_ms = bench(baseline, x, iters)
    fused_ms = bench(fused, x, iters)

    print(f"  Baseline (Linear+ReLU): {baseline_ms:.4f} ms/iter")
    print(f"  Fused    (Hypatia):     {fused_ms:.4f} ms/iter")
    if fused_ms <= baseline_ms:
        print(f"  ✅ Speedup: {baseline_ms / fused_ms:.3f}x faster")
    else:
        print(f"  ⚠️ Slowdown: {fused_ms / baseline_ms:.3f}x slower")


def main():
    # 1) CPU correctness
    check_forward_backward(device="cpu")

    # 2) CUDA correctness (varsa)
    if torch.cuda.is_available():
        check_forward_backward(device="cuda")

    # 3) Microbench
    microbench(device="cpu", iters=500)
    if torch.cuda.is_available():
        microbench(device="cuda", iters=1000)


if __name__ == "__main__":
    main()
