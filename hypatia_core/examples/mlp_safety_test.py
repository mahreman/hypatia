"""
MLP Safety/Correctness Test

Tests Hypatia compilation in STRICT mode to verify:
- No crashes or NaN values
- Numerical accuracy (comparing eager vs compiled outputs)
- Proper error detection via checksum validation
"""

import os
import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend

# Enable strict checksum validation mode
os.environ["HYPATIA_CHECKSUM_MODE"] = "strict"


class MLP(nn.Module):
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
        print(f"  {name:30s}  ||param|| = {param.norm().item():.6f}")



def main():
    print("=" * 70)
    print("MLP Safety Test (Strict Mode)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checksum Mode: {os.environ.get('HYPATIA_CHECKSUM_MODE', 'not set')}")
    print()

    # Create model and input
    model = MLP().to(device)
    model.eval()  # Inference mode
    x = torch.randn(32, 784, device=device)

    print("Step 1: Running eager (reference) model...")
    with torch.no_grad():
        y_ref, (ref_x1, ref_x2, ref_x3) = model(x)

    # Check for NaN in reference
    if torch.isnan(y_ref).any():
        print("ERROR: Reference output contains NaN!")
        return
    print(f"  Reference output shape: {y_ref.shape}")
    print(f"  Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")
    print()

    print("Step 2: Compiling with Hypatia backend...")
    opt = torch.compile(model, backend="hypatia")
    print()

    print("Step 3: Running compiled model...")
    with torch.no_grad():
        y_opt, (opt_x1, opt_x2, opt_x3) = opt(x)

    # Check for NaN in optimized output
    if torch.isnan(y_opt).any():
        print("ERROR: Optimized output contains NaN!")
        return
    print(f"  Optimized output shape: {y_opt.shape}")
    print(f"  Optimized output range: [{y_opt.min():.4f}, {y_opt.max():.4f}]")
    print()

    # Compare outputs
    print("Step 4: Comparing outputs...")
    abs_diff = (y_ref - y_opt).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Layer-by-layer diff
    print("\n  Layer-by-layer differences:")
    layer1_diff = (ref_x1 - opt_x1).abs().max().item()
    layer2_diff = (ref_x2 - opt_x2).abs().max().item()
    layer3_diff = (ref_x3 - opt_x3).abs().max().item()
    print(f"    After fc1+relu: {layer1_diff:.6e}")
    print(f"    After fc2+relu: {layer2_diff:.6e}")
    print(f"    After fc3:      {layer3_diff:.6e}")

    # Weight norms comparison
    dump_weight_norms(model, "eager")
    dump_weight_norms(opt, "hypatia")
    print()

    # Validate
    TOLERANCE = 1e-5
    if max_diff < TOLERANCE:
        print(f"✅ PASS: Max difference ({max_diff:.6e}) < tolerance ({TOLERANCE})")
    else:
        print(f"❌ FAIL: Max difference ({max_diff:.6e}) >= tolerance ({TOLERANCE})")

    print("=" * 70)


if __name__ == "__main__":
    main()
