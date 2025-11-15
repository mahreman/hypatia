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
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


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
        y_ref = model(x)

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
        y_opt = opt(x)

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
