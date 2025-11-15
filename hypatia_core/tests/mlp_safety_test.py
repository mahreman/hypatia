#!/usr/bin/env python3
"""
MLP Safety Test - Hypatia Backend Correctness Verification
Bu test, Hypatia backend'inin doğru sonuçlar ürettiğini doğrular.
"""

import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend

# Ensure backend is registered
hypatia_core.register_backend()


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for testing"""
    def __init__(self, in_features=256, hidden_dim=512, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def test_backend_registration():
    """Test 1: Backend kaydı kontrolü"""
    print("\n" + "="*80)
    print("TEST 1: Backend Registration Check")
    print("="*80)

    backends = torch._dynamo.list_backends()
    assert "hypatia" in backends, f"❌ Hypatia not in backends: {backends}"
    print("✅ Hypatia backend is registered")
    print(f"   Available backends: {backends}")


def test_output_correctness():
    """Test 2: Output doğruluğu - Eager vs Hypatia"""
    print("\n" + "="*80)
    print("TEST 2: Output Correctness (Eager vs Hypatia)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Model ve input
    model = MLP().to(device).eval()
    x = torch.randn(128, 256, device=device)

    # Eager mode çıktısı
    with torch.no_grad():
        eager_output = model(x)

    # Hypatia compiled çıktısı
    compiled_model = torch.compile(model, backend="hypatia")
    with torch.no_grad():
        hypatia_output = compiled_model(x)

    # Karşılaştırma
    max_diff = torch.max(torch.abs(eager_output - hypatia_output)).item()
    print(f"Max absolute difference: {max_diff}")

    assert torch.allclose(eager_output, hypatia_output, atol=1e-4), \
        f"❌ Outputs don't match! Max diff: {max_diff}"

    print("✅ Outputs match within tolerance (atol=1e-4)")


def test_multiple_forward_passes():
    """Test 3: Multiple forward pass consistency"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Forward Pass Consistency")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP().to(device).eval()
    compiled_model = torch.compile(model, backend="hypatia")

    # Aynı input ile 3 kez çalıştır
    x = torch.randn(64, 256, device=device)

    outputs = []
    with torch.no_grad():
        for i in range(3):
            out = compiled_model(x)
            outputs.append(out)
            print(f"   Pass {i+1}: output shape {out.shape}, mean={out.mean().item():.6f}")

    # Tüm çıktılar aynı olmalı
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-6), \
            f"❌ Pass {i+1} doesn't match pass 1"

    print("✅ All forward passes produce identical results")


def test_batch_size_variations():
    """Test 4: Farklı batch size'lar ile doğruluk"""
    print("\n" + "="*80)
    print("TEST 4: Batch Size Variations")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP().to(device).eval()
    compiled_model = torch.compile(model, backend="hypatia")

    batch_sizes = [1, 16, 64, 128]

    for bs in batch_sizes:
        x = torch.randn(bs, 256, device=device)

        with torch.no_grad():
            eager_out = model(x)
            hypatia_out = compiled_model(x)

        max_diff = torch.max(torch.abs(eager_out - hypatia_out)).item()

        assert torch.allclose(eager_out, hypatia_out, atol=1e-4), \
            f"❌ Batch size {bs}: outputs don't match! Max diff: {max_diff}"

        print(f"   ✓ Batch size {bs:3d}: max diff = {max_diff:.2e}")

    print("✅ All batch sizes produce correct results")


def main():
    print("\n" + "="*80)
    print("HYPATIA MLP SAFETY TEST SUITE")
    print("="*80)

    try:
        test_backend_registration()
        test_output_correctness()
        test_multiple_forward_passes()
        test_batch_size_variations()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80 + "\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80 + "\n")
        raise
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    main()
