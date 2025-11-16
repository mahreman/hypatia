import os
import sys
# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend


def test_backend_is_registered():
    """Hypatia backend'inin kayıtlı olduğunu test et"""
    backends = torch._dynamo.list_backends()
    assert "hypatia" in backends, f"Hypatia not in backends: {backends}"
    print("✅ Hypatia backend is registered")

def test_simple_compile():
    """Basit bir modeli compile et"""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU()
    ).eval()
    
    x = torch.randn(1, 10)
    
    # Original output
    with torch.no_grad():
        original_out = model(x)
    
    # Compile with Hypatia
    try:
        compiled = torch.compile(model, backend="hypatia")
        with torch.no_grad():
            compiled_out = compiled(x)
        
        # Check output matches
        assert torch.allclose(original_out, compiled_out, atol=1e-4)
        print("✅ Simple model compilation works!")
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        raise

if __name__ == "__main__":
    test_backend_is_registered()
    test_simple_compile()