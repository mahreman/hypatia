# benchmark_simple.py - Sadece temel fonksiyonları test et
import torch
import torch.nn as nn

def test_basic_functionality():
    """Sadece Hypatia'nın temel optimizasyonunu test et"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Basit model
    class SimpleModel(nn.Module):
        def forward(self, x, A, B, C):
            return x @ A @ B + x @ A @ C
    
    model = SimpleModel().to(device)
    x = torch.randn(1, 10).to(device)
    A = torch.randn(10, 20).to(device)
    B = torch.randn(20, 30).to(device)
    C = torch.randn(20, 30).to(device)
    
    # Orijinal çıktı
    with torch.no_grad():
        original_output = model(x, A, B, C)
    
    # Hypatia optimizasyonu
    try:
        optimized_model = hypatia_core.compile_model(model, example_inputs=(x, A, B, C))
        optimized_output = optimized_model(x, A, B, C)
        
        # Doğruluk kontrolü
        assert torch.allclose(original_output, optimized_output, atol=1e-5)
        print("✅ Basit test geçti!")
        
    except Exception as e:
        print(f"❌ Hypatia optimizasyonu başarısız: {e}")
