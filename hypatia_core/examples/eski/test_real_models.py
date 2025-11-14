#!/usr/bin/env python3
"""
Hypatia Gerçek Model Test Suite'i (Faz 2)

Amaç:
1. 'complete_fx_parser.py' içindeki parser'ı kullanarak, ResNet ve
   Vision Transformer (ViT) gibi gerçek, karmaşık modelleri trace etmek.
2. Bu modellerin ne kadar büyük S-ifadeleri ürettiğini görmek.
3. Bu S-ifadelerini 'hypatia_core.optimize_ast'ye gönderip
   optimizasyonun çöküp çökmediğini test etmek (crash test).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from complete_fx_parser import CompleteFXToHypatiaParser
import hypatia_core as hc
import time  # Zaman ölçümü için eklendi
import sys   # sys.exit için eklendi

# Opsiyonel: Vision Transformer testi için
try:
    from transformers import ViTModel, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("UYARI: 'transformers' kütüphanesi yüklü değil. ViT testi atlanacak.")

def test_resnet18():
    """ResNet-18 ile tam test"""
    print("\n" + "=" * 60)
    print("TEST: ResNet-18")
    print("=" * 60)
    
    # Modeli yükle
    model = models.resnet18(weights=None)
    model.eval()
    
    # FX ile trace et
    try:
        example_input = torch.randn(1, 3, 224, 224)
        graph_module = torch.fx.symbolic_trace(model)
        
        print("✅ FX trace başarılı.")
    except Exception as e:
        print(f"❌ FX trace başarısız: {e}")
        return False
        
    # Hypatia parser'ı kullan
    parser = CompleteFXToHypatiaParser()
    result = parser.parse_fx_graph(graph_module)
    
    if not result['expression']:
        print("❌ S-ifadesi oluşturulamadı.")
        return False

    original_expr = result['expression']
    print(f"\nOluşturulan S-ifadesi (İlk 500 karakter):")
    print(original_expr[:500] + "...")
    print(f"Toplam parametre (ağırlık vb.): {len(result['parameters'])}")
    print(f"S-ifadesi toplam uzunluğu: {len(original_expr)}")
    
    # Optimize et (Crash Test)
    print("\n'hypatia_core.optimize_ast' ile optimizasyon (crash test) deneniyor...")
    try:
        start_time = time.time()
        optimized = hc.optimize_ast(original_expr)
        duration = time.time() - start_time
        
        # ✅ DÜZELTME: Optimizasyonun gerçekten başarılı olup olmadığını kontrol et
        if optimized.startswith("(error"):
            print(f"❌ Optimizasyon hatası (Rust Parser): {optimized}")
            return False
            
        print(f"✅ Optimizasyon başarılı! ({duration:.4f} saniye)")
        print(f"Optimize edilmiş ifade (İlk 500 karakter):")
        print(optimized[:500] + "...")
        print(f"Optimize edilmiş uzunluk: {len(optimized)}")
        
        if len(optimized) < len(original_expr):
            print("✨ Başarı: Optimizasyon ifadeyi kısalttı!")
        else:
            print("ℹ️ Bilgi: Optimizasyon ifadeyi kısaltmadı (mevcut kurallarla beklenir).")
            
        print("✅ ResNet-18 testi tamamlandı.")
        return True
        
    except Exception as e:
        print(f"❌ Optimizasyon hatası (Python): {e}")
        return False

def test_vit_model():
    """Vision Transformer testi"""
    if not TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 60)
        print("TEST: Vision Transformer (ATLANDI)")
        print("=" * 60)
        return True # Atlandığı için başarısız sayılmasın

    print("\n" + "=" * 60)
    print("TEST: Vision Transformer (ViT)")
    print("=" * 60)
    
    try:
        config = ViTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            image_size=32,
            patch_size=16
        )
        model = ViTModel(config)
        model.eval()
        
        # FX trace
        example_input = torch.randn(1, 3, 32, 32)
        # ✅ DÜZELTME: ViT trace hatasını yakalamak için try-except
        print("FX trace deneniyor...")
        graph_module = torch.fx.symbolic_trace(model)
        print("✅ FX trace başarılı.")
        
        parser = CompleteFXToHypatiaParser()
        result = parser.parse_fx_graph(graph_module)
        
        if not result['expression']:
            print("❌ S-ifadesi oluşturulamadı.")
            return False

        original_expr = result['expression']
        print(f"\nOluşturulan S-ifadesi (İlk 500 karakter):")
        print(original_expr[:500] + "...")
        print(f"Toplam parametre (ağırlık vb.): {len(result['parameters'])}")
        print(f"S-ifadesi toplam uzunluğu: {len(original_expr)}")

        print("\n'hypatia_core.optimize_ast' ile optimizasyon (crash test) deneniyor...")
        try:
            start_time = time.time()
            optimized = hc.optimize_ast(original_expr)
            duration = time.time() - start_time
            
            if optimized.startswith("(error"):
                print(f"❌ Optimizasyon hatası (Rust Parser): {optimized}")
                return False

            print(f"✅ Optimizasyon başarılı! ({duration:.4f} saniye)")
            print(f"Optimize edilmiş ifade (İlk 500 karakter):")
            print(optimized[:500] + "...")
            print(f"Optimize edilmiş uzunluk: {len(optimized)}")

            print("✅ Vision Transformer testi tamamlandı.")
            return True
        
        except Exception as e:
            print(f"❌ Optimizasyon hatası (Python): {e}")
            return False
        
    except Exception as e:
        print(f"❌ ViT testi başarısız (FX Trace hatası): {e}")
        return False

def run_all_real_model_tests():
    """Tüm gerçek model testlerini çalıştır"""
    results = {}
    results["resnet18"] = test_resnet18()
    results["vit"] = test_vit_model() 
    
    print("\n" + "=" * 60)
    print("GERÇEK MODEL TEST ÖZETİ")
    print("=" * 60)
    
    all_passed = True
    for model_name, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"- {model_name}: {status}")
        if not success:
            all_passed = False
            
    print("=" * 60)
    if not all_passed:
        print("🔥 Faz 2 testlerinde hatalar bulundu.")
        sys.exit(1)
    else:
        print("🎉 Faz 2 (FX -> S-ifadesi -> Optimizasyon) pipeline'ı başarıyla çalıştı!")

if __name__ == "__main__":
    # 'complete_fx_parser.py' dosyasının bu betikle aynı dizinde olduğunu varsayar
    run_all_real_model_tests()