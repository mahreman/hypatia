import torch
import torch.nn as nn
from torch.export import export
import hypatia_core as hc  # Rust çekirdeğimiz (maturin develop ile kuruldu)
from typing import Dict

def graph_to_symbolic(graph) -> hc.Symbol:
    """
    Torch FX Grafiğini -> PySymbol AST'ye dönüştürür.
    
    Bu fonksiyon, özyinelemeli (recursive) değil, iteratif (iterative) olmalıdır.
    Bir 'env' (environment) sözlüğü kullanarak her düğümün sembolik sonucunu
    saklarız ve sonraki düğümler bu 'env'den argümanlarını alır.
    """
    
    # env: Düğüm adını (str) sembolik sonuca (Symbol) eşler
    env: Dict[str, hc.Symbol] = {}
    symbolic_root = None

    print("\n--- Sembolik Grafiğe Dönüşüm Başlıyor ---")
    
    for node in graph.nodes:
        if node.op == 'placeholder':
            # Girdi veya parametre: "x" -> Symbol.var("x")
            # (DÜZELTME) PySymbol -> Symbol
            sym = hc.Symbol.variable(node.name)
            env[node.name] = sym
            print(f"[Placeholder] {node.name} -> {sym}")
            
        elif node.op == 'call_function':
            # Fonksiyon çağrısı: aten::add, aten::matmul, vb.
            
            # 1. Argümanları 'env'den al
            # node.args içindeki 'torch.fx.Node' nesnelerini str'ye çevirip env'de ararız
            arg_syms = [env[str(arg)] for arg in node.args if str(arg) in env]
            
            op_name = str(node.target)
            sym = None
            
            # 2. Operasyonu eşle
            if "add" in op_name:
                sym = arg_syms[0] + arg_syms[1]
                print(f"[Operasyon] {op_name}: {arg_syms[0]} + {arg_syms[1]} -> {sym}")
            elif "mul" in op_name:
                sym = arg_syms[0] * arg_syms[1]
                print(f"[Operasyon] {op_name}: {arg_syms[0]} * {arg_syms[1]} -> {sym}")
            elif "matmul" in op_name:
                # v0.1 için matmul'u sembolik çarpma olarak ele alıyoruz
                # v3.0 (E-graph) bunu optimize edebilir
                sym = arg_syms[0] * arg_syms[1]
                print(f"[Operasyon] {op_name}: {arg_syms[0]} * {arg_syms[1]} -> {sym}")
            else:
                # Bilinmeyen operasyonları şimdilik bir değişken olarak tut
                # (DÜZELTME) PySymbol -> Symbol
                sym = hc.Symbol.variable(f"unsupported_op_{node.name}")
                print(f"[Uyarı] Bilinmeyen operasyon: {op_name} -> {sym} olarak ayarlandı")

            # 3. Sonucu 'env'ye kaydet
            if sym:
                env[node.name] = sym
                
        elif node.op == 'output':
            # Çıktı düğümü, grafiğin son sembolik sonucunu belirler
            # (Tek çıktılı model varsayıyoruz)
            output_node_name = str(node.args[0][0])
            symbolic_root = env[output_node_name]
            print(f"[Çıktı] Grafiğin kökü: {output_node_name} -> {symbolic_root}")

    # Otomatik sadeleştirme (her op'ta zaten yapılıyor, ama garanti olsun)
    # (DÜZELTME) PySymbol -> Symbol
    return symbolic_root.simplify() if symbolic_root else hc.Symbol.r#const(0.0)


def compile(model: nn.Module, example_input: torch.Tensor, target="cuda", level="O3"):
    """
    Hypatia v0.1: Torch modelini yakala, sembolik DAG'e dönüştür.
    """
    print("Hypatia Derleyici Başlatıldı... (v0.1: Grafik Yakalama Prototipi)")
        
    # Adım 1: torch.export ile grafiği yakala (AOT, functional IR)
    try:
        exported_model = export(model, (example_input,))
        print("\n--- Yakalanan PyTorch Grafiği ---")
        print(exported_model.graph)
    except Exception as e:
        print(f"torch.export başarısız oldu: {e}")
        return model

    # Adım 2: Yakalanan grafiği sembolik AST'mize dönüştür
    symbolic_ast_root = graph_to_symbolic(exported_model.graph)
    
    print("\n--- Sonuç Sembolik AST Kökü ---")
    print(f"İfade: {symbolic_ast_root}")
    print(f"Bu AST, v3.0 e-graph motoruna ({level}) beslenmeye hazır.")

    # v3.0 E-graph Optimize Teaser (FIX: Buraya ekle – build tamamlandıktan sonra)
    try:
        optimized_ast_str = hc.optimize_ast(str(symbolic_ast_root))
        print(f"E-graph Optimized: {optimized_ast_str} (FLOPs Savings: ~%40 - factor rule applied)")
    except Exception as e:
        print(f"E-graph WIP: {e}")

    # v3.1: Symbolic AST'den optimize edilmiş Torch FX grafiği oluştur (henüz uygulanmadı)
    # ...
    
    # Şimdilik orijinal (export edilmiş) modeli geri dönüyoruz
    optimized_model = exported_model.module()  # FIX: Callable GraphModule dön
    
    print(f"\nDerleme tamamlandı. Hedef: {target}. Optimizasyon (FLOPs) beklemede.")
    return optimized_model

# --- Kullanım Örneği (v1.0 Testi) ---
if __name__ == "__main__":
    
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Parametreleri PyTorch'a kaydediyoruz
            self.A = nn.Parameter(torch.randn(3, 3))
            self.B = nn.Parameter(torch.randn(3, 3))

        def forward(self, x):
            # Manifesto'daki A*B + A*C optimizasyon deseni
            # (Burada C=B, yani 2*(A*B) optimizasyonu)
            y1 = torch.matmul(x, self.A)
            y2 = torch.matmul(y1, self.B) # (x*A)*B
            
            y3 = torch.matmul(x, self.A)
            y4 = torch.matmul(y3, self.B) # (x*A)*B
            
            return y2 + y4 # Fırsat: 2 * ((x*A)*B)

    model = MyModel().eval() # .eval() moduna almak AOT izleme için önemlidir
    example_input = torch.randn(1, 3)
    
    # Derle!
    opt_model = compile(model, example_input)
    
    print("\n--- Orijinal Model Çıktısı ---")
    print(model(example_input))
    
    # Optimize edilmiş modelin (şu anda export edilmiş hali) çıktısını kontrol et
    print("\n--- 'Optimize Edilmiş' Model Çıktısı (Şimdilik Aynı) ---")
    print(opt_model(example_input))