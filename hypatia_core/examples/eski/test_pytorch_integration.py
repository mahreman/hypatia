import torch
import torch.fx
from torch.fx.node import Node
import hypatia_core as hc
from typing import Dict

# ======================================================================
# FEZ 6: PYTORCH FX GRAFİĞİ -> HYPATIA S-EXPRESSION DERLEYİCİSİ
# ======================================================================

def compile_fx_to_hypatia(graph_module: torch.fx.GraphModule) -> str:
    """
    Bir PyTorch FX Grafiğini Hypatia E-graph motorunun anladığı
    S-expression (LISP) formatına çevirir.
    """
    
    # Hangi PyTorch fonksiyonunun hangi Hypatia S-exp'e karşılık geldiği
    OPCODE_MAP = {
        torch.add: "add",
        torch.sub: "sub",
        torch.mul: "mul",
        torch.div: "div",
        torch.neg: "neg",
        torch.exp: "exp",
        torch.log: "log",
        torch.sqrt: "sqrt",
        torch.pow: "pow",
        torch.relu: "relu",
        torch.sigmoid: "sigmoid",
        torch.tanh: "tanh",
        torch.matmul: "mul", 
    }
    
    env: Dict[str, str] = {}
    final_expr = ""

    print("\n--- ADIM 2.1: PYTORCH FX GRAFİK ANALİZİ ---")
    
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            # GİRDİ
            env[node.name] = node.name
            print(f"  [Girdi]   '{node.name}' (Placeholder)")
            
        elif node.op == 'call_function':
            # İŞLEM
            op_name = OPCODE_MAP.get(node.target)
            if not op_name:
                raise NotImplementedError(f"Desteklenmeyen PyTorch işlemi: {node.target}")

            args_sexpr = [env[arg.name] for arg in node.args]
            current_sexpr = f"({op_name} {' '.join(args_sexpr)})"
            env[node.name] = current_sexpr
            
            print(f"  [İşlem]   '{node.name}' = {current_sexpr}")

        elif node.op == 'output':
            # ÇIKTI
            final_node_name = node.args[0].name
            final_expr = env[final_node_name]
            print(f"  [Çıktı]   return '{final_node_name}'")

    print("---------------------------------------------")
    return final_expr


# ======================================================================
# TEST SENARYOSU: v3.0 (FLOPs Optimizasyonu)
# ======================================================================

print("=" * 70)
print("HYPATIA PYTORCH ENTEGRASYON TESTİ (FEZ 6) - İZLENEBİLİR MOD")
print("=" * 70)
print("Strateji: 'Truva Atı' v1.0")
print("Hedef: (A*B + A*C) grafiğini yakala ve (A*(B+C)) olarak optimize et.")

# --- ADIM 1: MODELİ TANIMLA ---
class MyModel(torch.nn.Module):
    def forward(self, A, B, C):
        y1 = torch.matmul(A, B)
        y2 = torch.matmul(A, C)
        return torch.add(y1, y2)

print("\n--- ADIM 1: PYTORCH MODELİ TANIMLANDI ---")
print(MyModel())


# --- ADIM 2: MODELİ TORCH.FX İLE İZLE (TRACE) ---
try:
    model = MyModel()
    graph_module = torch.fx.symbolic_trace(model)
    print("\n--- ADIM 2: TORCH.FX GRAFİĞİ BAŞARIYLA YAKALANDI ---")
    print("Ham PyTorch FX Grafiği:")
    print("---------------------------------------------")
    print(graph_module.graph)
    print("---------------------------------------------")

except Exception as e:
    print(f"\nHATA: torch.fx modeli izleyemedi. 'torch' kurulu mu? Hata: {e}")
    exit(1)


# --- ADIM 3: FX GRAFİĞİNİ S-EXPRESSION'A ÇEVİR ---
try:
    original_sexpr = compile_fx_to_hypatia(graph_module)
    print("\n--- ADIM 3: S-EXPRESSION BAŞARIYLA OLUŞTURULDU ---")
    print(f"GİRDİ (E-graph için): {original_sexpr}")
    
    assert original_sexpr == "(add (mul a b) (mul a c))"

except Exception as e:
    print(f"\nDERLEME HATASI: FX -> S-expression çevirisi başarısız oldu: {e}")
    raise


# --- ADIM 4: HYPATIA V3.0 E-GRAPH MOTORUNU ÇAĞIR ---
try:
    print("\n--- ADIM 4: HYPATIA V3.0 E-GRAPH MOTORU ÇALIŞIYOR ---")
    optimized_sexpr = hc.optimize_ast(original_sexpr)
    print(f"ÇIKTI (E-graph'ten): {optimized_sexpr}")

except hc.HypatiaError as e:
    print(f"\nOPTİMİZASYON HATASI: optimize_ast başarısız oldu: {e}")
    raise
except AttributeError:
    print("\nÖNEMLİ HATA: hc.optimize_ast bulunamadı. Derleme hatası?")
    raise


# --- ADIM 5: SONUCU DOĞRULA ---
print("\n--- ADIM 5: SONUÇ DOĞRULANIYOR ---")
expected_forms = ["(mul a (add b c))", "(mul (add b c) a)"]

assert optimized_sexpr in expected_forms, \
    f"Optimizasyon başarısız! Beklenen: {expected_forms}, Gelen: {optimized_sexpr}"

print(f"Beklenen: {expected_forms}")
print(f"Gelen:    {optimized_sexpr}")
print("Durum:    BAŞARILI")

print("\n" + "✅" * 20)
print(" BAŞARILI: FEZ 6 TAMAMLANDI!")
print("PyTorch FX Grafiği başarıyla yakalandı, S-expression'a çevrildi")
print("ve Hypatia v3.0 E-graph motoru tarafından optimize edildi!")
print("✅" * 20)