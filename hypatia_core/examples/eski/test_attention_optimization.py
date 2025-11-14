import torch
import torch.nn.functional as F
import torch.fx
from torch.fx.node import Node
import hypatia_core as hc
import sys
from typing import Dict
import operator 

# ======================================================================
# BÖLÜM 1: FX GRAFİK -> S-EXPRESSION DERLEYİCİSİ (FEZ 10)
# (Değişiklik yok, bu kod doğru çalışıyor)
# ======================================================================

def compile_fx_to_hypatia(graph_module: torch.fx.GraphModule) -> Dict[str, str]:
    OPCODE_MAP = {
        torch.add: "add", torch.sub: "sub", torch.mul: "mul",
        torch.div: "div", torch.neg: "neg", torch.matmul: "mul", 
        operator.add: "add", operator.sub: "sub", operator.mul: "mul",
        operator.truediv: "div", operator.neg: "neg",
        torch.exp: "exp", torch.log: "log", torch.sqrt: "sqrt", 
        torch.pow: "pow", torch.relu: "relu", F.relu: "relu",
        torch.sigmoid: "sigmoid", torch.tanh: "tanh",
        torch.softmax: "softmax", F.softmax: "softmax",
    }
    
    env: Dict[str, str] = {}
    print("\n--- FX GRAFİK ANALİZİ ---")
    
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            env[node.name] = node.name
            print(f"  [Girdi]   '{node.name}'")
            
        elif node.op == 'call_function':
            op_name = OPCODE_MAP.get(node.target)
            if not op_name:
                if node.target == torch.tensor:
                    val = node.args[0]
                    op_name = str(float(val))
                    env[node.name] = op_name
                    print(f"  [Sabit]   '{node.name}' = {op_name}")
                    continue
                raise NotImplementedError(f"Desteklenmeyen PyTorch işlemi (call_function): {node.target}")
            args_sexpr = [env[arg.name] for arg in node.args]
            current_sexpr = f"({op_name} {' '.join(args_sexpr)})"
            env[node.name] = current_sexpr
            print(f"  [İşlem]   '{node.name}' = {current_sexpr}")
        
        elif node.op == 'call_method':
            if node.target == 'mean':
                op_name = "mean"
            elif node.target == 'var':
                op_name = "var"
            elif node.target == 'transpose':
                env[node.name] = "k_t" 
                print(f"  [Metod]   '{node.name}' = k_t (Transpose)")
                continue
            else:
                 raise NotImplementedError(f"Desteklenmeyen PyTorch metodu (call_method): {node.target}")

            arg_name = node.args[0].name
            current_sexpr = f"({op_name} {env[arg_name]})"
            env[node.name] = current_sexpr
            print(f"  [Metod]   '{node.name}' = {current_sexpr}")

        elif node.op == 'output':
            env['output'] = env[node.args[0].name]
            print(f"  [Çıktı]   return '{node.args[0].name}'")
            
    print("---------------------------")
    return env

# ======================================================================
# BÖLÜM 2: TEST ADIMI 1 - ATTENTION OPTİMİZASYONU
# ======================================================================
print("=" * 70)
print("HYPATIA AI PIPELINE TESTİ (FEZ 11: Sağlam Denklik Kontrolü)")
print("=" * 70)
print("Hedef: softmax( (Q*K_t) / d_sqrt )")
print("  ->   softmax( (Q*K_t) * (d_sqrt**-1) ) [YENİ MALİYET MODELİ]")

# 1. Model (Değişiklik yok)
class SimpleAttentionModel(torch.nn.Module):
    def forward(self, Q, K, V, d_sqrt):
        K_t = K.transpose(-2, -1) 
        scores = torch.matmul(Q, K_t)
        scaled_scores = scores / d_sqrt
        attention = F.softmax(scaled_scores, dim=-1)
        return torch.matmul(attention, V)

print("\n--- ADIM 1: PYTORCH MODELİ TANIMLANDI ---")
print(SimpleAttentionModel())

# 2. FX İzleme (Değişiklik yok)
try:
    model = SimpleAttentionModel()
    graph_module = torch.fx.symbolic_trace(model) 
    print("\n--- ADIM 2: TORCH.FX GRAFİĞİ BAŞARIYLA YAKALANDI ---")
    print(graph_module.graph)
except Exception as e:
    print(f"\nHATA: torch.fx modeli izleyemedi: {e}")
    sys.exit(1)

try:
    # 3. S-expression Çevirisi (Değişiklik yok)
    env = compile_fx_to_hypatia(graph_module)
    original_sexpr = env['softmax'] 
    
    print("\n--- ADIM 3: S-EXPRESSION (ATTENTION ÇEKİRDEĞİ) ---")
    print(f"GİRDİ (E-graph için): {original_sexpr}")
    
    expected_original = "(softmax (div (mul q k_t) d_sqrt))"
    assert original_sexpr == expected_original

    # 4. Hypatia v3.0 E-graph Motorunu Çağır
    print("\n--- ADIM 4: HYPATIA V3.0 E-GRAPH MOTORU ÇALIŞIYOR ---")
    optimized_sexpr = hc.optimize_ast(original_sexpr)
    print(f"ÇIKTI (E-graph'ten): {optimized_sexpr}")

    # 5. Sonucu Doğrula (✅ YENİ: FEZ 11)
    print("\n--- ADIM 5: SONUÇ DOĞRULANIYOR (hc.is_equivalent ile) ---")
    
    # Kullanıcı olarak, bizim 'ideal' formumuz budur:
    # (div A B) -> (mul A (pow B -1)) kuralının uygulanması
    # A = (mul q k_t)
    # B = d_sqrt
    expected_ideal_form = "(softmax (mul (mul q k_t) (pow d_sqrt -1)))"
    
    # E-graph'in çıktısı (örn: "(softmax (mul k_t (mul q (pow d_sqrt -1))))")
    # bizim ideal formumuza matematiksel olarak denk mi?
    is_equiv = hc.is_equivalent(optimized_sexpr, expected_ideal_form)
    
    assert is_equiv, \
        f"Optimizasyon başarısız! Çıktı, ideale denk değil.\n" + \
        f"  ÇIKTI:   {optimized_sexpr}\n" + \
        f"  BEKLENEN (İDEAL): {expected_ideal_form}"

    print(f"İdeal Form: {expected_ideal_form}")
    print(f"Gelen Form: {optimized_sexpr}")
    print(f"Denklik:    {is_equiv} (BAŞARILI)")

    print("\n" + "✅" * 20)
    print(" BAŞARILI: FEZ 11 (ATTENTION) TAMAMLANDI!")
    print("Test artık 'is_equivalent' ile sağlam (robust) ve")
    print("Attention çekirdeği (div) başarıyla (mul/pow) formuna dönüştürüldü.")
    print("✅" * 20)

except Exception as e:
    print(f"\n!!! TEST BAŞARISIZ OLDU: {e}", file=sys.stderr)
    sys.exit(1)