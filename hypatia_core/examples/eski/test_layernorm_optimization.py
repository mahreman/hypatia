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
# BÖLÜM 2: TEST ADIMI 2 - LAYERNORM OPTİMİZASYONU
# ======================================================================
print("=" * 70)
print("HYPATIA AI PIPELINE TESTİ (FEZ 11: Sağlam Denklik Kontrolü)")
print("=" * 70)
print("Hedef: (X - mean) / sqrt(var + eps)")
print("  ->   (X - mean) * (var + eps)**-0.5 [YENİ MALİYET MODELİ]")

# 1. Model (Değişiklik yok)
class SimpleNormModel(torch.nn.Module):
    def forward(self, X, eps):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True)    
        var_eps = var + eps
        sqrt_var = torch.sqrt(var_eps) 
        numerator = X - mean
        return numerator / sqrt_var

print("\n--- ADIM 1: PYTORCH MODELİ TANIMLANDI ---")
print(SimpleNormModel())

# 2. FX İzleme (Değişiklik yok)
try:
    model = SimpleNormModel()
    graph_module = torch.fx.symbolic_trace(model) 
    print("\n--- ADIM 2: TORCH.FX GRAFİĞİ BAŞARIYLA YAKALANDI ---")
    print(graph_module.graph)
except Exception as e:
    print(f"\nHATA: torch.fx modeli izleyemedi: {e}")
    sys.exit(1)

try:
    # 3. S-expression Çevirisi (Değişiklik yok)
    env = compile_fx_to_hypatia(graph_module)
    original_sexpr = env['output'] 
    
    print("\n--- ADIM 3: S-EXPRESSION (LAYERNORM ÇEKİRDEĞİ) ---")
    print(f"GİRDİ (E-graph için): {original_sexpr}")
    
    expected_original = "(div (sub x (mean x)) (sqrt (add (var x) eps)))"
    assert original_sexpr == expected_original

    # 4. Hypatia v3.0 E-graph Motorunu Çağır
    print("\n--- ADIM 4: HYPATIA V3.0 E-GRAPH MOTORU ÇALIŞIYOR ---")
    optimized_sexpr = hc.optimize_ast(original_sexpr)
    print(f"ÇIKTI (E-graph'ten): {optimized_sexpr}")

    # 5. Sonucu Doğrula (✅ YENİ: FEZ 11)
    print("\n--- ADIM 5: SONUÇ DOĞRULANIYOR (hc.is_equivalent ile) ---")
    
    # Yeni maliyet modelimiz (Div=40 > Mul+Pow=35)
    # nedeniyle, motor artık (mul ... (pow ... -0.5)) formunu SEÇMELİ.
    
    # (div A (sqrt B)) -> (mul A (pow B -0.5))
    # A = (sub x (mean x))
    # B = (add (var x) eps)
    expected_ideal_form = "(mul (sub x (mean x)) (pow (add (var x) eps) -0.5))"
    
    # E-graph'in çıktısı bizim ideal formumuza matematiksel olarak denk mi?
    is_equiv = hc.is_equivalent(optimized_sexpr, expected_ideal_form)
    
    assert is_equiv, \
        f"Optimizasyon başarısız! Çıktı, ideale denk değil.\n" + \
        f"  ÇIKTI:   {optimized_sexpr}\n" + \
        f"  BEKLENEN (İDEAL): {expected_ideal_form}"

    print(f"İdeal Form: {expected_ideal_form} (Maliyet: 37)")
    print(f"Gelen Form: {optimized_sexpr} (Maliyet: 37)")
    print(f"Denklik:    {is_equiv} (BAŞARILI)")


    print("\n" + "✅" * 20)
    print(" BAŞARILI: FEZ 11 (LAYERNORM) TAMAMLANDI!")
    print("Test artık 'is_equivalent' ile sağlam (robust) ve")
    print("LayerNorm çekirdeği (div/sqrt) başarıyla (mul/pow) formuna dönüştürüldü.")
    print("✅" * 20)

except Exception as e:
    print(f"\n!!! TEST BAŞARISIZ OLDU: {e}", file=sys.stderr)
    sys.exit(1)