# FEZ 10: Shape-Guard + FLOPs Maliyet Modeli ile Güvenli Factoring
# Çalıştır: python examples/test_shape_guard_and_cost.py

import math
from typing import Dict, Tuple, Any, List, Union
import operator as op  # built-in operator hedefleri için
import torch
import torch.fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp
import hypatia_core as hc  # editable install

# ---------------------------
# Basit S-Expr Parser/Formatter
# ---------------------------
Token = str
SExpr = Union[str, List["SExpr"]]

def _tokenize(src: str) -> List[Token]:
    src = src.replace("(", " ( ").replace(")", " ) ")
    toks = [t for t in src.strip().split() if t]
    return toks

def _parse(tokens: List[Token]) -> SExpr:
    stack: List[List[SExpr]] = []
    cur: List[SExpr] = []
    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            finished = cur
            cur = stack.pop()
            cur.append(finished)
        else:
            cur.append(tok)
    if stack:
        raise ValueError("Unbalanced parentheses at end")
    if len(cur) == 1:
        return cur[0]
    return cur

def parse_sexpr(src: str) -> SExpr:
    return _parse(_tokenize(src))

def sexpr_to_str(se: SExpr) -> str:
    if isinstance(se, str):
        return se
    return "(" + " ".join(sexpr_to_str(x) for x in se) + ")"

# ---------------------------
# Pattern: (add (mul A B) (mul A C)) → A, B, C
# ---------------------------
def match_add_mul_mul(se: SExpr):
    if not isinstance(se, list) or len(se) != 3:
        return None
    if se[0] != "add":
        return None
    left, right = se[1], se[2]
    if not (isinstance(left, list) and len(left) == 3 and left[0] == "mul"):
        return None
    if not (isinstance(right, list) and len(right) == 3 and right[0] == "mul"):
        return None
    A1, B = left[1], left[2]
    A2, C = right[1], right[2]
    if not (isinstance(A1, str) and isinstance(A2, str) and A1 == A2):
        return None
    if not (isinstance(B, str) and isinstance(C, str)):
        return None
    return (A1, B, C)

# ---------------------------
# Shape inference yardımcıları
# ---------------------------
Shape = Tuple[int, ...]
def matmul_shape(a: Shape, b: Shape) -> Shape:
    if len(a) == 2 and len(b) == 2:
        m, k1 = a
        k2, n = b
        if k1 != k2:
            raise ValueError(f"Matmul shape mismatch: {a} x {b}")
        return (m, n)
    if len(a) >= 2 and len(b) >= 2:
        *ba, m, k1 = a
        *bb, k2, n = b
        if ba != bb:
            raise ValueError(f"Batch dims must match exactly: {a} x {b}")
        if k1 != k2:
            raise ValueError(f"K mismatch: {a} x {b}")
        return tuple(ba) + (m, n)
    raise ValueError(f"Unsupported ranks for matmul: {a}, {b}")

def add_shape(a: Shape, b: Shape) -> Shape:
    if a != b:
        raise ValueError(f"Add shape mismatch: {a} vs {b}")
    return a

def flops_matmul(a: Shape, b: Shape) -> int:
    if len(a) == 2 and len(b) == 2:
        m, k = a
        _, n = b
        return 2 * m * k * n
    if len(a) >= 2 and len(b) >= 2:
        *batch, m, k = a
        *_, _, n = b
        B = math.prod(batch) if batch else 1
        return 2 * B * m * k * n
    return 0

def flops_add(shape: Shape) -> int:
    return math.prod(shape)

# ---------------------------
# Guarded factoring + maliyet denetimi
# ---------------------------
def try_factor_with_guards(original: str, shapes: Dict[str, Shape]) -> str:
    root = parse_sexpr(original)
    hit = match_add_mul_mul(root)
    if not hit:
        return original
    A, B, C = hit
    if A not in shapes or B not in shapes or C not in shapes:
        return original

    try:
        ab = matmul_shape(shapes[A], shapes[B])
        ac = matmul_shape(shapes[A], shapes[C])
        out = add_shape(ab, ac)
        flops_orig = flops_matmul(shapes[A], shapes[B]) + flops_matmul(shapes[A], shapes[C]) + flops_add(out)
    except Exception:
        return original

    try:
        bc = add_shape(shapes[B], shapes[C])
        _abc = matmul_shape(shapes[A], bc)
        flops_fact = flops_add(bc) + flops_matmul(shapes[A], bc)
    except Exception:
        return original

    if flops_fact >= flops_orig:
        return original

    factored = ["mul", A, ["add", B, C]]
    return sexpr_to_str(factored)

# ---------------------------
# FX → S-Expr (sağlamlaştırılmış op eşleme)
# ---------------------------
def _op_to_symbol(target) -> str:
    # 1) built-in operator.*
    if target in (op.add,): return "add"
    if target in (op.sub,): return "sub"
    if target in (op.mul,): return "mul"
    if target in (op.truediv, op.floordiv): return "div"
    if target in (op.matmul,): return "mul"
    # 2) torch üst-seviye
    if target in (torch.add,): return "add"
    if target in (torch.sub,): return "sub"
    if target in (torch.mul,): return "mul"
    if target in (torch.div,): return "div"
    if target in (torch.matmul,): return "mul"
    # 3) aten overload
    try:
        if target is torch.ops.aten.add.Tensor: return "add"
        if target is torch.ops.aten.sub.Tensor: return "sub"
        if target is torch.ops.aten.mul.Tensor: return "mul"
        if target is torch.ops.aten.div.Tensor: return "div"
        if target is torch.ops.aten.matmul.default: return "mul"
    except Exception:
        pass
    # 4) string fallback
    s = str(target)
    if "matmul" in s: return "mul"
    if "add" in s: return "add"
    if "sub" in s: return "sub"
    if "mul" in s: return "mul"
    if "div" in s: return "div"
    raise NotImplementedError(f"Unsupported op: {target}")

def fx_to_sexpr(gm: torch.fx.GraphModule) -> str:
    env: Dict[str, str] = {}
    out_expr = None
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            env[n.name] = n.name.lower()  # A→a
        elif n.op == "call_function":
            op_sym = _op_to_symbol(n.target)
            args = []
            for a in n.args:
                if isinstance(a, torch.fx.Node):
                    args.append(env[a.name])
                else:
                    raise NotImplementedError("Const args not handled in this demo")
            env[n.name] = f"({op_sym} {' '.join(args)})"
        elif n.op == "output":
            out_expr = env[n.args[0].name]
    if out_expr is None:
        raise RuntimeError("No output expr")
    return out_expr

def fx_shapes(gm: torch.fx.GraphModule, example_inputs: Dict[str, torch.Tensor]) -> Dict[str, Shape]:
    """
    Placeholder sırasına göre example_inputs besler.
    Anahtar eşleşmesi büyük/küçük harf duyarsızdır:
      - Graph'taki 'a' için {"A": tensör} da geçerlidir.
    """
    sp = ShapeProp(gm)
    args_in_order = []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            key = n.name
            t = (example_inputs.get(key) or
                 example_inputs.get(key.lower()) or
                 example_inputs.get(key.upper()))
            if t is None:
                raise KeyError(
                    f"Missing example input for placeholder '{key}'. "
                    f"Provided keys: {list(example_inputs.keys())}"
                )
            args_in_order.append(t)
    sp.propagate(*args_in_order)

    shapes: Dict[str, Shape] = {}
    for n in gm.graph.nodes:
        tm = n.meta.get("tensor_meta", None)
        if tm is not None:
            shapes[n.name] = tuple(tm.shape)

    # S-Expr küçük harf (a,b,c) kullandığı için map edelim
    mapped: Dict[str, Shape] = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder" and n.name in shapes:
            mapped[n.name.lower()] = shapes[n.name]
    return mapped

# ---------------------------
# Demo / Test
# ---------------------------
class Simple(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A, B, C):
        return torch.matmul(A, B) + torch.matmul(A, C)

def main():
    print("=== FEZ 10: Shape-Guard + FLOPs Cost ===")
    torch.manual_seed(0)
    m, k, n = 64, 128, 32
    A = torch.randn(m, k, dtype=torch.double)
    B = torch.randn(k, n, dtype=torch.double)
    C = torch.randn(k, n, dtype=torch.double)

    model = Simple()
    gm = torch.fx.symbolic_trace(model)

    original = fx_to_sexpr(gm)
    print("[FX→S-Expr]:", original)

    # Büyük/küçük fark etmeyecek; aşağıdaki gibi geçebilir:
    shapes = fx_shapes(gm, {"A": A, "B": B, "C": C})
    print("[Shapes]:", shapes)

    guarded = try_factor_with_guards(original, shapes)
    print("[Guarded ]:", guarded)

    try:
        optimized = hc.optimize_ast(original)
    except AttributeError:
        optimized = hc.optimizeast(original)
    print("[E-graph ]:", optimized)

    # Sayısal doğrulama (forward) — varsa guarded yolunu kontrol et
    lhs = A @ B + A @ C
    if guarded.startswith("(mul"):
        rhs_guard = A @ (B + C)
        assert torch.allclose(lhs, rhs_guard, atol=1e-10, rtol=1e-10), "Numeric mismatch!"
        print("[Numeric ]: OK (forward)")

    print("\nÖZET")
    print(" - Orijinal :", original)
    print(" - Guarded  :", guarded)
    print(" - E-graph  :", optimized)

if __name__ == "__main__":
    main()
