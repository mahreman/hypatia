import hypatia_core as hc

# Test numeric multivectors
mv2d = hc.PyMultiVector2D.vector(1.0, 2.0)
print(mv2d)  # Should print: MV2D(s=0.000, e1=1.000, e2=2.000, e12=0.000)

# Test symbolic parsing & optimization
expr = "(add (mul x 2) 3)"
parsed = hc.parse_expr(expr)
optimized = hc.optimize_ast(expr)
print(f"Original: {expr}")
print(f"Optimized: {optimized}")  # Should simplify to "(add x 5)" or similar

# Test FX graph compilation (Phase 2: identity pass-through)
import torch
from torch.fx import symbolic_trace

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1.0)

model = SimpleModel()
traced = symbolic_trace(model)
optimized_gm = hc.compile_fx_graph(traced)
print("FX compilation succeeded:", optimized_gm is not None)  # True