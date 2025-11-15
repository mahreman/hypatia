import torch
from torch import nn

torch._dynamo.reset()

class ToyModel(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024):
        super().__init__()
        self.a = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.c = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        # x @ a @ b + x @ a @ c
        a = x @ self.a
        ab = a @ self.b
        ac = a @ self.c
        return ab + ac

def test_hypatia_speedup_and_correctness():
    device = "cuda"
    model = ToyModel().to(device)
    x = torch.randn(256, 1024, device=device)

    # Eager
    with torch.no_grad():
        y_eager = model(x)
    eager_time = torch.utils.benchmark.Timer(
        stmt="model(x)",
        globals={"model": model, "x": x},
    ).blocked_autorange().mean

    # Hypatia backend
    opt_model = torch.compile(model, backend="hypatia")
    with torch.no_grad():
        y_opt = opt_model(x)

    # Sayısal doğruluk
    max_diff = (y_eager - y_opt).abs().max().item()
    rel_err = max_diff / (y_eager.abs().max().item() + 1e-8)

    assert rel_err < 1e-5, f"Numeric mismatch: rel_err={rel_err}"

    opt_time = torch.utils.benchmark.Timer(
        stmt="opt_model(x)",
        globals={"opt_model": opt_model, "x": x},
    ).blocked_autorange().mean

    speedup = eager_time / opt_time
    print(f"Eager: {eager_time*1e3:.3f} ms, Opt: {opt_time*1e3:.3f} ms, speedup={speedup:.2f}x")

    # Çok agresif olma, ama en azından 1.1x bekle
    assert speedup > 1.1, f"Beklenen hızlanma yok: speedup={speedup}"
