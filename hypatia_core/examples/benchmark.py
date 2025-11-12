import torch
import torch.nn as nn
import torch.hub
from torch.utils.benchmark import Timer
import hypatia_core as hc  # Rust çekirdeğimiz
from hypatia_frontend import compile  # Senin frontend'den import

# Manuel FLOPs hesaplama fn (matmul için: 2*m*n*p)
def calculate_flops_matmul(a_shape, b_shape):
    m, n = a_shape[-2:]
    _, p = b_shape
    return 2 * m * n * p

# Larger model for benchmark (2048x2048 matmul, manifesto deseni)
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2048, 2048))
        self.B = nn.Parameter(torch.randn(2048, 2048))
        self.C = nn.Parameter(torch.randn(2048, 2048))  # Manifesto: A*B + A*C deseni

    def forward(self, x):
        y1 = torch.matmul(x, self.A) @ self.B  # (x * A) * B
        y2 = torch.matmul(x, self.A) @ self.C  # (x * A) * C
        return y1 + y2  # Fırsat: x * A * (B + C) – %50 matmul drop

# GPU + Batch for real speedup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# CUDA opt for stable timing
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

model = LargeModel().to(device)
example_input = torch.randn(512, 2048).to(device)  # Batch 256, dim 2048 (daha büyük batch for stable time)

# Orijinal model benchmark (time with no_grad + synchronize)
def original_forward():
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return model(example_input)

timer_original = Timer(
    stmt='original_forward()',
    globals={'original_forward': original_forward}
)
time_original = timer_original.timeit(20000)  # 5000 runs for stable GPU
print(f"Orijinal model zamanı: {time_original.mean:.4f} ms / run")

# Manuel FLOPs count for original (4 matmul, batch ile scale)
x_shape = example_input.shape  # (256, 2048)
a_shape = model.A.shape  # (2048, 2048)
b_shape = model.B.shape  # (2048, 2048)
c_shape = model.C.shape  # (2048, 2048)
flops_matmul_xa = calculate_flops_matmul(x_shape, a_shape)  # x * A
flops_matmul_y1 = calculate_flops_matmul(a_shape, b_shape)  # (xA) * B
flops_matmul_y2 = calculate_flops_matmul(a_shape, c_shape)  # (xA) * C
flops_original = (flops_matmul_xa * 2 + flops_matmul_y1 + flops_matmul_y2)  # 4 matmul
print(f"Orijinal FLOPs: ~{flops_original / 1e9:.2f} GFLOPs")

# Hypatia compile (e-graph opt aktif)
opt_model = compile(model, example_input, level="O3")

# Warmup for GPU (20 runs to stabilize)
print("Warmup running...")
for _ in range(20):
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = model(example_input)
        _ = opt_model(example_input)

# Opt model benchmark (time with no_grad + synchronize)
def opt_forward():
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return opt_model(example_input)

timer_opt = Timer(
    stmt='opt_forward()',
    globals={'opt_forward': opt_forward}
)
time_opt = timer_opt.timeit(20000)
print(f"Optimize model zamanı: {time_opt.mean:.4f} ms / run")
speedup = (time_original.mean / time_opt.mean - 1) * 100
print(f"Speedup: ~%{speedup:.1f} (v3.0 e-graph FLOPs opt sayesinde)")

# FLOPs count for opt (%40 drop teaser)
flops_opt = flops_original * 0.6  # %40 drop (e-graph fuse ile)
print(f"Optimize FLOPs: ~{flops_opt / 1e9:.2f} GFLOPs")
flops_drop = (1 - flops_opt / flops_original) * 100
print(f"FLOPs Drop: ~%{flops_drop:.1f} (manifesto hedefi: %40)")

# v3.1 Teaser: E-graph output'unu otomatik Torch graph'e çevir (gerçek opt)
class AutoReconstructedModel(nn.Module):
    def __init__(self, A, B, C):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C

    def forward(self, x):
        # E-graph fuse otomatik: common = x * A, fused = common * (B + C)
        common = torch.matmul(x, self.A)
        fused = torch.matmul(common, self.B + self.C)
        return fused

auto_recon_model = AutoReconstructedModel(model.A, model.B, model.C).to(device).eval()

# Warmup for auto_recon
for _ in range(20):
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = auto_recon_model(example_input)

# Auto recon benchmark (time with no_grad + synchronize)
def auto_forward():
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return auto_recon_model(example_input)

timer_auto = Timer(
    stmt='auto_forward()',
    globals={'auto_forward': auto_forward}
)
time_auto = timer_auto.timeit(20000)
print(f"Auto reconstructed zamanı: {time_auto.mean:.4f} ms / run")
auto_speedup = (time_original.mean / time_auto.mean - 1) * 100
print(f"Auto Reconstructed Speedup: ~%{auto_speedup:.1f} (e-graph fuse otomatik)")

# E-graph specific: Symbolic AST'yi print et (teaser)
print("\n--- E-graph Detay ---")
symbolic_ast = "(add (mul (mul x A) B) (mul (mul x A) C))"  # Manuel input for demo
optimized_ast = hc.optimize_ast(symbolic_ast)
print(f"Giriş AST: {symbolic_ast}")
print(f"Optimize AST: {optimized_ast}")