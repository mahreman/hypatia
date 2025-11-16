import os
import sys
import time
from dataclasses import dataclass
from typing import List

# Add hypatia_core to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import hypatia_core  # Auto-registers 'hypatia' backend


# ---------------------------------------------------------------------
# Basit MLP tanımı (parametrik)
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc in self.fc_hidden:
            x = self.act(fc(x))
        x = self.fc_out(x)
        return x


# ---------------------------------------------------------------------
# Basit benchmark yardımcıları
# ---------------------------------------------------------------------
def bench(fn, x, iters: int = 200, device: torch.device = torch.device("cuda")) -> float:
    """Return ms/iter."""
    fn(x)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) * 1000.0 / iters


@dataclass
class Config:
    input_size: int
    hidden_size: int
    num_hidden_layers: int
    output_size: int
    batch_size: int

    def describe(self) -> str:
        return (f"in={self.input_size}, hid={self.hidden_size}×{self.num_hidden_layers}, "
                f"out={self.output_size}, batch={self.batch_size}")


def compute_flops(config: Config) -> float:
    """
    Çok kaba bir FLOP hesabı (Linear için 2 * m * n * k varsayımı):
    FLOPs/iter ~ 2 * batch * (in*hid + (L-1)*hid*hid + hid*out)
    """
    b = config.batch_size
    h = config.hidden_size
    L = config.num_hidden_layers
    inp = config.input_size
    out = config.output_size

    flops = 0.0
    # input -> hidden
    flops += 2.0 * b * inp * h
    # hidden layers
    flops += 2.0 * b * (L - 1) * h * h
    # last hidden -> out
    flops += 2.0 * b * h * out
    return flops


def gflops_per_sec(ms_per_iter: float, flops_per_iter: float) -> float:
    sec = ms_per_iter / 1000.0
    return flops_per_iter / sec / 1e9


# ---------------------------------------------------------------------
# Ana sweep
# ---------------------------------------------------------------------
def run_benchmark(configs: List[Config], iters: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================================================================================")
    print("Hypatia MLP Fusion Sweep")
    print("================================================================================")
    print(f"Device: {device}")
    print(f"Iterations per config: {iters}")
    print(f"HYPATIA_DEBUG_FX={os.getenv('HYPATIA_DEBUG_FX', '0')}")
    print(f"HYPATIA_ENABLE_LINRELU_FUSION={os.getenv('HYPATIA_ENABLE_LINRELU_FUSION', '0')}")
    print("Note: Use HYPATIA_ENABLE_LINRELU_FUSION=1 and backend='hypatia' for fusion.\n")

    header = (
        f"{'Config':55s} | {'Mode':7s} | {'ms/iter':8s} | {'GFLOP/s':9s} | {'Speedup vs eager':15s}"
    )
    print(header)
    print("-" * len(header))

    for cfg in configs:
        flops = compute_flops(cfg)

        # Model & input
        model_eager = MLP(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            output_size=cfg.output_size,
        ).to(device)

        x = torch.randn(cfg.batch_size, cfg.input_size, device=device)

        # Eager
        eager_ms = bench(model_eager, x, iters=iters, device=device)
        eager_gflops = gflops_per_sec(eager_ms, flops)

        cfg_str = cfg.describe()
        print(f"{cfg_str:55s} | {'eager':7s} | {eager_ms:8.4f} | {eager_gflops:9.2f} | {'-':15s}")

        # Hypatia compiled
        compiled = torch.compile(model_eager, backend="hypatia")

        hypatia_ms = bench(compiled, x, iters=iters, device=device)
        hypatia_gflops = gflops_per_sec(hypatia_ms, flops)
        speedup = eager_ms / hypatia_ms

        print(
            f"{'':55s} | {'hypatia':7s} | {hypatia_ms:8.4f} | {hypatia_gflops:9.2f} | {speedup:15.4f}"
        )
        print("-" * len(header))


def main():
    # Senin büyük workload'unu da içeren bir dizi config
    configs = [
        # Küçük MLP
        Config(input_size=784, hidden_size=256, num_hidden_layers=2, output_size=10, batch_size=256),
        # Orta
        Config(input_size=784, hidden_size=512, num_hidden_layers=3, output_size=100, batch_size=512),
        # Büyük (senin son testine yakın)
        Config(input_size=784, hidden_size=2048, num_hidden_layers=4, output_size=1000, batch_size=1024),
        # Daha da büyük istersen:
        Config(input_size=784, hidden_size=4096, num_hidden_layers=4, output_size=1000, batch_size=1024),
    ]

    run_benchmark(configs, iters=200)


if __name__ == "__main__":
    main()
