import torch
import torch.nn as nn
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class BenchmarkResult:
    model_name: str
    input_shape: tuple
    original_time_ms: float
    compiled_time_ms: float
    speedup_percent: float
    param_count: int
    fallback: bool

class HypatiaBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.results: List[BenchmarkResult] = []
    
    def benchmark_model(self, model: nn.Module, input_shape: tuple, 
                       n_warmup=10, n_runs=100, name="model"):
        """Tek bir model için benchmark"""
        model = model.eval().to(self.device)
        x = torch.randn(*input_shape).to(self.device)
        
        # Orijinal model timing
        original_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                _ = model(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Timing
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                original_times.append((time.perf_counter() - start) * 1000)  # ms
        
        # Compiled model timing
        try:
            compiled = torch.compile(model, backend="hypatia")
            compiled_times = []
            fallback = False
            
            with torch.no_grad():
                # Warmup
                for _ in range(n_warmup):
                    _ = compiled(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # Timing
                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = compiled(x)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    compiled_times.append((time.perf_counter() - start) * 1000)
            
        except Exception as e:
            print(f"⚠️  Compilation failed for {name}: {e}")
            compiled_times = original_times  # Fallback
            fallback = True
        
        # Sonuçları hesapla
        original_mean = sum(original_times) / len(original_times)
        compiled_mean = sum(compiled_times) / len(compiled_times)
        speedup = (original_mean - compiled_mean) / original_mean * 100
        
        result = BenchmarkResult(
            model_name=name,
            input_shape=input_shape,
            original_time_ms=original_mean,
            compiled_time_ms=compiled_mean,
            speedup_percent=speedup,
            param_count=sum(p.numel() for p in model.parameters()),
            fallback=fallback
        )
        
        self.results.append(result)
        return result
    
    def run_suite(self):
        """Tüm modelleri test et"""
        # 1. MLP
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 10)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        self.benchmark_model(SimpleMLP(), (32, 784), name="MLP")
        
        # 2. CNN
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.fc = nn.Linear(32 * 6 * 6, 10)
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, 2)
                return self.fc(x.view(x.size(0), -1))
        
        self.benchmark_model(SimpleCNN(), (1, 3, 32, 32), name="CNN")
        
        # 3. Factorization test (benchmark.py'den)
        class FactorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.A = nn.Parameter(torch.randn(1024, 1024))
                self.B = nn.Parameter(torch.randn(1024, 1024))
                self.C = nn.Parameter(torch.randn(1024, 1024))
            def forward(self, x):
                return torch.matmul(torch.matmul(x, self.A), self.B) + \
                       torch.matmul(torch.matmul(x, self.A), self.C)
        
        self.benchmark_model(FactorModel(), (1024, 1024), name="MatMul-Factor")
        
    def print_results(self):
        """Sonuçları yazdır"""
        print("\n" + "="*80)
        print("HYPATIA BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'Input Shape':<20} {'Original (ms)':<15} {'Compiled (ms)':<15} {'Speedup':<10} {'Fallback'}")
        print("-"*80)
        
        for r in self.results:
            print(f"{r.model_name:<20} {str(r.input_shape):<20} "
                  f"{r.original_time_ms:>10.3f} ms   "
                  f"{r.compiled_time_ms:>10.3f} ms   "
                  f"{r.speedup_percent:>6.1f}%     "
                  f"{'✅' if not r.fallback else '⚠️'}")
        
        print("="*80)
        
        # Özet istatistikler
        successful = [r for r in self.results if not r.fallback]
        if successful:
            avg_speedup = sum(r.speedup_percent for r in successful) / len(successful)
            print(f"\nAverage Speedup (non-fallback): {avg_speedup:.1f}%")
            print(f"Success Rate: {len(successful)}/{len(self.results)} ({len(successful)/len(self.results)*100:.1f}%)")
        
    def save_results(self, filename="benchmark_results.json"):
        """Sonuçları JSON'a kaydet"""
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"\nResults saved to {filename}")

# Kullanım
if __name__ == "__main__":
    benchmark = HypatiaBenchmark(device='cuda')
    benchmark.run_suite()
    benchmark.print_results()
    benchmark.save_results()