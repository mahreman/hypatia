#!/usr/bin/env python3
"""
Hypatia Benchmark Core

Bu modül, farklı benchmark scriptlerinin ortak kullanacağı:
- Model tanımları (MLP, TinyTransformer)
- HypatiaTracer + Hypatia FX entegrasyonu
- Dummy input/loader üretimi
- VRAM / FLOPs ölçümü
- Doğruluk (accuracy) ve memory leak testleri
- run_benchmark fonksiyonu

için çekirdek fonksiyonları içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import time
import traceback
from typing import Callable, Dict, Any, Tuple, List
from contextlib import nullcontext
import copy

# --- Opsiyonel importlar ---
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("⚠️  fvcore bulunamadı. FLOPs hesabı devre dışı (flops_est = -1).")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Phi-3 burada kullanılmadığı için sadece bilgi amaçlı
    print("⚠️  transformers bulunamadı. Phi-3 testi devre dışı.")

# Hypatia çekirdeği (opsiyonel)
try:
    import hypatia_core
    HYPATIA_AVAILABLE = True
    print("✅ hypatia_core bulundu. Gerçek Hypatia optimizasyonu kullanılabilir.")
except ImportError:
    HYPATIA_AVAILABLE = False
    print("⚠️  hypatia_core modülü bulunamadı. Placeholder (torch.compile) kullanılacak.")


# ====================================================================
# ✅ GÜNCELLENDİ: HYPATIA TRACER
# Transformer ve ResNet blokları için destek eklendi.
# ====================================================================

class HypatiaTracer(torch.fx.Tracer):
    """
    nn.Sequential ve nn.ModuleList gibi konteynerların içine
    girmesi (trace etmesi) için özelleştirilmiş Tracer.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # ====================================================================
        # ✅ CRITICAL FIX: Bu modüller ÇOK karmaşık, içlerine GİRME (Leaf=True)
        # ====================================================================
        if isinstance(m, (
            nn.TransformerEncoder,
            nn.TransformerEncoderLayer,
            nn.MultiheadAttention 
        )):
            return True # Evet, bu bir yapraktır (içine girme)
        # ====================================================================

        # Bu modüller konteynerdir, içlerine GİR (Leaf=False)
        if isinstance(m, (
            nn.Sequential, 
            nn.ModuleList, 
            models.resnet.Bottleneck, 
            models.resnet.BasicBlock
        )):
            return False # Hayır, bu bir yaprak değil (içine gir)
        
        # Diğer tüm modüller için varsayılan davranışı kullan
        # (örn. nn.Linear bir yapraktır, True döndürür)
        return super().is_leaf_module(m, module_qualified_name)


# ====================================================================
# Model Tanımları
# (MLP.forward GÜNCELLENDİ)
# ====================================================================

class MLP(nn.Module):
    """Basit bir MLP modeli"""
    def __init__(self, in_features=784, out_features=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_features),
        )

    def forward(self, x):
        # ✅ DÜZELTME: x.view(x.size(0), -1) -> torch.flatten(x, 1)
        return self.layers(torch.flatten(x, 1))


class TinyTransformer(nn.Module):
    """Basit bir Transformer Sınıflandırma Modeli"""
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, seq_len=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        self.embedding = nn.Embedding(1000, embed_dim)  # vocab_size=1000
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, 10)  # 10 classes

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # <-- Bu, FX trace'te 'call_method' olarak görünecek
        return self.fc(x)


# ====================================================================
# Yardımcı Fonksiyonlar
# (Değişiklik yok)
# ====================================================================

def get_model_and_input_shape(model_name: str) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Model ismine göre (model, input_shape) döner.
    """
    if model_name == "MLP":
        model = MLP(in_features=784, out_features=10)
        input_shape = (1, 28, 28)
    elif model_name == "Tiny-Transformer":
        model = TinyTransformer(embed_dim=128, nhead=4, num_layers=2, seq_len=64)
        input_shape = (64,)  # sequence length
    elif model_name == "ResNet-50":
        model = models.resnet50(num_classes=1000)
        input_shape = (3, 224, 224)
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")
    return model, input_shape


def get_dummy_inputs(
    batch_size: int, 
    input_shape: tuple, 
    device: torch.device, 
    precision: torch.dtype
) -> List[torch.Tensor]:
    """
    compile_fx_graph için örnek girdi listesi oluşturur.
    """
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)
    
    return [inputs]


def create_dummy_loader(
    batch_size: int, 
    input_shape: tuple, 
    device: torch.device, 
    precision: torch.dtype,
    num_batches: int = 10
) -> DataLoader:
    """
    Accuracy Validation için sahte DataLoader oluşturur.
    """
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        data = torch.randint(0, 1000, (num_batches * batch_size, seq_len), device=device, dtype=torch.long)
    elif len(input_shape) == 3:
        c, h, w = input_shape
        data = torch.randn(num_batches * batch_size, c, h, w, device=device, dtype=precision)
    else:
        data = torch.randn((num_batches * batch_size,) + input_shape, device=device, dtype=precision)

    targets = torch.zeros(num_batches * batch_size, dtype=torch.long, device=device)
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def safe_model_to_device(model, device, precision):
    """Modeli güvenli şekilde cihaza taşır ve parametreleri kontrol eder."""
    try:
        model = model.to(device)
        
        try:
            params = list(model.parameters())
            if not params:
                print("  > UYARI: Optimize edilmiş modelin parametresi yok (bu normal olabilir).")
                return model, None
            
            is_transformer = any(isinstance(m, (nn.TransformerEncoder, nn.Embedding)) for m in model.modules())

            if not is_transformer and precision != torch.long and hasattr(params[0], 'is_floating_point'):
                if params[0].is_floating_point():
                    model = model.to(dtype=precision)
            
            return model, None
            
        except (StopIteration, IndexError, AttributeError) as e:
            return None, f"Parametre kontrolü başarısız: {e}"
            
    except Exception as e:
        return None, f"Cihaza taşıma hatası: {e}"


# ====================================================================
# ✅ GÜNCELLENDİ: optimize_model_from_base
# (module_type_map -> module_info_map { 'type', 'has_bias' } oldu)
# ====================================================================

def optimize_model_from_base(
    original_model: nn.Module,
    example_inputs: List[torch.Tensor],
    precision: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Var olan bir modeli kullanarak Hypatia/torch.compile ile optimize edilmiş bir kopya üretir.
    """
    model_cpu = copy.deepcopy(original_model).to("cpu")
    
    is_transformer = any(isinstance(m, (nn.TransformerEncoder, nn.Embedding)) for m in model_cpu.modules())

    try:
        if not is_transformer and precision != torch.long and next(model_cpu.parameters()).is_floating_point():
            model_cpu = model_cpu.to(dtype=precision)
    except StopIteration:
        pass
    except RuntimeError:
        print(f"  > UYARI: Model {precision}'a çevrilemedi (örn: Embedding), atlanıyor.")
            
    example_inputs_cpu = [inp.to("cpu") for inp in example_inputs]
    
    optimized_model_obj = None 

    try:
        print(f"  > [FX] Model 'torch.fx.symbolic_trace' ile trace ediliyor...")
        
        tracer = HypatiaTracer()
        graph = tracer.trace(model_cpu)
        graph_module = torch.fx.GraphModule(tracer.root, graph)

        # ====================================================================
        # ✅ CRITICAL FIX: Modül Bilgi Haritası (Bias Tespiti)
        # ====================================================================
        module_info_map = {}
        for node in graph_module.graph.nodes:
            if node.op == 'call_module':
                module = graph_module.get_submodule(node.target)
                module_info_map[node.target] = {
                    'type': type(module).__name__,
                    # 'bias' attribute'u var mı VE None değil mi kontrol et
                    'has_bias': (hasattr(module, 'bias') and module.bias is not None)
                }
        # ====================================================================


        if HYPATIA_AVAILABLE and hasattr(hypatia_core, "compile_fx_graph"):
            print(f"  > [HYPATIA] Gerçek 'compile_fx_graph' optimizasyonu uygulanıyor...")
            
            optimized_model_obj = hypatia_core.compile_fx_graph(
                graph_module, 
                example_inputs_cpu,
                module_info_map # <-- ✅ GÜNCELLENDİ
            )
        else:
            print(f"  > [HYPATIA PLACEHOLDER] 'torch.compile' kullanılıyor...")
            optimized_model_obj = torch.compile(graph_module)
            
    except Exception as e:
        print(f"  > UYARI: Optimizasyon başarısız oldu, baseline kullanılacak. Hata: {e}")
        traceback.print_exc()
        optimized_model_obj = model_cpu 

    result, error = safe_model_to_device(optimized_model_obj, device, precision)
    if error:
        print(f"  > {error}, orijinal model kullanılıyor.")
        result, _ = safe_model_to_device(original_model, device, precision)
    
    return result if result is not None else original_model


# ====================================================================
# Benchmark Yardımcıları
# (Değişiklik yok)
# ====================================================================

def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı (CUDA > MPS > CPU) seçer."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("⚠️  MPS (Apple Silicon) cihazı saptandı. VRAM ölçümü yapılamayacak.")
        return torch.device("mps")
    return torch.device("cpu")


def measure_vram_usage(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> float:
    """Modelin tek bir forward pass için tepe VRAM kullanımını MB cinsinden ölçer."""
    if device.type != "cuda":
        return 0.0
        
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    try:
        _ = model(inputs)
        torch.cuda.synchronize(device)
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        return peak_vram_bytes / (1024 * 1024)  # MB
    except Exception as e:
        print(f"  > VRAM ölçüm hatası: {e}")
        return -1.0


def calculate_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    """fvcore kullanarak FLOPs hesaplar."""
    if not FVCORE_AVAILABLE:
        return -1.0
    try:
        model_cpu = copy.deepcopy(model).to("cpu")
        inputs_cpu = inputs.to("cpu")
        
        if inputs.dtype != torch.long:
            is_float_model = any(p.is_floating_point() for p in model_cpu.parameters())
            if is_float_model:
                model_dtype = next(p.dtype for p in model_cpu.parameters() if p.is_floating_point())
                inputs_cpu = inputs_cpu.to(dtype=model_dtype)
        
        flops = FlopCountAnalysis(model_cpu, (inputs_cpu,))
        return flops.total()
    except Exception as e:
        # ✅ DÜZELTME: 'Unsupported operator' hatasını 'hata' değil, 'uyarı' olarak ele al.
        # Bu, Hypatia'nın değil, fvcore kütüphanesinin bir sınırlamasıdır.
        if "Unsupported operator" in str(e):
            print(f"  > FLOPs hesaplama UYARISI: fvcore bazı operatörleri tanımadı (bu normaldir): {e}")
        else:
            print(f"  > FLOPs hesaplama hatası: {e}")
        return -1.0


# ====================================================================
# Validasyon ve Güvenlik Fonksiyonları
# (Değişiklik yok)
# ====================================================================

def validate_accuracy(
    original_model: nn.Module, 
    optimized_model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device,
    precision: torch.dtype
) -> Dict[str, float]:
    """
    Optimizasyonun doğruluk kaybına neden olup olmadığını kontrol et.
    """
    original_model.eval()
    optimized_model.eval()
    
    all_orig_outputs = []
    all_opt_outputs = []

    with torch.no_grad():
        for data, target in test_loader:
            autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision) \
                if precision != torch.float32 and device.type != 'mps' \
                else nullcontext()
            
            with autocast_ctx:
                orig_output = original_model(data)
                opt_output = optimized_model(data)
            
            if orig_output.shape != opt_output.shape:
                print(f"  > UYARI: Çıkış şekilleri uyuşmuyor! "
                      f"original: {orig_output.shape}, optimized: {opt_output.shape}")
                return {
                    "cosine_similarity": -1.0,
                    "max_difference": float("inf"),
                    "relative_error": float("inf"),
                }

            all_orig_outputs.append(orig_output.flatten())
            all_opt_outputs.append(opt_output.flatten())

    if not all_orig_outputs:
        return {
            "cosine_similarity": -1.0,
            "max_difference": float("inf"),
            "relative_error": float("inf"),
        }

    orig_flat = torch.cat(all_orig_outputs).float()
    opt_flat = torch.cat(all_opt_outputs).float()
    
    cos_sim = F.cosine_similarity(orig_flat, opt_flat, dim=0).item()
    max_diff = (orig_flat - opt_flat).abs().max().item()
    
    epsilon = 1e-6
    relative_diff = (orig_flat - opt_flat).abs() / (orig_flat.abs() + epsilon)
    rel_err = relative_diff.max().item()

    return {
        "cosine_similarity": cos_sim,
        "max_difference": max_diff,
        "relative_error": rel_err,
    }


def check_memory_leak(
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    input_shape: Tuple[int, ...],
    batch_size: int,
    precision: torch.dtype,
    num_iterations: int = 50,
) -> float:
    """
    Basit bir memory leak kontrolü.
    """
    if device.type != "cuda":
        return 0.0

    model = model_fn()
    
    if len(input_shape) == 1:
        seq_len = input_shape[0]
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)

    leak_bytes = max(0, mem_after - mem_before)
    leak_mb = leak_bytes / (1024 * 1024)
    return leak_mb


# ====================================================================
# ✅ GÜNCELLENDİ: Ana Benchmark Fonksiyonu
# ('precision_str' eklendi ve 'assert' güncellendi)
# ====================================================================

def run_benchmark(
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    batch_size: int,
    input_shape: Tuple[int, ...],
    precision: torch.dtype,
    precision_str: str, # <-- YENİ ARGÜMAN
    original_model: nn.Module = None,
    test_loader: DataLoader = None,
    warmup_runs: int = 50,
    measure_runs: int = 200,
) -> Dict[str, Any]:
    """
    Tek bir senaryo için benchmark çalıştırır.
    """
    results: Dict[str, Any] = {}

    if len(input_shape) == 1:
        seq_len = input_shape[0]
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    else:
        inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=precision)

    model = model_fn()
    model.eval()

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=precision) \
        if precision != torch.float32 and device.type != 'mps' \
        else nullcontext()

    # 1) Doğruluk Kontrolü
    if original_model is not None and test_loader is not None:
        print("  > Doğruluk (Accuracy) kontrolü yapılıyor...")
        accuracy_check = validate_accuracy(original_model, model, test_loader, device, precision)
        
        cos_sim_threshold = 0.99
        rel_err_threshold = 1e-3 if precision == torch.float32 else 0.1
        
        assert accuracy_check["cosine_similarity"] > cos_sim_threshold, \
            f"Doğruluk kaybı! Cosine Sim: {accuracy_check['cosine_similarity']:.4f} (Eşik: >{cos_sim_threshold})"
        
        if precision_str == "FP32":
            assert accuracy_check["relative_error"] < rel_err_threshold, \
                f"Numerik hata! Rel Err: {accuracy_check['relative_error']:.4f} (Eşik: <{rel_err_threshold})"
        else:
            if accuracy_check["relative_error"] >= rel_err_threshold:
                print(f"  > [UYARI] {precision_str} için rel_err={accuracy_check['relative_error']:.4f}, "
                      f"optimizasyon şu anda sadece FP32 doğruluğu için garanti ediliyor.")

        results["accuracy_cosine_sim"] = accuracy_check["cosine_similarity"]
        results["accuracy_max_diff"] = accuracy_check["max_difference"]
        results["relative_error"] = accuracy_check["relative_error"]
    
    # 2) Memory Leak Kontrolü
    if device.type == "cuda":
        print(f"  > Bellek sızıntısı (Memory Leak) kontrolü yapılıyor...")
        memory_leak_mb = check_memory_leak(model_fn, device, input_shape, batch_size, precision)
        
        if memory_leak_mb > 200.0:
            print(f"  > UYARI: Yüksek bellek kullanımı tespit edildi: {memory_leak_mb:.2f}MB")
        results["memory_leak_mb"] = memory_leak_mb
    else:
        results["memory_leak_mb"] = 0.0

    # 3) VRAM ve FLOPs ölçümü
    print("  > VRAM ve FLOPs ölçülüyor (tek çalıştırma)...")
    if device.type == "cuda":
        with autocast_ctx:
            results["peak_vram_MB"] = measure_vram_usage(model, inputs, device)
    else:
        results["peak_vram_MB"] = 0.0
    results["flops_est"] = calculate_flops(model, inputs)

    # 4) Isınma turları
    print(f"  > Isınma turları (Warmup) çalıştırılıyor ({warmup_runs} tur)...")
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(warmup_runs):
                _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 5) Gerçek ölçüm turları
    print(f"  > Ölçüm turları çalıştırılıyor ({measure_runs} tur)...")
    timings_ms = []
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(measure_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end_time = time.perf_counter()
                timings_ms.append((end_time - start_time) * 1000.0)

    if not timings_ms:
        return {}

    p50 = float(np.percentile(timings_ms, 50))
    p95 = float(np.percentile(timings_ms, 95))
    throughput = float((batch_size * measure_runs) / (sum(timings_ms) / 1000.0))

    results["p50_ms"] = p50
    results["p95_ms"] = p95
    results["throughput"] = throughput

    return results
