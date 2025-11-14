import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hypatia import bloğu (aynı)
try:
    import hypatia_core
    HYPATIA_AVAILABLE = True
    print("✅ hypatia_core bulundu. Gerçek Hypatia optimizasyonu kullanılabilir.")
except ImportError:
    HYPATIA_AVAILABLE = False
    print("⚠️ hypatia_core modülü bulunamadı. Placeholder (torch.compile) kullanılacak.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def setup_phi3():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading Phi-3: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    print(f"Model dtype: {dtype}, VRAM: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")
    return model, tokenizer

def benchmark_phi3(model, tokenizer, optimized=False, compile_skip=False):
    prompt = "Hello, this is a short benchmark prompt for Phi-3."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if optimized and not compile_skip:
        if HYPATIA_AVAILABLE and hasattr(hypatia_core, 'compile_fx_graph'):
            print("Optimizing with hypatia_core.compile_fx_graph...")
            traced = torch.fx.symbolic_trace(model)
            example_inputs = [inputs['input_ids']]
            model = hypatia_core.compile_fx_graph(traced, example_inputs)
        else:
            print("Optimizing with torch.compile (placeholder, reduce-overhead mode)...")
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")  # DÜZELTME: LLM için optimize
    elif compile_skip:
        print("Skipping compile for fair baseline comparison.")

    # VRAM clear (DÜZELTME: Şişkinlik önle)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Warmup artırıldı (50 iter, çok kısa tokens)
    with torch.no_grad():
        for _ in range(50):
            _ = model.generate(**inputs, max_new_tokens=4)  # Çok kısa warmup

    # Measure 5 iter ortalaması
    latencies = []
    with torch.no_grad():
        for _ in range(5):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            _ = model.generate(**inputs, max_new_tokens=8)  # Kısa measure
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()
            latencies.append((end - start) * 1000.0)

    avg_latency_ms = np.mean(latencies)
    print(f"{'Hypatia/Opt' if optimized else 'Baseline'} Latency (avg 5 runs): {avg_latency_ms:.2f} ms")
    return avg_latency_ms

# Run test
print("\n--- Baseline (no compile) ---")
model_base, tokenizer_base = setup_phi3()
latency_base = benchmark_phi3(model_base, tokenizer_base, False, compile_skip=True)

print("\n--- Optimized (with compile) ---")
model_opt, tokenizer_opt = setup_phi3()
latency_opt = benchmark_phi3(model_opt, tokenizer_opt, True)

# Speedup hesapla
if latency_base and latency_opt:
    speedup = latency_base / latency_opt
    print(f"\nSpeedup: {speedup:.2f}x (Baseline: {latency_base:.2f} ms → Opt: {latency_opt:.2f} ms)")
    if HYPATIA_AVAILABLE:
        print("✅ Gerçek Hypatia hazır – e-graph ile speedup artar.")
    else:
        print("⚠️ Placeholder aktif – Rust entegrasyonu bekleniyor.")
else:
    print("Test failed.")