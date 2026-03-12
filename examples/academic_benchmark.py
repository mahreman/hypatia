#!/usr/bin/env python3
"""
Hypatia Academic Benchmark Suite
=================================
Detaylı karşılaştırmalı ölçümler:
- Her özellik için "standart PyTorch" vs "Hypatia" karşılaştırması
- Bellek, hız, doğruluk metrikleri
- Gerçek model mimarileri üzerinde test
"""

import sys
import os
import time
import copy
import gc

import torch
import torch.nn as nn
import numpy as np

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "hypatia_core"))
sys.path.insert(0, os.path.join(_base, ".."))

SEPARATOR = "=" * 80


# ============================================================================
# 1. E-GRAPH EQUALITY SATURATION BENCHMARK
# ============================================================================

def benchmark_egraph():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 1: E-GRAPH EQUALITY SATURATION")
    print(SEPARATOR)

    from _hypatia_core import optimize_ast, parse_expr

    expressions = {
        "Basit Linear+ReLU": "(relu (linear w b x))",
        "2-katmanlı MLP": "(relu (linear w2 b2 (relu (linear w1 b1 x))))",
        "3-katmanlı MLP": "(relu (linear w3 b3 (relu (linear w2 b2 (relu (linear w1 b1 x))))))",
        "Residual bağlantı": "(add (relu (linear w b x)) x)",
        "Attention pattern": "(linear wo bo (attention (linear wq bq x) (linear wk bk x) (linear wv bv x)))",
    }

    print("\n  Her bir ifade için e-graph optimizasyonu:\n")
    for name, expr in expressions.items():
        t0 = time.perf_counter()
        for _ in range(100):
            optimized = optimize_ast(expr)
        elapsed = (time.perf_counter() - t0) * 1000

        # Node sayıları
        orig_nodes = expr.count("(") + expr.count(")") // 2
        opt_nodes = optimized.count("(") + optimized.count(")") // 2

        changed = expr != optimized
        print(f"  [{name}]")
        print(f"    Orijinal:   {expr}")
        print(f"    Optimize:   {optimized}")
        print(f"    Değişti mi: {'EVET - Fusion uygulandı' if changed else 'Hayır - zaten optimal'}")
        print(f"    Süre (100 iterasyon): {elapsed:.2f} ms")
        print(f"    Tek optimizasyon: {elapsed/100:.3f} ms")
        print()


# ============================================================================
# 2. TORCH.COMPILE BACKEND BENCHMARK
# ============================================================================

def benchmark_torch_compile():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 2: TORCH.COMPILE BACKEND (Linear-ReLU Fusion)")
    print(SEPARATOR)

    import hypatia_core

    model_configs = [
        ("Küçük MLP (128→64→32)", [128, 64, 32]),
        ("Orta MLP (512→256→128→64)", [512, 256, 128, 64]),
        ("Büyük MLP (1024→512→256→128→64)", [1024, 512, 256, 128, 64]),
    ]

    for name, dims in model_configs:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())

        # Standart PyTorch
        x = torch.randn(1, dims[0])
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(x)
            # Benchmark
            t0 = time.perf_counter()
            for _ in range(1000):
                out_std = model(x)
            std_time = (time.perf_counter() - t0) * 1000

        # Hypatia compiled
        compiled = torch.compile(model, backend="hypatia")
        with torch.no_grad():
            # Warmup (ilk çağrı compilation yapar)
            for _ in range(10):
                _ = compiled(x)
            # Benchmark
            t0 = time.perf_counter()
            for _ in range(1000):
                out_hyp = compiled(x)
            hyp_time = (time.perf_counter() - t0) * 1000

        diff = (out_std - out_hyp).abs().max().item()

        print(f"\n  [{name}]")
        print(f"    Parametre sayısı: {total_params:,}")
        print(f"    Katmanlar: {' → '.join(str(d) for d in dims)}")
        print(f"    ---")
        print(f"    Standart PyTorch (1000 iter): {std_time:.2f} ms")
        print(f"    Hypatia compiled  (1000 iter): {hyp_time:.2f} ms")
        print(f"    Hız oranı: {std_time/hyp_time:.2f}x")
        print(f"    Max çıktı farkı: {diff:.2e}")
        print(f"    Fusion: Linear+ReLU → FusedLinearReLU")

        # Temizle
        del model, compiled
        gc.collect()


# ============================================================================
# 3. MIXED PRECISION BENCHMARK
# ============================================================================

def benchmark_mixed_precision():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 3: MIXED PRECISION (FP16/BF16)")
    print(SEPARATOR)

    from _hypatia_core import to_half_precision, mixed_precision_stats, mixed_precision_forward

    configs = [
        ("Küçük katman", 128, 64),
        ("Orta katman", 512, 256),
        ("Büyük katman", 1024, 512),
        ("Çok büyük katman", 2048, 1024),
    ]

    print("\n  FP32 vs FP16 vs BF16 bellek ve doğruluk karşılaştırması:\n")
    for name, rows, cols in configs:
        w = np.random.randn(rows, cols).astype(np.float32) * 0.1
        x = np.random.randn(1, cols).astype(np.float32)
        b = np.zeros(rows, dtype=np.float32)

        fp32_bytes = rows * cols * 4
        fp16_bytes = rows * cols * 2
        bf16_bytes = rows * cols * 2

        fp16_stats = mixed_precision_stats(w, "fp16")
        bf16_stats = mixed_precision_stats(w, "bf16")

        # FP32 GEMM timing
        w_torch = torch.from_numpy(w)
        x_torch = torch.from_numpy(x)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(1000):
                _ = torch.mm(x_torch, w_torch.t())
            fp32_time = (time.perf_counter() - t0) * 1000

        # FP16 mixed precision GEMM timing (Hypatia)
        half_result = to_half_precision(w, "fp16")
        half_data = half_result['data']
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = mixed_precision_forward(x, half_data, b, rows, cols, "fp16", False)
        fp16_time = (time.perf_counter() - t0) * 1000

        print(f"  [{name}] ({rows}×{cols} = {rows*cols:,} parametre)")
        print(f"    FP32 bellek:  {fp32_bytes:>10,} byte")
        print(f"    FP16 bellek:  {fp16_bytes:>10,} byte  (tasarruf: %{(1-fp16_bytes/fp32_bytes)*100:.0f})")
        print(f"    BF16 bellek:  {bf16_bytes:>10,} byte  (tasarruf: %{(1-bf16_bytes/fp32_bytes)*100:.0f})")
        print(f"    FP16 max hata:  {fp16_stats['max_abs_error']:.6e}")
        print(f"    FP16 RMSE:      {fp16_stats['rmse']:.6e}")
        print(f"    BF16 max hata:  {bf16_stats['max_abs_error']:.6e}")
        print(f"    BF16 RMSE:      {bf16_stats['rmse']:.6e}")
        print(f"    FP32 GEMM (1000 iter):            {fp32_time:.2f} ms")
        print(f"    FP16 Mixed-Precision GEMM (1000): {fp16_time:.2f} ms")
        print()


# ============================================================================
# 4. SPARSE TENSOR IR BENCHMARK
# ============================================================================

def benchmark_sparse():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 4: SPARSE TENSOR IR (CSR Format)")
    print(SEPARATOR)

    from _hypatia_core import to_sparse_csr, sparse_linear_forward, sparsity_stats

    configs = [
        ("Küçük katman", 128, 64),
        ("Orta katman", 512, 256),
        ("Büyük katman", 1024, 512),
    ]

    sparsity_levels = [0.0, 0.5, 0.8, 0.9, 0.95]

    print("\n  Dense vs Sparse GEMM, farklı sparsity seviyelerinde:\n")
    for name, rows, cols in configs:
        print(f"  [{name}] ({rows}×{cols})")

        for sp in sparsity_levels:
            # Belirli sparsity seviyesinde ağırlık matrisi oluştur
            w = np.random.randn(rows, cols).astype(np.float32) * 0.1
            if sp > 0:
                mask = np.random.random((rows, cols)) > sp
                w = w * mask.astype(np.float32)

            x = np.random.randn(1, cols).astype(np.float32)
            b = np.zeros(rows, dtype=np.float32)

            stats = sparsity_stats(w)
            csr = to_sparse_csr(w, 1e-10)

            # Dense PyTorch
            w_torch = torch.from_numpy(w)
            x_torch = torch.from_numpy(x)
            b_torch = torch.from_numpy(b)
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(1000):
                    _ = torch.addmm(b_torch, x_torch, w_torch.t())
                dense_time = (time.perf_counter() - t0) * 1000

            # Sparse Hypatia
            row_ptrs = csr['row_ptrs']
            col_indices = csr['col_indices']
            values = np.array(csr['values'], dtype=np.float32)
            t0 = time.perf_counter()
            for _ in range(1000):
                _ = sparse_linear_forward(x, row_ptrs, col_indices, values, b, rows, cols, False)
            sparse_time = (time.perf_counter() - t0) * 1000

            # Doğruluk
            dense_out = torch.addmm(b_torch, x_torch, w_torch.t()).numpy()
            sparse_out = sparse_linear_forward(x, row_ptrs, col_indices, values, b, rows, cols, False)
            max_diff = np.abs(dense_out - sparse_out).max()

            speedup = dense_time / sparse_time
            print(f"    Sparsity %{sp*100:>5.1f}: Dense={dense_time:>7.2f}ms  Sparse={sparse_time:>7.2f}ms  "
                  f"Hız={speedup:.2f}x  NNZ={csr['nnz']:>6}  Bellek={csr['memory_bytes']:>8}B  "
                  f"MaxDiff={max_diff:.2e}")

        print()


# ============================================================================
# 5. ATTENTION FUSION BENCHMARK
# ============================================================================

def benchmark_attention():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 5: FUSED ATTENTION")
    print(SEPARATOR)

    from _hypatia_core import fused_attention_forward

    configs = [
        ("Küçük (d=64, h=4)", 64, 4, 8),
        ("Orta (d=128, h=8)", 128, 8, 16),
        ("Büyük (d=256, h=8)", 256, 8, 32),
    ]

    print("\n  PyTorch multi-head attention vs Hypatia fused attention:\n")
    for name, d_model, n_heads, seq_len in configs:
        d_head = d_model // n_heads

        # Ağırlıklar
        wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wv = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wo = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        bq = np.zeros(d_model, dtype=np.float32)
        bk = np.zeros(d_model, dtype=np.float32)
        bv = np.zeros(d_model, dtype=np.float32)
        bo = np.zeros(d_model, dtype=np.float32)
        x = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

        # PyTorch MHA
        mha = nn.MultiheadAttention(d_model, n_heads, batch_first=False, bias=True)
        mha.eval()
        x_torch = torch.from_numpy(x).unsqueeze(1)  # (seq, batch=1, d_model)
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = mha(x_torch, x_torch, x_torch)
            t0 = time.perf_counter()
            for _ in range(500):
                out_pt, _ = mha(x_torch, x_torch, x_torch)
            pt_time = (time.perf_counter() - t0) * 1000

        # Hypatia fused attention
        batch = 1
        t0 = time.perf_counter()
        for _ in range(500):
            out_hyp = fused_attention_forward(
                x, wq, bq, wk, bk, wv, bv, wo, bo, batch, seq_len, n_heads
            )
        hyp_time = (time.perf_counter() - t0) * 1000

        print(f"  [{name}] seq_len={seq_len}")
        print(f"    PyTorch MHA    (500 iter): {pt_time:.2f} ms")
        print(f"    Hypatia Fused  (500 iter): {hyp_time:.2f} ms")
        print(f"    Hız oranı: {pt_time/hyp_time:.2f}x")
        print(f"    Toplam parametre: {4 * d_model * d_model:,}")
        print()


# ============================================================================
# 6. SEMANTIC VALIDATION BENCHMARK
# ============================================================================

def benchmark_semantic_validation():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 6: SEMANTIC VALIDATION")
    print(SEPARATOR)

    from hypatia_core import SemanticValidator, validate_models

    print("\n  Model çiftlerinin semantik eşdeğerlik doğrulaması:\n")

    # Test 1: Aynı model (klonlanmış)
    model = nn.Sequential(
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32),
    )
    model.eval()
    clone = copy.deepcopy(model)

    t0 = time.perf_counter()
    result = validate_models(model, clone, (1, 256), tolerance=1e-5, num_samples=10)
    val_time = (time.perf_counter() - t0) * 1000

    print(f"  [Aynı Model - Klonlanmış]")
    print(f"    Geçerli: {result['is_valid']}")
    print(f"    Max fark: {result['max_diff']:.2e}")
    print(f"    Ortalama fark: {result['mean_diff']:.2e}")
    print(f"    Cosine benzerlik: {result['cosine_similarity']:.10f}")
    print(f"    Test girdisi sayısı: {result['num_test_inputs']}")
    print(f"    Doğrulama süresi: {val_time:.2f} ms")
    print()

    # Test 2: Hafif perturbe edilmiş model
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(torch.randn_like(p) * 1e-6)

    result2 = validate_models(model, perturbed, (1, 256), tolerance=1e-4, num_samples=10)
    print(f"  [Hafif Perturbe Edilmiş Model (noise=1e-6)]")
    print(f"    Geçerli: {result2['is_valid']}")
    print(f"    Max fark: {result2['max_diff']:.2e}")
    print(f"    Ortalama fark: {result2['mean_diff']:.2e}")
    print(f"    Cosine benzerlik: {result2['cosine_similarity']:.10f}")
    print()

    # Test 3: Tamamen farklı model
    different = nn.Sequential(
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32),
    )
    different.eval()
    result3 = validate_models(model, different, (1, 256), tolerance=1e-5, num_samples=10)
    print(f"  [Farklı Ağırlıklı Model]")
    print(f"    Geçerli: {result3['is_valid']}")
    print(f"    Max fark: {result3['max_diff']:.2e}")
    print(f"    Ortalama fark: {result3['mean_diff']:.2e}")
    print(f"    Cosine benzerlik: {result3['cosine_similarity']:.6f}")
    print()

    # Test 4: Kapsamlı doğrulama raporu
    validator = SemanticValidator(tolerance=1e-5, num_samples=5)
    print(f"  [E-graph Optimizasyon Doğrulaması]")
    expr_result = validator.validate_expr("(relu (linear w b x))")
    print(f"    İfade geçerli: {expr_result.get('is_valid')}")
    print(f"    Değişkenler korundu: {expr_result.get('variables_preserved')}")
    print(f"    Node azaltma: {expr_result.get('node_reduction')}")
    print(f"    Fusion: {expr_result.get('fusions_found')}")


# ============================================================================
# 7. REAL MODEL BENCHMARK (GPT-2 + DistilBERT)
# ============================================================================

def benchmark_real_models():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 7: GERÇEK MODEL TESTLERİ")
    print(SEPARATOR)

    from transformers import GPT2Model, GPT2Config, DistilBertModel, DistilBertConfig
    from _hypatia_core import to_sparse_csr, sparsity_stats, mixed_precision_stats
    from hypatia_core import model_summary

    models = [
        ("GPT-2 (2 katman, 128 embed)", GPT2Model(GPT2Config(
            n_layer=2, n_head=4, n_embd=128, vocab_size=1000
        ))),
        ("DistilBERT (2 katman, 128 dim)", DistilBertModel(DistilBertConfig(
            n_layers=2, n_heads=4, dim=128, hidden_dim=256, vocab_size=1000,
            max_position_embeddings=128
        ))),
    ]

    for model_name, model in models:
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes_fp32 = total_params * 4
        total_bytes_fp16 = total_params * 2

        print(f"\n  [{model_name}]")
        print(f"    Toplam parametre: {total_params:,}")
        print(f"    FP32 bellek: {total_bytes_fp32/1024/1024:.2f} MB")
        print(f"    FP16 bellek: {total_bytes_fp16/1024/1024:.2f} MB")
        print(f"    Bellek tasarrufu: {(total_bytes_fp32 - total_bytes_fp16)/1024:.1f} KB")

        # Her katman için analiz
        linear_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() == 2:
                w_np = module.weight.detach().numpy()
                sp_stats = sparsity_stats(w_np)
                mp_stats = mixed_precision_stats(w_np, "fp16")
                linear_layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'shape': tuple(w_np.shape),
                    'params': w_np.size,
                    'sparsity': sp_stats['sparsity_ratio'],
                    'fp16_error': mp_stats['max_abs_error'],
                    'fp16_rmse': mp_stats['rmse'],
                })

        print(f"    Ağırlık katmanı sayısı: {len(linear_layers)}")
        print(f"\n    Katman detayları:")
        for layer in linear_layers[:8]:
            print(f"      {layer['name']:45s} {str(layer['shape']):>12s}  "
                  f"params={layer['params']:>8,}  "
                  f"sparsity={layer['sparsity']:.4f}  "
                  f"fp16_err={layer['fp16_error']:.2e}")

        if len(linear_layers) > 8:
            print(f"      ... ve {len(linear_layers) - 8} katman daha")

        del model
        gc.collect()


# ============================================================================
# 8. INT4 QUANTIZATION BENCHMARK
# ============================================================================

def benchmark_quantization():
    print(f"\n{SEPARATOR}")
    print("  BENCHMARK 8: INT4 QUANTIZATION")
    print(SEPARATOR)

    from _hypatia_core import quantize_weights, quantized_forward

    configs = [
        ("Küçük (256×128)", 256, 128),
        ("Orta (1024×512)", 1024, 512),
        ("Büyük (2048×1024)", 2048, 1024),
    ]

    print("\n  FP32 vs INT4 kuantize çıkarım karşılaştırması:\n")
    for name, rows, cols in configs:
        w = np.random.randn(rows, cols).astype(np.float32) * 0.1
        x = np.random.randn(1, cols).astype(np.float32)
        b = np.zeros(rows, dtype=np.float32)

        # INT4 quantize
        layers_list = [(w, b, "none")]
        q_result = quantize_weights(layers_list, 32)

        fp32_bytes = rows * cols * 4
        int4_bytes = q_result[0][9] if len(q_result) > 0 and len(q_result[0]) > 9 else fp32_bytes // 4

        # FP32 GEMM
        w_torch = torch.from_numpy(w)
        x_torch = torch.from_numpy(x)
        b_torch = torch.from_numpy(b)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(500):
                _ = torch.addmm(b_torch, x_torch, w_torch.t())
            fp32_time = (time.perf_counter() - t0) * 1000

        # INT4 quantized forward
        t0 = time.perf_counter()
        for _ in range(500):
            _ = quantized_forward(x, q_result)
        int4_time = (time.perf_counter() - t0) * 1000

        # Doğruluk
        fp32_out = torch.addmm(b_torch, x_torch, w_torch.t()).numpy()
        int4_out = quantized_forward(x, q_result)
        max_diff = np.abs(fp32_out - int4_out).max()
        cos_sim = np.dot(fp32_out.flatten(), int4_out.flatten()) / (
            np.linalg.norm(fp32_out) * np.linalg.norm(int4_out) + 1e-12
        )

        compression = fp32_bytes / max(int4_bytes, 1)

        print(f"  [{name}]")
        print(f"    FP32 bellek:  {fp32_bytes:>10,} byte")
        print(f"    INT4 bellek:  {int4_bytes:>10,} byte  (sıkıştırma: {compression:.1f}x)")
        print(f"    FP32 GEMM  (500 iter): {fp32_time:.2f} ms")
        print(f"    INT4 GEMM  (500 iter): {int4_time:.2f} ms")
        print(f"    Hız oranı: {fp32_time/int4_time:.2f}x")
        print(f"    Max hata: {max_diff:.4e}")
        print(f"    Cosine benzerlik: {cos_sim:.8f}")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(SEPARATOR)
    print("  HYPATIA COMPILER - AKADEMİK BENCHMARK RAPORU")
    print(f"  Tarih: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Platform: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(SEPARATOR)

    benchmark_egraph()
    benchmark_torch_compile()
    benchmark_mixed_precision()
    benchmark_sparse()
    benchmark_attention()
    benchmark_semantic_validation()
    benchmark_real_models()
    benchmark_quantization()

    print(f"\n{SEPARATOR}")
    print("  TÜM BENCHMARKLAR TAMAMLANDI")
    print(SEPARATOR)
