#!/usr/bin/env python3
"""
ResNet-50 Benchmark Script (Hypatia)

Senaryolar:
- ResNet-50-Small  (batch=16)
- ResNet-50-Large  (batch=64)
"""

import argparse
import os
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from bench_core import (
    get_device,
    get_model_and_input_shape,
    create_dummy_loader,
    get_dummy_inputs,
    optimize_model_from_base,
    run_benchmark,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=str,
        default=f"results/benchmark_resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Benchmark sonuçlarının yazılacağı CSV dosyası",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    device = get_device()
    print("=" * 80)
    print("HYPATIA RESNET-50 BENCHMARK")
    print("=" * 80)
    print(f"Cihaz:      {device}")
    print(f"Çıktı CSV:  {args.output_csv}")
    print("-" * 80)

    SCENARIOS = {
        "ResNet-50-Small": (16, "ResNet-50"),
        "ResNet-50-Large": (64, "ResNet-50"),
    }

    MIXED_PRECISION_SCENARIOS = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }

    all_results = []
    baseline_perfs = {}

    for precision_str, precision in MIXED_PRECISION_SCENARIOS.items():
        if precision == torch.float16 and device.type == "cpu":
            print(f"\n--- {precision_str} CPU'da desteklenmiyor, atlanıyor. ---")
            continue
        if precision == torch.bfloat16 and (
            device.type == "cpu"
            or (device.type == "cuda" and not torch.cuda.is_bf16_supported())
        ):
            print(f"\n--- {precision_str} bu cihazda desteklenmiyor, atlanıyor. ---")
            continue

        print(f"\n{'=' * 80}\nÇALIŞTIRILIYOR: Precision = {precision_str}\n{'=' * 80}")

        for scenario_name, (batch_size, model_name) in SCENARIOS.items():
            baseline_key = (scenario_name, precision_str)
            original_model = None

            # -----------------------------
            # 1) Baseline
            # -----------------------------
            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Baseline")

            try:
                original_model, input_shape = get_model_and_input_shape(model_name)
                original_model = original_model.to(device=device)

                if precision != torch.long:
                    original_model = original_model.to(dtype=precision)

                dummy_loader = create_dummy_loader(
                    batch_size, input_shape, device, precision
                )

                model_fn_base = lambda om=original_model: om

                res_base = run_benchmark(
                    model_fn=model_fn_base,
                    device=device,
                    batch_size=batch_size,
                    input_shape=input_shape,
                    precision=precision,
                    original_model=None,
                    test_loader=dummy_loader,
                )

                if res_base:
                    res_base["scenario"] = f"{scenario_name}-Baseline"
                    res_base["precision"] = precision_str
                    res_base["speedup"] = 1.0
                    res_base["compilation_time_s"] = 0.0
                    all_results.append(res_base)
                    baseline_perfs[baseline_key] = res_base.get("p50_ms", np.nan)
                else:
                    baseline_perfs[baseline_key] = np.nan

            except Exception as e:
                print(f"  > !~ FATAL: Baseline senaryosu başarısız oldu: {e}")
                import traceback as _tb

                _tb.print_exc()
                baseline_perfs[baseline_key] = np.nan
                original_model = None

            # -----------------------------
            # 2) Hypatia
            # -----------------------------
            if original_model is None:
                print("  > !~ Baseline modeli yok, Hypatia senaryosu atlanıyor.")
                continue

            print(f"\n[SENARYO] {scenario_name} | {precision_str} | Hypatia")

            try:
                import time as _time

                start_compile = _time.time()

                example_inputs = get_dummy_inputs(
                    batch_size, input_shape, device, precision
                )

                optimized_model = optimize_model_from_base(
                    original_model=original_model,
                    example_inputs=example_inputs,
                    precision=precision,
                    device=device,
                )

                compilation_time_s = _time.time() - start_compile

                model_fn_opt = lambda om=optimized_model: om

                res_opt = run_benchmark(
                    model_fn=model_fn_opt,
                    device=device,
                    batch_size=batch_size,
                    input_shape=input_shape,
                    precision=precision,
                    original_model=original_model,
                    test_loader=dummy_loader,
                )

                if res_opt:
                    res_opt["scenario"] = f"{scenario_name}-Hypatia"
                    res_opt["precision"] = precision_str
                    res_opt["compilation_time_s"] = compilation_time_s

                    base_p50 = baseline_perfs.get(baseline_key, np.nan)
                    if not np.isnan(base_p50) and "p50_ms" in res_opt:
                        res_opt["speedup"] = base_p50 / res_opt["p50_ms"]
                    else:
                        res_opt["speedup"] = np.nan

                    all_results.append(res_opt)

            except Exception as e:
                print(f"  > !~ FATAL: Hypatia senaryosu başarısız oldu: {e}")
                import traceback as _tb

                _tb.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n✅ ResNet-50 benchmark sonuçları şuraya yazıldı: {args.output_csv}")

        print("\n" + "=" * 100)
        print("DETAYLI RESNET-50 BENCHMARK ÖZETİ")
        print("=" * 100)
        cols_to_show = [
            "scenario",
            "precision",
            "p50_ms",
            "speedup",
            "peak_vram_MB",
            "memory_leak_mb",
            "compilation_time_s",
        ]
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        print(df[cols_to_show].to_string(index=False))
    else:
        print("\nUYARI: Hiç sonuç üretilmedi, CSV yazılmadı.")


if __name__ == "__main__":
    main()
