# hypatia_core/dashboard.py
#
# Interactive HTML Benchmark Dashboard for Hypatia.
#
# Generates a self-contained HTML file with:
#   - Hardware info card
#   - Model info card
#   - Auto-tuner recommendation
#   - Benchmark results table + bar chart (pure CSS, no JS deps)
#   - Roofline analysis
#   - Strategy comparison
#
# Usage:
#   from hypatia_core.dashboard import BenchmarkDashboard
#   dash = BenchmarkDashboard("Qwen2.5-0.5B")
#   dash.set_hardware(hw)
#   dash.set_model_info(n_params=494e6, arch="qwen2", hidden=896, layers=24)
#   dash.add_result("CPU FP32", latency_ms=1449)
#   dash.add_result("GPU FP16+compile", latency_ms=8.9)
#   dash.set_tuner_recommendation("Transformer (Rust-native block)")
#   dash.save("benchmark_report.html")

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    latency_ms: float
    throughput_tps: float = 0.0  # tokens per second
    memory_mb: float = 0.0
    compression: float = 0.0    # e.g. 6.4x for INT4
    category: str = "general"   # cpu, gpu, quantized, compiled
    notes: str = ""


@dataclass
class BenchmarkDashboard:
    """Benchmark dashboard generator.

    Collects benchmark results and hardware info, then generates
    a self-contained interactive HTML report.
    """
    model_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    # Hardware
    hw_cpu: str = ""
    hw_gpu: str = ""
    hw_vram_gb: float = 0.0
    hw_compute: str = ""
    hw_tensor_cores: str = ""
    hw_peak_tflops: float = 0.0
    hw_bandwidth_gbs: float = 0.0
    hw_precision: str = ""
    # Model
    n_params: float = 0.0
    arch: str = ""
    hidden_size: int = 0
    n_layers: int = 0
    n_heads: int = 0
    model_memory_mb: float = 0.0
    input_shape: str = ""
    estimated_gflops: float = 0.0
    # Tuner
    tuner_recommendation: str = ""
    tuner_details: Dict[str, str] = field(default_factory=dict)
    # Roofline
    roofline_ridge: float = 0.0
    roofline_ai: float = 0.0
    roofline_status: str = ""  # compute-bound / memory-bound
    # Generation test
    prompt: str = ""
    generated_text: str = ""

    def set_hardware(self, hw) -> "BenchmarkDashboard":
        """Set hardware info from a HardwareInfo object."""
        self.hw_cpu = hw.cpu_name
        if hw.has_cuda:
            self.hw_gpu = hw.gpu_name
            self.hw_vram_gb = hw.gpu_memory_gb
            cc = hw.gpu_compute_capability
            self.hw_compute = f"SM {cc[0]}.{cc[1]}"
            self.hw_tensor_cores = hw.tensor_core_gen
            self.hw_peak_tflops = hw.peak_tflops_fp32
            self.hw_bandwidth_gbs = hw.memory_bandwidth_gbs
            precision = []
            if hw.supports_fp16: precision.append("FP16")
            if hw.supports_bf16: precision.append("BF16")
            if hw.supports_tf32: precision.append("TF32")
            if hw.supports_int8: precision.append("INT8")
            self.hw_precision = ", ".join(precision)
        return self

    def set_model_info(self, n_params: float, arch: str = "",
                       hidden: int = 0, layers: int = 0, heads: int = 0,
                       input_shape: str = "", estimated_gflops: float = 0.0) -> "BenchmarkDashboard":
        """Set model information."""
        self.n_params = n_params
        self.arch = arch
        self.hidden_size = hidden
        self.n_layers = layers
        self.n_heads = heads
        self.model_memory_mb = n_params * 4 / 1024**2
        self.input_shape = input_shape
        self.estimated_gflops = estimated_gflops
        return self

    def add_result(self, name: str, latency_ms: float,
                   throughput_tps: float = 0.0, memory_mb: float = 0.0,
                   compression: float = 0.0, category: str = "general",
                   notes: str = "") -> "BenchmarkDashboard":
        """Add a benchmark result."""
        self.results.append(BenchmarkResult(
            name=name, latency_ms=latency_ms,
            throughput_tps=throughput_tps, memory_mb=memory_mb,
            compression=compression, category=category, notes=notes
        ))
        return self

    def set_tuner_recommendation(self, name: str,
                                  details: Optional[Dict[str, str]] = None) -> "BenchmarkDashboard":
        """Set auto-tuner recommendation."""
        self.tuner_recommendation = name
        if details:
            self.tuner_details = details
        return self

    def set_roofline(self, ridge_point: float, arithmetic_intensity: float,
                     status: str) -> "BenchmarkDashboard":
        """Set roofline analysis data."""
        self.roofline_ridge = ridge_point
        self.roofline_ai = arithmetic_intensity
        self.roofline_status = status
        return self

    def set_generation_test(self, prompt: str, output: str) -> "BenchmarkDashboard":
        """Set token generation test result."""
        self.prompt = prompt
        self.generated_text = output
        return self

    def save(self, path: str = "hypatia_benchmark.html") -> str:
        """Generate and save the HTML dashboard.

        Args:
            path: Output file path.

        Returns:
            The generated HTML string.
        """
        html = self._render()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return html

    # ------------------------------------------------------------------
    # Private rendering
    # ------------------------------------------------------------------

    def _render(self) -> str:
        sorted_results = sorted(self.results, key=lambda r: r.latency_ms)
        baseline_ms = max((r.latency_ms for r in self.results
                           if "cpu" in r.name.lower() and "fp32" in r.name.lower()),
                          default=sorted_results[-1].latency_ms if sorted_results else 1.0)
        best = sorted_results[0] if sorted_results else None
        max_ms = sorted_results[-1].latency_ms if sorted_results else 1.0

        # Build chart data as JSON for the JS chart
        chart_data = []
        for r in sorted_results:
            speedup = baseline_ms / r.latency_ms if r.latency_ms > 0 else 0
            chart_data.append({
                "name": r.name,
                "ms": round(r.latency_ms, 2),
                "speedup": round(speedup, 1),
                "category": r.category,
            })
        chart_json = json.dumps(chart_data)

        # Results table rows
        table_rows = ""
        for r in sorted_results:
            speedup = baseline_ms / r.latency_ms if r.latency_ms > 0 else 0
            if r.latency_ms >= 1000:
                time_str = f"{r.latency_ms/1000:.2f} s"
            else:
                time_str = f"{r.latency_ms:.1f} ms"
            is_best = r == best
            row_class = ' class="best-row"' if is_best else ''
            badge = ' <span class="badge">FASTEST</span>' if is_best else ''
            cat_class = f"cat-{r.category}"
            notes_str = f'<span class="notes">{_esc(r.notes)}</span>' if r.notes else ""
            table_rows += f"""<tr{row_class}>
                <td><span class="cat-dot {cat_class}"></span>{_esc(r.name)}{badge}</td>
                <td class="num">{time_str}</td>
                <td class="num">{speedup:.1f}x</td>
                <td>{notes_str}</td>
            </tr>\n"""

        # Tuner details
        tuner_details_html = ""
        if self.tuner_details:
            tuner_details_html = '<div class="tuner-grid">'
            for k, v in self.tuner_details.items():
                tuner_details_html += f'<div class="tuner-item"><span class="tuner-key">{_esc(k)}</span><span class="tuner-val">{_esc(v)}</span></div>'
            tuner_details_html += '</div>'

        # Roofline section
        roofline_html = ""
        if self.roofline_status:
            status_class = "compute" if "compute" in self.roofline_status.lower() else "memory"
            roofline_html = f"""
            <div class="card">
                <h2><span class="icon">&#9650;</span> Roofline Analysis</h2>
                <div class="roofline-grid">
                    <div class="roofline-item">
                        <div class="roofline-val">{self.roofline_ridge:.0f}</div>
                        <div class="roofline-label">Ridge Point (FLOPs/byte)</div>
                    </div>
                    <div class="roofline-item">
                        <div class="roofline-val">{self.roofline_ai:.0f}</div>
                        <div class="roofline-label">Arithmetic Intensity</div>
                    </div>
                    <div class="roofline-item">
                        <div class="roofline-val status-{status_class}">{_esc(self.roofline_status.upper())}</div>
                        <div class="roofline-label">Bottleneck</div>
                    </div>
                </div>
            </div>"""

        # Generation test
        gen_html = ""
        if self.prompt and self.generated_text:
            gen_html = f"""
            <div class="card">
                <h2><span class="icon">&#9997;</span> Token Generation Test</h2>
                <div class="gen-prompt"><strong>Prompt:</strong> {_esc(self.prompt)}</div>
                <div class="gen-output">{_esc(self.generated_text)}</div>
            </div>"""

        # Format model size
        if self.n_params >= 1e9:
            param_str = f"{self.n_params/1e9:.2f}B"
        else:
            param_str = f"{self.n_params/1e6:.0f}M"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hypatia Benchmark - {_esc(self.model_name)}</title>
<style>
:root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e4e6f0;
    --text-dim: #8b8fa3;
    --accent: #6366f1;
    --accent2: #818cf8;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
    --blue: #3b82f6;
    --orange: #f97316;
    --cyan: #06b6d4;
    --purple: #a855f7;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
.header {{
    text-align: center;
    padding: 40px 0 30px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 30px;
}}
.header h1 {{
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
.header .subtitle {{
    font-size: 1.1em;
    color: var(--text-dim);
}}
.header .model-badge {{
    display: inline-block;
    margin-top: 12px;
    padding: 6px 16px;
    background: var(--card);
    border: 1px solid var(--accent);
    border-radius: 20px;
    font-size: 0.95em;
    color: var(--accent2);
}}
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}}
.card h2 {{
    font-size: 1.15em;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--text);
}}
.card h2 .icon {{ margin-right: 8px; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

/* Stats */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
}}
.stat {{
    background: rgba(99, 102, 241, 0.08);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}}
.stat .val {{
    font-size: 1.5em;
    font-weight: 700;
    color: var(--accent2);
}}
.stat .lbl {{
    font-size: 0.8em;
    color: var(--text-dim);
    margin-top: 2px;
}}
.hw-stat .val {{ color: var(--cyan); }}

/* Table */
table {{ width: 100%; border-collapse: collapse; }}
th {{
    text-align: left;
    padding: 10px 12px;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
}}
td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(42, 45, 58, 0.6);
    font-size: 0.95em;
}}
td.num {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; text-align: right; }}
tr:hover {{ background: rgba(99, 102, 241, 0.05); }}
tr.best-row {{ background: rgba(34, 197, 94, 0.08); }}
tr.best-row td {{ font-weight: 600; }}
.badge {{
    display: inline-block;
    padding: 1px 8px;
    background: var(--green);
    color: #000;
    font-size: 0.7em;
    font-weight: 700;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}}
.notes {{ color: var(--text-dim); font-size: 0.85em; }}

/* Category dots */
.cat-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}}
.cat-cpu {{ background: var(--blue); }}
.cat-gpu {{ background: var(--green); }}
.cat-quantized {{ background: var(--orange); }}
.cat-compiled {{ background: var(--purple); }}
.cat-general {{ background: var(--text-dim); }}

/* Chart */
.chart-container {{ margin-top: 16px; }}
.bar-row {{
    display: flex;
    align-items: center;
    margin-bottom: 6px;
    gap: 12px;
}}
.bar-label {{
    width: 180px;
    text-align: right;
    font-size: 0.85em;
    color: var(--text-dim);
    flex-shrink: 0;
}}
.bar-track {{
    flex: 1;
    height: 26px;
    background: rgba(255,255,255,0.04);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}}
.bar-fill {{
    height: 100%;
    border-radius: 4px;
    min-width: 2px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 0.75em;
    font-weight: 600;
    color: white;
    transition: width 0.8s ease;
}}
.bar-val {{
    width: 90px;
    text-align: right;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.85em;
    flex-shrink: 0;
}}

/* Tuner */
.tuner-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
    margin-top: 12px;
}}
.tuner-item {{
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 6px;
}}
.tuner-key {{ color: var(--text-dim); font-size: 0.85em; }}
.tuner-val {{ color: var(--accent2); font-weight: 600; font-size: 0.85em; }}

/* Roofline */
.roofline-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    text-align: center;
}}
.roofline-val {{
    font-size: 1.4em;
    font-weight: 700;
    color: var(--accent2);
}}
.roofline-label {{ font-size: 0.8em; color: var(--text-dim); margin-top: 4px; }}
.status-compute {{ color: var(--red); }}
.status-memory {{ color: var(--yellow); }}

/* Generation test */
.gen-prompt {{
    padding: 12px;
    background: rgba(99, 102, 241, 0.08);
    border-radius: 6px;
    margin-bottom: 10px;
    font-size: 0.95em;
}}
.gen-output {{
    padding: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.9em;
    line-height: 1.5;
    color: var(--green);
}}

/* Footer */
.footer {{
    text-align: center;
    padding: 30px 0 20px;
    color: var(--text-dim);
    font-size: 0.8em;
    border-top: 1px solid var(--border);
    margin-top: 20px;
}}
.footer a {{ color: var(--accent2); text-decoration: none; }}

/* Legend */
.legend {{
    display: flex;
    gap: 20px;
    margin-top: 12px;
    flex-wrap: wrap;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8em;
    color: var(--text-dim);
}}
</style>
</head>
<body>
<div class="container">
    <!-- Header -->
    <div class="header">
        <h1>Hypatia Benchmark Dashboard</h1>
        <div class="subtitle">Hardware-Aware Symbolic Compiler for PyTorch</div>
        <div class="model-badge">{_esc(self.model_name)} ({param_str} params)</div>
    </div>

    <!-- Hardware + Model Info -->
    <div class="grid-2">
        <div class="card">
            <h2><span class="icon">&#9889;</span> Hardware</h2>
            <div class="stat-grid">
                <div class="stat hw-stat"><div class="val">{_esc(self.hw_gpu or 'CPU')}</div><div class="lbl">GPU</div></div>
                <div class="stat hw-stat"><div class="val">{self.hw_vram_gb:.1f} GB</div><div class="lbl">VRAM</div></div>
                <div class="stat hw-stat"><div class="val">{_esc(self.hw_compute)}</div><div class="lbl">Compute</div></div>
                <div class="stat hw-stat"><div class="val">{self.hw_peak_tflops:.1f}</div><div class="lbl">Peak TFLOPS</div></div>
                <div class="stat hw-stat"><div class="val">{self.hw_bandwidth_gbs:.0f}</div><div class="lbl">BW (GB/s)</div></div>
                <div class="stat hw-stat"><div class="val">{_esc(self.hw_tensor_cores or 'N/A')}</div><div class="lbl">Tensor Cores</div></div>
            </div>
        </div>
        <div class="card">
            <h2><span class="icon">&#129302;</span> Model</h2>
            <div class="stat-grid">
                <div class="stat"><div class="val">{param_str}</div><div class="lbl">Parameters</div></div>
                <div class="stat"><div class="val">{_esc(self.arch) or 'N/A'}</div><div class="lbl">Architecture</div></div>
                <div class="stat"><div class="val">{self.hidden_size}</div><div class="lbl">Hidden Size</div></div>
                <div class="stat"><div class="val">{self.n_layers}</div><div class="lbl">Layers</div></div>
                <div class="stat"><div class="val">{self.model_memory_mb:.0f} MB</div><div class="lbl">FP32 Memory</div></div>
                <div class="stat"><div class="val">{self.estimated_gflops:.1f}</div><div class="lbl">GFLOPs/inf</div></div>
            </div>
        </div>
    </div>

    {gen_html}

    <!-- Auto-Tuner Recommendation -->
    <div class="card">
        <h2><span class="icon">&#9881;</span> Auto-Tuner Recommendation</h2>
        <div style="font-size:1.3em; font-weight:700; color:var(--green); margin-bottom:8px;">
            {_esc(self.tuner_recommendation)}
        </div>
        {tuner_details_html}
    </div>

    {roofline_html}

    <!-- Benchmark Results -->
    <div class="card">
        <h2><span class="icon">&#128200;</span> Benchmark Results</h2>
        <div class="legend">
            <div class="legend-item"><span class="cat-dot cat-cpu"></span>CPU</div>
            <div class="legend-item"><span class="cat-dot cat-gpu"></span>GPU</div>
            <div class="legend-item"><span class="cat-dot cat-quantized"></span>Quantized</div>
            <div class="legend-item"><span class="cat-dot cat-compiled"></span>Compiled</div>
        </div>
        <div class="chart-container" id="chart"></div>
    </div>

    <!-- Results Table -->
    <div class="card">
        <h2><span class="icon">&#128202;</span> Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th style="text-align:right">Latency</th>
                    <th style="text-align:right">Speedup</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">
        Generated by <a href="#">Hypatia Compiler</a> | {timestamp} | {_esc(self.input_shape)}
    </div>
</div>

<script>
// Render bar chart
const data = {chart_json};
const chart = document.getElementById('chart');
const maxMs = Math.max(...data.map(d => d.ms));
const colors = {{
    cpu: '#3b82f6',
    gpu: '#22c55e',
    quantized: '#f97316',
    compiled: '#a855f7',
    general: '#8b8fa3'
}};

// Use log scale for better visibility of fast results
const logMax = Math.log10(maxMs + 1);

data.forEach((d, i) => {{
    const pct = Math.max(2, (Math.log10(d.ms + 1) / logMax) * 100);
    const color = colors[d.category] || colors.general;
    const barText = d.ms < 1 ? d.ms.toFixed(2) + ' ms' :
                    d.ms >= 1000 ? (d.ms/1000).toFixed(2) + ' s' :
                    d.ms.toFixed(1) + ' ms';
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
        <div class="bar-label">${{d.name}}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width:${{pct}}%;background:${{color}};">
                ${{pct > 15 ? barText : ''}}
            </div>
        </div>
        <div class="bar-val">${{d.speedup}}x</div>
    `;
    // Animate on load
    const fill = row.querySelector('.bar-fill');
    fill.style.width = '0%';
    setTimeout(() => {{ fill.style.width = pct + '%'; }}, 100 + i * 80);
    chart.appendChild(row);
}});
</script>
</body>
</html>"""
        return html


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ============================================================================
# Convenience: generate dashboard from demo results
# ============================================================================

def generate_benchmark_dashboard(
    model_name: str,
    results: Dict[str, float],
    hw=None,
    model_info: Optional[Dict[str, Any]] = None,
    tuner_name: str = "",
    tuner_details: Optional[Dict[str, str]] = None,
    roofline: Optional[Dict[str, float]] = None,
    generation: Optional[Dict[str, str]] = None,
    output_path: str = "hypatia_benchmark.html",
) -> str:
    """One-call convenience to generate a benchmark dashboard.

    Args:
        model_name: Display name for the model
        results: Dict of {strategy_name: latency_ms}
        hw: HardwareInfo object (optional)
        model_info: Dict with keys: n_params, arch, hidden, layers, heads, input_shape, gflops
        tuner_name: Auto-tuner recommended strategy name
        tuner_details: Dict of tuner config details
        roofline: Dict with keys: ridge, ai, status
        generation: Dict with keys: prompt, output
        output_path: Path to save HTML

    Returns:
        HTML string
    """
    dash = BenchmarkDashboard(model_name)

    if hw is not None:
        dash.set_hardware(hw)

    if model_info:
        dash.set_model_info(
            n_params=model_info.get("n_params", 0),
            arch=model_info.get("arch", ""),
            hidden=model_info.get("hidden", 0),
            layers=model_info.get("layers", 0),
            heads=model_info.get("heads", 0),
            input_shape=model_info.get("input_shape", ""),
            estimated_gflops=model_info.get("gflops", 0.0),
        )

    # Categorize results
    for name, ms in results.items():
        cat = _categorize(name)
        dash.add_result(name, ms, category=cat)

    if tuner_name:
        dash.set_tuner_recommendation(tuner_name, tuner_details)

    if roofline:
        dash.set_roofline(
            ridge_point=roofline.get("ridge", 0),
            arithmetic_intensity=roofline.get("ai", 0),
            status=roofline.get("status", ""),
        )

    if generation:
        dash.set_generation_test(
            prompt=generation.get("prompt", ""),
            output=generation.get("output", ""),
        )

    return dash.save(output_path)


def _categorize(name: str) -> str:
    """Guess category from strategy name."""
    nl = name.lower()
    if "compile" in nl or "triton" in nl:
        return "compiled"
    if "int4" in nl or "int8" in nl or "quant" in nl:
        return "quantized"
    if "gpu" in nl or "cuda" in nl:
        return "gpu"
    if "cpu" in nl:
        return "cpu"
    return "general"


__all__ = [
    "BenchmarkDashboard",
    "BenchmarkResult",
    "generate_benchmark_dashboard",
]
