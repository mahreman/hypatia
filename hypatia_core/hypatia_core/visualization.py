# hypatia_core/visualization.py
#
# Visualization tools for Hypatia compiler optimization pipeline.
# - DOT graph export (GraphViz)
# - ASCII tree visualization
# - Before/after optimization comparison
# - HTML optimization report
# - Model architecture summary

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Union

import torch
import torch.nn as nn

try:
    from _hypatia_core import (
        expr_to_dot,
        expr_to_ascii_tree,
        optimization_report,
        optimize_ast,
    )
except ImportError:
    expr_to_dot = None
    expr_to_ascii_tree = None
    optimization_report = None
    optimize_ast = None


# ============================================================================
# DOT / GraphViz Export
# ============================================================================


def visualize_expr(
    expr: str,
    output_path: Optional[str] = None,
    graph_name: str = "hypatia",
    format: str = "png",
) -> str:
    """Visualize an S-expression as a graph.

    Args:
        expr: S-expression string, e.g. "(relu (linear w b x))"
        output_path: path to save rendered image (None = return DOT only)
        graph_name: name for the graph
        format: output format if rendering ("png", "svg", "pdf")

    Returns:
        DOT format string (always), image saved to output_path if provided
    """
    dot = expr_to_dot(expr, graph_name)

    if output_path:
        _render_dot(dot, output_path, format)

    return dot


def visualize_optimization(
    input_expr: str,
    output_path: Optional[str] = None,
    format: str = "png",
) -> Dict:
    """Visualize the before/after of an optimization.

    Runs the optimizer on the input expression and generates side-by-side
    DOT graphs for the original and optimized versions.

    Args:
        input_expr: original S-expression
        output_path: base path for output files (adds _before/_after suffixes)
        format: output format ("png", "svg", "pdf")

    Returns:
        dict with keys: before_dot, after_dot, report
    """
    output_expr = optimize_ast(input_expr)

    before_dot = expr_to_dot(input_expr, "before")
    after_dot = expr_to_dot(output_expr, "after")
    report = optimization_report(input_expr, output_expr)

    if output_path:
        base = Path(output_path).stem
        parent = Path(output_path).parent
        _render_dot(before_dot, str(parent / f"{base}_before.{format}"), format)
        _render_dot(after_dot, str(parent / f"{base}_after.{format}"), format)

    return {
        "before_dot": before_dot,
        "after_dot": after_dot,
        "report": dict(report),
    }


def _render_dot(dot_str: str, output_path: str, format: str = "png"):
    """Render a DOT string to an image file using GraphViz."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
            f.write(dot_str)
            dot_file = f.name

        subprocess.run(
            ["dot", f"-T{format}", dot_file, "-o", output_path],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "GraphViz 'dot' command not found. Install with: apt install graphviz"
        )
    finally:
        try:
            os.unlink(dot_file)
        except OSError:
            pass


# ============================================================================
# ASCII Tree
# ============================================================================


def print_expr_tree(expr: str) -> str:
    """Print an S-expression as an ASCII tree.

    Args:
        expr: S-expression string

    Returns:
        ASCII tree string (also printed to stdout)
    """
    tree = expr_to_ascii_tree(expr)
    print(tree)
    return tree


# ============================================================================
# Optimization Report
# ============================================================================


def compare_optimizations(input_expr: str) -> str:
    """Run optimizer and print a detailed comparison report.

    Args:
        input_expr: S-expression to optimize

    Returns:
        Formatted report string
    """
    output_expr = optimize_ast(input_expr)
    report = optimization_report(input_expr, output_expr)

    lines = []
    lines.append("=" * 60)
    lines.append("  HYPATIA OPTIMIZATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Input:  {report['input_expr']}")
    lines.append(f"  Output: {report['output_expr']}")
    lines.append("")
    lines.append(f"  Nodes: {report['input_node_count']} -> {report['output_node_count']}"
                 f"  (reduction: {report['node_reduction']})")
    lines.append("")

    if report["fusions_found"]:
        lines.append("  Fusions Applied:")
        for f in report["fusions_found"]:
            lines.append(f"    + {f}")
        lines.append("")

    if report["rewrites_applied"]:
        lines.append("  Node Changes:")
        for r in report["rewrites_applied"]:
            lines.append(f"    ~ {r}")
        lines.append("")

    # Node type breakdown
    lines.append("  Node Types (before -> after):")
    all_types = set(list(report["input_node_types"].keys()) + list(report["output_node_types"].keys()))
    for t in sorted(all_types):
        before = report["input_node_types"].get(t, 0)
        after = report["output_node_types"].get(t, 0)
        marker = ""
        if after > before:
            marker = " [NEW]"
        elif after < before:
            marker = " [REMOVED]"
        lines.append(f"    {t:25s} {before:3d} -> {after:3d}{marker}")
    lines.append("")

    # ASCII trees
    lines.append("  Before:")
    before_tree = expr_to_ascii_tree(report["input_expr"])
    for line in before_tree.strip().split("\n"):
        lines.append(f"    {line}")
    lines.append("")

    lines.append("  After:")
    after_tree = expr_to_ascii_tree(report["output_expr"])
    for line in after_tree.strip().split("\n"):
        lines.append(f"    {line}")
    lines.append("")
    lines.append("=" * 60)

    result = "\n".join(lines)
    print(result)
    return result


# ============================================================================
# HTML Report
# ============================================================================


def generate_html_report(
    input_expr: str,
    output_path: str = "hypatia_report.html",
) -> str:
    """Generate an HTML optimization report with embedded SVG graphs.

    Args:
        input_expr: S-expression to optimize
        output_path: path to save HTML file

    Returns:
        HTML content string
    """
    output_expr = optimize_ast(input_expr)
    report = optimization_report(input_expr, output_expr)

    before_dot = expr_to_dot(input_expr, "before")
    after_dot = expr_to_dot(output_expr, "after")

    # Try to render SVGs inline
    before_svg = _dot_to_svg(before_dot)
    after_svg = _dot_to_svg(after_dot)

    fusions_html = ""
    if report["fusions_found"]:
        fusions_html = "<ul>" + "".join(
            f"<li><strong>{f}</strong></li>" for f in report["fusions_found"]
        ) + "</ul>"
    else:
        fusions_html = "<p>No fusions applied.</p>"

    node_types_html = _node_types_table(
        dict(report["input_node_types"]),
        dict(report["output_node_types"]),
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hypatia Optimization Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat-box {{ background: #ecf0f1; border-radius: 6px; padding: 15px 20px;
                     text-align: center; min-width: 120px; }}
        .stat-box .number {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .stat-box .label {{ font-size: 0.85em; color: #7f8c8d; }}
        .graphs {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .graph-box {{ flex: 1; min-width: 300px; }}
        .graph-box svg {{ max-width: 100%; height: auto; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px;
               overflow-x: auto; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f0f0f0; }}
        .new {{ color: #27ae60; font-weight: bold; }}
        .removed {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Hypatia Optimization Report</h1>

    <div class="card">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="number">{report['input_node_count']}</div>
                <div class="label">Input Nodes</div>
            </div>
            <div class="stat-box">
                <div class="number">{report['output_node_count']}</div>
                <div class="label">Output Nodes</div>
            </div>
            <div class="stat-box">
                <div class="number">{report['node_reduction']}</div>
                <div class="label">Nodes Reduced</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(report['fusions_found'])}</div>
                <div class="label">Fusions</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Expressions</h2>
        <p><strong>Input:</strong></p>
        <pre>{_escape_html(report['input_expr'])}</pre>
        <p><strong>Output:</strong></p>
        <pre>{_escape_html(report['output_expr'])}</pre>
    </div>

    <div class="card">
        <h2>Fusions Applied</h2>
        {fusions_html}
    </div>

    <div class="card">
        <h2>Graph Visualization</h2>
        <div class="graphs">
            <div class="graph-box">
                <h3>Before</h3>
                {before_svg if before_svg else '<pre>' + _escape_html(before_dot) + '</pre>'}
            </div>
            <div class="graph-box">
                <h3>After</h3>
                {after_svg if after_svg else '<pre>' + _escape_html(after_dot) + '</pre>'}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Node Types</h2>
        {node_types_html}
    </div>

    <footer style="text-align:center; color:#999; margin-top:30px; font-size:0.85em;">
        Generated by Hypatia Compiler v1.0
    </footer>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return html


def _dot_to_svg(dot_str: str) -> Optional[str]:
    """Try to render DOT to SVG using graphviz. Returns None if graphviz not available."""
    try:
        result = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_str,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Extract just the <svg> element
            svg = result.stdout
            start = svg.find("<svg")
            if start >= 0:
                return svg[start:]
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _node_types_table(before: Dict[str, int], after: Dict[str, int]) -> str:
    all_types = sorted(set(list(before.keys()) + list(after.keys())))
    rows = []
    for t in all_types:
        b = before.get(t, 0)
        a = after.get(t, 0)
        css = ""
        if a > b:
            css = ' class="new"'
        elif a < b:
            css = ' class="removed"'
        diff = a - b
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff != 0 else ""
        rows.append(f"<tr{css}><td>{t}</td><td>{b}</td><td>{a}</td><td>{diff_str}</td></tr>")

    return f"""<table>
<tr><th>Node Type</th><th>Before</th><th>After</th><th>Change</th></tr>
{''.join(rows)}
</table>"""


# ============================================================================
# Model Architecture Visualization
# ============================================================================


def model_summary(
    model: nn.Module,
    input_shape: Optional[tuple] = None,
    show_flops: bool = True,
) -> str:
    """Generate a text summary of model architecture with optional FLOPs analysis.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224)).
                     Required for FLOPs estimation.
        show_flops: If True and input_shape provided, include per-layer FLOPs.

    Returns:
        Formatted summary string with architecture, parameters, and FLOPs.

    Examples:
        summary = model_summary(model, input_shape=(1, 3, 224, 224))
        summary = model_summary(model)  # Without FLOPs
    """
    # Get FLOPs profile if input_shape provided
    profile = None
    if input_shape and show_flops:
        try:
            from .profiler import estimate_flops
            profile = estimate_flops(model, input_shape)
        except Exception:
            pass

    # Build per-layer FLOPs lookup
    layer_flops = {}
    if profile:
        for lp in profile.layers:
            layer_flops[lp.name] = lp

    lines = []
    lines.append("=" * 90)
    lines.append(f"  {model.__class__.__name__} Architecture")
    lines.append("=" * 90)

    if profile:
        header = f"  {'Layer':<28} {'Type':<22} {'Shape':<18} {'Params':>10} {'FLOPs':>14} {'%':>6}"
    else:
        header = f"  {'Layer':<28} {'Type':<22} {'Shape':<18} {'Params':>10}"
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if name == "":
            continue

        # Count parameters
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        n_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        total_params += n_params
        trainable_params += n_trainable

        # Module type
        mod_type = module.__class__.__name__

        # Shape info
        info = ""
        if hasattr(module, "in_features"):
            info = f"{module.in_features}->{module.out_features}"
        elif hasattr(module, "in_channels"):
            k = getattr(module, 'kernel_size', '')
            info = f"{module.in_channels}->{module.out_channels}"
            if k:
                info += f" k{k}"

        param_str = f"{n_params:,}" if n_params > 0 else "-"

        if profile:
            lp = layer_flops.get(name)
            if lp and lp.flops > 0:
                pct = (lp.flops / profile.total_flops * 100) if profile.total_flops > 0 else 0
                lines.append(
                    f"  {name:<28} {mod_type:<22} {info:<18} {param_str:>10} {lp.flops_str:>14} {pct:>5.1f}%"
                )
            else:
                lines.append(f"  {name:<28} {mod_type:<22} {info:<18} {param_str:>10} {'-':>14} {'-':>6}")
        else:
            lines.append(f"  {name:<28} {mod_type:<22} {info:<18} {param_str:>10}")

    lines.append("-" * 90)
    lines.append(f"  Total parameters:     {total_params:>12,}")
    lines.append(f"  Trainable parameters: {trainable_params:>12,}")
    lines.append(f"  FP32 memory:          {total_params * 4 / 1024 / 1024:>10.2f} MB")
    lines.append(f"  FP16 memory:          {total_params * 2 / 1024 / 1024:>10.2f} MB")

    if profile:
        lines.append(f"  Total FLOPs:          {profile.total_flops_str:>14}")
        lines.append(f"  Arithmetic Intensity: {profile.arithmetic_intensity:>10.1f} FLOPs/byte")

    lines.append("=" * 90)

    result = "\n".join(lines)
    print(result)
    return result


__all__ = [
    "visualize_expr",
    "visualize_optimization",
    "print_expr_tree",
    "compare_optimizations",
    "generate_html_report",
    "model_summary",
]
