"""
plot_results.py — Generate Publication-Style Benchmark Plots
============================================================
Reads benchmarks/results.json and generates two charts in assets/:
  1. speedup_vs_gamma.png   — Speedup ratio vs γ, grouped by domain
  2. acceptance_by_domain.png — Mean acceptance rate α per domain across γ

Usage
-----
  python benchmarks/plot_results.py [--input PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Try importing matplotlib; give helpful error if missing
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError as e:
    print(f"ERROR: {e}\nInstall with: pip install matplotlib numpy")
    sys.exit(1)

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

DOMAIN_COLORS = {
    "code":    "#2196F3",   # Blue
    "math":    "#FF9800",   # Orange
    "general": "#4CAF50",   # Green
}
DOMAIN_MARKERS = {
    "code":    "o",
    "math":    "s",
    "general": "^",
}
DOMAIN_LABELS = {
    "code":    "Code (Python/C++)",
    "math":    "Math / Reasoning",
    "general": "General Text",
}

STYLE = {
    "figure.dpi":           150,
    "font.family":          "sans-serif",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.labelsize":       12,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.3,
    "lines.linewidth":      2.2,
    "lines.markersize":     8,
    "legend.framealpha":    0.9,
    "legend.fontsize":      10,
}


def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def group_by_domain_gamma(spec_results: List[dict]) -> Dict[str, Dict[int, List[dict]]]:
    """Group speculative results by domain → gamma → list of runs."""
    grouped: Dict[str, Dict[int, List[dict]]] = {}
    for r in spec_results:
        domain = r.get("domain", "unknown")
        gamma = r.get("gamma", 0)
        grouped.setdefault(domain, {}).setdefault(gamma, []).append(r)
    return grouped


def mean_field(records: List[dict], field: str) -> float:
    vals = [r[field] for r in records if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Plot 1: Speedup vs Gamma
# ---------------------------------------------------------------------------

def plot_speedup_vs_gamma(
    spec_results: List[dict],
    gamma_values: List[int],
    output_path: Path,
):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = group_by_domain_gamma(spec_results)

    for domain in ["code", "math", "general"]:
        domain_data = grouped.get(domain, {})
        if not domain_data:
            continue
        gammas = sorted(domain_data.keys())
        speedups = [mean_field(domain_data[g], "speedup_ratio") for g in gammas]

        ax.plot(
            gammas, speedups,
            color=DOMAIN_COLORS[domain],
            marker=DOMAIN_MARKERS[domain],
            label=DOMAIN_LABELS[domain],
            zorder=3,
        )

    # Reference line: baseline (speedup = 1.0)
    all_gammas = sorted(set(r.get("gamma", 0) for r in spec_results))
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
               alpha=0.7, label="Baseline (1.0×)", zorder=2)

    ax.set_xlabel("Draft Window Size γ")
    ax.set_ylabel("Speedup over Baseline (×)")
    ax.set_title("Speculative Decoding Speedup vs. Draft Window Size γ")
    ax.set_xticks(all_gammas)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f×"))
    ax.legend(loc="lower right")

    # Annotate peak speedup
    peak_records = [r for r in spec_results if r.get("speedup_ratio")]
    if peak_records:
        peak = max(peak_records, key=lambda r: r["speedup_ratio"])
        ax.annotate(
            f"Peak: {peak['speedup_ratio']:.2f}×\n(γ={peak['gamma']}, {peak['domain']})",
            xy=(peak["gamma"], peak["speedup_ratio"]),
            xytext=(peak["gamma"] + 0.5, peak["speedup_ratio"] + 0.05),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Acceptance Rate by Domain
# ---------------------------------------------------------------------------

def plot_acceptance_by_domain(
    spec_results: List[dict],
    gamma_values: List[int],
    output_path: Path,
):
    apply_style()

    domains = ["code", "math", "general"]
    grouped = group_by_domain_gamma(spec_results)

    # Build data: domain × gamma matrix of alpha values
    all_gammas = sorted(set(r.get("gamma", 0) for r in spec_results))
    n_gammas = len(all_gammas)
    x = np.arange(len(domains))
    bar_width = 0.8 / n_gammas

    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("viridis")
    gamma_colors = [cmap(i / max(n_gammas - 1, 1)) for i in range(n_gammas)]

    for gi, gamma in enumerate(all_gammas):
        alphas = []
        for domain in domains:
            records = grouped.get(domain, {}).get(gamma, [])
            alphas.append(mean_field(records, "mean_acceptance_rate"))

        offset = (gi - n_gammas / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, alphas, bar_width,
            label=f"γ={gamma}",
            color=gamma_colors[gi],
            alpha=0.85,
            zorder=3,
        )
        # Add value labels on bars
        for bar, alpha in zip(bars, alphas):
            if alpha > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{alpha:.2f}",
                    ha="center", va="bottom", fontsize=8, color="dimgray",
                )

    ax.set_xlabel("Prompt Domain")
    ax.set_ylabel("Mean Acceptance Rate α")
    ax.set_title("Draft Token Acceptance Rate by Domain and γ")
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains])
    ax.set_ylim(0, 1.15)
    ax.legend(title="Draft window γ", loc="upper right", ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--input", type=str,
        default=str(_ROOT / "benchmarks" / "results.json"),
        help="Path to results.json"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_ROOT / "assets"),
        help="Directory for output PNG files"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"ERROR: Results file not found: {input_path}")
        print("Run benchmarks/run_benchmark.py first.")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    spec_results = data.get("speculative", [])
    gamma_values = data.get("meta", {}).get("gamma_values", [2, 4, 6, 8])

    if not spec_results:
        print("No speculative results found in results.json.")
        sys.exit(1)

    print(f"[plot] Loaded {len(spec_results)} speculative results.")

    plot_speedup_vs_gamma(
        spec_results, gamma_values,
        output_dir / "speedup_vs_gamma.png"
    )
    plot_acceptance_by_domain(
        spec_results, gamma_values,
        output_dir / "acceptance_by_domain.png"
    )
    print("\n[plot] All charts generated successfully.")


if __name__ == "__main__":
    main()
