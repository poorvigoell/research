"""
Publication-Quality Plotting & Visualization Utilities for IEEE/ACM Papers.
Generates:
1. Camouflage Degradation Curve (Illicit-F1 vs Camouflage %: 0% to 50%)
2. Multi-Metric Benchmark Comparison Bar Chart
3. Ablation Study Breakdown Chart
All figures saved directly into paper/figures/ at 300 DPI.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def setup_style():
    """Apply clean academic IEEE style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })


def plot_camouflage_robustness(
    noise_levels: List[float],
    model_results: Dict[str, List[float]],
    metric_name: str = "Illicit-F1",
    save_path: str = "./paper/figures/camouflage_robustness.pdf",
):
    """
    Plot model degradation under increasing graph camouflage noise.
    Proves that ACO-GNN preserves fraud ring connectivity while baselines deteriorate.
    """
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    markers = ["o", "s", "^", "D", "v", "p", "*"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    x_labels = [f"{int(nl * 100)}%" for nl in noise_levels]

    for idx, (model_name, scores) in enumerate(model_results.items()):
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        linewidth = 2.5 if "ACO-GNN" in model_name else 1.5
        linestyle = "-" if "ACO-GNN" in model_name else "--"

        ax.plot(
            x_labels,
            scores,
            marker=marker,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=model_name,
            markersize=7,
        )

    ax.set_xlabel("Camouflage Noise Ratio (Adversarial Benign Edges)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Camouflage Attack Robustness ({metric_name})")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    # Also save PNG for easy viewing
    png_path = save_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved camouflage plot to {save_path} and {png_path}")


def plot_ablation_study(
    variants: List[str],
    f1_scores: List[float],
    pr_aucs: List[float],
    save_path: str = "./paper/figures/ablation_study.pdf",
):
    """Plot ablation study comparing component contributions."""
    setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    x = np.arange(len(variants))
    width = 0.35

    rects1 = ax.bar(x - width / 2, f1_scores, width, label="Illicit-F1", color="#3274a1")
    rects2 = ax.bar(x + width / 2, pr_aucs, width, label="PR-AUC", color="#e1812c")

    ax.set_ylabel("Performance Score")
    ax.set_title("Ablation Study of ACO-GNN Components")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    png_path = save_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved ablation plot to {save_path} and {png_path}")
