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


# -------------------------------------------------------------------------
# t-SNE Embedding Visualization & Pheromone Distribution Heatmap Utilities
# -------------------------------------------------------------------------
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Optional


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str = 'ACO-GNN',
    save_path: str = './paper/figures/tsne_embeddings.pdf',
    perplexity: int = 30,
    seed: int = 42,
):
    """Generate t-SNE 2D visualization of node embeddings colored by class."""
    setup_style()
    sns.set_theme(style="whitegrid")

    # Convert inputs to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)
    labels = labels.squeeze()

    # Filter out unlabeled/masked nodes (-1) if present
    valid_mask = (labels == 0) | (labels == 1)
    if np.any(valid_mask) and np.sum(valid_mask) < len(labels):
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]

    n_samples = len(embeddings)
    if n_samples == 0:
        raise ValueError("No valid embedding samples available to compute t-SNE.")

    # Guard against perplexity >= n_samples
    eff_perplexity = min(perplexity, max(1, n_samples - 1))

    # Run t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=seed)
    tsne_2d = tsne.fit_transform(embeddings)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    benign_mask = (labels == 0)
    fraud_mask = (labels == 1)

    # Plot benign nodes first (blue, s=5, alpha=0.6)
    if np.any(benign_mask):
        ax.scatter(
            tsne_2d[benign_mask, 0],
            tsne_2d[benign_mask, 1],
            c="#1f77b4",
            s=5,
            alpha=0.6,
            label="Benign",
            edgecolors="none",
        )

    # Plot fraud nodes on top (red, s=15, alpha=0.9 for minority visibility)
    if np.any(fraud_mask):
        ax.scatter(
            tsne_2d[fraud_mask, 0],
            tsne_2d[fraud_mask, 1],
            c="#d62728",
            s=15,
            alpha=0.9,
            label="Fraud",
            edgecolors="none",
        )

    ax.set_title(f"t-SNE Embedding Space ({model_name})")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="upper right", frameon=True, markerscale=2.0)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    png_path = save_path.replace(".pdf", ".png") if save_path.endswith(".pdf") else save_path + ".png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved t-SNE plot to {save_path} and {png_path}")


def plot_pheromone_heatmap(
    pheromone_values: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    save_path: str = './paper/figures/pheromone_heatmap.pdf',
    top_k: int = 50,
):
    """Visualize pheromone intensity distribution on fraud vs benign edges."""
    setup_style()
    sns.set_theme(style="whitegrid")

    # Convert inputs to numpy if needed
    if isinstance(pheromone_values, torch.Tensor):
        pheromones = pheromone_values.detach().cpu().numpy().squeeze()
    else:
        pheromones = np.asarray(pheromone_values).squeeze()

    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = np.asarray(edge_index)

    if isinstance(labels, torch.Tensor):
        node_labels = labels.detach().cpu().numpy().squeeze()
    else:
        node_labels = np.asarray(labels).squeeze()

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")

    src, dst = edges[0], edges[1]
    src_labels = node_labels[src]
    dst_labels = node_labels[dst]

    # Categorize edges into 3 types:
    # 1. Fraud-Fraud: both endpoints are fraud (fraud ring edges)
    # 2. Fraud-Benign (Camouflage): one fraud endpoint and one benign endpoint
    # 3. Benign-Benign: both endpoints are benign
    ff_mask = (src_labels == 1) & (dst_labels == 1)
    fb_mask = ((src_labels == 1) & (dst_labels == 0)) | ((src_labels == 0) & (dst_labels == 1))
    bb_mask = (src_labels == 0) & (dst_labels == 0)

    ff_vals = pheromones[ff_mask]
    fb_vals = pheromones[fb_mask]
    bb_vals = pheromones[bb_mask]

    categories = []
    values = []

    categories.extend(["Fraud-Fraud"] * len(ff_vals))
    values.extend(ff_vals.tolist())

    categories.extend(["Fraud-Benign (Camouflage)"] * len(fb_vals))
    values.extend(fb_vals.tolist())

    categories.extend(["Benign-Benign"] * len(bb_vals))
    values.extend(bb_vals.tolist())

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    order = ["Fraud-Fraud", "Fraud-Benign (Camouflage)", "Benign-Benign"]
    palette = {
        "Fraud-Fraud": "#d62728",                  # Red
        "Fraud-Benign (Camouflage)": "#ff7f0e",     # Orange
        "Benign-Benign": "#1f77b4",                # Blue
    }

    if len(categories) > 0:
        sns.boxplot(
            x=categories,
            y=values,
            order=order,
            hue=categories,
            hue_order=order,
            palette=palette,
            legend=False,
            ax=ax,
            width=0.45,
            showmeans=True,
            meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        )

    # Statistical summaries
    mean_ff = float(np.mean(ff_vals)) if len(ff_vals) > 0 else 0.0
    std_ff = float(np.std(ff_vals)) if len(ff_vals) > 0 else 0.0

    mean_fb = float(np.mean(fb_vals)) if len(fb_vals) > 0 else 0.0
    std_fb = float(np.std(fb_vals)) if len(fb_vals) > 0 else 0.0

    mean_bb = float(np.mean(bb_vals)) if len(bb_vals) > 0 else 0.0
    std_bb = float(np.std(bb_vals)) if len(bb_vals) > 0 else 0.0

    summary_lines = [
        "Mean Pheromone (μ ± σ):",
        f"• Fraud-Fraud: {mean_ff:.3f} ± {std_ff:.3f} (N={len(ff_vals)})",
        f"• Camouflage:  {mean_fb:.3f} ± {std_fb:.3f} (N={len(fb_vals)})",
        f"• Benign-Benign: {mean_bb:.3f} ± {std_bb:.3f} (N={len(bb_vals)})",
    ]

    if top_k is not None and top_k > 0 and len(pheromones) > 0:
        k = min(top_k, len(pheromones))
        top_k_indices = np.argsort(pheromones)[-k:]
        top_ff_count = int(np.sum(ff_mask[top_k_indices]))
        summary_lines.append(f"• Top-{k} Edges: {top_ff_count / k * 100:.1f}% Fraud-Fraud")

    summary_text = "\n".join(summary_lines)

    ax.text(
        0.98,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    ax.set_title("Edge Pheromone Distribution: Fraud-Fraud vs Camouflage vs Benign")
    ax.set_xlabel("Edge Category")
    ax.set_ylabel("Pheromone Intensity (τ)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    png_path = save_path.replace(".pdf", ".png") if save_path.endswith(".pdf") else save_path + ".png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved pheromone heatmap plot to {save_path} and {png_path}")
    print(f"Statistical Summary:\n{summary_text}")


def extract_embeddings(
    model: torch.nn.Module,
    data: dict,
    aco_engine=None,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract penultimate layer embeddings from a trained model for t-SNE."""
    # Find the target classifier's first layer to hook into
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            target_layer = model.classifier[0]
        elif isinstance(model.classifier, (nn.ModuleList, list)):
            target_layer = model.classifier[0]
        else:
            target_layer = model.classifier
    elif hasattr(model, "fc3"):
        target_layer = model.fc3
    elif hasattr(model, "head"):
        target_layer = model.head[0] if isinstance(model.head, nn.Sequential) else model.head
    else:
        raise AttributeError("Could not identify classifier head on model to extract embeddings.")

    captured_embeddings = []

    def hook_fn(module, hook_input, hook_output):
        # hook_input[0] is the penultimate representation tensor [N, hidden_dim]
        captured_embeddings.append(hook_input[0].detach().cpu().numpy())

    handle = target_layer.register_forward_hook(hook_fn)

    was_training = model.training
    model.eval()
    model.to(device)

    x = data["x"].to(device)
    edge_index = data.get("edge_index")
    if edge_index is not None:
        edge_index = edge_index.to(device)
    edge_type = data.get("edge_type", None)
    if edge_type is not None:
        edge_type = edge_type.to(device)

    hat_tau = None
    if aco_engine is not None and edge_index is not None:
        hat_tau = aco_engine.get_normalized_pheromones(edge_index, edge_type=edge_type)

    kwargs = {}
    if edge_type is not None:
        kwargs["edge_type"] = edge_type
    if hat_tau is not None:
        kwargs["hat_tau"] = hat_tau

    try:
        with torch.no_grad():
            if edge_index is not None:
                model(x, edge_index, **kwargs)
            else:
                model(x, **kwargs)
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not captured_embeddings:
        raise RuntimeError("Embedding hook did not capture any activations during forward pass.")

    return captured_embeddings[0]
