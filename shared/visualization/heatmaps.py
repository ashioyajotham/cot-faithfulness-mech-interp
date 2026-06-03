"""Head restoration score heatmaps."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_head_restoration_heatmap(
    scores: Dict[str, float],
    n_layers: int,
    n_heads: int,
    title: str = "Head-Level Restoration Scores",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot a layer×head heatmap of restoration scores.

    *scores* keys should be ``"L{layer}H{head}"`` strings.
    """
    grid = np.full((n_layers, n_heads), np.nan)
    for key, val in scores.items():
        if key.startswith("L") and "H" in key:
            parts = key.replace("L", "").split("H")
            layer, head = int(parts[0]), int(parts[1])
            if layer < n_layers and head < n_heads:
                grid[layer, head] = val

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        grid,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=False,
        xticklabels=[str(h) for h in range(n_heads)],
        yticklabels=[str(l) for l in range(n_layers)],
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bootstrap_ci_chart(
    ci_results: Dict[str, tuple],
    top_k: int = 15,
    component_types: Optional[Dict[str, str]] = None,
    title: str = "Bootstrap CI — Restoration Scores",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Horizontal bar chart with bootstrap confidence interval error bars.

    *ci_results*: ``{component: (mean, ci_lower, ci_upper)}``
    """
    sorted_items = sorted(ci_results.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_k]
    names = [c for c, _ in sorted_items]
    means = [v[0] for _, v in sorted_items]
    err_lo = [v[0] - v[1] for _, v in sorted_items]
    err_hi = [v[2] - v[0] for _, v in sorted_items]

    if component_types:
        colors = [
            "#e74c3c" if component_types.get(n) == "faithful" else "#3498db"
            for n in names
        ]
    else:
        colors = ["#3498db"] * len(names)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(names))
    ax.barh(y_pos, means, xerr=[err_lo, err_hi],
            color=colors, alpha=0.8, capsize=3, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Restoration Score")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_selectivity_comparison(
    linguistic_acc: float,
    control_acc_mean: float,
    control_acc_std: float,
    scrambled_acc_mean: float = None,
    random_layer_acc: float = None,
    title: str = "Probe Selectivity Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing probe accuracy across conditions."""
    labels = ["Linguistic\n(circuit)", "Control\n(random labels)"]
    values = [linguistic_acc, control_acc_mean]
    errors = [0, control_acc_std]
    colors = ["#2ecc71", "#e74c3c"]

    if scrambled_acc_mean is not None:
        labels.append("Scrambled\n(shuffled acts)")
        values.append(scrambled_acc_mean)
        errors.append(0)
        colors.append("#f39c12")

    if random_layer_acc is not None:
        labels.append("Random Layer\n(Layer 8)")
        values.append(random_layer_acc)
        errors.append(0)
        colors.append("#9b59b6")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, yerr=errors, color=colors, alpha=0.8,
           capsize=5, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
