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
