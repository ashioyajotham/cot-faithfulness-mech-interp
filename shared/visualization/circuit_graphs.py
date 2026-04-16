"""NetworkX circuit graph export and visualization."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx


def plot_circuit_graph(
    scores: Dict[str, float],
    threshold: float = 0.05,
    title: str = "Circuit Graph",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Build and plot a circuit graph from component restoration scores.

    Nodes are components; edges connect components in adjacent layers.
    Node colour encodes the restoration score (red = shortcut/negative,
    blue = faithful/positive).
    """
    G = nx.DiGraph()

    for comp, score in scores.items():
        if abs(score) < threshold:
            continue
        G.add_node(comp, score=score)

    sorted_comps = sorted(
        G.nodes,
        key=lambda c: int(c.replace("L", "").split("H")[0].split("M")[0]),
    )
    for i in range(len(sorted_comps) - 1):
        G.add_edge(sorted_comps[i], sorted_comps[i + 1])

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["score"] for n in G.nodes]

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.RdBu_r,
        node_size=800,
        font_size=7,
        arrows=True,
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
