"""
Cross-model head-position alignment for comparing shortcut circuits.

Aligns shortcut heads across GPT-2, Qwen, and Gemma by normalised
layer position (layer / n_layers) so that "L7H6 in GPT-2" can be
meaningfully compared with "L21H14 in Qwen".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class HeadScore:
    model: str
    layer: int
    head: int
    n_layers: int
    restoration_score: float

    @property
    def normalised_position(self) -> float:
        return self.layer / self.n_layers


def align_shortcut_heads(
    scores_by_model: Dict[str, List[HeadScore]],
    top_k: int = 5,
) -> Dict[str, List[Dict]]:
    """For each model, return the top-k shortcut heads with normalised positions.

    This enables cross-architecture comparison: if GPT-2's dominant
    shortcut head is at normalised position 0.58 and Qwen's is at
    0.65, we can quantify how similar the circuit structure is.
    """
    aligned = {}
    for model_name, heads in scores_by_model.items():
        sorted_heads = sorted(heads, key=lambda h: abs(h.restoration_score), reverse=True)
        aligned[model_name] = [
            {
                "label": f"L{h.layer}H{h.head}",
                "normalised_position": h.normalised_position,
                "restoration_score": h.restoration_score,
            }
            for h in sorted_heads[:top_k]
        ]
    return aligned


def position_correlation(
    model_a_heads: List[HeadScore],
    model_b_heads: List[HeadScore],
    top_k: int = 10,
) -> float:
    """Spearman correlation of normalised positions between two models' top heads."""
    from scipy.stats import spearmanr

    a_sorted = sorted(model_a_heads, key=lambda h: abs(h.restoration_score), reverse=True)[:top_k]
    b_sorted = sorted(model_b_heads, key=lambda h: abs(h.restoration_score), reverse=True)[:top_k]

    k = min(len(a_sorted), len(b_sorted))
    a_pos = [h.normalised_position for h in a_sorted[:k]]
    b_pos = [h.normalised_position for h in b_sorted[:k]]

    if k < 3:
        return float("nan")

    rho, _ = spearmanr(a_pos, b_pos)
    return float(rho)
