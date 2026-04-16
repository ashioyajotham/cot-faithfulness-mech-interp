"""
Contrastive pair construction and runner for clean/corrupted activation patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from .restoration import run_full_patching_sweep


@dataclass
class ContrastivePair:
    """A single clean/corrupted pair for activation patching."""

    pair_id: str
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    pair_type: str  # "faithfulness", "shortcut_detection", "positional_bias", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


def _token_id(model: HookedTransformer, token_str: str) -> int:
    """Resolve a string token to its integer id, handling space-prefix."""
    ids = model.to_tokens(token_str, prepend_bos=False).squeeze()
    if ids.dim() == 0:
        return ids.item()
    return ids[-1].item()


def run_contrastive_pair(
    model: HookedTransformer,
    pair: ContrastivePair,
    strategy: str = "two_pass",
    top_k_layers: int = 6,
) -> Dict[str, float]:
    """Run activation patching on a single contrastive pair.

    Returns a dict of component → restoration score.
    """
    clean_tokens = model.to_tokens(pair.clean_prompt)
    corrupted_tokens = model.to_tokens(pair.corrupted_prompt)

    correct_id = _token_id(model, pair.correct_token)
    incorrect_id = _token_id(model, pair.incorrect_token)

    return run_full_patching_sweep(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        correct_id=correct_id,
        incorrect_id=incorrect_id,
        strategy=strategy,
        top_k_layers=top_k_layers,
    )


def run_contrastive_dataset(
    model: HookedTransformer,
    pairs: List[ContrastivePair],
    strategy: str = "two_pass",
    top_k_layers: int = 6,
) -> Dict[str, float]:
    """Run patching across multiple pairs, averaging restoration scores."""
    from collections import defaultdict

    accum: Dict[str, List[float]] = defaultdict(list)

    for pair in pairs:
        scores = run_contrastive_pair(model, pair, strategy, top_k_layers)
        for comp, score in scores.items():
            if score == score:  # skip NaN
                accum[comp].append(score)

    return {comp: sum(vals) / len(vals) for comp, vals in accum.items() if vals}
