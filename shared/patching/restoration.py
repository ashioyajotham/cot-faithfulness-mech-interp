"""
Restoration score computation for activation patching experiments.

The restoration score measures how much patching a single component
recovers the clean model's logit difference when running on corrupted
input:

    restoration = (patched_diff − corrupted_diff) / (clean_diff − corrupted_diff)

A score of 1.0 means the component fully restores the clean behaviour;
0.0 means no effect; negative means the component pushes toward the
corrupted answer.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Literal, Optional, Tuple
from transformer_lens import HookedTransformer

from .hooks import (
    make_head_patch_hook,
    make_mlp_patch_hook,
    make_attn_out_patch_hook,
)

DEGENERACY_THRESHOLD = 1e-4


def _logit_diff(
    logits: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    position: int = -1,
) -> float:
    """Return logit(correct) − logit(incorrect) at *position*."""
    return (logits[0, position, correct_id] - logits[0, position, incorrect_id]).item()


def compute_restoration_score(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    hook_name: str,
    correct_id: int,
    incorrect_id: int,
    clean_diff: float,
    corrupted_diff: float,
    position: int = -1,
) -> float:
    """Compute the restoration score for a single hook.

    Returns ``float('nan')`` when the clean/corrupted gap is below
    ``DEGENERACY_THRESHOLD`` (the pair is degenerate).
    """
    gap = clean_diff - corrupted_diff
    if abs(gap) < DEGENERACY_THRESHOLD:
        return float("nan")

    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, lambda act, hook: clean_cache[hook_name].clone())],
    )
    patched_diff = _logit_diff(patched_logits, correct_id, incorrect_id, position)
    return (patched_diff - corrupted_diff) / gap


def run_full_patching_sweep(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    strategy: Literal["exhaustive", "two_pass"] = "two_pass",
    top_k_layers: int = 6,
    position: int = -1,
) -> Dict[str, float]:
    """Run patching over all components, return averaged restoration scores.

    Parameters
    ----------
    strategy
        ``"exhaustive"`` — patch every head and MLP individually.
        ``"two_pass"``   — first find the top-*k* layers via layer-level
        patching, then go per-head only within those layers.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Baselines
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    clean_diff = _logit_diff(clean_logits, correct_id, incorrect_id, position)
    corrupted_diff = _logit_diff(corrupted_logits, correct_id, incorrect_id, position)

    gap = clean_diff - corrupted_diff
    if abs(gap) < DEGENERACY_THRESHOLD:
        return {}

    scores: Dict[str, float] = {}

    if strategy == "two_pass":
        # Pass 1: layer-level (attn_out + mlp_out)
        layer_scores: List[Tuple[int, float]] = []
        for layer in range(n_layers):
            for suffix, factory in [
                ("hook_attn_out", make_attn_out_patch_hook),
                ("hook_mlp_out", make_mlp_patch_hook),
            ]:
                hook_name = f"blocks.{layer}.{suffix}"
                hook_fn = factory(clean_cache, layer)
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                )
                pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
                s = (pd - corrupted_diff) / gap
                scores[hook_name] = s

            combined = abs(scores.get(f"blocks.{layer}.hook_attn_out", 0)) + abs(
                scores.get(f"blocks.{layer}.hook_mlp_out", 0)
            )
            layer_scores.append((layer, combined))

        layer_scores.sort(key=lambda x: x[1], reverse=True)
        top_layers = [l for l, _ in layer_scores[:top_k_layers]]

        # Pass 2: per-head within top layers
        for layer in top_layers:
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"
                hook_fn = make_head_patch_hook(clean_cache, layer, head)
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                )
                pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
                scores[f"L{layer}H{head}"] = (pd - corrupted_diff) / gap

    else:  # exhaustive
        for layer in range(n_layers):
            # MLP
            hook_name = f"blocks.{layer}.hook_mlp_out"
            hook_fn = make_mlp_patch_hook(clean_cache, layer)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, hook_fn)],
            )
            pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
            scores[f"L{layer}MLP"] = (pd - corrupted_diff) / gap

            # Per-head
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_z"
                hook_fn = make_head_patch_hook(clean_cache, layer, head)
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                )
                pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
                scores[f"L{layer}H{head}"] = (pd - corrupted_diff) / gap

    return scores
