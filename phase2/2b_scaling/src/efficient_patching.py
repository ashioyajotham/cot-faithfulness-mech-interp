"""
Two-pass efficient patching for large models.

Qwen2.5-Math-7B has 28 heads × 32 layers = 896 heads.  Exhaustive
per-head patching is expensive.  The two-pass approach:

1. **Pass 1** — Layer-level patching (hook_attn_out + hook_mlp_out)
   identifies the top-k layers (default 6) with the highest combined
   restoration score.  This costs 64 forward passes per pair.

2. **Pass 2** — Per-head patching (hook_z) within the top-k layers
   only.  With k=6 and 28 heads, this costs 168 passes per pair.

Total: ~232 passes vs ~896 for exhaustive.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from shared.patching.hooks import (
    make_attn_out_patch_hook,
    make_head_patch_hook,
    make_mlp_patch_hook,
)


def _logit_diff(logits, correct_id, incorrect_id, pos=-1):
    return (logits[0, pos, correct_id] - logits[0, pos, incorrect_id]).item()


def two_pass_patching(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int,
    top_k_layers: int = 6,
    position: int = -1,
) -> Dict[str, float]:
    """Run two-pass patching and return component restoration scores.

    Keys in the result dict:
    - ``"blocks.{L}.hook_attn_out"`` for all layers (Pass 1)
    - ``"blocks.{L}.hook_mlp_out"`` for all layers (Pass 1)
    - ``"L{L}H{H}"`` for per-head scores in top-k layers (Pass 2)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    clean_diff = _logit_diff(clean_logits, correct_id, incorrect_id, position)
    corrupted_diff = _logit_diff(corrupted_logits, correct_id, incorrect_id, position)
    gap = clean_diff - corrupted_diff

    if abs(gap) < 1e-4:
        return {}

    scores: Dict[str, float] = {}

    # ── Pass 1: layer-level ────────────────────────────────────────────
    layer_importance: List[Tuple[int, float]] = []

    for layer in range(n_layers):
        for suffix, factory in [
            ("hook_attn_out", make_attn_out_patch_hook),
            ("hook_mlp_out", make_mlp_patch_hook),
        ]:
            hook_name = f"blocks.{layer}.{suffix}"
            hook_fn = factory(clean_cache, layer)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )
            pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
            s = (pd - corrupted_diff) / gap
            scores[hook_name] = s

        combined = abs(scores[f"blocks.{layer}.hook_attn_out"]) + abs(
            scores[f"blocks.{layer}.hook_mlp_out"]
        )
        layer_importance.append((layer, combined))

    layer_importance.sort(key=lambda x: x[1], reverse=True)
    top_layers = [l for l, _ in layer_importance[:top_k_layers]]

    # ── Pass 2: per-head within top layers ─────────────────────────────
    for layer in top_layers:
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"
            hook_fn = make_head_patch_hook(clean_cache, layer, head)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )
            pd = _logit_diff(patched_logits, correct_id, incorrect_id, position)
            scores[f"L{layer}H{head}"] = (pd - corrupted_diff) / gap

    return scores
