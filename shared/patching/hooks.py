"""
Hook factories for activation patching on HookedTransformer models.

Provides model-agnostic hook functions for:
- Per-head patching via hook_z
- Layer-level attention patching via hook_attn_out
- MLP patching via hook_mlp_out
- Zero ablation (head-level or layer-level)
"""

import torch
from typing import Callable, Dict, Optional


def make_head_patch_hook(
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    head_idx: int,
) -> Callable:
    """Patch a single attention head's hook_z output with clean activations.

    The returned hook replaces ``activation[:, :, head_idx, :]`` at the
    given layer with the corresponding slice from *clean_cache*.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(activation, hook):
        activation[:, :, head_idx, :] = clean_cache[hook_name][:, :, head_idx, :]
        return activation

    return hook_fn


def make_mlp_patch_hook(
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
) -> Callable:
    """Patch a full MLP layer output with clean activations."""
    hook_name = f"blocks.{layer}.hook_mlp_out"

    def hook_fn(activation, hook):
        activation[:] = clean_cache[hook_name]
        return activation

    return hook_fn


def make_attn_out_patch_hook(
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
) -> Callable:
    """Patch a full attention-layer output (hook_attn_out) with clean activations."""
    hook_name = f"blocks.{layer}.hook_attn_out"

    def hook_fn(activation, hook):
        activation[:] = clean_cache[hook_name]
        return activation

    return hook_fn


def make_zero_ablation_hook(
    layer: int,
    head_idx: Optional[int] = None,
) -> Callable:
    """Zero out a component.

    If *head_idx* is ``None`` the entire attention-layer output is zeroed
    (use on ``hook_attn_out``).  Otherwise only a single head slice is
    zeroed (use on ``hook_z``).
    """

    def hook_fn(activation, hook):
        if head_idx is not None:
            activation[:, :, head_idx, :] = 0.0
        else:
            activation[:] = 0.0
        return activation

    return hook_fn
