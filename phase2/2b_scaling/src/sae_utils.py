"""
Gemma Scope 2 SAE loading and differential feature analysis.

For each shortcut head identified in Gemma 3, this module finds
SAE features that are differentially active on unfaithful vs faithful
examples — providing the first interpretable description of *what*
a shortcut head attends to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


def load_gemma_sae(
    layer: int,
    device: str = "cuda",
):
    """Load Gemma Scope 2 SAE for a specific layer.

    Returns the SAE encoder/decoder and feature labels.

    .. note:: This is a placeholder — the actual loading depends on
       the ``gemma_scope`` package version available at runtime.
    """
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            release="gemma-scope-3-12b-it-res",
            sae_id=f"layer_{layer}/width_16k/average_l0_71",
            device=device,
        )
        return sae
    except ImportError:
        raise ImportError(
            "Install gemma_scope or sae-lens to use SAE analysis: "
            "pip install sae-lens"
        )


def differential_features(
    sae,
    faithful_residuals: torch.Tensor,
    unfaithful_residuals: torch.Tensor,
    top_k: int = 20,
) -> List[Dict]:
    """Find SAE features differentially active on unfaithful examples.

    Parameters
    ----------
    sae
        A loaded SAE with an ``encode()`` method.
    faithful_residuals
        Residual stream activations on faithful examples ``(n, d_model)``.
    unfaithful_residuals
        Residual stream activations on unfaithful examples ``(n, d_model)``.
    top_k
        Number of top differential features to return.

    Returns
    -------
    List of dicts with ``feature_idx``, ``delta_activation``, and
    ``label`` (if available).
    """
    with torch.no_grad():
        faithful_feats = sae.encode(faithful_residuals).mean(dim=0)
        unfaithful_feats = sae.encode(unfaithful_residuals).mean(dim=0)

    delta = unfaithful_feats - faithful_feats
    top_indices = delta.topk(top_k).indices.cpu().tolist()

    results = []
    for idx in top_indices:
        entry = {
            "feature_idx": idx,
            "delta_activation": delta[idx].item(),
        }
        if hasattr(sae, "feature_labels") and sae.feature_labels is not None:
            entry["label"] = sae.feature_labels[idx]
        results.append(entry)

    return results
