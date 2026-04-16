"""
Hewitt-Liang control task label generation.

A control task replaces the real faithfulness labels with random labels
stratified by prompt identity.  A selective probe achieves high
accuracy on the real task and low accuracy on the control task.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def generate_control_labels(
    y: np.ndarray,
    pair_ids: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate random control labels preserving class balance.

    If *pair_ids* is provided, labels are assigned per unique pair id
    (matching Hewitt-Liang's stratification by "word type").  Otherwise
    labels are shuffled uniformly.
    """
    rng = np.random.RandomState(seed)
    control_y = np.empty_like(y)

    if pair_ids is not None:
        unique_ids = np.unique(pair_ids)
        vocab_size = max(len(np.unique(y)), 2)
        label_map = {pid: rng.randint(0, vocab_size) for pid in unique_ids}
        for i, pid in enumerate(pair_ids):
            control_y[i] = label_map[pid]
    else:
        control_y = rng.permutation(y)

    return control_y
