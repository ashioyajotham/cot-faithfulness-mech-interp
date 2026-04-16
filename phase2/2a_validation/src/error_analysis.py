"""
High-confidence false negative analysis — Experiment 02.

Clusters the 11+ high-confidence false negatives from Phase 1 by
carry requirement, sum magnitude, corruption severity, and CoT
corruption type to identify systematic failure modes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


def extract_false_negatives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metadata: List[Dict[str, Any]],
    confidence_threshold: float = 0.99,
) -> List[Dict[str, Any]]:
    """Extract false negatives with confidence above threshold.

    A false negative here means the probe predicted faithful (0) when
    the true label was unfaithful (1).
    """
    fn_examples = []
    for i in range(len(y_true)):
        is_fn = (y_true[i] == 1) and (y_pred[i] == 0)
        if is_fn and y_proba[i] >= confidence_threshold:
            fn_examples.append({
                "index": i,
                "confidence": float(y_proba[i]),
                **metadata[i],
            })
    return fn_examples


def cluster_by_features(
    fn_examples: List[Dict[str, Any]],
    features: List[str],
) -> Dict[str, Counter]:
    """Cluster false negatives by specified metadata features.

    Returns a dict of feature_name → Counter of feature values.
    """
    clusters = {}
    for feat in features:
        values = [ex.get(feat, "unknown") for ex in fn_examples]
        clusters[feat] = Counter(values)
    return clusters


def test_carry_hypothesis(
    fn_examples: List[Dict[str, Any]],
    all_examples_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Test whether carry-requiring sums are disproportionately represented
    among false negatives.

    Returns chi-squared-like statistics and the carry proportions in FN
    vs the full dataset.
    """
    fn_carry = sum(1 for ex in fn_examples if ex.get("carry_required", False))
    fn_total = len(fn_examples)

    all_carry = sum(1 for ex in all_examples_metadata if ex.get("carry_required", False))
    all_total = len(all_examples_metadata)

    fn_carry_rate = fn_carry / max(fn_total, 1)
    all_carry_rate = all_carry / max(all_total, 1)

    return {
        "fn_carry_count": fn_carry,
        "fn_total": fn_total,
        "fn_carry_rate": fn_carry_rate,
        "dataset_carry_rate": all_carry_rate,
        "overrepresentation_ratio": fn_carry_rate / max(all_carry_rate, 1e-6),
        "hypothesis_supported": fn_carry_rate > all_carry_rate * 1.5,
    }


def test_magnitude_hypothesis(
    fn_examples: List[Dict[str, Any]],
    magnitude_threshold: int = 80,
) -> Dict[str, Any]:
    """Test whether high-magnitude sums are overrepresented in false negatives."""
    magnitudes = [
        ex.get("a", 0) + ex.get("b", 0) for ex in fn_examples
        if "a" in ex and "b" in ex
    ]

    if not magnitudes:
        return {"error": "no magnitude data available"}

    high_mag = sum(1 for m in magnitudes if m > magnitude_threshold)
    return {
        "n_high_magnitude": high_mag,
        "n_total": len(magnitudes),
        "high_magnitude_rate": high_mag / len(magnitudes),
        "mean_magnitude": float(np.mean(magnitudes)),
        "std_magnitude": float(np.std(magnitudes)),
    }
