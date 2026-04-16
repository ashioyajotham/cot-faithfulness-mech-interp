"""
Bootstrap significance testing — Experiment 03.

Provides confidence intervals for:
- Each head's average restoration score
- The rank ordering of the top-k heads
- The linear probe's coefficient magnitude for L7H6
- Cross-pair-type rank stability (Spearman correlation)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def bootstrap_restoration_ci(
    per_pair_scores: Dict[str, List[float]],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap CI for each component's mean restoration score.

    Parameters
    ----------
    per_pair_scores
        ``{component_name: [score_pair_0, score_pair_1, ...]}``

    Returns
    -------
    ``{component_name: (mean, ci_lower, ci_upper)}``
    """
    rng = np.random.RandomState(seed)
    results = {}

    for comp, scores in per_pair_scores.items():
        arr = np.array(scores)
        n = len(arr)
        boot_means = []
        for _ in range(n_iterations):
            sample = arr[rng.choice(n, size=n, replace=True)]
            boot_means.append(sample.mean())

        boot_means = np.array(boot_means)
        results[comp] = (
            float(arr.mean()),
            float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        )

    return results


def bootstrap_rank_stability(
    per_pair_scores: Dict[str, List[float]],
    target_component: str = "L7H6",
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test how often *target_component* ranks #1 across bootstrap samples.

    Returns the fraction of samples where the target is rank 1,
    and the distribution of its rank.
    """
    rng = np.random.RandomState(seed)
    components = list(per_pair_scores.keys())
    n_pairs = len(next(iter(per_pair_scores.values())))

    rank_1_count = 0
    target_ranks = []

    for _ in range(n_iterations):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        means = {c: np.array(per_pair_scores[c])[idx].mean() for c in components}
        sorted_comps = sorted(means, key=lambda c: abs(means[c]), reverse=True)

        rank = sorted_comps.index(target_component) + 1 if target_component in sorted_comps else -1
        target_ranks.append(rank)
        if rank == 1:
            rank_1_count += 1

    return {
        "target": target_component,
        "rank_1_fraction": rank_1_count / n_iterations,
        "mean_rank": float(np.mean(target_ranks)),
        "rank_distribution": dict(zip(*np.unique(target_ranks, return_counts=True))),
    }


def cross_pair_type_stability(
    scores_by_type: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute Spearman rank correlation between pair-type-specific rankings.

    *scores_by_type*: ``{pair_type: {component: mean_restoration_score}}``

    Returns pairwise correlations between all pair types.
    """
    pair_types = list(scores_by_type.keys())
    if len(pair_types) < 2:
        return {"error": "need at least 2 pair types"}

    all_components = set()
    for scores in scores_by_type.values():
        all_components.update(scores.keys())
    all_components = sorted(all_components)

    correlations = {}
    for i in range(len(pair_types)):
        for j in range(i + 1, len(pair_types)):
            t1, t2 = pair_types[i], pair_types[j]
            r1 = [scores_by_type[t1].get(c, 0.0) for c in all_components]
            r2 = [scores_by_type[t2].get(c, 0.0) for c in all_components]
            rho, pval = stats.spearmanr(r1, r2)
            correlations[f"{t1}_vs_{t2}"] = {"rho": float(rho), "p_value": float(pval)}

    return correlations


def ablation_cascade(
    model,
    pairs,
    shortcut_heads: List[str],
    eval_fn,
) -> List[Dict[str, Any]]:
    """Run ablation cascade: ablate top-1, top-2, ..., top-N shortcut heads.

    *eval_fn(model, pairs, ablated_heads)* should return a dict of metrics
    (e.g. ``{"accuracy": 0.85, "shift_rate": 0.20}``).

    Returns a list of result dicts, one per cascade level.
    """
    results = []
    for n in range(1, len(shortcut_heads) + 1):
        heads_to_ablate = shortcut_heads[:n]
        metrics = eval_fn(model, pairs, heads_to_ablate)
        results.append({
            "n_ablated": n,
            "ablated_heads": heads_to_ablate,
            **metrics,
        })
    return results
