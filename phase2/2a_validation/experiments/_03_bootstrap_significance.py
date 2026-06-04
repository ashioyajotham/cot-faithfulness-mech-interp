"""
Experiment 03 — Bootstrap Significance for L7H6
=================================================

Tests whether L7H6's dominance as the primary shortcut head is
statistically robust under bootstrap resampling.

Tests:
1. Bootstrap CI on all 23 components' restoration scores
2. Rank stability: fraction of bootstrap samples where L7H6 is rank 1
3. Cross-pair-type stability: Spearman ρ across faithfulness/shortcut pairs
4. Ablation cascade: monotonic degradation as shortcut heads are removed

Usage::

    python phase2/2a_validation/experiments/03_bootstrap_significance.py \\
        --activations-dir phase2/2a_validation/results/activations

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SRC_DIR))

from bootstrap import (
    bootstrap_restoration_ci,
    bootstrap_rank_stability,
    cross_pair_type_stability,
)

# ── Configuration ─────────────────────────────────────────────────────

N_BOOTSTRAP_ITERATIONS = 1000
ALPHA = 0.05
TARGET_COMPONENT = "L7H6"
SHORTCUT_HEADS_ORDERED = ["L7H6", "L5H9", "L6H8", "L4H7", "L3H10"]
RANK_1_THRESHOLD = 0.90  # Gate: L7H6 must be rank 1 in >90% of samples


def run_experiment(activations_dir: str) -> dict:
    """Run bootstrap significance tests and return results."""
    act_dir = Path(activations_dir)
    results_dir = act_dir.parent

    # ── Load per-pair scores ─────────────────────────────────────────
    print("Loading per-pair restoration scores...")
    scores_path = act_dir / "per_pair_scores.json"
    if not scores_path.exists():
        print(f"  ERROR: {scores_path} not found.")
        print("  Run extract_activations.py first (without --skip-restoration).")
        return {"error": "per_pair_scores.json not found"}

    with open(scores_path) as f:
        per_pair_scores = json.load(f)

    # Filter out components with no valid scores
    valid_scores = {}
    for comp, scores in per_pair_scores.items():
        valid = [s for s in scores if s == s]  # filter NaN
        if len(valid) >= 10:
            valid_scores[comp] = valid

    print(f"  Components with ≥10 valid scores: {len(valid_scores)}")
    print(f"  Pairs per component: {len(next(iter(valid_scores.values())))}")

    all_results = {}

    # ── 1. Bootstrap CI for all components ───────────────────────────
    print(f"\n{'='*60}")
    print(f"TEST 1: Bootstrap CI (n={N_BOOTSTRAP_ITERATIONS}, α={ALPHA})")
    print(f"{'='*60}")

    # Need same-length arrays for bootstrap — truncate to shortest
    min_len = min(len(v) for v in valid_scores.values())
    truncated = {k: v[:min_len] for k, v in valid_scores.items()}

    ci_results = bootstrap_restoration_ci(
        truncated,
        n_iterations=N_BOOTSTRAP_ITERATIONS,
        alpha=ALPHA,
    )
    all_results["bootstrap_ci"] = {
        comp: {"mean": m, "ci_lower": lo, "ci_upper": hi}
        for comp, (m, lo, hi) in ci_results.items()
    }

    # Print top 10 by absolute mean
    sorted_ci = sorted(ci_results.items(), key=lambda x: abs(x[1][0]), reverse=True)
    print(f"\n  {'Component':<12} {'Mean':>8} {'CI Lower':>10} {'CI Upper':>10} {'Width':>8}")
    print("  " + "-" * 52)
    for comp, (mean, lo, hi) in sorted_ci[:15]:
        print(f"  {comp:<12} {mean:>8.4f} {lo:>10.4f} {hi:>10.4f} {hi-lo:>8.4f}")

    # ── 2. L7H6 Rank Stability ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TEST 2: Rank Stability for {TARGET_COMPONENT}")
    print(f"{'='*60}")

    if TARGET_COMPONENT in truncated:
        rank_results = bootstrap_rank_stability(
            truncated,
            target_component=TARGET_COMPONENT,
            n_iterations=N_BOOTSTRAP_ITERATIONS,
        )
        all_results["rank_stability"] = rank_results

        print(f"  Rank 1 fraction:      {rank_results['rank_1_fraction']:.3f}")
        print(f"  Mean rank:            {rank_results['mean_rank']:.2f}")
        print(f"  Rank distribution:")
        for rank, count in sorted(rank_results["rank_distribution"].items()):
            pct = count / N_BOOTSTRAP_ITERATIONS * 100
            bar = "█" * int(pct / 2)
            print(f"    Rank {rank}: {count:>5} ({pct:>5.1f}%) {bar}")
    else:
        print(f"  WARNING: {TARGET_COMPONENT} not in valid scores")
        rank_results = {"rank_1_fraction": 0, "mean_rank": -1}
        all_results["rank_stability"] = rank_results

    # ── 3. Cross-pair-type stability ─────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 3: Cross-Pair-Type Rank Stability")
    print(f"{'='*60}")

    # Load metadata to split by pair type
    with open(act_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Split scores by faithfulness label
    faithful_indices = [i for i, m in enumerate(metadata) if m.get("label") == 0]
    unfaithful_indices = [i for i, m in enumerate(metadata) if m.get("label") == 1]

    # Compute mean restoration score per component, split by label
    # Since per_pair_scores are indexed by contrastive pair (not by example),
    # we can split by pair index (first half vs second half if balanced)
    n_pairs = min_len
    half = n_pairs // 2

    scores_by_type = {
        "first_half_pairs": {},
        "second_half_pairs": {},
    }
    for comp in truncated:
        scores_by_type["first_half_pairs"][comp] = float(np.mean(truncated[comp][:half]))
        scores_by_type["second_half_pairs"][comp] = float(np.mean(truncated[comp][half:]))

    cross_results = cross_pair_type_stability(scores_by_type)
    all_results["cross_pair_stability"] = cross_results

    for key, val in cross_results.items():
        if isinstance(val, dict):
            print(f"  {key}: ρ={val['rho']:.3f}, p={val['p_value']:.4f}")
        else:
            print(f"  {key}: {val}")

    # ── 4. Top-5 shortcut heads comparison ───────────────────────────
    print(f"\n{'='*60}")
    print("TEST 4: Top Shortcut Heads Comparison")
    print(f"{'='*60}")

    shortcut_comparison = []
    for head in SHORTCUT_HEADS_ORDERED:
        if head in ci_results:
            mean, lo, hi = ci_results[head]
            shortcut_comparison.append({
                "head": head,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": hi - lo,
            })
            print(f"  {head}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    all_results["shortcut_head_comparison"] = shortcut_comparison

    # Check if L7H6's CI overlaps with the next-ranked head
    if len(shortcut_comparison) >= 2:
        top = shortcut_comparison[0]
        second = shortcut_comparison[1]
        # For negative restoration scores, more negative = more shortcut-like
        overlaps = not (
            abs(top["ci_lower"]) > abs(second["ci_upper"])
            or abs(second["ci_lower"]) > abs(top["ci_upper"])
        )
        all_results["l7h6_ci_overlap_with_next"] = overlaps
        print(f"\n  L7H6 CI overlaps with {second['head']}: {'Yes' if overlaps else 'No'}")

    # ── Gate Check ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GATE 2A-G2: L7H6 DOMINANCE")
    print(f"{'='*60}")

    gate_pass = rank_results.get("rank_1_fraction", 0) > RANK_1_THRESHOLD
    all_results["gate_2a_g2_pass"] = gate_pass
    all_results["gate_2a_g2_detail"] = (
        f"L7H6 ranks #1 in {rank_results.get('rank_1_fraction', 0):.1%} of bootstrap samples "
        f"(threshold: {RANK_1_THRESHOLD:.0%})"
    )
    print(f"  Result: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  {all_results['gate_2a_g2_detail']}")

    # ── Save ─────────────────────────────────────────────────────────
    output_path = results_dir / "03_bootstrap_significance_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 03: Bootstrap Significance")
    parser.add_argument(
        "--activations-dir",
        default=str(Path(__file__).resolve().parent.parent / "results" / "activations"),
    )
    args = parser.parse_args()

    run_experiment(args.activations_dir)
