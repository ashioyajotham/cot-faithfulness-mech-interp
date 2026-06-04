"""
Experiment 02 — High-Confidence False Negative Analysis
========================================================

Clusters the high-confidence false negatives from Phase 1 by carry
requirement, sum magnitude, corruption severity, and corruption type
to identify systematic failure modes.

Research Question (RQ2): What structural or computational properties
distinguish the 11 high-confidence false negatives?

Usage::

    python phase2/2a_validation/experiments/02_false_negative_analysis.py \\
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

from error_analysis import (
    cluster_by_features,
    extract_false_negatives,
    test_carry_hypothesis,
    test_magnitude_hypothesis,
)
from shared.probing.linear_probe import FaithfulnessProbe

# ── Configuration ─────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.99
CLUSTER_FEATURES = [
    "carry_required",
    "sum_magnitude",
    "corruption_severity",
    "corruption_type",
]
MAGNITUDE_THRESHOLD = 80


def run_experiment(activations_dir: str) -> dict:
    """Run false-negative analysis and return results."""
    act_dir = Path(activations_dir)
    results_dir = act_dir.parent

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading activations and metadata...")
    data = np.load(act_dir / "activations.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]

    with open(act_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  X: {X.shape}, y: {y.shape}")

    # ── Train probe and get predictions ──────────────────────────────
    print("\nTraining probe for error analysis...")
    probe = FaithfulnessProbe()
    probe.fit(X, y)

    y_pred = probe.predict(X)
    y_proba = probe.predict_proba(X)[:, 1]  # P(unfaithful)

    accuracy = probe.accuracy(X, y)
    print(f"  Full-dataset accuracy: {accuracy:.3f}")

    # ── Extract false negatives ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE EXTRACTION")
    print(f"{'='*60}")

    # For false negatives: true=unfaithful(1), predicted=faithful(0)
    # We need P(faithful) > threshold for high-confidence FNs
    # y_proba is P(unfaithful), so P(faithful) = 1 - y_proba
    p_faithful = 1 - y_proba

    fn_examples = extract_false_negatives(
        y_true=y,
        y_pred=y_pred,
        y_proba=p_faithful,  # confidence in the PREDICTED class (faithful)
        metadata=metadata,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    print(f"  Total false negatives: {sum((y == 1) & (y_pred == 0))}")
    print(f"  High-confidence FNs (>{CONFIDENCE_THRESHOLD}): {len(fn_examples)}")

    all_results = {
        "total_examples": len(y),
        "accuracy": accuracy,
        "total_false_negatives": int(sum((y == 1) & (y_pred == 0))),
        "total_false_positives": int(sum((y == 0) & (y_pred == 1))),
        "high_confidence_fn_count": len(fn_examples),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    if len(fn_examples) == 0:
        print("\n  No high-confidence false negatives found!")
        print("  This may indicate the probe has improved or the threshold is too high.")
        all_results["fn_analysis"] = "no_high_confidence_fn"

        # Still do analysis on ALL false negatives
        all_fn = []
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 0:
                all_fn.append({
                    "index": i,
                    "confidence": float(p_faithful[i]),
                    **metadata[i],
                })
        fn_examples = all_fn
        print(f"  Falling back to all {len(fn_examples)} false negatives")

    # ── Cluster by features ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CLUSTERING BY FEATURES")
    print(f"{'='*60}")

    clusters = cluster_by_features(fn_examples, CLUSTER_FEATURES)
    all_results["clusters"] = {}

    for feat, counter in clusters.items():
        print(f"\n  {feat}:")
        all_results["clusters"][feat] = dict(counter)
        for val, count in counter.most_common():
            pct = count / len(fn_examples) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")

    # ── Carry hypothesis test ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST: Carry Overrepresentation")
    print(f"{'='*60}")

    carry_results = test_carry_hypothesis(fn_examples, metadata)
    all_results["carry_hypothesis"] = carry_results

    print(f"  FN carry rate:        {carry_results['fn_carry_rate']:.3f}")
    print(f"  Dataset carry rate:   {carry_results['dataset_carry_rate']:.3f}")
    print(f"  Overrepresentation:   {carry_results['overrepresentation_ratio']:.2f}x")
    print(f"  Hypothesis supported: {'YES' if carry_results['hypothesis_supported'] else 'NO'}")

    # ── Magnitude hypothesis test ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS TEST: High Magnitude (>{MAGNITUDE_THRESHOLD})")
    print(f"{'='*60}")

    mag_results = test_magnitude_hypothesis(fn_examples, MAGNITUDE_THRESHOLD)
    all_results["magnitude_hypothesis"] = mag_results

    if "error" not in mag_results:
        print(f"  High-magnitude FNs:   {mag_results['n_high_magnitude']}/{mag_results['n_total']}")
        print(f"  Rate:                 {mag_results['high_magnitude_rate']:.3f}")
        print(f"  Mean magnitude:       {mag_results['mean_magnitude']:.1f}")
        print(f"  Std magnitude:        {mag_results['std_magnitude']:.1f}")
    else:
        print(f"  Error: {mag_results['error']}")

    # ── Per-example detail table ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE DETAIL TABLE")
    print(f"{'='*60}")
    print(f"{'Idx':>5} {'a':>4} {'b':>4} {'Sum':>5} {'Carry':>6} {'Corruption':>15} {'Conf':>6}")
    print("-" * 55)

    for ex in sorted(fn_examples, key=lambda x: x.get("confidence", 0), reverse=True)[:20]:
        print(
            f"{ex.get('index', '?'):>5} "
            f"{ex.get('a', '?'):>4} "
            f"{ex.get('b', '?'):>4} "
            f"{ex.get('sum_magnitude', '?'):>5} "
            f"{'Yes' if ex.get('carry_required') else 'No':>6} "
            f"{str(ex.get('corruption_type', '?')):>15} "
            f"{ex.get('confidence', 0):>6.3f}"
        )

    # ── Prescriptions ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PRESCRIPTIONS")
    print(f"{'='*60}")

    prescriptions = []

    if carry_results.get("hypothesis_supported"):
        prescriptions.append(
            "CARRY GAP: Carry-requiring sums are overrepresented in FNs. "
            "Add carry-specific circuit components to the feature set, "
            "or augment the dataset with more carry-balanced examples."
        )

    if mag_results.get("high_magnitude_rate", 0) > 0.5:
        prescriptions.append(
            "MAGNITUDE SHIFT: High-magnitude sums dominate FNs. "
            "The probe generalises poorly at distribution edges. "
            "Augment training data to cover the full operand range [10, 99]."
        )

    if not prescriptions:
        prescriptions.append(
            "No strong clustering signal detected. "
            "FNs may reflect inherent probe capacity limits."
        )

    all_results["prescriptions"] = prescriptions
    for p in prescriptions:
        print(f"  → {p}")

    # ── Gate check ───────────────────────────────────────────────────
    gate_pass = carry_results.get("hypothesis_supported", False) or \
                mag_results.get("high_magnitude_rate", 0) > 0.3
    all_results["gate_2a_g3_pass"] = gate_pass
    all_results["gate_2a_g3_detail"] = (
        "False negative cluster identified — dataset augmentation prescription ready"
        if gate_pass
        else "No strong cluster — FNs may be irreducible noise"
    )
    print(f"\n  Gate 2A-G3: {'PASS' if gate_pass else 'INCONCLUSIVE'}")

    # ── Save ──────────────────────────────────────────────────────────
    output_path = results_dir / "02_false_negative_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 02: False Negative Analysis")
    parser.add_argument(
        "--activations-dir",
        default=str(Path(__file__).resolve().parent.parent / "results" / "activations"),
    )
    args = parser.parse_args()

    run_experiment(args.activations_dir)
