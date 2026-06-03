"""
Experiment 01 — Probe Selectivity (Hewitt-Liang)
=================================================

Tests whether the Phase 1 linear probe detects genuine representation
structure or surface features of prompt construction.

Three tests:
1. **Hewitt-Liang selectivity**: Train on real labels vs random control labels.
   selectivity = linguistic_acc - control_acc.  Target: > 0.15.
2. **Scramble ablation**: Shuffle activation vectors across examples.
   If probe degrades → it uses structure, not surface features.
3. **Random-layer baseline**: Compare circuit activations vs Layer 8
   (non-circuit).  If probe is comparable → circuit ID added no value.
4. **MDL score**: Minimum description length compression (Voita & Titov 2020).

Usage::

    python phase2/2a_validation/experiments/01_probe_selectivity.py \\
        --activations-dir phase2/2a_validation/results/activations

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow imports from project root and 2a_validation/src
# (2a_validation starts with a digit — not a valid Python package name)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SRC_DIR))

from selectivity import (
    run_random_layer_baseline,
    run_scramble_ablation,
    run_selectivity_test,
)
from shared.probing.linear_probe import FaithfulnessProbe

# ── Configuration ─────────────────────────────────────────────────────

SELECTIVITY_THRESHOLD = 0.15
N_RANDOM_LABELS = 5
N_SCRAMBLE_SEEDS = 5


def run_experiment(activations_dir: str) -> dict:
    """Run all selectivity tests and return combined results."""
    act_dir = Path(activations_dir)
    results_dir = act_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading activations...")
    data = np.load(act_dir / "activations.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    components = data["components"]

    random_data = np.load(act_dir / "random_layer_activations.npz")
    X_random = random_data["X"]

    with open(act_dir / "metadata.json") as f:
        metadata = json.load(f)
    pair_ids = np.array([m.get("pair_id", i) for i, m in enumerate(metadata)])

    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  X_random: {X_random.shape}")
    print(f"  Classes: {np.bincount(y.astype(int))}")

    all_results = {}

    # ── 1. Hewitt-Liang Selectivity ───────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 1: Hewitt-Liang Selectivity")
    print(f"{'='*60}")

    sel_results = run_selectivity_test(
        X, y, pair_ids=pair_ids, n_random_labels=N_RANDOM_LABELS,
    )
    all_results["selectivity"] = sel_results

    selective = sel_results["selectivity"] > SELECTIVITY_THRESHOLD
    print(f"  Linguistic accuracy:  {sel_results['linguistic_acc']:.3f}")
    print(f"  Control accuracy:     {sel_results['control_acc_mean']:.3f} ± {sel_results['control_acc_std']:.3f}")
    print(f"  Selectivity:          {sel_results['selectivity']:.3f}")
    print(f"  Threshold:            {SELECTIVITY_THRESHOLD}")
    print(f"  PASS: {'✓' if selective else '✗'}")

    # ── 2. Scramble Ablation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 2: Scramble Ablation")
    print(f"{'='*60}")

    scramble_results = run_scramble_ablation(X, y, n_seeds=N_SCRAMBLE_SEEDS)
    all_results["scramble_ablation"] = scramble_results

    print(f"  Original accuracy:    {scramble_results['original_acc']:.3f}")
    print(f"  Scrambled accuracy:   {scramble_results['scrambled_acc_mean']:.3f} ± {scramble_results['scrambled_acc_std']:.3f}")
    print(f"  Degradation:          {scramble_results['degradation']:.3f}")
    print(f"  Uses structure: {'✓' if scramble_results['degradation'] > 0.05 else '✗'}")

    # ── 3. Random-Layer Baseline ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 3: Random-Layer Baseline (Layer 8)")
    print(f"{'='*60}")

    random_results = run_random_layer_baseline(X, X_random, y)
    all_results["random_layer_baseline"] = random_results

    print(f"  Circuit accuracy:     {random_results['circuit_acc']:.3f}")
    print(f"  Random layer accuracy:{random_results['random_layer_acc']:.3f}")
    print(f"  Advantage:            {random_results['advantage']:.3f}")
    print(f"  Circuit adds value: {'✓' if random_results['advantage'] > 0.05 else '✗'}")

    # ── 4. MDL Score ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 4: Minimum Description Length")
    print(f"{'='*60}")

    probe = FaithfulnessProbe()
    probe.fit(X, y)
    mdl = probe.mdl_score(X, y)
    all_results["mdl_score"] = mdl

    # Compare with random-layer MDL
    probe_random = FaithfulnessProbe()
    probe_random.fit(X_random, y)
    mdl_random = probe_random.mdl_score(X_random, y)
    all_results["mdl_score_random_layer"] = mdl_random

    print(f"  Circuit MDL:          {mdl:.1f} bits")
    print(f"  Random-layer MDL:     {mdl_random:.1f} bits")
    print(f"  Circuit compresses more: {'✓' if mdl < mdl_random else '✗'}")

    # ── 5. Bootstrap CI on probe accuracy ─────────────────────────────
    print(f"\n{'='*60}")
    print("TEST 5: Bootstrap CI on Probe Accuracy")
    print(f"{'='*60}")

    lower, upper = probe.bootstrap_ci(X, y, n_iterations=500, metric="accuracy")
    all_results["bootstrap_ci_accuracy"] = {"lower": lower, "upper": upper}

    lower_auc, upper_auc = probe.bootstrap_ci(X, y, n_iterations=500, metric="roc_auc")
    all_results["bootstrap_ci_roc_auc"] = {"lower": lower_auc, "upper": upper_auc}

    print(f"  Accuracy CI (95%):    [{lower:.3f}, {upper:.3f}]")
    print(f"  ROC-AUC CI (95%):     [{lower_auc:.3f}, {upper_auc:.3f}]")

    # ── Gate Check ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GATE 2A-G1: PROBE SELECTIVITY")
    print(f"{'='*60}")

    gate_pass = sel_results["selectivity"] > 0
    all_results["gate_2a_g1_pass"] = gate_pass
    all_results["gate_2a_g1_detail"] = (
        "Probe reflects genuine representation structure"
        if gate_pass
        else "Probe may rely on surface features — reframe claims"
    )
    print(f"  Result: {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print(f"  {all_results['gate_2a_g1_detail']}")

    # ── Save ──────────────────────────────────────────────────────────
    output_path = results_dir / "01_probe_selectivity_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 01: Probe Selectivity")
    parser.add_argument(
        "--activations-dir",
        default=str(Path(__file__).resolve().parent.parent / "results" / "activations"),
        help="Directory containing activations.npz",
    )
    args = parser.parse_args()

    run_experiment(args.activations_dir)
