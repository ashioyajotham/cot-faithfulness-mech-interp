"""
Probe selectivity analysis — Experiment 01.

Implements the Hewitt-Liang selectivity test and ablation baselines
to determine whether the Phase 1 linear probe detects genuine
representation structure or surface features.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from shared.probing.linear_probe import FaithfulnessProbe
from shared.probing.control_tasks import generate_control_labels


def run_selectivity_test(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: Optional[np.ndarray] = None,
    n_random_labels: int = 5,
) -> Dict[str, float]:
    """Run the full Hewitt-Liang selectivity test.

    Returns a dict with ``selectivity``, ``linguistic_acc``,
    ``control_acc_mean``, ``control_acc_std``.
    """
    probe = FaithfulnessProbe()
    probe.fit(X, y)

    linguistic_acc = probe.accuracy(X, y)

    control_accs = []
    for seed in range(n_random_labels):
        ctrl_y = generate_control_labels(y, pair_ids=pair_ids, seed=seed)
        ctrl_probe = FaithfulnessProbe(random_state=seed)
        ctrl_probe.fit(X, ctrl_y)
        control_accs.append(ctrl_probe.accuracy(X, ctrl_y))

    return {
        "selectivity": linguistic_acc - float(np.mean(control_accs)),
        "linguistic_acc": linguistic_acc,
        "control_acc_mean": float(np.mean(control_accs)),
        "control_acc_std": float(np.std(control_accs)),
    }


def run_scramble_ablation(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int = 5,
) -> Dict[str, float]:
    """Scramble activation vectors across examples.

    If the probe degrades, it is using structure, not surface features.
    """
    probe = FaithfulnessProbe()
    probe.fit(X, y)
    original_acc = probe.accuracy(X, y)

    scrambled_accs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        X_scrambled = X.copy()
        rng.shuffle(X_scrambled)
        scrambled_probe = FaithfulnessProbe(random_state=seed)
        scrambled_probe.fit(X_scrambled, y)
        scrambled_accs.append(scrambled_probe.accuracy(X_scrambled, y))

    return {
        "original_acc": original_acc,
        "scrambled_acc_mean": float(np.mean(scrambled_accs)),
        "scrambled_acc_std": float(np.std(scrambled_accs)),
        "degradation": original_acc - float(np.mean(scrambled_accs)),
    }


def run_random_layer_baseline(
    circuit_X: np.ndarray,
    random_layer_X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Compare probe accuracy on circuit activations vs random-layer activations.

    If the probe performs comparably on random layers, the circuit
    identification added no value over arbitrary layer selection.
    """
    circuit_probe = FaithfulnessProbe()
    circuit_probe.fit(circuit_X, y)

    random_probe = FaithfulnessProbe()
    random_probe.fit(random_layer_X, y)

    return {
        "circuit_acc": circuit_probe.accuracy(circuit_X, y),
        "random_layer_acc": random_probe.accuracy(random_layer_X, y),
        "advantage": circuit_probe.accuracy(circuit_X, y) - random_probe.accuracy(random_layer_X, y),
    }
