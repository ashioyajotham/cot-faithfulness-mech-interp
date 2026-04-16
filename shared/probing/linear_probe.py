"""
Linear probe for faithfulness detection with built-in selectivity testing.

Implements:
- Logistic regression probe on circuit activations
- Hewitt-Liang selectivity (Experiment 1)
- Minimum description length score (Voita & Titov 2020)
- Bootstrap confidence intervals
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional, Tuple

from .control_tasks import generate_control_labels


class FaithfulnessProbe:
    """Linear probe with built-in selectivity testing.

    Wraps ``LogisticRegression`` and exposes:
    - ``fit(X, y)``
    - ``predict(X)``
    - ``selectivity(X, y, n_control_seeds)``
    - ``bootstrap_ci(X, y, n_iterations)``
    """

    def __init__(
        self,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaithfulnessProbe":
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def roc_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)

    # --- Hewitt-Liang selectivity -------------------------------------------

    def selectivity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_seeds: int = 5,
        pair_ids: Optional[np.ndarray] = None,
    ) -> float:
        """Hewitt-Liang selectivity: linguistic_acc − mean(control_task_acc).

        *pair_ids* (if provided) are used to stratify the random label
        assignment, matching the original protocol.
        """
        linguistic_acc = self.accuracy(X, y)

        control_accs = []
        for seed in range(n_seeds):
            control_y = generate_control_labels(y, pair_ids=pair_ids, seed=seed)
            control_probe = FaithfulnessProbe(random_state=seed)
            control_probe.fit(X, control_y)
            control_accs.append(control_probe.accuracy(X, control_y))

        return linguistic_acc - float(np.mean(control_accs))

    # --- MDL score ----------------------------------------------------------

    def mdl_score(self, X: np.ndarray, y: np.ndarray, n_portions: int = 10) -> float:
        """Minimum description length compression score (Voita & Titov 2020).

        Returns the online coding cost in bits.  Lower values mean the
        probe compresses the label information more efficiently (i.e.
        the representations encode the target).
        """
        n = len(y)
        portion_sizes = np.linspace(0, n, n_portions + 1, dtype=int)[1:]
        indices = np.arange(n)
        np.random.RandomState(42).shuffle(indices)

        total_bits = 0.0
        for i in range(len(portion_sizes) - 1):
            train_idx = indices[: portion_sizes[i]]
            test_idx = indices[portion_sizes[i] : portion_sizes[i + 1]]
            if len(train_idx) < 2 or len(test_idx) == 0:
                continue

            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(X[train_idx], y[train_idx])
            proba = probe.predict_proba(X[test_idx])

            for j, idx in enumerate(test_idx):
                p = np.clip(proba[j, int(y[idx])], 1e-10, 1.0)
                total_bits -= np.log2(p)

        return total_bits

    # --- Bootstrap CI -------------------------------------------------------

    def bootstrap_ci(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 1000,
        alpha: float = 0.05,
        metric: str = "accuracy",
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for probe performance."""
        rng = np.random.RandomState(42)
        n = len(y)
        scores = []

        for _ in range(n_iterations):
            idx = rng.choice(n, size=n, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            probe = FaithfulnessProbe(random_state=42)
            probe.fit(X[idx], y[idx])
            if metric == "accuracy":
                scores.append(probe.accuracy(X[idx], y[idx]))
            elif metric == "roc_auc":
                try:
                    scores.append(probe.roc_auc(X[idx], y[idx]))
                except ValueError:
                    continue

        lower = float(np.percentile(scores, 100 * alpha / 2))
        upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return lower, upper
