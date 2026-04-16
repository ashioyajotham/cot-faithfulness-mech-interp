"""
Steering vector detector for faithfulness classification.

Uses the difference-of-means approach on circuit activations:
faithful_mean − unfaithful_mean gives a direction in activation space
that separates the two classes.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class SteeringVectorDetector:
    """Diff-of-means steering vector with projection-based thresholding."""

    def __init__(self):
        self.steering_vector: Optional[np.ndarray] = None
        self.threshold: float = 0.0
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_method: str = "median",
    ) -> "SteeringVectorDetector":
        """Compute the steering vector as mean(faithful) − mean(unfaithful)."""
        faithful_mask = y == 1
        unfaithful_mask = y == 0

        mean_faithful = X[faithful_mask].mean(axis=0)
        mean_unfaithful = X[unfaithful_mask].mean(axis=0)

        self.steering_vector = mean_faithful - mean_unfaithful
        norm = np.linalg.norm(self.steering_vector)
        if norm > 0:
            self.steering_vector /= norm

        projections = X @ self.steering_vector
        if threshold_method == "median":
            self.threshold = float(np.median(projections))
        elif threshold_method == "mean":
            self.threshold = float(np.mean(projections))
        else:
            self.threshold = 0.0

        self._is_fitted = True
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project activations onto the steering direction."""
        assert self._is_fitted, "Must call fit() first"
        return X @ self.steering_vector

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict faithfulness labels (1=faithful, 0=unfaithful)."""
        projections = self.project(X)
        return (projections >= self.threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
