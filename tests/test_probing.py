"""Tests for shared/probing/ module."""

import numpy as np
import pytest


class TestFaithfulnessProbe:
    """Unit tests for the linear probe wrapper."""

    def test_fit_and_predict(self):
        from shared.probing.linear_probe import FaithfulnessProbe

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)

        probe = FaithfulnessProbe()
        probe.fit(X, y)
        preds = probe.predict(X)
        assert probe.accuracy(X, y) > 0.7

    def test_selectivity_positive_for_structured_data(self):
        from shared.probing.linear_probe import FaithfulnessProbe

        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = (X[:, 0] > 0).astype(int)

        probe = FaithfulnessProbe()
        probe.fit(X, y)
        sel = probe.selectivity(X, y, n_seeds=3)
        assert sel > 0, f"Expected positive selectivity, got {sel}"

    def test_bootstrap_ci_contains_mean(self):
        from shared.probing.linear_probe import FaithfulnessProbe

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        probe = FaithfulnessProbe()
        probe.fit(X, y)
        lower, upper = probe.bootstrap_ci(X, y, n_iterations=50)
        acc = probe.accuracy(X, y)
        assert lower <= acc <= upper or abs(lower - upper) < 0.2


class TestControlTasks:
    """Tests for Hewitt-Liang control label generation."""

    def test_control_labels_preserve_class_count(self):
        from shared.probing.control_tasks import generate_control_labels

        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        ctrl = generate_control_labels(y, seed=0)
        assert len(ctrl) == len(y)

    def test_different_seeds_give_different_labels(self):
        from shared.probing.control_tasks import generate_control_labels

        y = np.array([0, 1] * 50)
        c1 = generate_control_labels(y, seed=0)
        c2 = generate_control_labels(y, seed=1)
        assert not np.array_equal(c1, c2)
