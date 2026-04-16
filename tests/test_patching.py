"""Tests for shared/patching/ module."""

import numpy as np
import pytest


class TestRestorationScore:
    """Unit tests for the restoration score computation."""

    def test_perfect_restoration_returns_one(self):
        """When patched_diff == clean_diff, score should be 1.0."""
        clean_diff = 5.0
        corrupted_diff = -2.0
        patched_diff = 5.0
        gap = clean_diff - corrupted_diff
        score = (patched_diff - corrupted_diff) / gap
        assert abs(score - 1.0) < 1e-6

    def test_no_effect_returns_zero(self):
        """When patched_diff == corrupted_diff, score should be 0.0."""
        clean_diff = 5.0
        corrupted_diff = -2.0
        patched_diff = -2.0
        gap = clean_diff - corrupted_diff
        score = (patched_diff - corrupted_diff) / gap
        assert abs(score) < 1e-6

    def test_negative_restoration(self):
        """When patching makes things worse, score should be negative."""
        clean_diff = 5.0
        corrupted_diff = -2.0
        patched_diff = -4.0
        gap = clean_diff - corrupted_diff
        score = (patched_diff - corrupted_diff) / gap
        assert score < 0


class TestHooks:
    """Tests for hook factory functions (require no model)."""

    def test_make_zero_ablation_hook_zeroes_head(self):
        import torch
        from shared.patching.hooks import make_zero_ablation_hook

        hook_fn = make_zero_ablation_hook(layer=0, head_idx=2)
        activation = torch.randn(1, 10, 4, 64)
        result = hook_fn(activation, None)
        assert (result[:, :, 2, :] == 0).all()
        assert (result[:, :, 0, :] != 0).any()

    def test_make_zero_ablation_hook_zeroes_layer(self):
        import torch
        from shared.patching.hooks import make_zero_ablation_hook

        hook_fn = make_zero_ablation_hook(layer=0, head_idx=None)
        activation = torch.randn(1, 10, 768)
        result = hook_fn(activation, None)
        assert (result == 0).all()
