"""Tests for chaosbench.scoring.difficulty — composite difficulty scoring."""

import pytest

from chaosbench.scoring.difficulty import composite_difficulty, weighted_score


class TestCompositeDifficulty:
    def test_minimum(self):
        """h_ks=0, depth=0, noise=0 → difficulty = 1.0."""
        assert composite_difficulty(0.0, 0, 0.0) == pytest.approx(1.0)

    def test_chaos_increases(self):
        d0 = composite_difficulty(0.0)
        d1 = composite_difficulty(0.5)
        d2 = composite_difficulty(1.0)
        assert d0 < d1 < d2

    def test_depth_increases(self):
        d0 = composite_difficulty(0.5, grammar_depth=0)
        d1 = composite_difficulty(0.5, grammar_depth=1)
        assert d1 > d0

    def test_noise_increases(self):
        d0 = composite_difficulty(0.5, noise_std=0.0)
        d1 = composite_difficulty(0.5, noise_std=0.1)
        assert d1 > d0

    def test_formula(self):
        """Verify explicit formula: (1+h_ks) × (1+depth) × (1+10σ)."""
        h, d, s = 0.5, 1, 0.05
        expected = (1 + h) * (1 + d) * (1 + 10 * s)
        assert composite_difficulty(h, d, s) == pytest.approx(expected)

    def test_non_chaotic_not_zero(self):
        """Non-chaotic systems with noise still have difficulty > 1."""
        d = composite_difficulty(0.0, 0, 0.01)
        assert d > 1.0


class TestWeightedScore:
    def test_basic(self):
        assert weighted_score(0.5, 2.0) == pytest.approx(1.0)

    def test_zero_score(self):
        assert weighted_score(0.0, 5.0) == pytest.approx(0.0)

    def test_higher_difficulty_higher_weighted(self):
        w1 = weighted_score(0.8, 1.5)
        w2 = weighted_score(0.8, 3.0)
        assert w2 > w1
