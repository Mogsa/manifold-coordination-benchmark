"""Tests for chaosbench.validation.baselines — Stage 2 baseline battery."""

import numpy as np
import pytest

from chaosbench.validation.baselines import (
    baseline_persistence,
    baseline_mean_reversion,
    baseline_ar5,
    baseline_classify_moments,
    baseline_identify_return_map,
    validate_stage2_predict,
)


class TestBaselinePersistence:
    def test_shape(self):
        data = np.array([1.0, 2.0, 3.0])
        pred = baseline_persistence(data, 5)
        assert pred.shape == (5,)

    def test_value(self):
        data = np.array([1.0, 2.0, 3.0])
        pred = baseline_persistence(data, 3)
        np.testing.assert_array_equal(pred, [3.0, 3.0, 3.0])


class TestBaselineMeanReversion:
    def test_shape(self):
        data = np.array([1.0, 2.0, 3.0])
        pred = baseline_mean_reversion(data, 4)
        assert pred.shape == (4,)

    def test_value(self):
        data = np.array([1.0, 2.0, 3.0])
        pred = baseline_mean_reversion(data, 2)
        np.testing.assert_array_equal(pred, [2.0, 2.0])


class TestBaselineAR5:
    def test_shape(self):
        data = np.random.RandomState(42).random(100)
        pred = baseline_ar5(data, 10)
        assert pred.shape == (10,)

    def test_short_data_fallback(self):
        """With very short data, should fall back to persistence."""
        data = np.array([1.0])
        pred = baseline_ar5(data, 3)
        np.testing.assert_array_equal(pred, [1.0, 1.0, 1.0])

    def test_convergent_detected(self):
        """AR5 on a convergent series should predict near the fixed point."""
        data = np.array([5.0 * 0.5**i for i in range(50)])
        pred = baseline_ar5(data, 5)
        # Should predict values near zero
        assert all(abs(p) < 1.0 for p in pred)


class TestBaselineClassifyMoments:
    def test_fixed_point(self):
        """Convergent data should classify as fixed_point."""
        data = np.array([5.0 * 0.5**i for i in range(100)])
        assert baseline_classify_moments(data) == "fixed_point"

    def test_periodic(self):
        """Periodic data should classify as periodic."""
        data = np.tile([0.2, 0.8], 100)
        assert baseline_classify_moments(data) == "periodic"

    def test_chaotic(self):
        """Chaotic-looking data should classify as chaotic."""
        from chaosbench.grammar.atoms import LogisticAtom
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 2000)[500:]
        assert baseline_classify_moments(traj) == "chaotic"


class TestBaselineIdentifyReturnMap:
    def test_damped_linear(self):
        """Convergent trajectory should identify as damped_linear."""
        data = np.array([5.0 * 0.5**i for i in range(100)])
        assert baseline_identify_return_map(data) == "damped_linear"

    def test_returns_known_family(self):
        """Result should be one of the known families."""
        data = np.random.RandomState(42).random(100)
        result = baseline_identify_return_map(data)
        assert result in ["logistic", "tent", "damped_linear", "rotation"]


class TestValidateStage2Predict:
    def test_has_structure(self):
        result = validate_stage2_predict({"k_eff": 3}, tau_lambda=5.0)
        assert result["has_structure"]
        assert not result["is_trivial"]
        assert result["passed"]

    def test_no_structure(self):
        result = validate_stage2_predict({"k_eff": 0}, tau_lambda=5.0)
        assert not result["has_structure"]
        assert not result["passed"]

    def test_trivial(self):
        result = validate_stage2_predict({"k_eff": 20}, tau_lambda=5.0)
        assert result["has_structure"]
        assert result["is_trivial"]
        assert not result["passed"]

    def test_non_chaotic_not_trivial(self):
        """Non-chaotic (tau=inf) systems are never flagged as trivial."""
        result = validate_stage2_predict({"k_eff": 50}, tau_lambda=float("inf"))
        assert not result["is_trivial"]
