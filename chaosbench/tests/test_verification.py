"""Tests for chaosbench.problems.verification — 3 verify functions."""

import numpy as np
import pytest

from chaosbench.problems.verification import (
    verify_classify,
    verify_identify,
    verify_predict,
)


class TestVerifyClassify:
    def test_exact_match(self):
        assert verify_classify("chaotic", "chaotic") == 1.0

    def test_case_insensitive(self):
        assert verify_classify("CHAOTIC", "chaotic") == 1.0
        assert verify_classify("Periodic", "periodic") == 1.0

    def test_whitespace_stripped(self):
        assert verify_classify("  chaotic  ", "chaotic") == 1.0

    def test_mismatch(self):
        assert verify_classify("periodic", "chaotic") == 0.0

    def test_non_string_answer_returns_zero(self):
        assert verify_classify(None, "chaotic") == 0.0
        assert verify_classify(123, "chaotic") == 0.0


class TestVerifyIdentify:
    def test_exact_match(self):
        assert verify_identify("logistic", "logistic") == 1.0

    def test_case_insensitive(self):
        assert verify_identify("LOGISTIC", "logistic") == 1.0

    def test_mismatch(self):
        assert verify_identify("tent", "logistic") == 0.0

    def test_non_string_answer_returns_zero(self):
        assert verify_identify(None, "logistic") == 0.0
        assert verify_identify(123, "logistic") == 0.0


class TestVerifyPredict:
    def test_perfect_prediction(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        train = np.array([0.0, 5.0])  # diameter = 5, epsilon = 0.5
        result = verify_predict(pred, true, train)
        assert result["k_eff"] == 4
        assert result["K"] == 4
        assert result["raw_score"] == 1.0
        assert result["epsilon"] == pytest.approx(0.5)

    def test_first_wrong(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([10.0, 2.0, 3.0])
        train = np.array([0.0, 5.0])
        result = verify_predict(pred, true, train)
        assert result["k_eff"] == 0
        assert result["raw_score"] == 0.0

    def test_partial_correct(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 30.0, 4.0])
        train = np.array([0.0, 5.0])  # epsilon = 0.5
        result = verify_predict(pred, true, train)
        assert result["k_eff"] == 2
        assert result["raw_score"] == pytest.approx(0.5)

    def test_short_prediction_padded(self):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0])  # Too short, padded with NaN
        train = np.array([0.0, 10.0])
        result = verify_predict(pred, true, train)
        assert result["k_eff"] == 2
        assert result["K"] == 4

    def test_epsilon_floor(self):
        """When attractor diameter is tiny, epsilon floors at 0.01."""
        true = np.array([0.5, 0.5, 0.5])
        pred = np.array([0.5, 0.5, 0.5])
        train = np.array([0.5, 0.5])  # diameter = 0, epsilon = max(0, 0.01) = 0.01
        result = verify_predict(pred, true, train)
        assert result["epsilon"] == pytest.approx(0.01)

    def test_long_prediction_truncated(self):
        true = np.array([1.0, 2.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        train = np.array([0.0, 10.0])
        result = verify_predict(pred, true, train)
        assert result["K"] == 2
        assert result["k_eff"] == 2
