"""
Tests for Telepathic Benchmark v2 Scoring System.

Tests BLC bit counting, error penalty, probe penalty, and total score.
"""

import math
import pytest

from telepathic.core.scoring import (
    # Constants
    ERROR_SCALE_K,
    PROBE_COST,
    SCALE,
    # Functions
    count_bits,
    error_penalty,
    probe_penalty,
    compute_score,
    # Classes
    Score,
    # Analysis
    compare_scores,
    score_breakdown,
)


# =============================================================================
# BLC Bit Counting Tests (Checkpoint 3.1)
# =============================================================================

class TestBLCBitCounting:
    """Test BLC bit string counting."""

    def test_count_bits_identity(self):
        """Identity function should be 4 bits."""
        # Identity: lambda x: x = 0010 in BLC
        assert count_bits("0010") == 4

    def test_count_bits_true(self):
        """TRUE combinator should be 7 bits."""
        # TRUE: lambda x: lambda y: x = 0000110 in BLC
        assert count_bits("0000110") == 7

    def test_count_bits_false(self):
        """FALSE combinator should be 6 bits."""
        # FALSE: lambda x: lambda y: y = 000010 in BLC
        assert count_bits("000010") == 6

    def test_count_bits_empty(self):
        """Empty string should be 0 bits."""
        assert count_bits("") == 0

    def test_count_bits_validates_binary(self):
        """Should reject non-binary strings."""
        with pytest.raises(ValueError):
            count_bits("012")
        with pytest.raises(ValueError):
            count_bits("abc")


# =============================================================================
# Error Penalty Tests (Checkpoint 3.2)
# =============================================================================

class TestErrorPenalty:
    """Test MDL-style error penalty computation."""

    def test_error_penalty_zero_error(self):
        """Zero error should give zero penalty."""
        # log2(1 + 0 * 100) = log2(1) = 0
        penalty = error_penalty([0.0])
        assert penalty == pytest.approx(0.0)

    def test_error_penalty_small_error(self):
        """Small error (0.01) should give ~1 bit."""
        # log2(1 + 0.01 * 100) = log2(2) = 1
        penalty = error_penalty([0.01])
        assert penalty == pytest.approx(1.0)

    def test_error_penalty_medium_error(self):
        """Medium error (0.1) should give ~3.5 bits."""
        # log2(1 + 0.1 * 100) = log2(11) ~= 3.46
        penalty = error_penalty([0.1])
        assert 3.4 < penalty < 3.5

    def test_error_penalty_large_error(self):
        """Large error (1.0) should give ~6.7 bits."""
        # log2(1 + 1.0 * 100) = log2(101) ~= 6.66
        penalty = error_penalty([1.0])
        assert 6.6 < penalty < 6.7

    def test_error_penalty_multiple_errors(self):
        """Should sum penalties for multiple errors."""
        errors = [0.01, 0.01]  # Each gives 1 bit
        penalty = error_penalty(errors)
        assert penalty == pytest.approx(2.0)

    def test_error_penalty_uses_absolute_value(self):
        """Should use absolute value of errors."""
        penalty_pos = error_penalty([0.1])
        penalty_neg = error_penalty([-0.1])
        assert penalty_pos == penalty_neg

    def test_error_penalty_custom_k(self):
        """Should respect custom k value."""
        # With k=1: log2(1 + 0.1 * 1) = log2(1.1) ~= 0.138
        penalty = error_penalty([0.1], k=1)
        assert 0.1 < penalty < 0.2


# =============================================================================
# Probe Penalty Tests (Checkpoint 3.3)
# =============================================================================

class TestProbePenalty:
    """Test probe penalty computation."""

    def test_probe_penalty_zero(self):
        """Zero probes should give zero penalty."""
        assert probe_penalty(0) == 0

    def test_probe_penalty_one(self):
        """One probe should cost 5 bits."""
        assert probe_penalty(1) == 5

    def test_probe_penalty_ten(self):
        """Ten probes should cost 50 bits."""
        assert probe_penalty(10) == 50

    def test_probe_penalty_custom_cost(self):
        """Should respect custom cost per probe."""
        assert probe_penalty(5, cost_per_probe=10) == 50


# =============================================================================
# Total Score Tests (Checkpoint 3.4)
# =============================================================================

class TestTotalScore:
    """Test total score computation."""

    def test_compute_score_basic(self):
        """Basic score computation."""
        blc = "0010"  # 4 bits
        errors = [0.01, 0.01]  # 2 bits total error
        n_probes = 2  # 10 bits probe penalty

        score = compute_score(blc, errors, n_probes)

        assert score.compression_bits == 4
        assert score.error_penalty == pytest.approx(2.0)
        assert score.probe_penalty == 10
        assert score.total == pytest.approx(16.0)

    def test_compute_score_includes_errors_list(self):
        """Score should include original errors."""
        errors = [0.1, 0.2, 0.3]
        score = compute_score("0010", errors, 0)
        assert score.errors == errors

    def test_compute_score_includes_reasoning(self):
        """Score should preserve reasoning."""
        score = compute_score("0010", [0.0], 0, reasoning="Test reasoning")
        assert score.reasoning == "Test reasoning"


# =============================================================================
# Score Dataclass Tests (Checkpoint 3.5)
# =============================================================================

class TestScoreDataclass:
    """Test Score dataclass behavior."""

    def test_score_repr(self):
        """Score should have readable repr."""
        score = compute_score("0010", [0.0] * 20, 5)
        repr_str = repr(score)
        assert "total=" in repr_str
        assert "bits=" in repr_str

    def test_score_to_dict(self):
        """Score should serialize to dict."""
        score = compute_score("0010", [0.1, 0.2], 3, reasoning="test")
        d = score.to_dict()

        assert "compression_bits" in d
        assert "error_penalty" in d
        assert "probe_penalty" in d
        assert "total" in d
        assert "reasoning" in d

    def test_timeout_score(self):
        """Timeout score should have infinite total."""
        score = Score.timeout_score(reasoning="Execution timeout", n_probes=5)

        assert score.timed_out is True
        assert math.isinf(score.total)
        assert score.probe_penalty == 25  # 5 * 5
        assert score.reasoning == "Execution timeout"

    def test_timeout_score_repr(self):
        """Timeout score repr should indicate timeout."""
        score = Score.timeout_score()
        assert "TIMEOUT" in repr(score)


# =============================================================================
# Score Analysis Tests
# =============================================================================

class TestScoreAnalysis:
    """Test score analysis utilities."""

    def test_compare_scores(self):
        """Should compare two scores."""
        score1 = compute_score("0010", [0.0] * 20, 0)
        score2 = compute_score("0010" * 10, [0.1] * 20, 10)

        comparison = compare_scores(score1, score2)
        assert "Score 1" in comparison
        assert "Score 2" in comparison
        assert "better" in comparison

    def test_score_breakdown(self):
        """Should generate score breakdown."""
        score = compute_score("0010", [0.01] * 20, 4)
        breakdown = score_breakdown(score)

        assert "Compression" in breakdown
        assert "Error" in breakdown
        assert "Probes" in breakdown
        assert "TOTAL" in breakdown

    def test_score_breakdown_timeout(self):
        """Timeout score breakdown should indicate timeout."""
        score = Score.timeout_score()
        breakdown = score_breakdown(score)
        assert "TIMEOUT" in breakdown


# =============================================================================
# Integration Tests
# =============================================================================

class TestScoringIntegration:
    """Integration tests for scoring system."""

    def test_lower_score_is_better(self):
        """Verify lower scores represent better performance."""
        # Good: short program, low error, few probes
        good_score = compute_score("0010", [0.001] * 20, 2)

        # Bad: long program, high error, many probes
        bad_score = compute_score("0010" * 100, [0.5] * 20, 50)

        assert good_score.total < bad_score.total

    def test_perfect_vs_imperfect(self):
        """Perfect predictions should beat imperfect."""
        perfect = compute_score("0010", [0.0] * 20, 5)
        imperfect = compute_score("0010", [0.1] * 20, 5)

        assert perfect.error_penalty < imperfect.error_penalty
        assert perfect.total < imperfect.total
