"""
Evaluation engine for the Telepathic benchmark.

Scores predictions and computes the Phi (Φ) metric.
"""

from dataclasses import dataclass
from typing import List, Optional
import math


# Tolerance settings
RELATIVE_TOLERANCE = 0.01  # 1% relative error allowed
ABSOLUTE_TOLERANCE = 0.01  # For near-zero values


@dataclass
class TrialScore:
    """Score for a single trial."""
    success: bool
    predicted: float
    expected: float
    absolute_error: float
    relative_error: Optional[float]  # None if expected ≈ 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "predicted": self.predicted,
            "expected": self.expected,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


@dataclass
class PhaseMetrics:
    """Aggregated metrics for a phase."""
    phase_name: str
    num_trials: int
    num_successes: int
    accuracy: float
    scores: List[TrialScore]

    def to_dict(self) -> dict:
        return {
            "phase_name": self.phase_name,
            "num_trials": self.num_trials,
            "num_successes": self.num_successes,
            "accuracy": self.accuracy,
        }


class Evaluator:
    """Evaluates Doer predictions."""

    def __init__(
        self,
        relative_tolerance: float = RELATIVE_TOLERANCE,
        absolute_tolerance: float = ABSOLUTE_TOLERANCE
    ):
        """
        Initialize evaluator.

        Args:
            relative_tolerance: Max relative error for success (default 1%)
            absolute_tolerance: Max absolute error for near-zero values
        """
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def score(self, predicted: float, expected: float) -> TrialScore:
        """
        Score a single prediction.

        Uses relative tolerance for most values, absolute tolerance
        when expected value is near zero.

        Args:
            predicted: Doer's prediction
            expected: True function output

        Returns:
            TrialScore with success status and error metrics
        """
        # Handle NaN or Inf
        if math.isnan(predicted) or math.isinf(predicted):
            return TrialScore(
                success=False,
                predicted=predicted,
                expected=expected,
                absolute_error=float('inf'),
                relative_error=None
            )

        absolute_error = abs(predicted - expected)

        # Near-zero: use absolute tolerance
        if abs(expected) < self.absolute_tolerance:
            success = absolute_error < self.absolute_tolerance
            return TrialScore(
                success=success,
                predicted=predicted,
                expected=expected,
                absolute_error=absolute_error,
                relative_error=None  # Not meaningful for near-zero
            )

        # Normal case: use relative tolerance
        relative_error = absolute_error / abs(expected)
        success = relative_error < self.relative_tolerance

        return TrialScore(
            success=success,
            predicted=predicted,
            expected=expected,
            absolute_error=absolute_error,
            relative_error=relative_error
        )

    def score_binary(self, predicted: float, expected: float) -> int:
        """
        Score as binary (1 or 0).

        Args:
            predicted: Doer's prediction
            expected: True function output

        Returns:
            1 if success, 0 otherwise
        """
        return 1 if self.score(predicted, expected).success else 0

    def compute_phase_metrics(
        self,
        scores: List[TrialScore],
        phase_name: str
    ) -> PhaseMetrics:
        """
        Aggregate scores into phase metrics.

        Args:
            scores: List of trial scores
            phase_name: Name of the phase

        Returns:
            PhaseMetrics with accuracy
        """
        num_trials = len(scores)
        num_successes = sum(1 for s in scores if s.success)
        accuracy = num_successes / num_trials if num_trials > 0 else 0.0

        return PhaseMetrics(
            phase_name=phase_name,
            num_trials=num_trials,
            num_successes=num_successes,
            accuracy=accuracy,
            scores=scores
        )

    def compute_phi(
        self,
        primitive_accuracy: float,
        novel_accuracy: float
    ) -> float:
        """
        Compute the Fluidity Ratio Φ.

        Φ = accuracy_on_novel_compositions / accuracy_on_primitives

        Interpretation:
            Φ ≈ 1.0: Agents learned compositional grammar
            Φ ≈ 0.0: Agents memorized lookup table

        Args:
            primitive_accuracy: Accuracy on Phase 1 (primitives)
            novel_accuracy: Accuracy on Phase 2 (novel compositions)

        Returns:
            Phi value (0 to ~1, can exceed 1 theoretically)
        """
        if primitive_accuracy == 0:
            return 0.0
        return novel_accuracy / primitive_accuracy


# Module-level convenience using default evaluator
_default_evaluator = Evaluator()


def score_prediction(predicted: float, expected: float) -> TrialScore:
    """Score using default evaluator."""
    return _default_evaluator.score(predicted, expected)


def score_binary(predicted: float, expected: float) -> int:
    """Binary score using default evaluator."""
    return _default_evaluator.score_binary(predicted, expected)


def compute_phi(primitive_accuracy: float, novel_accuracy: float) -> float:
    """Compute Phi using default evaluator."""
    return _default_evaluator.compute_phi(primitive_accuracy, novel_accuracy)
