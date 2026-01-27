"""
Scoring System for Telepathic Benchmark v2.

Implements the MDL-based scoring formula:
    SCORE = |P|_BLC + ERROR_PENALTY + PROBE_PENALTY

Lower scores are better.

Checkpoint 3.1: BLC bit counting
Checkpoint 3.2: Error penalty computation
Checkpoint 3.3: Probe penalty computation
Checkpoint 3.4: Total score aggregation
Checkpoint 3.5: Score dataclass (includes reasoning for research analysis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# Constants (from spec Section 5.4)
# =============================================================================

ERROR_SCALE_K = 100  # Error scaling factor in penalty
PROBE_COST = 5  # Bits per probe
SCALE = 1000  # Fixed-point scaling factor


# =============================================================================
# Penalty Functions
# =============================================================================

def count_bits(blc_program: str) -> int:
    """
    Count the number of bits in a BLC program string.

    Checkpoint 3.1: BLC bit counting

    Args:
        blc_program: Binary string (e.g., "0010" for identity)

    Returns:
        Length of the bit string.

    Raises:
        ValueError: If string contains non-binary characters.
    """
    # Validate it's a binary string
    if not all(c in '01' for c in blc_program):
        raise ValueError(f"BLC program must be binary string, got: {blc_program[:50]}...")

    return len(blc_program)


def error_penalty(errors: List[float], k: float = ERROR_SCALE_K) -> float:
    """
    Compute MDL-style error penalty.

    Checkpoint 3.2: Error penalty computation

    Formula: Σ log₂(1 + |error_i| × k)

    The logarithmic scaling follows information theory:
    - Small errors (0.01): ~1 bit
    - Medium errors (0.1): ~3.5 bits
    - Large errors (1.0): ~6.7 bits

    Args:
        errors: List of absolute errors |predicted - actual|.
        k: Scaling factor (default 100).

    Returns:
        Total error penalty in bits.
    """
    penalty = 0.0
    for e in errors:
        # log₂(1 + |e| × k)
        penalty += math.log2(1 + abs(e) * k)
    return penalty


def probe_penalty(n_probes: int, cost_per_probe: int = PROBE_COST) -> int:
    """
    Compute probe penalty.

    Checkpoint 3.3: Probe penalty computation

    Formula: n_probes × cost_per_probe

    Args:
        n_probes: Number of probes made.
        cost_per_probe: Bits per probe (default 5).

    Returns:
        Probe penalty in bits.
    """
    return n_probes * cost_per_probe


# =============================================================================
# Score Dataclass
# =============================================================================

@dataclass
class Score:
    """
    Complete score for a synthesis result.

    Checkpoint 3.5: Score dataclass

    SCORE = compression_bits + error_penalty + probe_penalty
    LOWER IS BETTER
    """

    # Score components
    compression_bits: int
    error_penalty: float
    probe_penalty: int

    # Total score (computed)
    total: float

    # Analysis data (not part of score)
    errors: List[float]  # Individual test point errors
    reasoning: str  # Agent's reasoning (preserved for research)

    # Optional metadata
    program: Optional[str] = None  # Original lambda program
    blc_program: Optional[str] = None  # Compiled BLC
    n_probes: Optional[int] = None  # Number of probes made
    n_test_points: Optional[int] = None  # Number of test points
    timed_out: bool = False  # Whether execution timed out

    def __repr__(self) -> str:
        if self.timed_out:
            return "Score(TIMEOUT)"
        return (
            f"Score(total={self.total:.2f}, "
            f"bits={self.compression_bits}, "
            f"error={self.error_penalty:.2f}, "
            f"probes={self.probe_penalty})"
        )

    def to_dict(self) -> dict:
        """Serialize for logging/export."""
        return {
            "compression_bits": self.compression_bits,
            "error_penalty": self.error_penalty,
            "probe_penalty": self.probe_penalty,
            "total": self.total,
            "errors": self.errors,
            "reasoning": self.reasoning,
            "n_probes": self.n_probes,
            "n_test_points": self.n_test_points,
            "timed_out": self.timed_out,
        }

    @classmethod
    def timeout_score(cls, reasoning: str = "", n_probes: int = 0) -> Score:
        """
        Create a timeout score (infinite).

        Args:
            reasoning: Agent's reasoning (if available).
            n_probes: Number of probes made before timeout.

        Returns:
            Score with total = infinity.
        """
        return cls(
            compression_bits=0,
            error_penalty=float('inf'),
            probe_penalty=n_probes * PROBE_COST,
            total=float('inf'),
            errors=[],
            reasoning=reasoning,
            n_probes=n_probes,
            timed_out=True,
        )


# =============================================================================
# Score Computation
# =============================================================================

def compute_score(
    blc_program: str,
    errors: List[float],
    n_probes: int,
    reasoning: str = "",
    program: Optional[str] = None
) -> Score:
    """
    Compute total score for a synthesis result.

    Checkpoint 3.4: Total score aggregation

    SCORE = |P|_BLC + error_penalty + probe_penalty
    LOWER IS BETTER

    Args:
        blc_program: Compiled BLC bit string.
        errors: List of absolute errors on test points.
        n_probes: Number of probes made.
        reasoning: Agent's reasoning (for research analysis).
        program: Original lambda program (for logging).

    Returns:
        Score dataclass with all components.
    """
    bits = count_bits(blc_program)
    err_penalty = error_penalty(errors)
    prb_penalty = probe_penalty(n_probes)

    total = bits + err_penalty + prb_penalty

    return Score(
        compression_bits=bits,
        error_penalty=err_penalty,
        probe_penalty=prb_penalty,
        total=total,
        errors=errors,
        reasoning=reasoning,
        program=program,
        blc_program=blc_program,
        n_probes=n_probes,
        n_test_points=len(errors),
        timed_out=False,
    )


# =============================================================================
# Score Analysis Utilities
# =============================================================================

def compare_scores(score1: Score, score2: Score) -> str:
    """
    Generate a comparison of two scores.

    Args:
        score1: First score.
        score2: Second score.

    Returns:
        Human-readable comparison string.
    """
    diff = score1.total - score2.total

    lines = []
    lines.append(f"Score 1: {score1.total:.2f}")
    lines.append(f"Score 2: {score2.total:.2f}")
    lines.append(f"Difference: {diff:+.2f}")

    if diff < 0:
        lines.append("→ Score 1 is better (lower)")
    elif diff > 0:
        lines.append("→ Score 2 is better (lower)")
    else:
        lines.append("→ Scores are equal")

    return "\n".join(lines)


def score_breakdown(score: Score) -> str:
    """
    Generate a detailed breakdown of a score.

    Args:
        score: The score to analyze.

    Returns:
        Human-readable breakdown string.
    """
    if score.timed_out:
        return "TIMEOUT - Execution exceeded limit"

    lines = []
    lines.append("Score Breakdown:")
    lines.append(f"  Compression: {score.compression_bits} bits ({score.compression_bits / score.total * 100:.1f}%)")
    lines.append(f"  Error:       {score.error_penalty:.2f} bits ({score.error_penalty / score.total * 100:.1f}%)")
    lines.append(f"  Probes:      {score.probe_penalty} bits ({score.probe_penalty / score.total * 100:.1f}%)")
    lines.append(f"  TOTAL:       {score.total:.2f} bits")

    if score.errors:
        avg_error = sum(score.errors) / len(score.errors)
        max_error = max(score.errors)
        lines.append(f"\n  Avg error:   {avg_error:.4f}")
        lines.append(f"  Max error:   {max_error:.4f}")

    return "\n".join(lines)
