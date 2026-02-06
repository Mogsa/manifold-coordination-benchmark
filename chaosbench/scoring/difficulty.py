"""
Composite difficulty scoring for ChaosBench v2.

Difficulty increases with chaos (h_ks), grammar depth, and noise level.
Minimum difficulty is 1.0 (trivial non-chaotic, depth-0, no noise).
"""


def composite_difficulty(
    h_ks: float, grammar_depth: int = 0, noise_std: float = 0.01
) -> float:
    """Compute composite difficulty score.

    Formula: (1 + h_ks) × (1 + grammar_depth) × (1 + 10σ)

    Returns at least 1.0 for all inputs.
    """
    return (1.0 + h_ks) * (1.0 + grammar_depth) * (1.0 + 10.0 * noise_std)


def weighted_score(raw_score: float, difficulty: float) -> float:
    """Weight a raw score by difficulty.

    Higher difficulty → higher weighted score for same raw performance.
    """
    return raw_score * difficulty
