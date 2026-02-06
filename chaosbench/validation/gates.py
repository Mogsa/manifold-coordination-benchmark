"""
Stage 1 hard gates — quality filters for generated problems.

Each gate returns (passed: bool, reason: str).
Gates are regime-aware: PE and ACF only apply to chaotic systems.
"""

from typing import Dict, List, Tuple

import numpy as np


def check_parameter_bounds(
    family: str, params: dict, valid_ranges: dict
) -> Tuple[bool, str]:
    """Check all parameters are within valid ranges."""
    for name, value in params.items():
        if name not in valid_ranges:
            return False, f"Unknown parameter '{name}' for family '{family}'"
        lo, hi = valid_ranges[name]
        if not lo <= value <= hi:
            return False, f"{name}={value} outside [{lo}, {hi}]"
    return True, "Parameters within bounds"


def check_trajectory_stability(trajectory: np.ndarray) -> Tuple[bool, str]:
    """Check trajectory has no NaN/Inf and stays bounded."""
    if not np.all(np.isfinite(trajectory)):
        n_bad = int(np.sum(~np.isfinite(trajectory)))
        return False, f"{n_bad} non-finite values in trajectory"
    max_abs = float(np.max(np.abs(trajectory)))
    if max_abs > 1e10:
        return False, f"Max |x| = {max_abs:.2e} exceeds 1e10"
    return True, f"Stable (max |x| = {max_abs:.4f})"


def check_attractor_bounds(
    trajectory: np.ndarray,
    domain: Tuple[float, float],
    margin: float = 0.1,
) -> Tuple[bool, str]:
    """Check trajectory stays within domain ± margin."""
    lo, hi = domain
    t_min = float(np.min(trajectory))
    t_max = float(np.max(trajectory))
    if t_min < lo - margin:
        return False, f"Min {t_min:.4f} below domain [{lo}, {hi}] - margin"
    if t_max > hi + margin:
        return False, f"Max {t_max:.4f} above domain [{lo}, {hi}] + margin"
    return True, f"Within bounds [{t_min:.4f}, {t_max:.4f}]"


def check_minimum_lyapunov(
    lambda_max: float, threshold: float = 0.01
) -> Tuple[bool, str]:
    """Check chaotic system has positive Lyapunov exponent (chaotic only)."""
    if lambda_max < threshold:
        return False, f"λ={lambda_max:.4f} below threshold {threshold}"
    return True, f"λ={lambda_max:.4f} ≥ {threshold}"


def check_periodicity(
    trajectory: np.ndarray, max_period: int = 1000, min_repeats: int = 3
) -> Tuple[bool, str]:
    """Check that trajectory is NOT periodic (chaotic only).

    A chaotic trajectory should not repeat exactly. This gate is strict:
    if the trajectory is too short to certify non-periodicity up to
    ``max_period``, it fails with an explicit reason.
    """
    if max_period < 1:
        raise ValueError(f"max_period must be >= 1, got {max_period}")
    if min_repeats < 2:
        raise ValueError(f"min_repeats must be >= 2, got {min_repeats}")

    n = len(trajectory)
    tail = trajectory[n // 2 :]

    max_testable_period = len(tail) // min_repeats
    if max_testable_period < max_period:
        return (
            False,
            (
                f"Insufficient tail length {len(tail)} for max_period={max_period} "
                f"(can only test up to {max_testable_period})"
            ),
        )

    for period in range(1, max_period + 1):
        reference = tail[:period]
        periodic = True
        for repeat in range(1, min_repeats):
            start = repeat * period
            end = (repeat + 1) * period
            segment = tail[start:end]
            if len(segment) < period or not np.allclose(reference, segment, atol=1e-10):
                periodic = False
                break
        if periodic:
            return False, f"Periodic with period {period}"
    return True, f"Not periodic (checked periods 1..{max_period})"


def check_permutation_entropy(
    trajectory: np.ndarray, order: int = 5, threshold: float = 0.5
) -> Tuple[bool, str]:
    """Check permutation entropy is high enough for chaotic trajectories.

    PE is normalized to [0, 1]. Chaotic systems should have PE > threshold.
    """
    n = len(trajectory)
    if n < order + 1:
        return False, f"Trajectory too short ({n}) for PE order {order}"

    # Count ordinal patterns
    from math import factorial

    max_patterns = factorial(order)
    pattern_counts: Dict[tuple, int] = {}
    count = 0

    for i in range(n - order):
        window = trajectory[i : i + order]
        pattern = tuple(np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        count += 1

    if count == 0:
        return False, "No patterns extracted"

    # Normalized Shannon entropy
    probs = np.array(list(pattern_counts.values()), dtype=float) / count
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(max_patterns)
    pe = entropy / max_entropy if max_entropy > 0 else 0.0

    if pe < threshold:
        return False, f"PE={pe:.3f} below threshold {threshold}"
    return True, f"PE={pe:.3f} ≥ {threshold}"


def check_autocorrelation(
    trajectory: np.ndarray, lag: int = 1, threshold: float = 0.95
) -> Tuple[bool, str]:
    """Check autocorrelation at given lag is not near-perfect (chaotic only).

    Catches periodic/fixed-point trajectories mislabelled as chaotic.
    Threshold is high (0.95) because chaotic maps like logistic r<4
    legitimately have significant ACF (e.g. r=3.891 has ACF(1)≈-0.53).
    """
    n = len(trajectory)
    if n < lag + 2:
        return False, f"Trajectory too short ({n}) for lag {lag}"

    x = trajectory - np.mean(trajectory)
    var = np.var(trajectory)
    if var < 1e-15:
        return False, "Zero variance trajectory"

    acf = np.mean(x[:-lag] * x[lag:]) / var
    if abs(acf) > threshold:
        return False, f"|ACF({lag})|={abs(acf):.4f} above threshold {threshold}"
    return True, f"|ACF({lag})|={abs(acf):.4f} ≤ {threshold}"


def validate_stage1(
    trajectory: np.ndarray,
    family: str,
    params: dict,
    valid_ranges: dict,
    domain: Tuple[float, float],
    regime: str,
    lambda_max: float,
    periodicity_max_period: int = 1000,
) -> Tuple[bool, List[Tuple[str, bool, str]]]:
    """Run all applicable Stage 1 gates.

    Returns (all_passed, [(gate_name, passed, reason), ...]).
    Gate applicability by regime:
    - All: parameter_bounds, trajectory_stability, attractor_bounds
    - Chaotic only: minimum_lyapunov, periodicity, autocorrelation
    - Chaotic only: permutation_entropy
    """
    results: List[Tuple[str, bool, str]] = []

    # Universal gates
    p, r = check_parameter_bounds(family, params, valid_ranges)
    results.append(("parameter_bounds", p, r))

    p, r = check_trajectory_stability(trajectory)
    results.append(("trajectory_stability", p, r))

    p, r = check_attractor_bounds(trajectory, domain)
    results.append(("attractor_bounds", p, r))

    # Chaotic-only gates
    if regime == "chaotic":
        p, r = check_minimum_lyapunov(lambda_max)
        results.append(("minimum_lyapunov", p, r))

        p, r = check_periodicity(trajectory, max_period=periodicity_max_period)
        results.append(("periodicity", p, r))

        p, r = check_autocorrelation(trajectory)
        results.append(("autocorrelation", p, r))

    # Chaotic-only structure gate
    if regime == "chaotic":
        p, r = check_permutation_entropy(trajectory)
        results.append(("permutation_entropy", p, r))

    all_passed = all(passed for _, passed, _ in results)
    return all_passed, results
