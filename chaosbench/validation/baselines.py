"""
Stage 2 baseline battery — sanity-check baselines for problem validation.

PREDICT baselines: persistence, mean-reversion, AR(5)
CLASSIFY baseline: moment-based heuristic
IDENTIFY baseline: return-map heuristic
"""

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PREDICT baselines
# ---------------------------------------------------------------------------

def baseline_persistence(data: np.ndarray, K: int) -> np.ndarray:
    """Repeat last observed value K times."""
    return np.full(K, data[-1])


def baseline_mean_reversion(data: np.ndarray, K: int) -> np.ndarray:
    """Repeat mean of observed data K times."""
    return np.full(K, np.mean(data))


def baseline_ar5(data: np.ndarray, K: int) -> np.ndarray:
    """Fit AR(5) model and extrapolate recursively.

    Falls back to persistence if data is too short or fit fails.
    """
    order = min(5, len(data) - 1)
    if order < 1:
        return baseline_persistence(data, K)

    n = len(data)
    # Build regression matrix: X[i] = [data[i-1], ..., data[i-order]]
    X = np.column_stack([data[order - j - 1 : n - j - 1] for j in range(order)])
    y = data[order:]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return baseline_persistence(data, K)

    # Recursive extrapolation
    buffer = list(data[-order:])
    predictions = np.empty(K)
    for i in range(K):
        x_in = np.array(buffer[-order:][::-1])  # Most recent first
        pred = float(x_in @ coeffs)
        predictions[i] = pred
        buffer.append(pred)

    return predictions


# ---------------------------------------------------------------------------
# CLASSIFY baseline
# ---------------------------------------------------------------------------

def baseline_classify_moments(data: np.ndarray) -> str:
    """Heuristic regime classifier based on trajectory statistics.

    - Very low variance (overall or tail) → fixed_point
    - Strong ACF peak → periodic
    - Otherwise → chaotic
    """
    var = np.var(data)
    if var < 1e-6:
        return "fixed_point"

    # Check if tail has converged (damped systems)
    n = len(data)
    if n > 20:
        tail_var = np.var(data[-n // 4:])
        if tail_var < 1e-6:
            return "fixed_point"

    # Check for periodicity via autocorrelation peaks
    x = data - np.mean(data)
    n = len(x)
    if n > 10:
        # Normalized autocorrelation at multiple lags
        acf_vals = []
        denom = np.sum(x ** 2)
        if denom > 1e-15:
            for lag in range(1, min(n // 3, 100)):
                acf = np.sum(x[:-lag] * x[lag:]) / denom
                acf_vals.append(acf)
            # If any ACF peak > 0.8, likely periodic
            if acf_vals and max(acf_vals) > 0.8:
                return "periodic"

    return "chaotic"


# ---------------------------------------------------------------------------
# IDENTIFY baseline
# ---------------------------------------------------------------------------

def baseline_identify_return_map(
    data: np.ndarray, families: Optional[List[str]] = None
) -> str:
    """Heuristic family identifier based on return-map shape.

    Examines the (x_n, x_{n+1}) return map:
    - Converging to point → damped_linear
    - Linear return map with slope ~1 → rotation
    - Piecewise-linear (V-shape) → tent
    - Curved (parabolic) → logistic
    """
    if families is None:
        families = ["logistic", "tent", "damped_linear", "rotation"]

    n = len(data)
    if n < 10:
        return families[0]

    x = data[:-1]
    y = data[1:]

    # Check for convergence (damped_linear)
    tail_var = np.var(data[-n // 4 :])
    if tail_var < 1e-6:
        return "damped_linear" if "damped_linear" in families else families[0]

    # Fit linear model to return map
    try:
        coeffs = np.polyfit(x, y, 1)
        linear_residual = np.mean((y - np.polyval(coeffs, x)) ** 2)
    except (np.linalg.LinAlgError, ValueError):
        return families[0]

    # Fit quadratic
    try:
        coeffs2 = np.polyfit(x, y, 2)
        quad_residual = np.mean((y - np.polyval(coeffs2, x)) ** 2)
    except (np.linalg.LinAlgError, ValueError):
        quad_residual = linear_residual

    # If linear fit is very good and slope ≈ 1 → rotation
    if linear_residual < 1e-4 and abs(coeffs[0] - 1.0) < 0.1:
        return "rotation" if "rotation" in families else families[0]

    # If quadratic is much better than linear → logistic (parabolic return map)
    if linear_residual > 1e-8 and quad_residual < 0.5 * linear_residual:
        return "logistic" if "logistic" in families else families[0]

    # Check for V-shape (tent): split at midpoint and fit two lines
    mid = np.median(x)
    left = x < mid
    right = ~left
    if np.sum(left) > 3 and np.sum(right) > 3:
        try:
            c_l = np.polyfit(x[left], y[left], 1)
            c_r = np.polyfit(x[right], y[right], 1)
            # Tent: positive slope on left, negative on right
            if c_l[0] > 0.5 and c_r[0] < -0.5:
                return "tent" if "tent" in families else families[0]
        except (np.linalg.LinAlgError, ValueError):
            pass

    return "logistic" if "logistic" in families else families[0]


# ---------------------------------------------------------------------------
# Stage 2 validation
# ---------------------------------------------------------------------------

def validate_stage2_predict(
    results: dict, tau_lambda: float
) -> Dict[str, object]:
    """Validate PREDICT baseline results.

    Checks:
    - k_eff > 0 (baseline captures some structure)
    - k_eff < 3 * tau_lambda (not trivially predictable for chaotic systems)
    """
    k_eff = results.get("k_eff", 0)

    structure = k_eff > 0
    if tau_lambda < float("inf"):
        trivial = k_eff >= 3 * tau_lambda
    else:
        trivial = False  # Non-chaotic systems can be trivially predictable

    return {
        "has_structure": structure,
        "is_trivial": trivial,
        "k_eff": k_eff,
        "tau_lambda": tau_lambda,
        "passed": structure and not trivial,
    }
