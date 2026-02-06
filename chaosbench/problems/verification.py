"""
Verification functions for ChaosBench v2 problem types.

Three question types, three verifiers:
- CLASSIFY: exact match on regime label
- IDENTIFY: exact match on family name
- PREDICT: k_eff scoring (largest k where ALL predictions are within epsilon)
"""

import numpy as np


def _normalize_label(answer: object) -> str:
    """Normalize a label-like answer for robust exact matching."""
    if isinstance(answer, str):
        return answer.strip().lower()
    return ""


def verify_classify(agent_answer: object, ground_truth: object) -> float:
    """Verify a CLASSIFY answer. Case-insensitive exact match."""
    return 1.0 if _normalize_label(agent_answer) == _normalize_label(ground_truth) else 0.0


def verify_identify(agent_answer: object, ground_truth: object) -> float:
    """Verify an IDENTIFY answer. Case-insensitive exact match."""
    return 1.0 if _normalize_label(agent_answer) == _normalize_label(ground_truth) else 0.0


def verify_predict(
    predictions: np.ndarray,
    true_future: np.ndarray,
    training_data: np.ndarray,
) -> dict:
    """Verify a PREDICT answer using k_eff scoring.

    Args:
        predictions: Agent's predicted future values.
        true_future: Ground-truth future trajectory.
        training_data: The observation data given to the agent (for attractor diameter).

    Returns:
        dict with keys: k_eff, K, raw_score, epsilon
    """
    K = len(true_future)
    predictions = np.asarray(predictions, dtype=float)
    true_future = np.asarray(true_future, dtype=float)
    training_data = np.asarray(training_data, dtype=float)

    # Pad short predictions with NaN
    if len(predictions) < K:
        padded = np.full(K, np.nan)
        padded[: len(predictions)] = predictions
        predictions = padded

    # Truncate if too long
    predictions = predictions[:K]

    # Epsilon = 0.1 * attractor diameter, floored at 0.01
    attractor_diameter = float(np.max(training_data) - np.min(training_data))
    epsilon = max(0.1 * attractor_diameter, 0.01)

    # k_eff = largest k where ALL |pred[i] - true[i]| < epsilon for i < k
    k_eff = 0
    for i in range(K):
        if np.isnan(predictions[i]):
            break
        if abs(predictions[i] - true_future[i]) >= epsilon:
            break
        k_eff = i + 1

    raw_score = k_eff / K if K > 0 else 0.0

    return {
        "k_eff": k_eff,
        "K": K,
        "raw_score": raw_score,
        "epsilon": epsilon,
    }
