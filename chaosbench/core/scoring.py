"""NLL bin-based scoring for ChaosBench.

Implements proper probabilistic scoring where:
- Agent outputs prediction ± sigma (uncertainty)
- Prediction is a truncated Gaussian over [0, 1]
- State space is discretized into bins
- Score = probability mass in the true bin
- NLL = -log(probability mass)
"""
import numpy as np
from scipy import stats


def compute_nll(
    prediction: float,
    actual: float,
    sigma: float = 0.1,
    n_bins: int = 20,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Compute NLL score for a prediction.

    Uses a truncated Gaussian centered on prediction with std=sigma,
    discretized into bins. Score is the probability mass in the bin
    containing the actual value.

    Args:
        prediction: Agent's predicted value
        actual: True value
        sigma: Uncertainty (std dev of Gaussian)
        n_bins: Number of bins for discretization
        bounds: (low, high) bounds for state space

    Returns:
        (nll, score) where nll = -log(prob) and score = prob
    """
    low, high = bounds

    # Clamp actual to bounds
    actual_clamped = np.clip(actual, low, high)

    # Create bin edges
    edges = np.linspace(low, high, n_bins + 1)

    # Find which bin the actual value falls in
    true_bin = np.searchsorted(edges[1:-1], actual_clamped)  # 0 to n_bins-1
    bin_low = edges[true_bin]
    bin_high = edges[true_bin + 1]

    # Create truncated normal distribution
    # scipy's truncnorm uses standardized bounds: a = (low - loc) / scale
    a = (low - prediction) / sigma
    b = (high - prediction) / sigma
    dist = stats.truncnorm(a, b, loc=prediction, scale=sigma)

    # Compute probability mass in the true bin
    prob_mass = dist.cdf(bin_high) - dist.cdf(bin_low)

    # Avoid log(0) — minimum probability is uniform over bins
    min_prob = 1e-6
    prob_mass = max(prob_mass, min_prob)

    nll = -np.log(prob_mass)
    score = prob_mass

    return nll, score


def compute_score(
    prediction: float,
    actual: float,
    sigma: float = 0.1,
    n_bins: int = 20,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Compute exp(-NLL) score for Phi accumulation.

    This is a convenience wrapper that returns just the score.
    For [0,1] predictions, score = probability mass in true bin.

    Args:
        prediction: Agent's predicted value
        actual: True value
        sigma: Uncertainty (std dev)
        n_bins: Number of bins
        bounds: State space bounds

    Returns:
        Score in [0, 1] — probability mass in true bin
    """
    _, score = compute_nll(prediction, actual, sigma, n_bins, bounds)
    return score


def get_bin_edges(n_bins: int = 20, bounds: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Get bin edges for visualization.

    Args:
        n_bins: Number of bins
        bounds: (low, high) bounds

    Returns:
        Array of n_bins+1 edge values
    """
    return np.linspace(bounds[0], bounds[1], n_bins + 1)


def get_bin_index(value: float, n_bins: int = 20, bounds: tuple[float, float] = (0.0, 1.0)) -> int:
    """Get bin index for a value.

    Args:
        value: Value to bin
        n_bins: Number of bins
        bounds: (low, high) bounds

    Returns:
        Bin index (0 to n_bins-1)
    """
    low, high = bounds
    value_clamped = np.clip(value, low, high)
    edges = np.linspace(low, high, n_bins + 1)
    return int(np.searchsorted(edges[1:-1], value_clamped))
