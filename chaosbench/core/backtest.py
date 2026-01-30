"""Backtest models against observations."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .models import create_model


@dataclass
class BacktestResult:
    """Result of backtesting a model against observations."""
    mae: float  # Mean absolute error on one-step predictions
    predicted_next: float  # Model's prediction for x_50


def backtest_model(
    family: str,
    params: Dict[str, Any],
    observations: np.ndarray,
) -> BacktestResult:
    """Test a model against observations.

    Computes one-step prediction error: for each x_i, predict x_{i+1}
    using the model, compare to actual x_{i+1}.

    Args:
        family: Model family ("logistic", "tent", etc.)
        params: Model parameters (e.g., {"r": 3.9})
        observations: Array of x_0, x_1, ..., x_49

    Returns:
        BacktestResult with MAE and predicted x_50
    """
    model = create_model(family, params)
    obs = observations.flatten()

    # One-step predictions: predict x_{i+1} from x_i
    errors = []
    for i in range(len(obs) - 1):
        x_i = obs[i:i+1] if model.dim == 1 else obs[i]
        predicted = model.step(x_i)
        # step() returns ndarray, extract scalar
        predicted = float(predicted.flat[0])
        actual = float(obs[i + 1])
        errors.append(abs(predicted - actual))

    mae = float(np.mean(errors))

    # Predict x_50
    x_last = obs[-1:] if model.dim == 1 else obs[-model.dim:]
    next_pred = model.step(x_last)
    predicted_next = float(next_pred.flat[0])

    return BacktestResult(mae=mae, predicted_next=predicted_next)
