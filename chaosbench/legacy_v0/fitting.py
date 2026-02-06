"""Fit model parameters to observations."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize_scalar, minimize

from .backtest import backtest_model


@dataclass
class FitResult:
    """Result of fitting a model to observations."""
    params: Dict[str, float]
    mae: float
    predicted_next: float


def fit_model(family: str, observations: np.ndarray) -> FitResult:
    """Fit model parameters to observations.

    Uses scipy.optimize to find parameters that minimize
    one-step prediction error.

    Args:
        family: Model family ("logistic", "tent", etc.)
        observations: Array of x_0, x_1, ..., x_49

    Returns:
        FitResult with estimated params, MAE, and predicted x_50
    """
    obs = observations.flatten()

    if family == "logistic":
        # Fit r in [3.5, 4.0] (chaotic regime)
        def loss(r):
            result = backtest_model("logistic", {"r": r}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(3.5, 4.0), method='bounded')
        best_r = opt.x
        best_result = backtest_model("logistic", {"r": best_r}, obs)
        return FitResult(
            params={"r": float(best_r)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "tent":
        # Fit mu in [1.0, 2.0]
        def loss(mu):
            result = backtest_model("tent", {"mu": mu}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(1.0, 2.0), method='bounded')
        best_mu = opt.x
        best_result = backtest_model("tent", {"mu": best_mu}, obs)
        return FitResult(
            params={"mu": float(best_mu)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "henon":
        # Fit a, b
        def loss(params):
            a, b = params
            result = backtest_model("henon", {"a": a, "b": b}, obs)
            return result.mae

        opt = minimize(loss, x0=[1.4, 0.3], bounds=[(1.0, 1.5), (0.1, 0.5)], method='L-BFGS-B')
        best_a, best_b = opt.x
        best_result = backtest_model("henon", {"a": best_a, "b": best_b}, obs)
        return FitResult(
            params={"a": float(best_a), "b": float(best_b)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "standard":
        # Fit K
        def loss(K):
            result = backtest_model("standard", {"K": K}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(0.5, 2.0), method='bounded')
        best_K = opt.x
        best_result = backtest_model("standard", {"K": best_K}, obs)
        return FitResult(
            params={"K": float(best_K)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "lorenz":
        # Use standard Lorenz params, just fit rho
        def loss(rho):
            result = backtest_model("lorenz", {"sigma": 10.0, "rho": rho, "beta": 8/3}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(20.0, 35.0), method='bounded')
        best_rho = opt.x
        best_result = backtest_model("lorenz", {"sigma": 10.0, "rho": best_rho, "beta": 8/3}, obs)
        return FitResult(
            params={"sigma": 10.0, "rho": float(best_rho), "beta": 8/3},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    else:
        raise ValueError(f"Unknown model family: {family}")
