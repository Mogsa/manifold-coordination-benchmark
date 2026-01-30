"""Model factory for creating chaotic systems from family name + params."""
from typing import Dict, Any, List

from .Chaosbench_v3 import (
    ChaoticSystem,
    LogisticMap,
    TentMap,
    HenonMap,
    StandardMap,
    LorenzDisc,
)


# Maps family name -> list of parameter names
MODEL_PARAMS: Dict[str, List[str]] = {
    "logistic": ["r"],
    "tent": ["mu"],
    "henon": ["a", "b"],
    "standard": ["K"],
    "lorenz": ["sigma", "rho", "beta"],
}


def create_model(family: str, params: Dict[str, Any]) -> ChaoticSystem:
    """Create a chaotic system from family name and parameters.

    Args:
        family: One of "logistic", "tent", "henon", "standard", "lorenz"
        params: Dict of parameter values (e.g., {"r": 3.9} for logistic)

    Returns:
        ChaoticSystem instance

    Raises:
        ValueError: If family unknown or params invalid
    """
    if family == "logistic":
        return LogisticMap(r=params.get("r", 4.0))
    elif family == "tent":
        return TentMap(mu=params.get("mu", 2.0))
    elif family == "henon":
        return HenonMap(a=params.get("a", 1.4), b=params.get("b", 0.3))
    elif family == "standard":
        return StandardMap(K=params.get("K", 1.0))
    elif family == "lorenz":
        return LorenzDisc(
            sigma=params.get("sigma", 10.0),
            rho=params.get("rho", 28.0),
            beta=params.get("beta", 8/3),
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
