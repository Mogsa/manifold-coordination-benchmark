"""
Function library for Telepathic Benchmark v2.

Defines test functions in 4 tiers for the compression benchmark:
- Tier 1: Polynomials (low complexity)
- Tier 2: Transcendentals (medium complexity)
- Tier 3: Compositions (high complexity)
- Tier 4: Incompressible (control - agent should fail)

Checkpoint 2.1: Function library (all 4 tiers)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class TestFunction:
    """A test function for the benchmark."""

    id: str
    name: str
    tier: int  # 1=Polynomial, 2=Transcendental, 3=Composition, 4=Incompressible
    formula: str
    func: Callable[[float], float]
    description: str

    def __call__(self, x: float) -> float:
        """Evaluate function at x."""
        return self.func(x)

    def to_dict(self) -> dict:
        """Serialize (without the callable)."""
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "formula": self.formula,
            "description": self.description,
        }


# =============================================================================
# TIER 1: Polynomials (Low Complexity)
# Expected: Short programs, easy compression
# =============================================================================

TIER_1_POLYNOMIALS: Dict[str, TestFunction] = {
    "P1": TestFunction(
        id="P1",
        name="identity",
        tier=1,
        formula="f(x) = x",
        func=lambda x: x,
        description="Identity function",
    ),
    "P2": TestFunction(
        id="P2",
        name="square",
        tier=1,
        formula="f(x) = x²",
        func=lambda x: x ** 2,
        description="Square function",
    ),
    "P3": TestFunction(
        id="P3",
        name="cube",
        tier=1,
        formula="f(x) = x³",
        func=lambda x: x ** 3,
        description="Cube function",
    ),
    "P4": TestFunction(
        id="P4",
        name="linear",
        tier=1,
        formula="f(x) = 2x + 1",
        func=lambda x: 2 * x + 1,
        description="Linear function",
    ),
    "P5": TestFunction(
        id="P5",
        name="quadratic",
        tier=1,
        formula="f(x) = (x+1)²",
        func=lambda x: (x + 1) ** 2,
        description="Quadratic: (x+1)²",
    ),
}


# =============================================================================
# TIER 2: Transcendentals (Medium Complexity)
# Expected: Medium programs, need Taylor series approximations
# =============================================================================

TIER_2_TRANSCENDENTALS: Dict[str, TestFunction] = {
    "T1": TestFunction(
        id="T1",
        name="sin",
        tier=2,
        formula="f(x) = sin(x)",
        func=lambda x: math.sin(x),
        description="Sine function",
    ),
    "T2": TestFunction(
        id="T2",
        name="cos",
        tier=2,
        formula="f(x) = cos(x)",
        func=lambda x: math.cos(x),
        description="Cosine function",
    ),
    "T3": TestFunction(
        id="T3",
        name="exp",
        tier=2,
        formula="f(x) = exp(x)",
        func=lambda x: math.exp(x),
        description="Exponential function",
    ),
    "T4": TestFunction(
        id="T4",
        name="ln1p",
        tier=2,
        formula="f(x) = ln(1+x)",
        func=lambda x: math.log(1 + x),
        description="Logarithm: ln(1+x)",
    ),
}


# =============================================================================
# TIER 3: Compositions (High Complexity)
# Expected: Longer programs, need to compose operations
# =============================================================================

TIER_3_COMPOSITIONS: Dict[str, TestFunction] = {
    "C1": TestFunction(
        id="C1",
        name="sin_of_square",
        tier=3,
        formula="f(x) = sin(x²)",
        func=lambda x: math.sin(x ** 2),
        description="Sine of square",
    ),
    "C2": TestFunction(
        id="C2",
        name="gaussian",
        tier=3,
        formula="f(x) = exp(-x²)",
        func=lambda x: math.exp(-(x ** 2)),
        description="Gaussian kernel",
    ),
    "C3": TestFunction(
        id="C3",
        name="pythagorean_identity",
        tier=3,
        formula="f(x) = sin²(x) + cos²(x) = 1",
        func=lambda x: math.sin(x) ** 2 + math.cos(x) ** 2,
        description="Pythagorean identity (always 1)",
    ),
}


# =============================================================================
# TIER 4: Incompressible (Control)
# Expected: Agent SHOULD FAIL - validates benchmark isn't trivially gameable
# =============================================================================

def _create_random_lookup(seed: int, n_points: int = 20) -> Callable[[float], float]:
    """
    Create a random lookup table function.

    Args:
        seed: Random seed for reproducibility.
        n_points: Number of points in the lookup.

    Returns:
        Function that interpolates random values.
    """
    rng = random.Random(seed)
    # Generate random y values for uniform x points in (0, 1]
    xs = [(i + 1) / n_points for i in range(n_points)]
    ys = [rng.uniform(0, 1) for _ in range(n_points)]

    def lookup(x: float) -> float:
        # Clamp to domain
        x = max(0.001, min(1.0, x))
        # Find nearest point (simple nearest-neighbor)
        idx = int(x * n_points) - 1
        idx = max(0, min(n_points - 1, idx))
        return ys[idx]

    return lookup


def _create_pseudo_random(seed: int) -> Callable[[float], float]:
    """
    Create a pseudo-random function using a deterministic hash.

    Args:
        seed: Seed for the PRNG.

    Returns:
        Function that returns pseudo-random values.
    """
    def prng(x: float) -> float:
        # Use a simple hash-based PRNG
        # This is deterministic for the same x and seed
        import hashlib
        data = f"{seed}:{x:.10f}".encode()
        h = hashlib.sha256(data).digest()
        # Convert first 8 bytes to float in [0, 1]
        value = int.from_bytes(h[:8], 'big')
        return value / (2 ** 64)

    return prng


TIER_4_INCOMPRESSIBLE: Dict[str, TestFunction] = {
    "R1": TestFunction(
        id="R1",
        name="random_lookup",
        tier=4,
        formula="f(x) = lookup[x] (random)",
        func=_create_random_lookup(seed=42),
        description="Random lookup table",
    ),
    "R2": TestFunction(
        id="R2",
        name="pseudo_random",
        tier=4,
        formula="f(x) = hash(x)",
        func=_create_pseudo_random(seed=42),
        description="Pseudo-random hash function",
    ),
}


# =============================================================================
# COMBINED FUNCTION REGISTRY
# =============================================================================

ALL_FUNCTIONS: Dict[str, TestFunction] = {
    **TIER_1_POLYNOMIALS,
    **TIER_2_TRANSCENDENTALS,
    **TIER_3_COMPOSITIONS,
    **TIER_4_INCOMPRESSIBLE,
}


def get_function(function_id: str) -> TestFunction:
    """
    Get a test function by ID.

    Args:
        function_id: e.g., "P1", "T2", "C1", "R1"

    Returns:
        TestFunction instance.

    Raises:
        KeyError: If function ID not found.
    """
    return ALL_FUNCTIONS[function_id]


def get_functions_by_tier(tier: int) -> List[TestFunction]:
    """
    Get all functions in a tier.

    Args:
        tier: 1, 2, 3, or 4

    Returns:
        List of TestFunction instances.
    """
    return [f for f in ALL_FUNCTIONS.values() if f.tier == tier]


def list_all_functions() -> List[TestFunction]:
    """Return all test functions."""
    return list(ALL_FUNCTIONS.values())


def list_function_ids() -> List[str]:
    """Return all function IDs."""
    return list(ALL_FUNCTIONS.keys())


# =============================================================================
# TIER METADATA
# =============================================================================

TIER_NAMES = {
    1: "Polynomials",
    2: "Transcendentals",
    3: "Compositions",
    4: "Incompressible",
}

TIER_DESCRIPTIONS = {
    1: "Low complexity - simple arithmetic operations",
    2: "Medium complexity - require Taylor series approximations",
    3: "High complexity - require composing operations",
    4: "Control - agent should fail (validates benchmark)",
}
