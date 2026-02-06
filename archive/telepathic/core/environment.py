"""
Benchmark Environment for Telepathic Benchmark v2.

Provides the active probing interface for agents to observe noisy samples
and defines held-out test points for evaluation.

Checkpoint 2.2: Noisy sample generation
Checkpoint 2.3: Active probing interface
Checkpoint 2.4: Held-out test point generation
Checkpoint 2.5: Environment class
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional


# =============================================================================
# Constants (from spec Section 5.4)
# =============================================================================

DEFAULT_NOISE_STD = 0.1  # Gaussian noise σ
DEFAULT_NUM_TEST_POINTS = 20  # Held-out test points
DOMAIN_MIN = 0.0  # Exclusive (0, 1]
DOMAIN_MAX = 1.0  # Inclusive


# =============================================================================
# Noisy Sampling (Checkpoint 2.2)
# =============================================================================

def sample_with_noise(
    f: Callable[[float], float],
    x: float,
    sigma: float = DEFAULT_NOISE_STD,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Return f(x) + Gaussian noise N(0, σ²).

    Args:
        f: The function to sample.
        x: The input point.
        sigma: Standard deviation of Gaussian noise.
        rng: Optional random number generator for reproducibility.

    Returns:
        Noisy sample y = f(x) + ε where ε ~ N(0, σ²)
    """
    if rng is None:
        rng = np.random.default_rng()

    y_true = f(x)
    noise = rng.normal(0, sigma)
    return y_true + noise


def generate_test_points(
    n_points: int = DEFAULT_NUM_TEST_POINTS,
    seed: Optional[int] = None
) -> List[float]:
    """
    Generate uniform test points in (0, 1].

    Points are deterministic for a given seed to ensure reproducibility.

    Args:
        n_points: Number of test points.
        seed: Random seed for reproducibility.

    Returns:
        List of x values in (0, 1].
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate uniform points in (0, 1]
    # Using linspace-style uniform distribution
    points = []
    for i in range(n_points):
        # Evenly spaced from (0, 1] with small jitter for variation
        base = (i + 1) / (n_points + 1)
        # Small jitter within the spacing
        jitter = rng.uniform(-0.5 / (n_points + 1), 0.5 / (n_points + 1))
        x = max(0.001, min(1.0, base + jitter))
        points.append(x)

    return sorted(points)


def generate_fixed_test_points(n_points: int = DEFAULT_NUM_TEST_POINTS) -> List[float]:
    """
    Generate fixed uniform test points in (0, 1].

    These are deterministic with no randomness for consistent evaluation.

    Args:
        n_points: Number of test points.

    Returns:
        List of x values evenly spaced in (0, 1].
    """
    # Evenly spaced points: 1/21, 2/21, ..., 20/21 for n_points=20
    return [(i + 1) / (n_points + 1) for i in range(n_points)]


# =============================================================================
# Probe Record
# =============================================================================

@dataclass
class Probe:
    """Record of a single probe."""
    x: float
    y_noisy: float
    y_true: Optional[float] = None  # Only set for analysis/debugging


# =============================================================================
# Environment Class (Checkpoint 2.5)
# =============================================================================

@dataclass
class Environment:
    """
    Active probing environment for the compression benchmark.

    The agent interacts with this environment to:
    1. Probe the function at chosen x values (receives noisy samples)
    2. Get held-out test points for evaluation (no noise)
    """

    function: Callable[[float], float]
    function_id: Optional[str] = None
    noise_std: float = DEFAULT_NOISE_STD
    test_seed: Optional[int] = None  # Seed for test point generation
    probe_seed: Optional[int] = None  # Seed for probe noise

    # Internal state
    probes: List[Probe] = field(default_factory=list)
    _probe_rng: Optional[np.random.Generator] = field(default=None, repr=False)
    _test_points: Optional[List[float]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize RNG and test points."""
        # Initialize probe RNG
        if self.probe_seed is not None:
            self._probe_rng = np.random.default_rng(self.probe_seed)
        else:
            self._probe_rng = np.random.default_rng()

        # Initialize test points (fixed per function)
        if self.test_seed is not None:
            self._test_points = generate_test_points(
                DEFAULT_NUM_TEST_POINTS, seed=self.test_seed
            )
        else:
            # Use deterministic fixed points for consistency
            self._test_points = generate_fixed_test_points(DEFAULT_NUM_TEST_POINTS)

    def probe(self, x: float) -> float:
        """
        Probe the function at x, returning a noisy sample.

        Records the probe for penalty calculation.

        Args:
            x: Input value in (0, 1].

        Returns:
            Noisy sample y = f(x) + ε where ε ~ N(0, σ²)
        """
        # Validate domain
        if x <= DOMAIN_MIN or x > DOMAIN_MAX:
            raise ValueError(f"x must be in ({DOMAIN_MIN}, {DOMAIN_MAX}], got {x}")

        # Get noisy sample
        y_true = self.function(x)
        noise = self._probe_rng.normal(0, self.noise_std)
        y_noisy = y_true + noise

        # Record probe
        probe = Probe(x=x, y_noisy=y_noisy, y_true=y_true)
        self.probes.append(probe)

        return y_noisy

    def get_test_points(self) -> List[float]:
        """
        Return the 20 held-out test points.

        These are fixed per function and have NO noise.

        Returns:
            List of x values in (0, 1].
        """
        return self._test_points.copy()

    def evaluate_at_test_points(self) -> List[Tuple[float, float]]:
        """
        Get ground truth (x, y) pairs at test points.

        No noise is applied - these are the true function values.

        Returns:
            List of (x, y_true) tuples.
        """
        return [(x, self.function(x)) for x in self._test_points]

    def evaluate(self, predictions: List[float]) -> List[float]:
        """
        Compute errors between predictions and ground truth.

        Args:
            predictions: Agent's predictions for each test point.

        Returns:
            List of absolute errors |predicted - actual|.

        Raises:
            ValueError: If number of predictions doesn't match test points.
        """
        if len(predictions) != len(self._test_points):
            raise ValueError(
                f"Expected {len(self._test_points)} predictions, got {len(predictions)}"
            )

        errors = []
        for x, y_pred in zip(self._test_points, predictions):
            y_true = self.function(x)
            errors.append(abs(y_pred - y_true))

        return errors

    def get_probe_count(self) -> int:
        """Return the number of probes made."""
        return len(self.probes)

    def get_probe_history(self) -> List[Tuple[float, float]]:
        """
        Return history of probes as (x, y_noisy) tuples.

        Useful for analysis and debugging.
        """
        return [(p.x, p.y_noisy) for p in self.probes]

    def reset(self) -> None:
        """
        Reset the environment for a new trial.

        Clears probe history but keeps the same test points.
        """
        self.probes = []
        # Reset probe RNG to ensure different noise
        if self.probe_seed is not None:
            self._probe_rng = np.random.default_rng(self.probe_seed)
        else:
            self._probe_rng = np.random.default_rng()


# =============================================================================
# Factory Functions
# =============================================================================

def create_environment(
    function: Callable[[float], float],
    function_id: Optional[str] = None,
    noise_std: float = DEFAULT_NOISE_STD,
    test_seed: Optional[int] = None,
    probe_seed: Optional[int] = None
) -> Environment:
    """
    Create a new environment for a function.

    Args:
        function: The function to test.
        function_id: Optional ID for logging.
        noise_std: Noise standard deviation (default 0.1).
        test_seed: Seed for test point generation.
        probe_seed: Seed for probe noise.

    Returns:
        Environment instance.
    """
    return Environment(
        function=function,
        function_id=function_id,
        noise_std=noise_std,
        test_seed=test_seed,
        probe_seed=probe_seed,
    )


def create_environment_from_function_id(
    function_id: str,
    noise_std: float = DEFAULT_NOISE_STD,
    test_seed: Optional[int] = None,
    probe_seed: Optional[int] = None
) -> Environment:
    """
    Create environment from a registered function ID.

    Args:
        function_id: Function ID like "P1", "T2", etc.
        noise_std: Noise standard deviation.
        test_seed: Seed for test point generation.
        probe_seed: Seed for probe noise.

    Returns:
        Environment instance.
    """
    from .functions_v2 import get_function

    test_func = get_function(function_id)
    return Environment(
        function=test_func.func,
        function_id=function_id,
        noise_std=noise_std,
        test_seed=test_seed,
        probe_seed=probe_seed,
    )
