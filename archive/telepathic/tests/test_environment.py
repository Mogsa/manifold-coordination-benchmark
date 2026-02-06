"""
Tests for Telepathic Benchmark v2 Environment.

Tests probing, noisy sampling, test points, and environment class.
"""

import math
import numpy as np
import pytest

from telepathic.core.environment import (
    # Constants
    DEFAULT_NOISE_STD,
    DEFAULT_NUM_TEST_POINTS,
    DOMAIN_MIN,
    DOMAIN_MAX,
    # Functions
    sample_with_noise,
    generate_test_points,
    generate_fixed_test_points,
    # Classes
    Probe,
    Environment,
    # Factory functions
    create_environment,
    create_environment_from_function_id,
)
from telepathic.core.functions_v2 import get_function


# =============================================================================
# Noisy Sampling Tests (Checkpoint 2.2)
# =============================================================================

class TestNoisySampling:
    """Test noisy sample generation."""

    def test_sample_with_noise_returns_float(self):
        """Should return a float value."""
        f = lambda x: x
        y = sample_with_noise(f, 0.5)
        assert isinstance(y, float)

    def test_sample_with_noise_adds_noise(self):
        """Samples should vary due to noise."""
        f = lambda x: x ** 2
        x = 0.5
        true_value = f(x)

        # Collect multiple samples
        samples = [sample_with_noise(f, x) for _ in range(100)]

        # Should have variation
        assert min(samples) != max(samples)

        # Mean should be close to true value
        mean = sum(samples) / len(samples)
        assert abs(mean - true_value) < 0.1

    def test_sample_with_noise_reproducible_with_rng(self):
        """Should be reproducible when using the same RNG seed."""
        f = lambda x: x
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        y1 = sample_with_noise(f, 0.5, rng=rng1)
        y2 = sample_with_noise(f, 0.5, rng=rng2)

        assert y1 == y2


# =============================================================================
# Test Point Generation Tests (Checkpoint 2.4)
# =============================================================================

class TestTestPointGeneration:
    """Test held-out test point generation."""

    def test_generate_test_points_count(self):
        """Should generate correct number of points."""
        points = generate_test_points(n_points=20)
        assert len(points) == 20

    def test_generate_test_points_in_domain(self):
        """All points should be in (0, 1]."""
        points = generate_test_points(n_points=20)
        for p in points:
            assert DOMAIN_MIN < p <= DOMAIN_MAX

    def test_generate_fixed_test_points_deterministic(self):
        """Fixed test points should be deterministic."""
        points1 = generate_fixed_test_points(20)
        points2 = generate_fixed_test_points(20)
        assert points1 == points2


# =============================================================================
# Probe Dataclass Tests
# =============================================================================

class TestProbeDataclass:
    """Test the Probe dataclass."""

    def test_probe_fields(self):
        """Probe should have required fields."""
        probe = Probe(x=0.5, y_noisy=0.26, y_true=0.25)
        assert probe.x == 0.5
        assert probe.y_noisy == 0.26
        assert probe.y_true == 0.25


# =============================================================================
# Environment Class Tests (Checkpoint 2.5)
# =============================================================================

class TestEnvironment:
    """Test the Environment class."""

    @pytest.fixture
    def square_env(self):
        """Create an environment with square function."""
        return create_environment(lambda x: x ** 2, function_id="test_square")

    def test_environment_creation(self, square_env):
        """Should create environment correctly."""
        assert square_env.function_id == "test_square"
        assert square_env.noise_std == DEFAULT_NOISE_STD
        assert len(square_env.probes) == 0

    def test_probe_returns_noisy_value(self, square_env):
        """probe() should return a noisy sample."""
        y = square_env.probe(0.5)
        assert isinstance(y, float)
        assert abs(y - 0.25) < 0.5

    def test_probe_records_history(self, square_env):
        """probe() should record probe history."""
        assert len(square_env.probes) == 0
        square_env.probe(0.5)
        assert len(square_env.probes) == 1
        square_env.probe(0.3)
        assert len(square_env.probes) == 2

    def test_probe_validates_domain(self, square_env):
        """probe() should reject values outside (0, 1]."""
        with pytest.raises(ValueError):
            square_env.probe(0.0)
        with pytest.raises(ValueError):
            square_env.probe(1.5)

    def test_get_test_points(self, square_env):
        """get_test_points() should return fixed test points."""
        points = square_env.get_test_points()
        assert len(points) == DEFAULT_NUM_TEST_POINTS

    def test_evaluate_computes_errors(self, square_env):
        """evaluate() should compute absolute errors."""
        points = square_env.get_test_points()
        perfect = [x ** 2 for x in points]
        errors = square_env.evaluate(perfect)
        assert all(e < 1e-10 for e in errors)

    def test_get_probe_count(self, square_env):
        """get_probe_count() should track probes."""
        assert square_env.get_probe_count() == 0
        square_env.probe(0.5)
        assert square_env.get_probe_count() == 1

    def test_reset_clears_probes(self, square_env):
        """reset() should clear probe history."""
        square_env.probe(0.5)
        assert square_env.get_probe_count() == 1
        square_env.reset()
        assert square_env.get_probe_count() == 0


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test environment factory functions."""

    def test_create_environment(self):
        """Should create environment from callable."""
        env = create_environment(lambda x: x * 2, function_id="double")
        assert env.function_id == "double"

    def test_create_environment_from_function_id(self):
        """Should create environment from function ID."""
        env = create_environment_from_function_id("P2")
        assert env.function_id == "P2"


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_probe_reproducible_with_seed(self):
        """Probes should be reproducible with same seed."""
        env1 = create_environment(lambda x: x, probe_seed=42)
        env2 = create_environment(lambda x: x, probe_seed=42)
        y1 = env1.probe(0.5)
        y2 = env2.probe(0.5)
        assert y1 == y2
