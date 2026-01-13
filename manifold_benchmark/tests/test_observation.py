"""
Tests for the ObservationGenerator class.

Tests verify correct generation of 1D slices for both agents,
boundary handling, and gradient inclusion.
"""

import pytest
from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.observation import ObservationGenerator


def test_horizontal_slice_samples():
    """Horizontal slice has correct number of samples."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    assert len(obs["slice"]) == 11


def test_vertical_slice_samples():
    """Vertical slice has correct number of samples."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_b(5.0, 5.0)
    assert len(obs["slice"]) == 11


def test_slice_range():
    """Slice covers correct x range."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    x_values = [s["x"] for s in obs["slice"]]
    assert abs(min(x_values) - 3.5) < 0.01  # 5.0 - 1.5
    assert abs(max(x_values) - 6.5) < 0.01  # 5.0 + 1.5


def test_vertical_slice_range():
    """Vertical slice covers correct y range."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_b(5.0, 5.0)
    y_values = [s["y"] for s in obs["slice"]]
    assert abs(min(y_values) - 3.5) < 0.01  # 5.0 - 1.5
    assert abs(max(y_values) - 6.5) < 0.01  # 5.0 + 1.5


def test_boundary_clipping():
    """Slice clips at domain boundary."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(0.5, 5.0)  # Near left edge
    x_values = [s["x"] for s in obs["slice"]]
    assert min(x_values) >= 0.0  # Clipped to boundary


def test_boundary_clipping_vertical():
    """Vertical slice clips at domain boundary."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_b(5.0, 0.5)  # Near bottom edge
    y_values = [s["y"] for s in obs["slice"]]
    assert min(y_values) >= 0.0  # Clipped to boundary


def test_gradient_in_observation():
    """Observation includes correct gradient."""
    surface = Surface([{"cx": 7.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    assert obs["gradient_x"] > 0  # Peak is to the right


def test_gradient_in_observation_vertical():
    """Vertical observation includes correct gradient."""
    surface = Surface([{"cx": 5.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_b(5.0, 5.0)
    assert obs["gradient_y"] > 0  # Peak is above


def test_observation_a_structure():
    """Agent A observation has correct structure."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)

    # Check all required keys present
    assert "position" in obs
    assert "value_at_position" in obs
    assert "gradient_x" in obs
    assert "slice" in obs

    # Check position structure
    assert "x" in obs["position"]
    assert "y" in obs["position"]
    assert obs["position"]["x"] == 5.0
    assert obs["position"]["y"] == 5.0

    # Check slice structure
    assert all("x" in s and "value" in s for s in obs["slice"])


def test_observation_b_structure():
    """Agent B observation has correct structure."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_b(5.0, 5.0)

    # Check all required keys present
    assert "position" in obs
    assert "value_at_position" in obs
    assert "gradient_y" in obs
    assert "slice" in obs

    # Check position structure
    assert "x" in obs["position"]
    assert "y" in obs["position"]
    assert obs["position"]["x"] == 5.0
    assert obs["position"]["y"] == 5.0

    # Check slice structure
    assert all("y" in s and "value" in s for s in obs["slice"])


def test_agents_see_different_slices():
    """Agent A and B see perpendicular slices (different data)."""
    surface = Surface([{"cx": 7.0, "cy": 3.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)

    obs_a = obs_gen.generate_observation_a(5.0, 5.0)
    obs_b = obs_gen.generate_observation_b(5.0, 5.0)

    # Same position
    assert obs_a["position"] == obs_b["position"]
    assert obs_a["value_at_position"] == obs_b["value_at_position"]

    # Different gradients (A gets x-gradient, B gets y-gradient)
    assert "gradient_x" in obs_a
    assert "gradient_y" in obs_b
    assert "gradient_y" not in obs_a
    assert "gradient_x" not in obs_b

    # Different slice orientations
    assert "x" in obs_a["slice"][0]
    assert "y" in obs_b["slice"][0]


def test_value_at_position_matches_slice_center():
    """Value at position should match middle of slice when not at boundary."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)

    # Middle sample should be at position x=5.0
    middle_idx = len(obs["slice"]) // 2
    middle_sample = obs["slice"][middle_idx]

    assert abs(middle_sample["x"] - 5.0) < 0.01
    assert abs(middle_sample["value"] - obs["value_at_position"]) < 0.01
