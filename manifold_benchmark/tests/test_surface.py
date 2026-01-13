"""
Tests for the Surface class.

Tests verify correct evaluation of Gaussian peaks, gradient computation,
and optimal point finding.
"""

import pytest
from manifold_benchmark.core.surface import Surface


def test_single_peak_evaluation():
    """Surface with single peak evaluates correctly at peak."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    assert abs(surface.evaluate(5.0, 5.0) - 1.0) < 0.01
    assert surface.evaluate(0.0, 0.0) < 0.1


def test_gradient_points_toward_peak():
    """Gradient should point toward peak."""
    surface = Surface([{"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    gx, gy = surface.gradient(5.0, 5.0)
    assert gx > 0  # Peak is to the right
    assert gy > 0  # Peak is above


def test_gradient_zero_at_peak():
    """Gradient should be near zero at peak."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    gx, gy = surface.gradient(5.0, 5.0)
    assert abs(gx) < 0.01
    assert abs(gy) < 0.01


def test_get_optimal():
    """get_optimal returns correct peak location."""
    surface = Surface([
        {"cx": 3.0, "cy": 3.0, "height": 0.5, "sigma": 1.0},
        {"cx": 7.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
    ])
    x_opt, y_opt, f_opt = surface.get_optimal()
    assert abs(x_opt - 7.0) < 0.1
    assert abs(y_opt - 8.0) < 0.1
    assert abs(f_opt - 1.0) < 0.01


def test_multi_peak_surface():
    """Multiple peaks superimpose correctly."""
    surface = Surface([
        {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
        {"cx": 8.0, "cy": 8.0, "height": 0.5, "sigma": 1.0}
    ])
    # At each peak
    assert abs(surface.evaluate(2.0, 2.0) - 0.5) < 0.01
    assert abs(surface.evaluate(8.0, 8.0) - 0.5) < 0.01
    # At center (far from both peaks)
    assert surface.evaluate(5.0, 5.0) < 0.1


def test_to_dict_from_dict():
    """Serialization and deserialization work correctly."""
    original = Surface([
        {"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.5}
    ], domain_size=10.0)

    # Serialize
    config = original.to_dict()

    # Deserialize
    restored = Surface.from_dict(config)

    # Verify they produce same results
    assert abs(restored.evaluate(5.0, 5.0) - original.evaluate(5.0, 5.0)) < 0.001
    assert restored.domain_size == original.domain_size
    assert len(restored.peaks) == len(original.peaks)


def test_invalid_peak_definition():
    """Constructor should raise error for invalid peak definitions."""
    with pytest.raises(ValueError):
        Surface([{"cx": 5.0, "cy": 5.0}])  # Missing height and sigma


def test_domain_size():
    """Custom domain size is respected."""
    surface = Surface([{"cx": 15.0, "cy": 15.0, "height": 1.0, "sigma": 2.0}], domain_size=20.0)
    assert surface.domain_size == 20.0
    x_opt, y_opt, f_opt = surface.get_optimal()
    # Optimal should be within custom domain bounds
    assert 0 <= x_opt <= 20.0
    assert 0 <= y_opt <= 20.0
