"""
Tests for visualization functions.

These tests verify that visualization functions run without errors
and produce the expected output structures.
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
from coordination.core.surface import Surface
from coordination.visualization.plot3d import (
    plot_surface, plot_episode, plot_surface_contour
)
from coordination.visualization.slices import (
    plot_slices, plot_observation_comparison
)


@pytest.fixture
def simple_surface():
    """Create a simple single-peak surface for testing."""
    peaks = [{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.5}]
    return Surface(peaks)


@pytest.fixture
def two_peak_surface():
    """Create a two-peak surface for testing."""
    peaks = [
        {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},
        {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}
    ]
    return Surface(peaks)


class TestPlotSurface:
    """Tests for plot_surface function."""

    def test_plot_surface_basic(self, simple_surface):
        """Test that plot_surface runs without errors."""
        ax = plot_surface(simple_surface, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_surface_with_save(self, simple_surface, tmp_path):
        """Test saving plot to file."""
        save_path = tmp_path / "test_surface.png"
        ax = plot_surface(simple_surface, show=False, save_path=str(save_path))
        assert ax is not None
        assert save_path.exists()
        plt.close('all')

    def test_plot_surface_two_peaks(self, two_peak_surface):
        """Test plotting surface with multiple peaks."""
        ax = plot_surface(two_peak_surface, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_surface_with_custom_ax(self, simple_surface):
        """Test providing custom axis."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        result_ax = plot_surface(simple_surface, ax=ax, show=False)
        assert result_ax is ax
        plt.close('all')


class TestPlotEpisode:
    """Tests for plot_episode function."""

    def test_plot_episode_simple_trajectory(self, simple_surface):
        """Test plotting with simple list of positions."""
        positions = [
            (5.0, 5.0),
            (6.0, 6.0),
            (7.0, 7.0)
        ]
        ax = plot_episode(simple_surface, positions, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_episode_dict_format(self, simple_surface):
        """Test plotting with dictionary format history."""
        history = [
            {"turn": 1, "position_after": [5.0, 5.0]},
            {"turn": 2, "position_after": [6.0, 6.0]},
            {"turn": 3, "position_after": [7.0, 7.0]}
        ]
        ax = plot_episode(simple_surface, history, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_episode_with_save(self, simple_surface, tmp_path):
        """Test saving episode plot."""
        positions = [(5.0, 5.0), (6.0, 6.0)]
        save_path = tmp_path / "test_episode.png"
        ax = plot_episode(simple_surface, positions, show=False,
                         save_path=str(save_path))
        assert ax is not None
        assert save_path.exists()
        plt.close('all')

    def test_plot_episode_empty_history(self, simple_surface):
        """Test handling of empty history."""
        ax = plot_episode(simple_surface, [], show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_episode_single_position(self, simple_surface):
        """Test with single position."""
        positions = [(5.0, 5.0)]
        ax = plot_episode(simple_surface, positions, show=False)
        assert ax is not None
        plt.close('all')


class TestPlotContour:
    """Tests for plot_surface_contour function."""

    def test_plot_contour_basic(self, simple_surface):
        """Test basic contour plot."""
        ax = plot_surface_contour(simple_surface, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_contour_with_trajectory(self, simple_surface):
        """Test contour plot with trajectory overlay."""
        positions = [(5.0, 5.0), (6.0, 6.0), (7.0, 7.0)]
        ax = plot_surface_contour(simple_surface, positions=positions, show=False)
        assert ax is not None
        plt.close('all')

    def test_plot_contour_with_save(self, simple_surface, tmp_path):
        """Test saving contour plot."""
        save_path = tmp_path / "test_contour.png"
        ax = plot_surface_contour(simple_surface, show=False,
                                 save_path=str(save_path))
        assert ax is not None
        assert save_path.exists()
        plt.close('all')


class TestPlotSlices:
    """Tests for plot_slices function."""

    def test_plot_slices_basic(self, simple_surface):
        """Test basic slice visualization."""
        fig, (ax1, ax2) = plot_slices(simple_surface, x_a=5.0, y_b=5.0,
                                     show=False)
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        plt.close('all')

    def test_plot_slices_at_peak(self, simple_surface):
        """Test slices at peak location."""
        x_opt, y_opt, _ = simple_surface.get_optimal()
        fig, axes = plot_slices(simple_surface, x_a=x_opt, y_b=y_opt,
                               show=False)
        assert fig is not None
        plt.close('all')

    def test_plot_slices_at_edge(self, simple_surface):
        """Test slices near domain boundary."""
        fig, axes = plot_slices(simple_surface, x_a=0.5, y_b=9.5,
                               radius=1.5, show=False)
        assert fig is not None
        plt.close('all')

    def test_plot_slices_with_save(self, simple_surface, tmp_path):
        """Test saving slice plot."""
        save_path = tmp_path / "test_slices.png"
        fig, axes = plot_slices(simple_surface, x_a=5.0, y_b=5.0,
                               show=False, save_path=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close('all')

    def test_plot_slices_custom_radius(self, simple_surface):
        """Test with custom observation radius."""
        fig, axes = plot_slices(simple_surface, x_a=5.0, y_b=5.0,
                               radius=2.0, show=False)
        assert fig is not None
        plt.close('all')


class TestPlotObservationComparison:
    """Tests for plot_observation_comparison function."""

    def test_comparison_single_position(self, simple_surface):
        """Test comparison with single position."""
        positions = [(5.0, 5.0)]
        fig, axes = plot_observation_comparison(simple_surface, positions,
                                               show=False)
        assert fig is not None
        assert axes.shape == (2, 1)
        plt.close('all')

    def test_comparison_multiple_positions(self, simple_surface):
        """Test comparison with multiple positions."""
        positions = [(5.0, 5.0), (6.0, 6.0), (7.0, 7.0)]
        fig, axes = plot_observation_comparison(simple_surface, positions,
                                               show=False)
        assert fig is not None
        assert axes.shape == (2, 3)
        plt.close('all')

    def test_comparison_empty_list(self, simple_surface):
        """Test that empty position list raises error."""
        with pytest.raises(ValueError, match="Must provide at least one position"):
            plot_observation_comparison(simple_surface, [], show=False)

    def test_comparison_with_save(self, simple_surface, tmp_path):
        """Test saving comparison plot."""
        positions = [(5.0, 5.0), (7.0, 7.0)]
        save_path = tmp_path / "test_comparison.png"
        fig, axes = plot_observation_comparison(simple_surface, positions,
                                               show=False, save_path=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close('all')


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_episode_visualization(self, two_peak_surface):
        """Test complete visualization workflow for an episode."""
        # Simulate an episode trajectory
        positions = [
            (5.0, 5.0),   # Start
            (6.0, 6.0),   # Move toward peak
            (7.0, 7.0),   # Continue
            (7.5, 7.5)    # Reach peak
        ]

        # Test all visualization functions
        ax_surface = plot_surface(two_peak_surface, show=False)
        assert ax_surface is not None

        ax_episode = plot_episode(two_peak_surface, positions, show=False)
        assert ax_episode is not None

        ax_contour = plot_surface_contour(two_peak_surface, positions=positions,
                                         show=False)
        assert ax_contour is not None

        fig_slices, _ = plot_slices(two_peak_surface, x_a=7.5, y_b=7.5,
                                   show=False)
        assert fig_slices is not None

        fig_comp, _ = plot_observation_comparison(two_peak_surface, positions,
                                                 show=False)
        assert fig_comp is not None

        plt.close('all')

    def test_visualization_consistency(self, simple_surface):
        """Test that visualizations are consistent across calls."""
        # Plot same surface twice
        ax1 = plot_surface(simple_surface, show=False)
        ax2 = plot_surface(simple_surface, show=False)

        # Both should succeed
        assert ax1 is not None
        assert ax2 is not None

        plt.close('all')
