"""
Slice visualization for agent observations.

This module provides functions for visualizing the 1D slices that agents observe.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_slices(surface, x_a: float, y_b: float, radius: float = 1.5,
               show: bool = True, save_path: Optional[str] = None):
    """
    Visualize the horizontal and vertical slices that agents observe.

    Args:
        surface: Surface object
        x_a: Agent A's x-position
        y_b: Agent B's y-position
        radius: Observation radius
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (fig, (ax1, ax2)) matplotlib objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Compute gradients at current position
    gx, gy = surface.gradient(x_a, y_b)
    value_at_pos = surface.evaluate(x_a, y_b)

    # === LEFT PLOT: Horizontal Slice (Agent A's view) ===
    # Sample range with boundary clamping
    x_min = max(0, x_a - radius)
    x_max = min(surface.domain_size, x_a + radius)
    x_samples = np.linspace(x_min, x_max, 100)

    # Evaluate horizontal slice
    h_slice = [surface.evaluate(x, y_b) for x in x_samples]

    # Plot horizontal slice
    ax1.plot(x_samples, h_slice, 'b-', linewidth=2, label='Horizontal Slice')
    ax1.axvline(x=x_a, color='red', linestyle='--', linewidth=1.5,
               label=f'Current x={x_a:.2f}')
    ax1.scatter([x_a], [value_at_pos], color='red', s=100, zorder=5)

    # Show gradient as arrow
    if abs(gx) > 0.001:  # Only show if gradient is significant
        arrow_length = 0.3  # Visual length in data units
        arrow_dx = arrow_length * np.sign(gx)
        arrow_dy = gx * arrow_dx * 0.1  # Scale for visual clarity
        ax1.arrow(x_a, value_at_pos, arrow_dx, arrow_dy,
                 head_width=0.05, head_length=0.1,
                 fc='orange', ec='orange', linewidth=2,
                 label=f'Gradient ∂f/∂x={gx:.3f}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x, y_b)')
    ax1.set_title(f'Agent A View: Horizontal Slice at y={y_b:.2f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(x_min - 0.1, x_max + 0.1)

    # === RIGHT PLOT: Vertical Slice (Agent B's view) ===
    # Sample range with boundary clamping
    y_min = max(0, y_b - radius)
    y_max = min(surface.domain_size, y_b + radius)
    y_samples = np.linspace(y_min, y_max, 100)

    # Evaluate vertical slice
    v_slice = [surface.evaluate(x_a, y) for y in y_samples]

    # Plot vertical slice
    ax2.plot(y_samples, v_slice, 'g-', linewidth=2, label='Vertical Slice')
    ax2.axvline(x=y_b, color='red', linestyle='--', linewidth=1.5,
               label=f'Current y={y_b:.2f}')
    ax2.scatter([y_b], [value_at_pos], color='red', s=100, zorder=5)

    # Show gradient as arrow
    if abs(gy) > 0.001:  # Only show if gradient is significant
        arrow_length = 0.3  # Visual length in data units
        arrow_dx = arrow_length * np.sign(gy)
        arrow_dy = gy * arrow_dx * 0.1  # Scale for visual clarity
        ax2.arrow(y_b, value_at_pos, arrow_dx, arrow_dy,
                 head_width=0.05, head_length=0.1,
                 fc='orange', ec='orange', linewidth=2,
                 label=f'Gradient ∂f/∂y={gy:.3f}')

    ax2.set_xlabel('y')
    ax2.set_ylabel('f(x_a, y)')
    ax2.set_title(f'Agent B View: Vertical Slice at x={x_a:.2f}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(y_min - 0.1, y_max + 0.1)

    # Overall title
    fig.suptitle(f'Agent Observations at Position ({x_a:.2f}, {y_b:.2f}) | '
                f'f(x,y)={value_at_pos:.3f}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig, (ax1, ax2)


def plot_observation_comparison(surface, positions: list,
                                radius: float = 1.5,
                                show: bool = True, save_path: Optional[str] = None):
    """
    Compare slices at multiple positions (useful for debugging episode progression).

    Args:
        surface: Surface object
        positions: List of (x, y) positions to compare
        radius: Observation radius
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (fig, axes) matplotlib objects
    """
    n_positions = len(positions)
    if n_positions == 0:
        raise ValueError("Must provide at least one position")

    # Create subplots: 2 rows (horizontal, vertical), n_positions columns
    fig, axes = plt.subplots(2, n_positions, figsize=(5 * n_positions, 8))

    # Handle single position case
    if n_positions == 1:
        axes = axes.reshape(2, 1)

    for idx, (x_a, y_b) in enumerate(positions):
        # Get current column
        ax_h = axes[0, idx]  # Horizontal slice
        ax_v = axes[1, idx]  # Vertical slice

        # Compute gradient and value
        gx, gy = surface.gradient(x_a, y_b)
        value_at_pos = surface.evaluate(x_a, y_b)

        # === Horizontal Slice ===
        x_min = max(0, x_a - radius)
        x_max = min(surface.domain_size, x_a + radius)
        x_samples = np.linspace(x_min, x_max, 100)
        h_slice = [surface.evaluate(x, y_b) for x in x_samples]

        ax_h.plot(x_samples, h_slice, 'b-', linewidth=2)
        ax_h.axvline(x=x_a, color='red', linestyle='--', linewidth=1.5)
        ax_h.scatter([x_a], [value_at_pos], color='red', s=80, zorder=5)
        ax_h.set_xlabel('x')
        ax_h.set_ylabel('f(x, y_b)')
        ax_h.set_title(f'Position {idx + 1}: ({x_a:.2f}, {y_b:.2f})\n'
                      f'Horizontal | ∂f/∂x={gx:.3f}')
        ax_h.grid(True, alpha=0.3)

        # === Vertical Slice ===
        y_min = max(0, y_b - radius)
        y_max = min(surface.domain_size, y_b + radius)
        y_samples = np.linspace(y_min, y_max, 100)
        v_slice = [surface.evaluate(x_a, y) for y in y_samples]

        ax_v.plot(y_samples, v_slice, 'g-', linewidth=2)
        ax_v.axvline(x=y_b, color='red', linestyle='--', linewidth=1.5)
        ax_v.scatter([y_b], [value_at_pos], color='red', s=80, zorder=5)
        ax_v.set_xlabel('y')
        ax_v.set_ylabel('f(x_a, y)')
        ax_v.set_title(f'Vertical | ∂f/∂y={gy:.3f}')
        ax_v.grid(True, alpha=0.3)

    fig.suptitle('Observation Slices at Multiple Positions',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return fig, axes


def plot_slice_evolution(surface, episode_history: list,
                        radius: float = 1.5,
                        show: bool = True, save_path: Optional[str] = None):
    """
    Show how slices evolve over the course of an episode.

    Args:
        surface: Surface object
        episode_history: Episode history with position information
        radius: Observation radius
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (fig, axes) matplotlib objects
    """
    # Extract positions from history
    positions = []
    if isinstance(episode_history, list) and len(episode_history) > 0:
        if isinstance(episode_history[0], dict):
            for entry in episode_history:
                if 'position_after' in entry:
                    pos = entry['position_after']
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        positions.append(tuple(pos))
        elif isinstance(episode_history[0], (list, tuple)):
            positions = [tuple(pos) for pos in episode_history]

    if len(positions) == 0:
        raise ValueError("No valid positions found in episode_history")

    # Use plot_observation_comparison for the visualization
    return plot_observation_comparison(surface, positions, radius, show, save_path)
