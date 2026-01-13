"""
3D visualization for surfaces and episode trajectories.

This module provides functions for:
- Plotting 3D surface visualizations
- Overlaying agent trajectories on surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Dict, Tuple


def plot_surface(surface, ax=None, show: bool = True, save_path: Optional[str] = None):
    """
    Plot a 3D visualization of the surface.

    Args:
        surface: Surface object to visualize
        ax: Optional matplotlib 3D axis. If None, creates new figure
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        The matplotlib axis object
    """
    # Create figure and axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Create grid of points (50x50 resolution)
    x = np.linspace(0, surface.domain_size, 50)
    y = np.linspace(0, surface.domain_size, 50)
    X, Y = np.meshgrid(x, y)

    # Evaluate surface at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.evaluate(X[i, j], Y[i, j])

    # Plot the surface with colormap
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                          linewidth=0, antialiased=True)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x,y)')

    # Mark the optimal point with red sphere
    x_opt, y_opt, f_opt = surface.get_optimal()
    ax.scatter([x_opt], [y_opt], [f_opt],
              color='red', s=100, marker='o',
              label='Global Maximum', zorder=10)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Surface Visualization')
    ax.legend()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return ax


def plot_episode(surface, episode_history: List[Dict],
                ax=None, show: bool = True, save_path: Optional[str] = None):
    """
    Plot surface with agent trajectory overlay.

    Args:
        surface: Surface object to visualize
        episode_history: List of position dictionaries with keys:
                        'turn', 'position_before', 'position_after', etc.
                        Or a simpler list of (x, y) tuples
        ax: Optional matplotlib 3D axis. If None, creates new figure
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        The matplotlib axis object
    """
    # Create figure and axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Create grid of points (50x50 resolution)
    x = np.linspace(0, surface.domain_size, 50)
    y = np.linspace(0, surface.domain_size, 50)
    X, Y = np.meshgrid(x, y)

    # Evaluate surface at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.evaluate(X[i, j], Y[i, j])

    # Plot the surface (more transparent to see trajectory)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5,
                          linewidth=0, antialiased=True)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x,y)')

    # Extract trajectory positions
    positions = []

    # Handle different history formats
    if isinstance(episode_history, list) and len(episode_history) > 0:
        if isinstance(episode_history[0], dict):
            # Format from Episode.get_history()
            for entry in episode_history:
                if 'position_after' in entry:
                    pos = entry['position_after']
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        positions.append(pos)
                elif 'position' in entry:
                    pos = entry['position']
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        positions.append(pos)
        elif isinstance(episode_history[0], (list, tuple)):
            # Simple list of (x, y) tuples
            positions = episode_history

    if len(positions) == 0:
        print("Warning: No positions found in episode_history")
        return ax

    # Extract x, y coordinates and evaluate z values
    traj_x = [pos[0] for pos in positions]
    traj_y = [pos[1] for pos in positions]
    traj_z = [surface.evaluate(pos[0], pos[1]) for pos in positions]

    # Plot trajectory line
    ax.plot(traj_x, traj_y, traj_z,
           color='white', linewidth=2, alpha=0.9,
           label='Trajectory', zorder=5)

    # Plot turn markers
    for i, (x, y, z) in enumerate(zip(traj_x, traj_y, traj_z)):
        if i == 0:
            # Start point (green)
            ax.scatter([x], [y], [z],
                      color='green', s=150, marker='o',
                      edgecolors='white', linewidths=2,
                      label='Start', zorder=8)
        elif i == len(traj_x) - 1:
            # End point (blue)
            ax.scatter([x], [y], [z],
                      color='blue', s=150, marker='o',
                      edgecolors='white', linewidths=2,
                      label='Final Position', zorder=8)
        else:
            # Intermediate positions (yellow)
            ax.scatter([x], [y], [z],
                      color='yellow', s=80, marker='o',
                      edgecolors='white', linewidths=1,
                      alpha=0.7, zorder=7)
            # Add turn number labels
            ax.text(x, y, z, f' {i}', fontsize=8, color='black')

    # Mark the optimal point with red sphere
    x_opt, y_opt, f_opt = surface.get_optimal()
    ax.scatter([x_opt], [y_opt], [f_opt],
              color='red', s=150, marker='*',
              edgecolors='white', linewidths=2,
              label='Global Maximum', zorder=10)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Episode Trajectory on Surface')
    ax.legend(loc='upper left')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return ax


def plot_surface_contour(surface, positions: Optional[List[Tuple[float, float]]] = None,
                        show: bool = True, save_path: Optional[str] = None):
    """
    Plot a 2D contour view of the surface (useful for debugging).

    Args:
        surface: Surface object to visualize
        positions: Optional list of (x, y) positions to overlay
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        The matplotlib axis object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid of points
    x = np.linspace(0, surface.domain_size, 100)
    y = np.linspace(0, surface.domain_size, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate surface at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.evaluate(X[i, j], Y[i, j])

    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax.contour(X, Y, Z, levels=10, colors='black',
                               alpha=0.3, linewidths=0.5)

    # Add colorbar
    fig.colorbar(contour, ax=ax, label='f(x,y)')

    # Mark optimal point
    x_opt, y_opt, f_opt = surface.get_optimal()
    ax.scatter([x_opt], [y_opt], color='red', s=200, marker='*',
              edgecolors='white', linewidths=2, label='Global Maximum', zorder=10)

    # Plot trajectory if provided
    if positions and len(positions) > 0:
        traj_x = [pos[0] for pos in positions]
        traj_y = [pos[1] for pos in positions]

        # Plot line
        ax.plot(traj_x, traj_y, color='white', linewidth=2,
               alpha=0.9, label='Trajectory', zorder=5)

        # Mark start and end
        ax.scatter([traj_x[0]], [traj_y[0]], color='green', s=150,
                  marker='o', edgecolors='white', linewidths=2,
                  label='Start', zorder=8)
        ax.scatter([traj_x[-1]], [traj_y[-1]], color='blue', s=150,
                  marker='o', edgecolors='white', linewidths=2,
                  label='Final', zorder=8)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Surface Contour Plot')
    ax.legend()
    ax.set_aspect('equal')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show if requested
    if show:
        plt.show()

    return ax
