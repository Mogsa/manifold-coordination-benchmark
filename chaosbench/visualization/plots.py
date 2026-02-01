"""Plotting functions for ChaosBench experiments."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_task(
    observations: np.ndarray,
    actual: float,
    prediction: Optional[float] = None,
    uncertainty: Optional[float] = None,
    title: str = "Task",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a single task: observations + prediction vs actual.

    Args:
        observations: Array of observed values [x_0, ..., x_49]
        actual: True value to predict (x_50)
        prediction: Agent's prediction (optional)
        uncertainty: Agent's uncertainty ± (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    n_obs = len(observations)
    t_obs = np.arange(n_obs)
    t_target = n_obs

    # Plot observations
    ax.plot(t_obs, observations, 'b.-', label='Observations', markersize=4, linewidth=1)

    # Plot actual value
    ax.scatter([t_target], [actual], color='green', s=100, marker='o',
               label=f'Actual: {actual:.3f}', zorder=5)

    # Plot prediction if provided
    if prediction is not None:
        ax.scatter([t_target], [prediction], color='red', s=100, marker='x',
                   label=f'Prediction: {prediction:.3f}', zorder=5)

        # Add uncertainty bar if provided
        if uncertainty is not None:
            ax.errorbar([t_target], [prediction], yerr=uncertainty,
                       color='red', capsize=5, capthick=2, linewidth=2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add vertical line separating observations from prediction
    ax.axvline(x=n_obs - 0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_phi_curve(
    phi_points: List[dict],
    title: str = "Learning Curve Φ(n)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the Φ(n) learning curve.

    Args:
        phi_points: List of dicts with 'tasks' and 'phi' keys
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = [p['tasks'] for p in phi_points]
    phi = [p['phi'] for p in phi_points]

    # Plot actual curve
    ax.plot(tasks, phi, 'b.-', linewidth=2, markersize=8, label='Φ(n)')

    # Plot linear reference (no learning)
    if len(tasks) > 1:
        avg_increment = phi[-1] / tasks[-1]
        linear_ref = [avg_increment * t for t in tasks]
        ax.plot(tasks, linear_ref, 'k--', alpha=0.5, label='Linear (no learning)')

    ax.set_xlabel('Tasks completed (n)')
    ax.set_ylabel('Cumulative Φ(n)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_session_summary(
    phi_points: List[dict],
    scores: List[float],
    title: str = "Session Summary",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot session summary: Φ(n) curve + per-task scores.

    Args:
        phi_points: List of dicts with 'tasks' and 'phi' keys
        scores: List of per-task scores
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Φ(n) curve
    tasks = [p['tasks'] for p in phi_points]
    phi = [p['phi'] for p in phi_points]

    ax1.plot(tasks, phi, 'b.-', linewidth=2, markersize=8, label='Φ(n)')

    if len(tasks) > 1:
        avg_increment = phi[-1] / tasks[-1]
        linear_ref = [avg_increment * t for t in tasks]
        ax1.plot(tasks, linear_ref, 'k--', alpha=0.5, label='Linear (no learning)')

    ax1.set_xlabel('Tasks completed (n)')
    ax1.set_ylabel('Cumulative Φ(n)')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Per-task scores
    task_nums = list(range(1, len(scores) + 1))
    colors = ['green' if s > 0.5 else 'orange' if s > 0.2 else 'red' for s in scores]
    ax2.bar(task_nums, scores, color=colors, alpha=0.7)
    ax2.axhline(y=np.mean(scores), color='blue', linestyle='--',
                label=f'Mean: {np.mean(scores):.2f}')
    ax2.set_xlabel('Task number')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Task Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
