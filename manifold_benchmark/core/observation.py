"""
Observation module for the Manifold Coordination Benchmark.

This module generates agent observations (1D slices) from 2D surfaces.
"""

from typing import List
import numpy as np
from manifold_benchmark.core.surface import Surface


class ObservationGenerator:
    """Generates agent observations from surface."""

    def __init__(self, surface: Surface, radius: float = 1.5, n_samples: int = 11):
        """
        Initialize observation generator.

        Args:
            surface: The Surface to observe
            radius: Observation radius R (agents see slice of length 2R)
            n_samples: Number of samples in each slice
        """
        self.surface = surface
        self.radius = radius
        self.n_samples = n_samples

    def generate_observation_a(self, x_a: float, y_b: float) -> dict:
        """
        Generate observation for Agent A (horizontal slice).

        Agent A controls x-coordinate and sees horizontal slice through (x_a, y_b).

        Args:
            x_a: Agent A's x-position
            y_b: Agent B's y-position

        Returns:
            Dictionary containing:
                - position: {"x": x_a, "y": y_b}
                - value_at_position: f(x_a, y_b)
                - gradient_x: ∂f/∂x at (x_a, y_b)
                - slice: List of {"x": float, "value": float} for horizontal slice
        """
        # Compute slice bounds (clamped to domain)
        x_min = max(0, x_a - self.radius)
        x_max = min(self.surface.domain_size, x_a + self.radius)

        # Generate evenly spaced samples within bounds
        x_samples = np.linspace(x_min, x_max, self.n_samples)

        # Evaluate surface at each sample point (horizontal slice)
        slice_data = [
            {"x": float(x), "value": float(self.surface.evaluate(x, y_b))}
            for x in x_samples
        ]

        # Get gradient in x-direction
        gradient_x, _ = self.surface.gradient(x_a, y_b)

        return {
            "position": {"x": x_a, "y": y_b},
            "value_at_position": float(self.surface.evaluate(x_a, y_b)),
            "gradient_x": float(gradient_x),
            "slice": slice_data
        }

    def generate_observation_b(self, x_a: float, y_b: float) -> dict:
        """
        Generate observation for Agent B (vertical slice).

        Agent B controls y-coordinate and sees vertical slice through (x_a, y_b).

        Args:
            x_a: Agent A's x-position
            y_b: Agent B's y-position

        Returns:
            Dictionary containing:
                - position: {"x": x_a, "y": y_b}
                - value_at_position: f(x_a, y_b)
                - gradient_y: ∂f/∂y at (x_a, y_b)
                - slice: List of {"y": float, "value": float} for vertical slice
        """
        # Compute slice bounds (clamped to domain)
        y_min = max(0, y_b - self.radius)
        y_max = min(self.surface.domain_size, y_b + self.radius)

        # Generate evenly spaced samples within bounds
        y_samples = np.linspace(y_min, y_max, self.n_samples)

        # Evaluate surface at each sample point (vertical slice)
        slice_data = [
            {"y": float(y), "value": float(self.surface.evaluate(x_a, y))}
            for y in y_samples
        ]

        # Get gradient in y-direction
        _, gradient_y = self.surface.gradient(x_a, y_b)

        return {
            "position": {"x": x_a, "y": y_b},
            "value_at_position": float(self.surface.evaluate(x_a, y_b)),
            "gradient_y": float(gradient_y),
            "slice": slice_data
        }
