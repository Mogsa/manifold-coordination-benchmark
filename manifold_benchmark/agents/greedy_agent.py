"""
Greedy agent module for the Manifold Coordination Benchmark.

This module implements a gradient-following baseline agent.
"""

from manifold_benchmark.agents.base import BaseAgent
import numpy as np


class GreedyAgent(BaseAgent):
    """Gradient-following baseline agent."""

    def __init__(self, role: str, step_size: float = 1.0, domain_size: float = 10.0):
        """
        Initialize a greedy agent.

        Args:
            role: 'A' or 'B'
            step_size: Movement magnitude per turn
            domain_size: Domain bounds [0, domain_size]
        """
        super().__init__(role)
        self.step_size = step_size
        self.domain_size = domain_size
        self.current_position = 5.0  # Start at center

    def generate_message(self) -> str:
        """
        Generate message to send to other agent.

        Greedy agent does not communicate.

        Returns:
            Empty string
        """
        return ""

    def decide_position(self) -> float:
        """
        Decide next position along controlled axis.

        Follows gradient direction: moves step_size in direction of gradient.

        Returns:
            New position coordinate in [0, domain_size]
        """
        # Get most recent observation
        if not self.observation_history:
            return self.current_position

        obs = self.observation_history[-1]

        # Extract gradient based on role
        if self.role == 'A':
            gradient = obs.get('gradient_x', 0.0)
            self.current_position = obs['position']['x']
        else:  # role == 'B'
            gradient = obs.get('gradient_y', 0.0)
            self.current_position = obs['position']['y']

        # Move in gradient direction
        new_position = self.current_position + self.step_size * np.sign(gradient)

        # Clamp to bounds
        new_position = max(0.0, min(self.domain_size, new_position))

        return float(new_position)

    def final_decision(self) -> float:
        """
        Make final position decision.

        Returns current position (no special logic for final turn).

        Returns:
            Current position coordinate
        """
        return self.current_position
