"""
Random agent module for the Manifold Coordination Benchmark.

This module implements a random baseline agent for lower-bound performance.
"""

from manifold_benchmark.agents.base import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    """Random baseline agent for lower-bound performance."""

    def __init__(self, role: str, seed: int = None, domain_size: float = 10.0):
        """
        Initialize a random agent.

        Args:
            role: 'A' or 'B'
            seed: Random seed for reproducibility
            domain_size: Domain bounds [0, domain_size]
        """
        super().__init__(role)
        self.domain_size = domain_size
        self.rng = np.random.default_rng(seed)

    def generate_message(self) -> str:
        """
        Generate message to send to other agent.

        Random agent does not communicate.

        Returns:
            Empty string
        """
        return ""

    def decide_position(self) -> float:
        """
        Decide next position along controlled axis.

        Returns random position in [0, domain_size].

        Returns:
            Random position coordinate
        """
        return float(self.rng.uniform(0, self.domain_size))

    def final_decision(self) -> float:
        """
        Make final position decision.

        Returns random position in [0, domain_size].

        Returns:
            Random final position coordinate
        """
        return float(self.rng.uniform(0, self.domain_size))
