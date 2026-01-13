"""
Episode module for the Manifold Coordination Benchmark.

This module manages episode state, turn progression, and scoring.
"""

from typing import Tuple, List
from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.observation import ObservationGenerator


class Episode:
    """Manages episode state and progression."""

    def __init__(
        self,
        surface: Surface,
        initial_position: Tuple[float, float] = (5.0, 5.0),
        n_turns: int = 10,
        observation_generator: ObservationGenerator = None
    ):
        """
        Initialize an episode.

        Args:
            surface: The Surface to navigate
            initial_position: Starting (x, y) position
            n_turns: Number of turns in the episode
            observation_generator: Optional ObservationGenerator (creates default if None)
        """
        self.surface = surface
        self.n_turns = n_turns
        self._turn_number = 0
        self._current_position = initial_position

        # Create observation generator if not provided
        if observation_generator is None:
            self.observation_generator = ObservationGenerator(surface)
        else:
            self.observation_generator = observation_generator

        # History tracking
        self._history = []

    @property
    def current_position(self) -> Tuple[float, float]:
        """Current (x_a, y_b) position."""
        return self._current_position

    @property
    def turn_number(self) -> int:
        """Current turn number (0-indexed, 0 = before first turn)."""
        return self._turn_number

    def get_state(self) -> dict:
        """
        Get current episode state.

        Returns:
            Dictionary with current position and turn number
        """
        return {
            "position": self._current_position,
            "turn_number": self._turn_number
        }

    def step(self, x_new: float, y_new: float) -> Tuple[dict, dict]:
        """
        Move to new position and generate observations.

        Args:
            x_new: New x-coordinate (Agent A's decision)
            y_new: New y-coordinate (Agent B's decision)

        Returns:
            Tuple of (observation_a, observation_b)
        """
        # Update position
        self._current_position = (x_new, y_new)
        self._turn_number += 1

        # Generate observations at new position
        obs_a = self.observation_generator.generate_observation_a(x_new, y_new)
        obs_b = self.observation_generator.generate_observation_b(x_new, y_new)

        # Record in history
        self._history.append({
            "turn": self._turn_number,
            "position": (x_new, y_new),
            "observation_a": obs_a,
            "observation_b": obs_b
        })

        return (obs_a, obs_b)

    def is_done(self) -> bool:
        """
        Check if episode is complete.

        Returns:
            True if all turns have been completed
        """
        return self._turn_number >= self.n_turns

    def get_score(self, x_final: float, y_final: float) -> float:
        """
        Compute normalized score for final position.

        Score = f(x_final, y_final) / f(x_optimal, y_optimal)

        Args:
            x_final: Final x-coordinate
            y_final: Final y-coordinate

        Returns:
            Score in range [0, 1] where 1.0 is optimal
        """
        # Evaluate surface at final position
        final_value = self.surface.evaluate(x_final, y_final)

        # Get optimal value
        _, _, optimal_value = self.surface.get_optimal()

        # Compute normalized score
        # Avoid division by zero (though optimal should always be > 0)
        if optimal_value == 0:
            return 0.0

        score = final_value / optimal_value

        # Clamp to [0, 1] (shouldn't exceed 1, but just in case of numerical issues)
        return max(0.0, min(1.0, score))

    def get_history(self) -> List[dict]:
        """
        Return full history of positions and observations.

        Returns:
            List of dictionaries, one per turn, containing:
                - turn: Turn number
                - position: (x, y) tuple
                - observation_a: Agent A's observation
                - observation_b: Agent B's observation
        """
        return self._history.copy()
