"""
Base agent module for the Manifold Coordination Benchmark.

This module defines the abstract base class for all agents.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, role: str):
        """
        Initialize a base agent.

        Args:
            role: 'A' (controls x) or 'B' (controls y)
        """
        self.role = role
        self.observation_history = []
        self.message_history = []

    def receive_observation(self, observation: dict) -> None:
        """
        Store observation from environment.

        Args:
            observation: Observation dict from ObservationGenerator
        """
        self.observation_history.append(observation)

    def receive_message(self, message: str) -> None:
        """
        Store message from other agent.

        Args:
            message: Message string from other agent
        """
        self.message_history.append(message)

    @abstractmethod
    def generate_message(self) -> str:
        """
        Generate message to send to other agent.

        Returns:
            Message string (can be empty for non-communicating agents)
        """
        pass

    @abstractmethod
    def decide_position(self) -> float:
        """
        Decide next position along controlled axis.

        Returns:
            Position coordinate in [0, 10]
        """
        pass

    @abstractmethod
    def final_decision(self) -> float:
        """
        Make final position decision after all turns.

        Returns:
            Final position coordinate in [0, 10]
        """
        pass

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.observation_history = []
        self.message_history = []
