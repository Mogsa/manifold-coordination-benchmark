"""
Base agent interfaces for the Manifold Benchmark suite.

This module defines minimal abstract interfaces that can be extended
by both coordination and temporal benchmark agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AgentBase(ABC):
    """
    Minimal abstract base class for all benchmark agents.

    This defines the core interface that any agent must implement.
    Specific benchmarks (coordination, temporal) extend this with
    their own requirements.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return agent configuration for logging.

        Returns:
            Dictionary with agent type and parameters
        """
        pass


class ObservationHistoryMixin:
    """
    Mixin providing observation history tracking.

    Use this in agents that need to track their observation history.
    """

    def __init__(self):
        self.observation_history: List[Dict[str, Any]] = []

    def receive_observation(self, observation: Dict[str, Any]) -> None:
        """Store an observation in history."""
        self.observation_history.append(observation)

    def clear_observation_history(self) -> None:
        """Clear all stored observations."""
        self.observation_history = []

    def get_observation_history(self) -> List[Dict[str, Any]]:
        """Return copy of observation history."""
        return list(self.observation_history)


class MessageHistoryMixin:
    """
    Mixin providing message history tracking.

    Use this in agents that communicate with other agents.
    """

    def __init__(self):
        self.message_history: List[str] = []

    def receive_message(self, message: str) -> None:
        """Store a received message in history."""
        self.message_history.append(message)

    def clear_message_history(self) -> None:
        """Clear all stored messages."""
        self.message_history = []

    def get_message_history(self) -> List[str]:
        """Return copy of message history."""
        return list(self.message_history)
