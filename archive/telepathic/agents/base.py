"""
Base agent interfaces for the Telepathic benchmark.

Defines abstract interfaces for Seer and Doer agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class SeerInput:
    """Input provided to the Seer agent."""
    samples: List[Tuple[float, float]]  # (x, y) pairs
    vocabulary: List[str]
    max_tokens: int


@dataclass
class SeerOutput:
    """Output from the Seer agent."""
    message: str
    raw_response: Optional[str] = None  # Full LLM response for debugging
    parse_attempts: int = 1  # How many attempts to get valid output
    parse_error: bool = False  # True if we had to salvage/fallback


@dataclass
class DoerInput:
    """Input provided to the Doer agent."""
    message: str
    test_input: float
    vocabulary: List[str]


@dataclass
class DoerOutput:
    """Output from the Doer agent."""
    prediction: float
    raw_response: Optional[str] = None
    parse_attempts: int = 1
    parse_error: bool = False


class SeerAgent(ABC):
    """
    Abstract base class for Seer agents.

    The Seer observes function samples and produces a token message
    encoding the function for the Doer.
    """

    @abstractmethod
    def generate_message(self, input: SeerInput) -> SeerOutput:
        """
        Generate a message encoding the observed function.

        Args:
            input: SeerInput containing samples, vocabulary, max_tokens

        Returns:
            SeerOutput with message and metadata
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration for logging."""
        pass


class DoerAgent(ABC):
    """
    Abstract base class for Doer agents.

    The Doer receives a token message and a test input,
    and must produce the function output.
    """

    @abstractmethod
    def generate_prediction(self, input: DoerInput) -> DoerOutput:
        """
        Generate a prediction for the test input.

        Args:
            input: DoerInput containing message, test_input, vocabulary

        Returns:
            DoerOutput with prediction and metadata
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration for logging."""
        pass
