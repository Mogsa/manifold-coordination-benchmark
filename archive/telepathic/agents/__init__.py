"""Agent implementations for the Telepathic benchmark."""

from .base import (
    SeerAgent,
    DoerAgent,
    SeerInput,
    SeerOutput,
    DoerInput,
    DoerOutput,
)

from .seer import LLMSeerAgent
from .doer import LLMDoerAgent
from .random_agent import (
    RandomSeerAgent,
    RandomDoerAgent,
    OracleSeerAgent,
    OracleDoerAgent,
)

__all__ = [
    # Base classes
    "SeerAgent",
    "DoerAgent",
    "SeerInput",
    "SeerOutput",
    "DoerInput",
    "DoerOutput",
    # LLM agents
    "LLMSeerAgent",
    "LLMDoerAgent",
    # Baseline agents
    "RandomSeerAgent",
    "RandomDoerAgent",
    "OracleSeerAgent",
    "OracleDoerAgent",
]
