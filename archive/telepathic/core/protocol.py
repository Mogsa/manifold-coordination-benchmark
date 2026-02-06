"""
Communication protocol for the Telepathic benchmark.

Defines vocabulary, message format, and validation rules.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ProtocolConfig:
    """Configuration for the communication protocol."""
    vocabulary: List[str]
    max_tokens: int
    separator: str


# MVP Protocol: 5 tokens, max 5 in a message
MVP_VOCABULARY: List[str] = ["α", "β", "γ", "δ", "ε"]
FULL_VOCABULARY: List[str] = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ"]

DEFAULT_CONFIG = ProtocolConfig(
    vocabulary=MVP_VOCABULARY,
    max_tokens=5,
    separator="-"
)


class Protocol:
    """Manages communication protocol and message validation."""

    def __init__(self, config: Optional[ProtocolConfig] = None):
        """
        Initialize protocol.

        Args:
            config: Protocol configuration (defaults to MVP)
        """
        self.config = config or DEFAULT_CONFIG

    @property
    def vocabulary(self) -> List[str]:
        """Available tokens."""
        return self.config.vocabulary

    @property
    def max_tokens(self) -> int:
        """Maximum message length in tokens."""
        return self.config.max_tokens

    @property
    def separator(self) -> str:
        """Token separator character."""
        return self.config.separator

    def validate_message(self, message: str) -> bool:
        """
        Check if a message is valid.

        A valid message:
        1. Contains only vocabulary tokens
        2. Uses proper separator
        3. Has at most max_tokens tokens

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        if not message or not message.strip():
            return False

        tokens = message.strip().split(self.separator)

        # Check token count
        if len(tokens) > self.max_tokens:
            return False

        # Check each token is in vocabulary
        for token in tokens:
            if token not in self.vocabulary:
                return False

        return True

    def parse_message(self, message: str) -> List[str]:
        """
        Parse message into token list.

        Args:
            message: Message to parse

        Returns:
            List of tokens

        Raises:
            ValueError: If message is invalid
        """
        if not self.validate_message(message):
            raise ValueError(f"Invalid message: '{message}'")
        return message.strip().split(self.separator)

    def format_vocabulary(self) -> str:
        """Format vocabulary for prompts."""
        return ", ".join(self.vocabulary)

    def count_tokens(self, message: str) -> int:
        """Count tokens in a message."""
        if not message or not message.strip():
            return 0
        return len(message.strip().split(self.separator))

    def is_primitive(self, message: str) -> bool:
        """Check if message represents a single primitive."""
        return self.validate_message(message) and self.count_tokens(message) == 1

    def is_composition(self, message: str) -> bool:
        """Check if message represents a composition."""
        return self.validate_message(message) and self.count_tokens(message) > 1


# Module-level convenience functions using default protocol
_default_protocol = Protocol()


def validate_message(message: str) -> bool:
    """Validate message using default protocol."""
    return _default_protocol.validate_message(message)


def parse_message(message: str) -> List[str]:
    """Parse message using default protocol."""
    return _default_protocol.parse_message(message)


def get_vocabulary() -> List[str]:
    """Get vocabulary from default protocol."""
    return _default_protocol.vocabulary
