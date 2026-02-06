"""
Random baseline agents for the Telepathic benchmark.

These agents make random choices and establish the lower bound
for performance comparison.
"""

import random
from typing import Dict, Any, Optional

from .base import SeerAgent, DoerAgent, SeerInput, SeerOutput, DoerInput, DoerOutput


class RandomSeerAgent(SeerAgent):
    """
    Random Seer baseline.

    Outputs random token(s) regardless of the samples observed.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_composition_depth: int = 2
    ):
        """
        Initialize random Seer.

        Args:
            seed: Random seed for reproducibility
            max_composition_depth: Maximum number of tokens in message
        """
        self.seed = seed
        self.max_composition_depth = max_composition_depth
        self.rng = random.Random(seed)

    def reset(self) -> None:
        """Reset random state."""
        self.rng = random.Random(self.seed)

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "RandomSeerAgent",
            "seed": self.seed,
            "max_composition_depth": self.max_composition_depth
        }

    def generate_message(self, input: SeerInput) -> SeerOutput:
        """
        Generate a random message.

        Args:
            input: SeerInput (ignored except for vocabulary)

        Returns:
            SeerOutput with random message
        """
        # Decide how many tokens (1 to max_composition_depth)
        num_tokens = self.rng.randint(1, min(self.max_composition_depth, input.max_tokens))

        # Pick random tokens (with replacement allowed)
        tokens = [self.rng.choice(input.vocabulary) for _ in range(num_tokens)]

        message = "-".join(tokens)

        return SeerOutput(
            message=message,
            raw_response=None,
            parse_attempts=1,
            parse_error=False
        )


class RandomDoerAgent(DoerAgent):
    """
    Random Doer baseline.

    Outputs a random number in a reasonable range regardless
    of the message received.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        output_range: tuple = (-10.0, 10.0)
    ):
        """
        Initialize random Doer.

        Args:
            seed: Random seed for reproducibility
            output_range: (min, max) range for random outputs
        """
        self.seed = seed
        self.output_range = output_range
        self.rng = random.Random(seed)

    def reset(self) -> None:
        """Reset random state."""
        self.rng = random.Random(self.seed)

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "RandomDoerAgent",
            "seed": self.seed,
            "output_range": self.output_range
        }

    def generate_prediction(self, input: DoerInput) -> DoerOutput:
        """
        Generate a random prediction.

        Args:
            input: DoerInput (ignored)

        Returns:
            DoerOutput with random prediction
        """
        prediction = self.rng.uniform(*self.output_range)

        return DoerOutput(
            prediction=prediction,
            raw_response=None,
            parse_attempts=1,
            parse_error=False
        )


class OracleSeerAgent(SeerAgent):
    """
    Oracle Seer that knows the correct answer.

    Used for validating the evaluation pipeline.
    Requires the correct message to be provided at creation.
    """

    def __init__(self, correct_message: str):
        """
        Initialize oracle Seer.

        Args:
            correct_message: The correct message to output
        """
        self.correct_message = correct_message

    def reset(self) -> None:
        """No state to reset."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "OracleSeerAgent",
            "correct_message": self.correct_message
        }

    def generate_message(self, input: SeerInput) -> SeerOutput:
        """Return the correct message."""
        return SeerOutput(
            message=self.correct_message,
            raw_response=None,
            parse_attempts=1,
            parse_error=False
        )


class OracleDoerAgent(DoerAgent):
    """
    Oracle Doer that knows the correct answer.

    Used for validating the evaluation pipeline.
    Requires a function that computes the correct output.
    """

    def __init__(self, compute_func):
        """
        Initialize oracle Doer.

        Args:
            compute_func: Function that takes test_input and returns correct output
        """
        self.compute_func = compute_func

    def reset(self) -> None:
        """No state to reset."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "OracleDoerAgent"
        }

    def generate_prediction(self, input: DoerInput) -> DoerOutput:
        """Return the correct prediction."""
        prediction = self.compute_func(input.test_input)

        return DoerOutput(
            prediction=prediction,
            raw_response=None,
            parse_attempts=1,
            parse_error=False
        )
