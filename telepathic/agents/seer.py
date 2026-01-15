"""
LLM Seer agent for the Telepathic benchmark.

The Seer observes function samples and produces a token message.
"""

import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import SeerAgent, SeerInput, SeerOutput
from ..core.protocol import Protocol, validate_message

# Try to import shared LLM utilities
try:
    from shared.llm_utils import call_llm
except ImportError:
    # Fallback if running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.llm_utils import call_llm


# Default paths to prompt files
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()


class LLMSeerAgent(SeerAgent):
    """
    LLM-based Seer agent.

    Uses few-shot examples to teach token meanings, then asks
    the LLM to encode new functions.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        temperature: float = 0.7,
        max_retries: int = 3,
        zero_shot: bool = False,
        verbose: bool = False
    ):
        """
        Initialize Seer agent.

        Args:
            model: LLM model identifier (LiteLLM format)
            temperature: Sampling temperature
            max_retries: Max attempts to get valid output
            zero_shot: If True, use zero-shot prompts (no examples)
            verbose: Print debug info
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.zero_shot = zero_shot
        self.verbose = verbose

        # Load prompts
        if zero_shot:
            self.system_prompt = load_prompt("seer_system_zero.txt")
            self.trial_template = load_prompt("seer_trial_zero.txt")
        else:
            self.system_prompt = load_prompt("seer_system.txt")
            self.trial_template = load_prompt("seer_trial.txt")

        self.protocol = Protocol()

    def reset(self) -> None:
        """Reset agent state."""
        pass  # Stateless agent

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "LLMSeerAgent",
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "zero_shot": self.zero_shot
        }

    def _format_trial_prompt(self, samples: List[tuple]) -> str:
        """Format the trial prompt with sample values."""
        # samples is list of (x, y) tuples
        # Template expects {y0}, {y1}, {y2}, {y3}, {y4}
        prompt = self.trial_template
        for i, (x, y) in enumerate(samples):
            prompt = prompt.replace(f"{{y{i}}}", f"{y:.4f}")
        return prompt

    def _parse_message(self, response: str) -> Optional[str]:
        """
        Extract message from LLM response.

        Looks for "MESSAGE: <tokens>" pattern.
        Returns None if no valid message found.
        """
        # Look for MESSAGE: pattern (case-insensitive)
        match = re.search(r'MESSAGE:\s*([^\n]+)', response, re.IGNORECASE)

        if not match:
            return None

        message = match.group(1).strip()

        # Clean up common issues
        # Remove any trailing punctuation
        message = message.rstrip('.,;:')

        # Remove quotes if present
        message = message.strip('"\'')

        # Validate the message
        if validate_message(message):
            return message

        # Try to salvage: extract just the Greek letters
        tokens = re.findall(r'[αβγδε]', message)
        if tokens:
            salvaged = "-".join(tokens)
            if validate_message(salvaged):
                return salvaged

        return None

    def _get_clarification_prompt(self, original_response: str) -> str:
        """Generate a prompt asking for clarification."""
        return (
            f"Your previous response was:\n{original_response}\n\n"
            "I could not extract a valid message. Please respond with ONLY:\n"
            "MESSAGE: <tokens>\n\n"
            "Where <tokens> is one or more of α, β, γ, δ, ε separated by hyphens.\n"
            "For example: MESSAGE: α or MESSAGE: α-γ"
        )

    def generate_message(self, input: SeerInput) -> SeerOutput:
        """
        Generate a message encoding the function.

        Args:
            input: SeerInput with samples, vocabulary, max_tokens

        Returns:
            SeerOutput with message and metadata
        """
        trial_prompt = self._format_trial_prompt(input.samples)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": trial_prompt}
        ]

        parse_error = False
        attempts = 0
        last_response = None

        for attempt in range(self.max_retries):
            attempts = attempt + 1

            try:
                response = call_llm(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=500
                )
                last_response = response

                if self.verbose:
                    print(f"[Seer attempt {attempts}] Response: {response[:200]}...")

                message = self._parse_message(response)

                if message:
                    return SeerOutput(
                        message=message,
                        raw_response=response,
                        parse_attempts=attempts,
                        parse_error=parse_error
                    )

                # Parsing failed - ask for clarification
                parse_error = True
                if self.verbose:
                    print(f"[Seer attempt {attempts}] Parse failed, asking for clarification")

                # Add the response and clarification request
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": self._get_clarification_prompt(response)
                })

            except Exception as e:
                if self.verbose:
                    print(f"[Seer attempt {attempts}] Error: {e}")
                parse_error = True

        # All retries exhausted - return fallback
        # Use first token as fallback
        fallback_message = input.vocabulary[0] if input.vocabulary else "α"
        if self.verbose:
            print(f"[Seer] All retries exhausted, using fallback: {fallback_message}")

        return SeerOutput(
            message=fallback_message,
            raw_response=last_response,
            parse_attempts=attempts,
            parse_error=True
        )
