"""
LLM Doer agent for the Telepathic benchmark.

The Doer receives a token message and computes the function output.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional

from .base import DoerAgent, DoerInput, DoerOutput

# Try to import shared LLM utilities
try:
    from shared.llm_utils import call_llm
except ImportError:
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


class LLMDoerAgent(DoerAgent):
    """
    LLM-based Doer agent.

    Uses few-shot examples to learn token meanings, then decodes
    messages and computes function outputs.
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
        Initialize Doer agent.

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
            self.system_prompt = load_prompt("doer_system_zero.txt")
            self.trial_template = load_prompt("doer_trial_zero.txt")
        else:
            self.system_prompt = load_prompt("doer_system.txt")
            self.trial_template = load_prompt("doer_trial.txt")

    def reset(self) -> None:
        """Reset agent state."""
        pass  # Stateless agent

    def get_config(self) -> Dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "LLMDoerAgent",
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "zero_shot": self.zero_shot
        }

    def _format_trial_prompt(self, message: str, test_input: float) -> str:
        """Format the trial prompt with message and test input."""
        prompt = self.trial_template
        prompt = prompt.replace("{message}", message)
        prompt = prompt.replace("{x_test}", f"{test_input}")
        return prompt

    def _parse_prediction(self, response: str) -> Optional[float]:
        """
        Extract numerical prediction from LLM response.

        Looks for "PREDICTION: <number>" pattern.
        Returns None if no valid number found.
        """
        # Look for PREDICTION: pattern (case-insensitive)
        match = re.search(r'PREDICTION:\s*([^\n]+)', response, re.IGNORECASE)

        if match:
            value_str = match.group(1).strip()

            # Try to parse as float
            try:
                # Remove any trailing text
                value_str = value_str.split()[0] if value_str.split() else value_str
                # Remove common suffixes
                value_str = value_str.rstrip('.,;:')
                return float(value_str)
            except ValueError:
                pass

        # Fallback: try to find any number that looks like a result
        # Look for numbers with optional negative sign and decimal
        numbers = re.findall(r'-?\d+\.?\d*', response)

        if numbers:
            # Prefer numbers that appear near "=" or "output" or at the end
            # First, try numbers near key words
            for pattern in [r'=\s*(-?\d+\.?\d*)', r'output[:\s]+(-?\d+\.?\d*)',
                           r'result[:\s]+(-?\d+\.?\d*)', r'answer[:\s]+(-?\d+\.?\d*)']:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue

            # Last resort: take the last number (most likely to be final answer)
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _get_clarification_prompt(self, original_response: str) -> str:
        """Generate a prompt asking for clarification."""
        return (
            f"Your previous response was:\n{original_response}\n\n"
            "I could not extract a numerical prediction. Please respond with ONLY:\n"
            "PREDICTION: <number>\n\n"
            "Where <number> is your computed output value.\n"
            "For example: PREDICTION: 0.4794 or PREDICTION: -1.5"
        )

    def generate_prediction(self, input: DoerInput) -> DoerOutput:
        """
        Generate a prediction for the test input.

        Args:
            input: DoerInput with message, test_input, vocabulary

        Returns:
            DoerOutput with prediction and metadata
        """
        trial_prompt = self._format_trial_prompt(input.message, input.test_input)

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
                    print(f"[Doer attempt {attempts}] Response: {response[:200]}...")

                prediction = self._parse_prediction(response)

                if prediction is not None:
                    return DoerOutput(
                        prediction=prediction,
                        raw_response=response,
                        parse_attempts=attempts,
                        parse_error=parse_error
                    )

                # Parsing failed - ask for clarification
                parse_error = True
                if self.verbose:
                    print(f"[Doer attempt {attempts}] Parse failed, asking for clarification")

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": self._get_clarification_prompt(response)
                })

            except Exception as e:
                if self.verbose:
                    print(f"[Doer attempt {attempts}] Error: {e}")
                parse_error = True

        # All retries exhausted - return fallback
        fallback = 0.0
        if self.verbose:
            print(f"[Doer] All retries exhausted, using fallback: {fallback}")

        return DoerOutput(
            prediction=fallback,
            raw_response=last_response,
            parse_attempts=attempts,
            parse_error=True
        )
