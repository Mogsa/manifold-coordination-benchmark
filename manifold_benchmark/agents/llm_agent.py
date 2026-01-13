"""
LLM agent module for the Manifold Coordination Benchmark.

This module implements an LLM-based agent that uses LiteLLM to support
100+ LLM providers including OpenAI, Anthropic, Google Gemini, and more.
"""

from manifold_benchmark.agents.base import BaseAgent
import os
import re
import time
from typing import List, Dict, Optional
from pathlib import Path
from litellm import completion
import litellm


class LLMAgent(BaseAgent):
    """LLM-based agent using LiteLLM (supports 100+ models)."""

    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30

    def __init__(
        self,
        role: str,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        system_prompt_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        domain_size: float = 10.0
    ):
        """
        Initialize an LLM agent using LiteLLM.

        Args:
            role: 'A' (controls x) or 'B' (controls y)
            model: Model identifier. Examples:
                   - OpenAI: "gpt-4", "gpt-3.5-turbo"
                   - Anthropic: "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"
                   - Google: "gemini/gemini-2.0-flash-exp", "gemini/gemini-1.5-pro"
                   - Full list: https://docs.litellm.ai/docs/providers
            api_key: API key (or reads from environment variable)
                     - OPENAI_API_KEY for OpenAI models
                     - ANTHROPIC_API_KEY for Anthropic models
                     - GOOGLE_API_KEY or GEMINI_API_KEY for Google models
            system_prompt_path: Path to system prompt file (default: prompts/agent_{role}_system.txt)
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens per response
            domain_size: Domain bounds [0, domain_size]
        """
        super().__init__(role)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.domain_size = domain_size

        # Set API key if provided (LiteLLM will auto-detect from env otherwise)
        if api_key:
            # Set the appropriate environment variable based on model
            if 'gemini' in model.lower():
                os.environ['GEMINI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            elif 'gpt' in model.lower() or 'o1' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key

        # Load system prompt
        self._load_system_prompt(system_prompt_path)

        # Track current position
        self.current_position = 5.0  # Start at center

        # Store last reasoning for transcript capture
        self.last_reasoning = ""

    def _load_system_prompt(self, prompt_path: Optional[str] = None) -> None:
        """
        Load system prompt from file.

        Args:
            prompt_path: Path to system prompt file (uses default if None)
        """
        if prompt_path is None:
            # Default path based on role - relative to package directory
            prompt_file = f"agent_{self.role.lower()}_system.txt"
            # Get the package directory (parent of agents/)
            package_dir = Path(__file__).parent.parent
            prompt_path = package_dir / "prompts" / prompt_file

        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()

    def _format_observation(self, observation: dict) -> str:
        """
        Convert observation dict to human-readable text.

        Args:
            observation: Observation dict from ObservationGenerator

        Returns:
            Formatted string describing the observation
        """
        pos = observation['position']
        value = observation['value_at_position']

        # Format position info
        text = f"Current position: ({pos['x']:.2f}, {pos['y']:.2f})\n"
        text += f"Surface value at this position: {value:.3f}\n\n"

        # Format gradient
        if self.role == 'A':
            grad = observation.get('gradient_x', 0.0)
            text += f"Gradient in x-direction: df/dx = {grad:.3f}\n\n"
        else:  # role == 'B'
            grad = observation.get('gradient_y', 0.0)
            text += f"Gradient in y-direction: df/dy = {grad:.3f}\n\n"

        # Format slice
        slice_data = observation.get('slice', [])
        coord_key = 'x' if self.role == 'A' else 'y'

        text += f"Your {coord_key}-slice (values along your axis):\n"
        for sample in slice_data:
            coord = sample.get(coord_key, 0.0)
            val = sample.get('value', 0.0)
            text += f"  {coord_key}={coord:.2f}: f={val:.3f}\n"

        return text

    def _build_prompt(self, include_decision: bool = False) -> List[Dict[str, str]]:
        """
        Build message list for API call (LiteLLM uses OpenAI-style format).

        Optimized for token efficiency:
        - Previous turns: summarized as position + value only
        - Current turn: full observation details

        Args:
            include_decision: Whether to add decision request at end

        Returns:
            List of message dicts with role and content keys
        """
        messages = []

        # System prompt
        messages.append({"role": "system", "content": self.system_prompt})

        n_obs = len(self.observation_history)

        # Summarize all previous turns (not the current one)
        if n_obs > 1:
            history_lines = ["Previous positions and values:"]
            for i in range(n_obs - 1):
                obs = self.observation_history[i]
                pos = obs['position']
                val = obs['value_at_position']
                history_lines.append(f"  Turn {i+1}: ({pos['x']:.1f}, {pos['y']:.1f}) -> value={val:.3f}")

            messages.append({"role": "user", "content": "\n".join(history_lines)})

        # Add current turn's full observation (the last one)
        if n_obs > 0:
            i = n_obs - 1
            obs = self.observation_history[i]
            obs_text = f"=== Turn {i + 1} (Current) ===\n\n"
            obs_text += self._format_observation(obs)

            # Add other agent's message if available (truncated)
            if i < len(self.message_history):
                other_msg = self.message_history[i]
                if other_msg:
                    # Truncate long messages
                    if len(other_msg) > 150:
                        other_msg = other_msg[:150] + "..."
                    obs_text += f"\n\nMessage from other agent:\n{other_msg}\n"

            messages.append({"role": "user", "content": obs_text})

        # Add decision request if needed
        if include_decision:
            coord_name = 'x' if self.role == 'A' else 'y'
            current_pos = self.current_position
            decision_prompt = (
                f"DECISION: Where should {coord_name} be next?\n"
                f"Current {coord_name} = {current_pos:.1f}, range [0-{self.domain_size}]\n\n"
                f"BE VERY BRIEF (1 sentence max), then state:\n"
                f"MY_POSITION: [number]\n\n"
                f"Example: Gradient positive, moving right. MY_POSITION: 6.5"
            )
            messages.append({"role": "user", "content": decision_prompt})

        return messages

    def _parse_coordinate(self, response: str) -> float:
        """
        Extract coordinate from LLM response.

        Parsing priority:
        1. Look for "MY_POSITION: X" pattern (most reliable)
        2. Look for "x = X" or "y = X" patterns
        3. Look for "position: X" or "move to X" patterns
        4. Fall back to last valid number in range [0, domain_size]
        5. Last resort: last number (clamped)

        Args:
            response: Raw text response from LLM

        Returns:
            Parsed coordinate value in [0, domain_size]

        Raises:
            ValueError: If no numbers found in response
        """
        # Helper to safely convert to float
        def safe_float(s):
            try:
                # Remove trailing periods that might cause issues
                s = s.rstrip('.')
                return float(s)
            except (ValueError, AttributeError):
                return None

        # Priority 1: Look for MY_POSITION: pattern (most reliable)
        # Match MY_POSITION: followed by a number (integer or decimal)
        my_pos_match = re.search(r'MY_POSITION:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if my_pos_match:
            value = safe_float(my_pos_match.group(1))
            if value is not None:
                return max(0.0, min(self.domain_size, value))

        # Priority 2: Look for explicit coordinate assignments like "x = 6.5" or "y=7"
        coord_name = 'x' if self.role == 'A' else 'y'
        coord_match = re.search(rf'{coord_name}\s*=\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if coord_match:
            value = safe_float(coord_match.group(1))
            if value is not None and 0 <= value <= self.domain_size:
                return value

        # Priority 3: Look for "move to X" or "position X" patterns
        move_match = re.search(r'(?:move to|position|choose|select|go to)\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if move_match:
            value = safe_float(move_match.group(1))
            if value is not None and 0 <= value <= self.domain_size:
                return value

        # Priority 4: Find all properly formatted numbers
        # This regex matches integers and decimals but not malformed ones like "0.000."
        all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)

        if not all_numbers:
            raise ValueError(f"Could not parse coordinate from: {response[:200]}")

        # Convert to floats, filtering out any that fail conversion
        float_numbers = []
        for n in all_numbers:
            val = safe_float(n)
            if val is not None:
                float_numbers.append(val)

        if not float_numbers:
            raise ValueError(f"Could not parse coordinate from: {response[:200]}")

        # Filter to numbers in valid domain range [0, 10]
        valid_numbers = [n for n in float_numbers if 0 <= n <= self.domain_size]

        # Prefer numbers that look like reasonable positions (not gradients/values)
        # Gradients are usually small decimals like 0.069, positions are usually larger
        position_candidates = [n for n in valid_numbers if n >= 1.0 or n == 0.0]

        if position_candidates:
            return position_candidates[-1]
        elif valid_numbers:
            return valid_numbers[-1]
        else:
            # Last resort - take last number and clamp
            return max(0.0, min(self.domain_size, float_numbers[-1]))

    def _call_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> str:
        """
        Call LLM API using LiteLLM with retry logic and exponential backoff.

        LiteLLM automatically handles provider differences (OpenAI, Anthropic, Google, etc.)

        Args:
            messages: List of message dicts to send
            retry_count: Current retry attempt number

        Returns:
            Response text from LLM

        Raises:
            Exception: If all retries exhausted
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.TIMEOUT_SECONDS
            )
            return response.choices[0].message.content

        except Exception as e:
            if retry_count < self.MAX_RETRIES:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** retry_count
                print(f"API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self._call_api(messages, retry_count + 1)
            else:
                print(f"API failed after {self.MAX_RETRIES} retries: {e}")
                raise

    def generate_message(self) -> str:
        """
        Generate message to send to other agent.

        Returns:
            Message string describing observations and reasoning
        """
        messages = self._build_prompt(include_decision=False)

        # Add message generation request
        messages.append({
            "role": "user",
            "content": "What message would you like to send to the other agent about your observations?"
        })

        try:
            response = self._call_api(messages)
            return response.strip()
        except Exception as e:
            print(f"Error generating message: {e}")
            return ""  # Return empty message on failure

    def decide_position(self) -> float:
        """
        Decide next position using LLM.

        Returns:
            New position coordinate in [0, domain_size]
        """
        # Update current position from latest observation
        if self.observation_history:
            obs = self.observation_history[-1]
            key = 'x' if self.role == 'A' else 'y'
            self.current_position = obs['position'][key]

        messages = self._build_prompt(include_decision=True)

        try:
            response = self._call_api(messages)
            self.last_reasoning = response  # Capture full response
            new_position = self._parse_coordinate(response)
            return new_position

        except ValueError as e:
            # Parsing failed - try with clarifying prompt
            print(f"Parsing error: {e}. Retrying with clarifying prompt...")
            messages.append({
                "role": "assistant",
                "content": response
            })
            coord_name = 'x' if self.role == 'A' else 'y'
            messages.append({
                "role": "user",
                "content": f"I couldn't parse your position. Please respond with ONLY:\nMY_POSITION: [your {coord_name} value]\n\nFor example: MY_POSITION: 6.5"
            })

            try:
                response = self._call_api(messages)
                new_position = self._parse_coordinate(response)
                return new_position
            except Exception:
                # Complete failure - return current position (no movement)
                print("Failed to parse coordinate. Staying at current position.")
                return self.current_position

        except Exception as e:
            # API failure - return current position
            print(f"API error in decide_position: {e}")
            return self.current_position

    def final_decision(self) -> float:
        """
        Make final position decision after all turns.

        Returns:
            Final position coordinate in [0, domain_size]
        """
        messages = self._build_prompt(include_decision=False)

        # Add final decision prompt
        coord_name = 'x' if self.role == 'A' else 'y'
        final_prompt = (
            f"=== FINAL DECISION ===\n"
            f"Based on all observations, what is your final {coord_name} for the global maximum?\n\n"
            f"BE VERY BRIEF (1-2 sentences), then state:\n"
            f"MY_POSITION: [number]"
        )
        messages.append({"role": "user", "content": final_prompt})

        try:
            response = self._call_api(messages)
            self.last_reasoning = response  # Capture full response
            final_pos = self._parse_coordinate(response)
            return final_pos
        except Exception as e:
            print(f"Error in final_decision: {e}")
            # Fallback to current position
            return self.current_position

    def reset(self) -> None:
        """Reset agent state for new episode."""
        super().reset()
        self.current_position = 5.0  # Reset to center
        self.last_reasoning = ""
