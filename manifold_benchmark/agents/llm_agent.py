"""
LLM agent module for the Manifold Coordination Benchmark.

This module implements an LLM-based agent that uses OpenAI or Anthropic APIs
to make decisions based on observations and communication with the other agent.
"""

from manifold_benchmark.agents.base import BaseAgent
import os
import re
import time
from typing import List, Dict, Optional
from pathlib import Path


class LLMAgent(BaseAgent):
    """LLM-based agent using OpenAI or Anthropic APIs."""

    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30

    def __init__(
        self,
        role: str,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        system_prompt_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        domain_size: float = 10.0
    ):
        """
        Initialize an LLM agent.

        Args:
            role: 'A' (controls x) or 'B' (controls y)
            model: Model identifier (e.g., "gpt-4", "gpt-4o", "claude-3-opus-20240229")
            api_key: API key (or reads from environment variable)
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

        # Initialize API client
        self._init_api_client(api_key)

        # Load system prompt
        self._load_system_prompt(system_prompt_path)

        # Track current position
        self.current_position = 5.0  # Start at center

    def _init_api_client(self, api_key: Optional[str] = None) -> None:
        """
        Initialize OpenAI or Anthropic client based on model.

        Args:
            api_key: API key (or reads from environment variable)

        Raises:
            ValueError: If model type cannot be determined
        """
        model_lower = self.model.lower()

        if 'gpt' in model_lower or 'o1' in model_lower:
            # OpenAI
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or os.environ.get('OPENAI_API_KEY')
            )
            self.api_type = 'openai'
        elif 'claude' in model_lower:
            # Anthropic
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get('ANTHROPIC_API_KEY')
            )
            self.api_type = 'anthropic'
        else:
            raise ValueError(f"Unknown model type: {self.model}. "
                           f"Model name must contain 'gpt', 'o1' (OpenAI) or 'claude' (Anthropic)")

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
        Build message list for API call.

        Args:
            include_decision: Whether to add decision request at end

        Returns:
            List of message dicts with role and content keys
        """
        messages = []

        # For OpenAI, system prompt is part of messages
        # For Anthropic, it's passed separately, but we still build messages list
        if self.api_type == 'openai':
            messages.append({"role": "system", "content": self.system_prompt})

        # Add observation and message history
        for i, obs in enumerate(self.observation_history):
            obs_text = f"=== Turn {i + 1} Observation ===\n\n"
            obs_text += self._format_observation(obs)

            # Add other agent's message if available
            if i < len(self.message_history):
                other_msg = self.message_history[i]
                if other_msg:  # Only add if non-empty
                    obs_text += f"\n\nMessage from other agent:\n{other_msg}\n"

            messages.append({"role": "user", "content": obs_text})

        # Add decision request if needed
        if include_decision:
            coord_name = 'x' if self.role == 'A' else 'y'
            decision_prompt = (
                f"Based on all observations and messages, "
                f"what {coord_name}-coordinate do you choose for your next position? "
                f"Respond with a single number between 0 and {self.domain_size}."
            )
            messages.append({"role": "user", "content": decision_prompt})

        return messages

    def _parse_coordinate(self, response: str) -> float:
        """
        Extract coordinate from LLM response.

        Handles various formats:
        - "7.5"
        - "My answer is 7.5"
        - "gradient 0.12, move to 6.8" -> 6.8 (takes LAST number)
        - "approximately 7 to 8, let's say 7.5" -> 7.5
        - "x=6.5" -> 6.5
        - Values outside domain are clamped

        Args:
            response: Raw text response from LLM

        Returns:
            Parsed coordinate value in [0, domain_size]

        Raises:
            ValueError: If no numbers found in response
        """
        # Find all numbers in response (including decimals and negatives)
        matches = re.findall(r'[-+]?\d*\.?\d+', response)

        if not matches:
            raise ValueError(f"Could not parse coordinate from: {response[:200]}")

        # Take the last number (usually the final answer)
        value = float(matches[-1])

        # Clamp to valid domain
        value = max(0.0, min(self.domain_size, value))

        return value

    def _call_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> str:
        """
        Call LLM API with retry logic and exponential backoff.

        Args:
            messages: List of message dicts to send
            retry_count: Current retry attempt number

        Returns:
            Response text from LLM

        Raises:
            Exception: If all retries exhausted
        """
        try:
            if self.api_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.TIMEOUT_SECONDS
                )
                return response.choices[0].message.content

            else:  # anthropic
                # For Anthropic, system prompt goes in separate parameter
                # Remove system messages from the messages list
                non_system_messages = [m for m in messages if m['role'] != 'system']

                response = self.client.messages.create(
                    model=self.model,
                    messages=non_system_messages,
                    system=self.system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.TIMEOUT_SECONDS
                )
                return response.content[0].text

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
            new_position = self._parse_coordinate(response)
            return new_position

        except ValueError as e:
            # Parsing failed - try with clarifying prompt
            print(f"Parsing error: {e}. Retrying with clarifying prompt...")
            messages.append({
                "role": "assistant",
                "content": response
            })
            messages.append({
                "role": "user",
                "content": f"Please respond with just a single number between 0 and {self.domain_size}."
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
        final_prompt = f"""
=== FINAL DECISION ===

You have completed all exploration turns.

Based on all your observations and the conversation with the other agent,
what is your final answer for the {coord_name}-coordinate of the global maximum?

Provide your reasoning, then state your final {coord_name} value
(a single number between 0 and {self.domain_size}).
"""
        messages.append({"role": "user", "content": final_prompt})

        try:
            response = self._call_api(messages)
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
