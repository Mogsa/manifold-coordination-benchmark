"""LLM-based metacognitive agent for ChaosBench."""
from pathlib import Path
from typing import Tuple

from shared.llm_utils import call_llm
from chaosbench.legacy_v0.agents.metacognitive_types import (
    AgentObservation,
    AgentAction,
    parse_action,
)


class MetacognitiveAgent:
    """LLM agent with persistent learnings."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        scaffolded: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.scaffolded = scaffolded
        self.system_prompt = self._load_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _load_system_prompt(self) -> str:
        """Load the appropriate system prompt."""
        prompt_dir = Path(__file__).parent.parent / "prompts"
        if self.scaffolded:
            prompt_file = prompt_dir / "hypothesis_system.txt"
        else:
            prompt_file = prompt_dir / "mvp_system.txt"
        if prompt_file.exists():
            return prompt_file.read_text()
        # Fallback inline prompt
        return """You are a scientist studying unknown dynamical systems.
Predict future states from observations. Output JSON actions."""

    def _format_observation(self, obs: AgentObservation) -> str:
        """Format observation as user message."""
        lines = []

        lines.append(f"## Task {obs.task_id}")
        if obs.family:
            lines.append(f"**System family:** {obs.family}")

        # Format observations compactly (flatten for 2D arrays from ChaosBench)
        flat_obs = obs.observations.flatten()[:10]
        obs_str = ", ".join(f"{x:.3f}" for x in flat_obs)
        if len(obs.observations) > 10:
            obs_str += f", ... ({len(obs.observations)} total)"
        lines.append(f"**Observations:** [{obs_str}]")
        lines.append(f"**Predict:** x_{len(obs.observations)}")

        # Include feedback if present
        if obs.last_feedback:
            lines.append("")
            lines.append("**Last attempt:**")
            lines.append(f"- Your prediction: {obs.last_feedback.prediction:.3f}")
            lines.append(f"- Actual value: {obs.last_feedback.actual:.3f}")
            lines.append(f"- Score: {obs.last_feedback.score:.2f}")

        # Include backtest feedback if present
        if obs.last_backtest:
            lines.append("")
            lines.append("**Backtest result:**")
            lines.append(obs.last_backtest.format())

        # Include learnings
        lines.append("")
        lines.append("---")
        lines.append("**Your Learnings:**")
        lines.append(obs.learnings)

        return "\n".join(lines)

    def __call__(self, obs: AgentObservation) -> Tuple[str, AgentAction]:
        """Get action from LLM.

        Returns:
            Tuple of (reasoning, action)
        """
        user_message = self._format_observation(obs)

        # Build messages for this turn
        messages = self.messages + [{"role": "user", "content": user_message}]

        # Call LLM
        response = call_llm(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse action from response
        action = parse_action(response)

        # Extract reasoning (everything before the JSON)
        reasoning = response.split('{"action"')[0].strip()

        return reasoning, action

    def reset(self) -> None:
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
