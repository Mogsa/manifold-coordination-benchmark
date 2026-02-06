#!/usr/bin/env python3
"""Quick test of hypothesis-driven agent with Gemini."""
from pathlib import Path

from shared.llm_utils import call_llm
from chaosbench.legacy_v0.experiments.session import SessionRunner, SessionConfig
from chaosbench.legacy_v0.agents.metacognitive_types import AgentObservation, AgentAction, parse_action, BacktestFeedback


class HypothesisAgent:
    """Agent that can use HYPOTHESIZE and FIT actions."""

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        self.model = model
        self.system_prompt = self._load_prompt()
        self.messages = []

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "hypothesis_system.txt"
        return prompt_path.read_text()

    def _format_observation(self, obs: AgentObservation) -> str:
        lines = []
        lines.append(f"## Task {obs.task_id}")

        # Show all 50 observations compactly
        flat_obs = obs.observations.flatten()
        # First 10
        first = ", ".join(f"{x:.4f}" for x in flat_obs[:10])
        # Last 5
        last = ", ".join(f"{x:.4f}" for x in flat_obs[-5:])
        lines.append(f"**Observations (x_0 to x_49):**")
        lines.append(f"[{first}, ..., {last}]")
        lines.append(f"(50 values total, range [{flat_obs.min():.3f}, {flat_obs.max():.3f}])")

        # Backtest feedback if present
        if hasattr(obs, 'last_backtest') and obs.last_backtest:
            bt = obs.last_backtest
            lines.append("")
            lines.append(bt.format())

        # Previous prediction feedback
        if obs.last_feedback:
            lines.append("")
            lines.append("**Previous prediction result:**")
            lines.append(f"- Your prediction: {obs.last_feedback.prediction:.4f}")
            lines.append(f"- Actual x_50: {obs.last_feedback.actual:.4f}")
            lines.append(f"- Score: {obs.last_feedback.score:.3f}")

        # Learnings
        lines.append("")
        lines.append("---")
        lines.append("**Your Learnings:**")
        lines.append(obs.learnings if obs.learnings else "(empty)")

        return "\n".join(lines)

    def __call__(self, obs: AgentObservation) -> tuple[str, AgentAction]:
        user_msg = self._format_observation(obs)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
            {"role": "user", "content": user_msg},
        ]

        response = call_llm(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        # Keep conversation history
        self.messages.append({"role": "user", "content": user_msg})
        self.messages.append({"role": "assistant", "content": response})

        # Parse action (retry once if missing)
        try:
            action = parse_action(response)
        except ValueError:
            # Gemini forgot the action - prompt for it
            retry_msg = "You must end with a JSON action. What action do you take?"
            self.messages.append({"role": "assistant", "content": response})
            self.messages.append({"role": "user", "content": retry_msg})

            response2 = call_llm(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                temperature=0.7,
                max_tokens=500,
            )
            self.messages.append({"role": "assistant", "content": response2})
            action = parse_action(response2)
            response = response + "\n" + response2

        # Extract reasoning (everything before the JSON)
        reasoning = response.split("{")[0].strip() if "{" in response else response

        return reasoning, action


def main():
    print("Testing hypothesis-driven agent with Gemini...")
    print("=" * 60)

    agent = HypothesisAgent()
    config = SessionConfig(n_tasks=2, timeout_seconds=120, conditional=False)
    runner = SessionRunner(config)

    result = runner.run(agent)

    print("=" * 60)
    print(f"Tasks completed: {result.tasks_completed}")
    print(f"Final Phi: {result.final_phi:.3f}")
    print()
    print("TRACE:")
    print("-" * 60)
    print(result.trace.to_markdown())


if __name__ == "__main__":
    main()
