"""Data types for the metacognitive agent protocol."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import json
import re


@dataclass
class Feedback:
    """What agent sees after PREDICT."""
    prediction: float
    actual: float
    score: float


@dataclass
class AgentObservation:
    """What the agent sees each turn."""
    task_id: int
    observations: np.ndarray
    obs_times: np.ndarray
    prediction_horizon: int
    family: str | None
    learnings: str
    last_feedback: Feedback | None


@dataclass
class AgentAction:
    """What agent can do."""
    action: Literal["PREDICT", "WRITE", "DELETE", "MOVE_ON", "HYPOTHESIZE", "FIT"]
    value: float | None = None
    text: str | None = None
    section: str | None = None
    model: str | None = None  # For HYPOTHESIZE/FIT
    params: dict | None = None  # For HYPOTHESIZE


@dataclass
class BacktestFeedback:
    """Feedback from testing a hypothesis."""
    model: str
    params: dict
    mae: float
    predicted_next: float

    def format(self) -> str:
        """Format as human-readable feedback."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        quality = "fits well" if self.mae < 0.05 else "doesn't reproduce the observations well"

        return f"""Model: {self.model} ({params_str})

Backtest (fitting x_0 → x_49):
  MAE: {self.mae:.3f}
  Your model {quality}.

If you trust this model, it predicts x_50 = {self.predicted_next:.4f}"""


def parse_action(response: str) -> AgentAction:
    """Parse an AgentAction from LLM response.

    Extracts JSON from response text (agent may write reasoning before JSON).

    Raises:
        ValueError: If no valid action JSON found.
    """
    # Find JSON object in response - handle nested objects for params
    json_match = re.search(r'\{[^{}]*"action"[^{}]*(?:\{[^{}]*\})?[^{}]*\}', response)
    if not json_match:
        # Try simpler pattern
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response)
    if not json_match:
        raise ValueError(f"No JSON action found in response: {response[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    action_type = data.get("action")
    valid_actions = ("PREDICT", "WRITE", "DELETE", "MOVE_ON", "HYPOTHESIZE", "FIT")
    if action_type not in valid_actions:
        raise ValueError(f"Invalid action type: {action_type}")

    return AgentAction(
        action=action_type,
        value=data.get("value"),
        text=data.get("text"),
        section=data.get("section"),
        model=data.get("model"),
        params=data.get("params"),
    )
