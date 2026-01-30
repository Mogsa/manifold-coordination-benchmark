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
    action: Literal["PREDICT", "WRITE", "DELETE", "MOVE_ON"]
    value: float | None = None
    text: str | None = None
    section: str | None = None


def parse_action(response: str) -> AgentAction:
    """Parse an AgentAction from LLM response.

    Extracts JSON from response text (agent may write reasoning before JSON).

    Raises:
        ValueError: If no valid action JSON found.
    """
    # Find JSON object in response
    json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response)
    if not json_match:
        raise ValueError(f"No JSON action found in response: {response[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    action_type = data.get("action")
    if action_type not in ("PREDICT", "WRITE", "DELETE", "MOVE_ON"):
        raise ValueError(f"Invalid action type: {action_type}")

    return AgentAction(
        action=action_type,
        value=data.get("value"),
        text=data.get("text"),
        section=data.get("section"),
    )
