"""Tests for metacognitive agent data types."""
import pytest
import numpy as np
from chaosbench.legacy_v0.agents.metacognitive_types import (
    Feedback,
    AgentObservation,
    AgentAction,
    parse_action,
    BacktestFeedback,
)


class TestFeedback:
    def test_feedback_creation(self):
        fb = Feedback(prediction=0.5, actual=0.4, score=0.8)
        assert fb.prediction == 0.5
        assert fb.actual == 0.4
        assert fb.score == 0.8


class TestAgentObservation:
    def test_observation_creation(self):
        obs = AgentObservation(
            task_id=1,
            observations=np.array([0.1, 0.2, 0.3]),
            obs_times=np.array([0, 1, 2]),
            prediction_horizon=1,
            family="logistic",
            learnings="# My Learnings\n",
            last_feedback=None,
        )
        assert obs.task_id == 1
        assert len(obs.observations) == 3
        assert obs.family == "logistic"


class TestAgentAction:
    def test_predict_action(self):
        action = AgentAction(action="PREDICT", value=0.42)
        assert action.action == "PREDICT"
        assert action.value == 0.42

    def test_write_action(self):
        action = AgentAction(action="WRITE", text="Some insight")
        assert action.action == "WRITE"
        assert action.text == "Some insight"

    def test_delete_action(self):
        action = AgentAction(action="DELETE", section="## Old Section")
        assert action.action == "DELETE"
        assert action.section == "## Old Section"

    def test_move_on_action(self):
        action = AgentAction(action="MOVE_ON")
        assert action.action == "MOVE_ON"


class TestParseAction:
    def test_parse_predict(self):
        json_str = '{"action": "PREDICT", "value": 0.42}'
        action = parse_action(json_str)
        assert action.action == "PREDICT"
        assert action.value == 0.42

    def test_parse_write(self):
        json_str = '{"action": "WRITE", "text": "My note"}'
        action = parse_action(json_str)
        assert action.action == "WRITE"
        assert action.text == "My note"

    def test_parse_move_on(self):
        json_str = '{"action": "MOVE_ON"}'
        action = parse_action(json_str)
        assert action.action == "MOVE_ON"

    def test_parse_from_reasoning_with_json(self):
        """Agent writes reasoning then JSON — extract just the JSON."""
        response = """Looking at the pattern, it seems like a logistic map.
        The values cluster around 0.2 and 0.8, suggesting r > 3.5.

        {"action": "PREDICT", "value": 0.35}"""
        action = parse_action(response)
        assert action.action == "PREDICT"
        assert action.value == 0.35

    def test_parse_invalid_action_raises(self):
        with pytest.raises(ValueError):
            parse_action('{"action": "INVALID"}')

    def test_parse_no_json_raises(self):
        with pytest.raises(ValueError):
            parse_action("Just some text with no JSON")


class TestParseHypothesisActions:
    def test_parse_hypothesize(self):
        """Parse HYPOTHESIZE action."""
        response = '''I think this is logistic.
        {"action": "HYPOTHESIZE", "model": "logistic", "params": {"r": 3.85}}'''

        action = parse_action(response)

        assert action.action == "HYPOTHESIZE"
        assert action.model == "logistic"
        assert action.params == {"r": 3.85}

    def test_parse_fit(self):
        """Parse FIT action."""
        response = '''Let me fit the model.
        {"action": "FIT", "model": "tent"}'''

        action = parse_action(response)

        assert action.action == "FIT"
        assert action.model == "tent"


class TestBacktestFeedback:
    def test_format_good_fit(self):
        """Format feedback for good fit."""
        fb = BacktestFeedback(
            model="logistic",
            params={"r": 3.9},
            mae=0.02,
            predicted_next=0.156,
        )
        text = fb.format()

        assert "logistic" in text
        assert "r=3.9" in text or "r: 3.9" in text
        assert "0.02" in text
        assert "0.156" in text

    def test_format_poor_fit(self):
        """Format feedback for poor fit."""
        fb = BacktestFeedback(
            model="logistic",
            params={"r": 3.5},
            mae=0.25,
            predicted_next=0.42,
        )
        text = fb.format()

        assert "doesn't reproduce" in text.lower() or "poor" in text.lower()
