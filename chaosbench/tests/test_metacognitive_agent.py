"""Tests for LLM-based metacognitive agent."""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from chaosbench.agents.metacognitive_agent import MetacognitiveAgent
from chaosbench.agents.metacognitive_types import AgentObservation, Feedback


class TestMetacognitiveAgent:
    def test_format_observation_message(self):
        """Test that observations are formatted correctly."""
        agent = MetacognitiveAgent(model="gemini/gemini-2.0-flash")

        obs = AgentObservation(
            task_id=1,
            observations=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            obs_times=np.array([0, 1, 2, 3, 4]),
            prediction_horizon=1,
            family="logistic",
            learnings="# My Learnings\n",
            last_feedback=None,
        )

        message = agent._format_observation(obs)

        assert "Task 1" in message
        assert "logistic" in message
        assert "0.1" in message
        assert "predict" in message.lower()

    def test_format_with_feedback(self):
        """Test that feedback is included when present."""
        agent = MetacognitiveAgent(model="gemini/gemini-2.0-flash")

        obs = AgentObservation(
            task_id=1,
            observations=np.array([0.1, 0.2, 0.3]),
            obs_times=np.array([0, 1, 2]),
            prediction_horizon=1,
            family="logistic",
            learnings="# My Learnings\n",
            last_feedback=Feedback(prediction=0.5, actual=0.3, score=0.6),
        )

        message = agent._format_observation(obs)

        assert "0.5" in message  # prediction
        assert "0.3" in message  # actual
        assert "0.6" in message  # score

    @patch('chaosbench.agents.metacognitive_agent.call_llm')
    def test_call_returns_action(self, mock_llm):
        """Test that agent call returns reasoning and action."""
        mock_llm.return_value = '''Looking at the pattern, values oscillate.

{"action": "PREDICT", "value": 0.42}'''

        agent = MetacognitiveAgent(model="gemini/gemini-2.0-flash")

        obs = AgentObservation(
            task_id=1,
            observations=np.array([0.1, 0.2, 0.3]),
            obs_times=np.array([0, 1, 2]),
            prediction_horizon=1,
            family="logistic",
            learnings="# My Learnings\n",
            last_feedback=None,
        )

        reasoning, action = agent(obs)

        assert "oscillate" in reasoning
        assert action.action == "PREDICT"
        assert action.value == 0.42
