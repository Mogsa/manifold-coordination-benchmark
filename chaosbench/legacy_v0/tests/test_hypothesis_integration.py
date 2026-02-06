"""Integration test for hypothesis-driven session."""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from chaosbench.legacy_v0.experiments.session import SessionRunner, SessionConfig
from chaosbench.legacy_v0.agents.metacognitive_types import AgentAction


class MockHypothesisAgent:
    """Mock agent that uses HYPOTHESIZE before PREDICT."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, observation):
        self.call_count += 1

        # First call: hypothesize
        if self.call_count == 1:
            return "Let me test logistic", AgentAction(
                action="HYPOTHESIZE",
                model="logistic",
                params={"r": 3.9}
            )
        # Second call: predict based on hypothesis
        elif self.call_count == 2:
            return "Using model prediction", AgentAction(
                action="PREDICT",
                value=0.42
            )
        # Third call: move on
        else:
            return "Done", AgentAction(action="MOVE_ON")


class TestHypothesisIntegration:
    def test_full_hypothesis_flow(self):
        """Agent can hypothesize, predict, and complete task."""
        agent = MockHypothesisAgent()
        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        with patch.object(runner, '_generate_tasks') as mock_gen:
            r = 3.9
            obs = np.zeros(50)
            obs[0] = 0.3
            for i in range(1, 50):
                obs[i] = r * obs[i-1] * (1 - obs[i-1])

            mock_task = Mock()
            mock_task.task_id = 1
            mock_task.system.family = "logistic"
            mock_task.h_ks = 0.5
            mock_task.observations = obs
            mock_task.obs_times = np.arange(50)
            mock_task.future_time = 1
            mock_task.true_future = np.array([r * obs[-1] * (1 - obs[-1])])
            mock_gen.return_value = [mock_task]

            result = runner.run(agent)

        assert result.tasks_completed == 1
        trace = result.trace.to_markdown()
        assert "HYPOTHESIZE" in trace
        assert "PREDICT" in trace
        assert "MOVE_ON" in trace


class TestFitIntegration:
    def test_full_fit_flow(self):
        """Agent can FIT, predict, and complete task."""

        class MockFitAgent:
            def __init__(self):
                self.call_count = 0

            def __call__(self, obs):
                self.call_count += 1
                if self.call_count == 1:
                    return "Fitting", AgentAction(action="FIT", model="logistic")
                elif self.call_count == 2:
                    return "Predicting", AgentAction(action="PREDICT", value=0.5)
                else:
                    return "Done", AgentAction(action="MOVE_ON")

        agent = MockFitAgent()
        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        with patch.object(runner, '_generate_tasks') as mock_gen:
            r = 3.85
            obs = np.zeros(50)
            obs[0] = 0.3
            for i in range(1, 50):
                obs[i] = r * obs[i-1] * (1 - obs[i-1])

            mock_task = Mock()
            mock_task.task_id = 1
            mock_task.system.family = "logistic"
            mock_task.h_ks = 0.5
            mock_task.observations = obs
            mock_task.obs_times = np.arange(50)
            mock_task.future_time = 1
            mock_task.true_future = np.array([r * obs[-1] * (1 - obs[-1])])
            mock_gen.return_value = [mock_task]

            result = runner.run(agent)

        assert result.tasks_completed == 1
        trace = result.trace.to_markdown()
        assert "FIT" in trace
