"""Tests for session runner."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from chaosbench.legacy_v0.experiments.session import SessionRunner, SessionConfig, SessionResult
from chaosbench.legacy_v0.agents.metacognitive_types import AgentAction


class MockAgent:
    """Mock agent for testing session runner."""
    def __init__(self, actions: list):
        self.actions = actions
        self.call_count = 0

    def __call__(self, observation):
        action = self.actions[self.call_count % len(self.actions)]
        self.call_count += 1
        return "Mock reasoning", action


class TestSessionRunner:
    def test_single_task_predict_and_move_on(self):
        """Agent predicts once, then moves on."""
        agent = MockAgent([
            AgentAction(action="PREDICT", value=0.5),
            AgentAction(action="MOVE_ON"),
        ])

        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        # Mock task generation
        with patch.object(runner, '_generate_tasks') as mock_gen:
            mock_task = Mock()
            mock_task.task_id = 1
            mock_task.system.family = "logistic"
            mock_task.h_ks = 0.5
            mock_task.observations = np.random.rand(50)
            mock_task.obs_times = np.arange(50)
            mock_task.future_time = 1
            mock_task.true_future = np.array([0.45])
            mock_task.discretizer.state_to_bin = Mock(return_value=5)
            mock_task.true_bin = 5
            mock_gen.return_value = [mock_task]

            result = runner.run(agent)

        assert result.tasks_completed == 1
        assert len(result.phi_curve) > 0

    def test_multiple_predictions_last_counts(self):
        """Multiple predictions — last one before MOVE_ON counts."""
        agent = MockAgent([
            AgentAction(action="PREDICT", value=0.3),  # First guess
            AgentAction(action="PREDICT", value=0.5),  # Better guess
            AgentAction(action="MOVE_ON"),
        ])

        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        with patch.object(runner, '_generate_tasks') as mock_gen:
            mock_task = Mock()
            mock_task.task_id = 1
            mock_task.system.family = "logistic"
            mock_task.h_ks = 0.5
            mock_task.observations = np.random.rand(50)
            mock_task.obs_times = np.arange(50)
            mock_task.future_time = 1
            mock_task.true_future = np.array([0.5])
            mock_task.discretizer.state_to_bin = Mock(return_value=10)
            mock_task.true_bin = 10
            mock_gen.return_value = [mock_task]

            result = runner.run(agent)

        # Last prediction (0.5) should match true value (0.5)
        assert result.tasks_completed == 1

    def test_write_updates_learnings(self):
        """WRITE action updates learnings."""
        agent = MockAgent([
            AgentAction(action="WRITE", text="Logistic maps are chaotic"),
            AgentAction(action="PREDICT", value=0.5),
            AgentAction(action="MOVE_ON"),
        ])

        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        with patch.object(runner, '_generate_tasks') as mock_gen:
            mock_task = Mock()
            mock_task.task_id = 1
            mock_task.system.family = "logistic"
            mock_task.h_ks = 0.5
            mock_task.observations = np.random.rand(50)
            mock_task.obs_times = np.arange(50)
            mock_task.future_time = 1
            mock_task.true_future = np.array([0.5])
            mock_task.discretizer.state_to_bin = Mock(return_value=10)
            mock_task.true_bin = 10
            mock_gen.return_value = [mock_task]

            result = runner.run(agent)

        assert "Logistic maps are chaotic" in result.final_learnings


class TestHypothesizeHandler:
    def test_hypothesize_returns_backtest(self):
        """HYPOTHESIZE should return backtest feedback."""
        # Agent hypothesizes, then predicts based on result
        agent = MockAgent([
            AgentAction(action="HYPOTHESIZE", model="logistic", params={"r": 3.9}),
            AgentAction(action="PREDICT", value=0.5),
            AgentAction(action="MOVE_ON"),
        ])

        config = SessionConfig(n_tasks=1, timeout_seconds=60)
        runner = SessionRunner(config)

        with patch.object(runner, '_generate_tasks') as mock_gen:
            # Generate actual logistic data
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
        # Check trace includes HYPOTHESIZE
        trace_md = result.trace.to_markdown()
        assert "HYPOTHESIZE" in trace_md


class TestFitHandler:
    def test_fit_returns_params(self):
        """FIT should return fitted parameters."""
        agent = MockAgent([
            AgentAction(action="FIT", model="logistic"),
            AgentAction(action="PREDICT", value=0.5),
            AgentAction(action="MOVE_ON"),
        ])

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
        trace_md = result.trace.to_markdown()
        assert "FIT" in trace_md
