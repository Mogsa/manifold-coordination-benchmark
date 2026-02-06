"""Tests for trace logging."""
import pytest
from chaosbench.legacy_v0.experiments.trace import TraceLogger, Turn, TaskTrace
from chaosbench.legacy_v0.agents.metacognitive_types import AgentAction, Feedback


class TestTraceLogger:
    def test_start_task(self):
        logger = TraceLogger()
        logger.start_task(task_id=1, family="logistic", h_ks=0.5)
        assert logger.current_task.task_id == 1
        assert logger.current_task.family == "logistic"

    def test_log_turn(self):
        logger = TraceLogger()
        logger.start_task(task_id=1, family="logistic", h_ks=0.5)
        logger.log_turn(
            reasoning="This looks like chaos",
            action=AgentAction(action="PREDICT", value=0.5),
            feedback=Feedback(prediction=0.5, actual=0.4, score=0.7),
        )
        assert len(logger.current_task.turns) == 1
        assert logger.current_task.turns[0].reasoning == "This looks like chaos"

    def test_end_task(self):
        logger = TraceLogger()
        logger.start_task(task_id=1, family="logistic", h_ks=0.5)
        logger.log_turn(
            reasoning="Reasoning",
            action=AgentAction(action="PREDICT", value=0.5),
            feedback=Feedback(prediction=0.5, actual=0.4, score=0.7),
        )
        logger.end_task(final_score=0.7)

        assert len(logger.tasks) == 1
        assert logger.tasks[0].final_score == 0.7
        assert logger.current_task is None

    def test_to_markdown(self):
        logger = TraceLogger()
        logger.start_task(task_id=1, family="logistic", h_ks=0.5)
        logger.log_turn(
            reasoning="Pattern analysis",
            action=AgentAction(action="PREDICT", value=0.42),
            feedback=Feedback(prediction=0.42, actual=0.4, score=0.9),
        )
        logger.end_task(final_score=0.9)

        md = logger.to_markdown()
        assert "Task 1" in md
        assert "logistic" in md
        assert "Pattern analysis" in md
        assert "PREDICT" in md
