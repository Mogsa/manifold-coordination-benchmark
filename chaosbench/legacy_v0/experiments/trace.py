"""Trace logging for metacognitive agent sessions."""
from dataclasses import dataclass, field
from typing import List, Optional
import time

from chaosbench.agents.metacognitive_types import AgentAction, Feedback, BacktestFeedback


@dataclass
class Turn:
    """A single turn in a task."""
    turn_number: int
    reasoning: str
    action: AgentAction
    feedback: Optional[Feedback]
    timestamp: float
    backtest: Optional[BacktestFeedback] = None


@dataclass
class TaskTrace:
    """Trace of a single task."""
    task_id: int
    family: str
    h_ks: float
    turns: List[Turn] = field(default_factory=list)
    final_score: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def n_attempts(self) -> int:
        return sum(1 for t in self.turns if t.action.action == "PREDICT")


class TraceLogger:
    """Logs traces for a session."""

    def __init__(self):
        self.tasks: List[TaskTrace] = []
        self.current_task: Optional[TaskTrace] = None
        self.session_start: float = time.time()

    def start_task(self, task_id: int, family: str, h_ks: float) -> None:
        self.current_task = TaskTrace(
            task_id=task_id,
            family=family,
            h_ks=h_ks,
        )

    def log_turn(
        self,
        reasoning: str,
        action: AgentAction,
        feedback: Optional[Feedback] = None,
        backtest: Optional[BacktestFeedback] = None,
    ) -> None:
        if self.current_task is None:
            raise RuntimeError("No active task")

        turn = Turn(
            turn_number=len(self.current_task.turns) + 1,
            reasoning=reasoning,
            action=action,
            feedback=feedback,
            timestamp=time.time(),
            backtest=backtest,
        )
        self.current_task.turns.append(turn)

    def end_task(self, final_score: float) -> None:
        if self.current_task is None:
            raise RuntimeError("No active task")

        self.current_task.final_score = final_score
        self.current_task.end_time = time.time()
        self.tasks.append(self.current_task)
        self.current_task = None

    def to_markdown(self) -> str:
        """Export trace as markdown for dissertation analysis."""
        lines = ["# Session Trace\n"]

        for task in self.tasks:
            lines.append(f"## Task {task.task_id}: {task.family} (h_KS={task.h_ks:.2f})\n")
            lines.append(f"**Duration:** {task.duration:.1f}s | **Attempts:** {task.n_attempts} | **Final Score:** {task.final_score:.2f}\n")

            for turn in task.turns:
                lines.append(f"### Turn {turn.turn_number}\n")
                lines.append(f"**Reasoning:**\n> {turn.reasoning}\n")
                lines.append(f"**Action:** {turn.action.action}")
                if turn.action.value is not None:
                    lines.append(f"({turn.action.value})")
                lines.append("\n")

                if turn.feedback:
                    lines.append(f"**Feedback:** pred={turn.feedback.prediction:.3f}, actual={turn.feedback.actual:.3f}, score={turn.feedback.score:.2f}\n")

                if turn.backtest:
                    params_str = ", ".join(f"{k}={v}" for k, v in turn.backtest.params.items())
                    lines.append(f"**Backtest:** model={turn.backtest.model}({params_str}), MAE={turn.backtest.mae:.4f}, predicted_next={turn.backtest.predicted_next:.4f}\n")

            lines.append("---\n")

        return "\n".join(lines)
