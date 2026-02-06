"""Agent protocol and solution dataclasses for ChaosBench v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from chaosbench.problems.factory import Problem


@dataclass
class TaskResult:
    """Record of a single solved problem (for task history)."""

    problem_id: str
    question_type: str
    observations_preview: list[float]  # First 20 values
    agent_answer: Any
    correct_answer: Any
    raw_score: float
    confidence: float


@dataclass
class Solution:
    """An agent's answer to a problem."""

    answer: Any          # str for CLASSIFY/IDENTIFY, list[float] for PREDICT
    confidence: float    # 0-1
    reasoning_trace: str  # Full LLM response (stored, not scored)


@runtime_checkable
class Agent(Protocol):
    """Protocol that all agents must satisfy."""

    @property
    def agent_id(self) -> str: ...

    def solve(self, problem: Problem, task_history: list[TaskResult]) -> Solution: ...
