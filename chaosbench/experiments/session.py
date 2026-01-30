"""Session runner for metacognitive agent experiments."""
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import time
import numpy as np

from chaosbench.core.Chaosbench_v3 import (
    TaskGenerator,
    TaskConfig,
    Task,
    DifficultyWeighting,
)
from chaosbench.agents.metacognitive_types import (
    AgentObservation,
    AgentAction,
    Feedback,
)
from chaosbench.agents.learnings import LearningsManager
from chaosbench.experiments.trace import TraceLogger


@dataclass
class SessionConfig:
    """Configuration for a session."""
    n_tasks: int = 50
    timeout_seconds: float = 600  # 10 minutes
    conditional: bool = True
    weighting: Callable[[float], float] = DifficultyWeighting.linear
    max_turns_per_task: int = 20  # Safety limit


@dataclass
class PhiPoint:
    """Point on the Phi(t) curve."""
    wall_time: float
    cumulative_phi: float
    tasks_completed: int


@dataclass
class SessionResult:
    """Result of a session."""
    tasks_completed: int
    phi_curve: List[PhiPoint]
    final_phi: float
    final_learnings: str
    trace: TraceLogger
    total_time: float


class SessionRunner:
    """Runs a metacognitive agent session."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self.learnings = LearningsManager()
        self.trace = TraceLogger()
        self.phi_curve: List[PhiPoint] = []
        self.cumulative_phi = 0.0

    def _generate_tasks(self) -> List[Task]:
        """Generate tasks for the session."""
        task_config = TaskConfig(
            conditional=self.config.conditional,
            n_obs=50,
        )
        generator = TaskGenerator(task_config)
        return generator.generate_batch(self.config.n_tasks, stratified=True)

    def _compute_score(self, prediction: float, task: Task) -> tuple[float, float]:
        """Compute score for a prediction.

        Returns: (score, actual_value)
        """
        actual = float(task.true_future[0]) if task.true_future.ndim > 0 else float(task.true_future)

        # Score based on distance (simple version)
        # Could use NLL on discretized space for full version
        error = abs(prediction - actual)
        score = np.exp(-error * 5)  # Exponential decay

        return score, actual

    def _build_observation(
        self,
        task: Task,
        last_feedback: Optional[Feedback],
    ) -> AgentObservation:
        """Build observation for the agent."""
        return AgentObservation(
            task_id=task.task_id,
            observations=task.observations,
            obs_times=task.obs_times,
            prediction_horizon=task.future_time,
            family=task.system.family if self.config.conditional else None,
            learnings=self.learnings.content,
            last_feedback=last_feedback,
        )

    def run(self, agent: Callable) -> SessionResult:
        """Run the session.

        Args:
            agent: Callable that takes AgentObservation and returns (reasoning, AgentAction)

        Returns:
            SessionResult with Phi(t) curve, trace, and final learnings.
        """
        tasks = self._generate_tasks()
        start_time = time.time()
        tasks_completed = 0

        for task in tasks:
            # Check timeout
            if time.time() - start_time > self.config.timeout_seconds:
                break

            self.trace.start_task(
                task_id=task.task_id,
                family=task.system.family,
                h_ks=task.h_ks,
            )

            last_feedback = None
            last_score = 0.0
            turn_count = 0

            while turn_count < self.config.max_turns_per_task:
                turn_count += 1

                # Build observation
                obs = self._build_observation(task, last_feedback)

                # Get agent response
                reasoning, action = agent(obs)

                # Handle action
                feedback = None

                if action.action == "PREDICT":
                    score, actual = self._compute_score(action.value, task)
                    last_score = score
                    feedback = Feedback(
                        prediction=action.value,
                        actual=actual,
                        score=score,
                    )
                    last_feedback = feedback

                elif action.action == "WRITE":
                    self.learnings.write(action.text)

                elif action.action == "DELETE":
                    self.learnings.delete(action.section)

                elif action.action == "MOVE_ON":
                    self.trace.log_turn(reasoning, action, feedback)
                    break

                self.trace.log_turn(reasoning, action, feedback)

            # Bank score
            weighted_score = self.config.weighting(task.h_ks) * last_score
            self.cumulative_phi += weighted_score

            self.phi_curve.append(PhiPoint(
                wall_time=time.time() - start_time,
                cumulative_phi=self.cumulative_phi,
                tasks_completed=tasks_completed + 1,
            ))

            self.trace.end_task(final_score=last_score)
            tasks_completed += 1

        return SessionResult(
            tasks_completed=tasks_completed,
            phi_curve=self.phi_curve,
            final_phi=self.cumulative_phi,
            final_learnings=self.learnings.content,
            trace=self.trace,
            total_time=time.time() - start_time,
        )
