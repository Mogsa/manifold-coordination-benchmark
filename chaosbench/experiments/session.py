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
    get_chaotic_systems,
)
from chaosbench.core.scoring import compute_score
from chaosbench.agents.metacognitive_types import (
    AgentObservation,
    AgentAction,
    Feedback,
    BacktestFeedback,
)
from chaosbench.core.backtest import backtest_model
from chaosbench.core.fitting import fit_model
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
    scaffolded: bool = True  # If False, disable HYPOTHESIZE/FIT/WRITE/DELETE
    # Difficulty settings
    horizon_multiplier: float = 1.5  # Horizon = k * lyapunov_time (higher = harder)
    noise_std: float = 0.01  # Observation noise (higher = harder)
    # System selection
    families: List[str] = field(default_factory=lambda: ["chaotic"])
    # Valid options: ["chaotic"] for logistic r=4 + tent μ=2 only,
    # or ["logistic", "tent", "henon", ...] for specific families


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
    tasks: List[Task] = field(default_factory=list)  # Store tasks for visualization


class SessionRunner:
    """Runs a metacognitive agent session."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self.learnings = LearningsManager()
        self.trace = TraceLogger()
        self.phi_curve: List[PhiPoint] = []
        self.cumulative_phi = 0.0

    def _generate_tasks(self) -> List[Task]:
        """Generate tasks for the session.

        Uses self.config.families to determine which systems to use:
        - ["chaotic"]: Only logistic r=4 and tent μ=2 (provably chaotic)
        - ["logistic", "tent", ...]: Specific families from create_system_family()
        """
        task_config = TaskConfig(
            conditional=self.config.conditional,
            n_obs=50,
            horizon_lyapunov_multiplier=self.config.horizon_multiplier,
            noise_std=self.config.noise_std,
        )
        generator = TaskGenerator(task_config)

        # Check if using chaotic-only mode
        if self.config.families == ["chaotic"]:
            # Use only provably chaotic systems (logistic r=4, tent μ=2)
            chaotic_systems = get_chaotic_systems()
            tasks = []
            for i in range(self.config.n_tasks):
                # Alternate between logistic and tent
                system = chaotic_systems[i % len(chaotic_systems)]
                tasks.append(generator.generate_task(system))
            return tasks
        else:
            # Use standard stratified generation from specified families
            # (TaskGenerator already filters by min_h_ks)
            return generator.generate_batch(self.config.n_tasks, stratified=True)

    def _compute_score(
        self,
        prediction: float,
        task: Task,
        sigma: float = 0.1,
    ) -> tuple[float, float]:
        """Compute NLL-based score for a prediction.

        Uses bin-based probabilistic scoring:
        - Prediction is treated as center of truncated Gaussian with std=sigma
        - Score = probability mass in the bin containing actual value
        - Punishes confident wrong predictions, rewards calibrated uncertainty

        Args:
            prediction: Agent's predicted value
            task: The task being evaluated
            sigma: Uncertainty (std dev), default 0.1

        Returns: (score, actual_value)
        """
        actual = float(task.true_future[0]) if task.true_future.ndim > 0 else float(task.true_future)

        # Get bounds based on system family
        bounds_map = {
            "logistic": (0.0, 1.0),
            "tent": (0.0, 1.0),
            "henon": (-1.5, 1.5),
            "standard": (0.0, 2 * np.pi),
            "lorenz": (-30.0, 30.0),
        }
        bounds = bounds_map.get(task.system.family, (0.0, 1.0))

        # Compute NLL-based score
        score = compute_score(
            prediction=prediction,
            actual=actual,
            sigma=sigma,
            n_bins=20,
            bounds=bounds,
        )

        return score, actual

    def _build_observation(
        self,
        task: Task,
        last_feedback: Optional[Feedback],
        last_backtest: Optional[BacktestFeedback] = None,
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
            last_backtest=last_backtest,
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

            last_prediction = None  # Store prediction, don't reveal answer
            last_backtest = None  # Store backtest feedback for next turn
            turn_count = 0

            while turn_count < self.config.max_turns_per_task:
                turn_count += 1

                # Build observation (no feedback until MOVE_ON, but include backtest)
                obs = self._build_observation(task, last_feedback=None, last_backtest=last_backtest)

                # Get agent response
                reasoning, action = agent(obs)

                # Handle action
                feedback = None

                if action.action == "PREDICT":
                    # Just record the prediction, don't compute or reveal score yet
                    last_prediction = action.value
                    # Log turn with no feedback (agent doesn't see answer)
                    self.trace.log_turn(reasoning, action, feedback=None)

                elif action.action == "WRITE":
                    if not self.config.scaffolded:
                        # Ignore WRITE in MVP mode, just log and continue
                        self.trace.log_turn(reasoning, action, feedback=None)
                        continue
                    self.learnings.write(action.text)
                    self.trace.log_turn(reasoning, action, feedback=None)

                elif action.action == "DELETE":
                    if not self.config.scaffolded:
                        self.trace.log_turn(reasoning, action, feedback=None)
                        continue
                    self.learnings.delete(action.section)
                    self.trace.log_turn(reasoning, action, feedback=None)

                elif action.action == "HYPOTHESIZE":
                    if not self.config.scaffolded:
                        self.trace.log_turn(reasoning, action, feedback=None)
                        continue
                    # Test the hypothesis against observations
                    result = backtest_model(
                        action.model,
                        action.params,
                        task.observations.flatten(),
                    )
                    backtest_fb = BacktestFeedback(
                        model=action.model,
                        params=action.params,
                        mae=result.mae,
                        predicted_next=result.predicted_next,
                    )
                    last_backtest = backtest_fb  # Pass to agent next turn
                    self.trace.log_turn(reasoning, action, feedback=None, backtest=backtest_fb)

                elif action.action == "FIT":
                    if not self.config.scaffolded:
                        self.trace.log_turn(reasoning, action, feedback=None)
                        continue
                    # Fit parameters for the model family
                    result = fit_model(action.model, task.observations.flatten())
                    backtest_fb = BacktestFeedback(
                        model=action.model,
                        params=result.params,
                        mae=result.mae,
                        predicted_next=result.predicted_next,
                    )
                    last_backtest = backtest_fb  # Pass to agent next turn
                    self.trace.log_turn(reasoning, action, feedback=None, backtest=backtest_fb)

                elif action.action == "MOVE_ON":
                    # NOW we score the last prediction and reveal the answer
                    if last_prediction is not None:
                        last_score, actual = self._compute_score(last_prediction, task)
                        feedback = Feedback(
                            prediction=last_prediction,
                            actual=actual,
                            score=last_score,
                        )
                    else:
                        last_score = 0.0
                        actual = None
                        feedback = None
                    self.trace.log_turn(reasoning, action, feedback)

                    # REFLECTION PHASE: Let agent see result and write learnings
                    if feedback is not None:
                        reflection_turns = 0
                        while reflection_turns < 3:  # Max 3 reflection turns
                            reflection_turns += 1
                            # Show result to agent
                            obs = self._build_observation(task, last_feedback=feedback)
                            reasoning, action = agent(obs)

                            if action.action == "WRITE":
                                self.learnings.write(action.text)
                                self.trace.log_turn(reasoning, action, feedback=None)
                            elif action.action == "DELETE":
                                self.learnings.delete(action.section)
                                self.trace.log_turn(reasoning, action, feedback=None)
                            else:
                                # Any other action (PREDICT, MOVE_ON) ends reflection
                                break
                    break

            # Compute final score (in case agent hit turn limit without MOVE_ON)
            if last_prediction is not None and 'last_score' not in dir():
                last_score, _ = self._compute_score(last_prediction, task)
            elif last_prediction is None:
                last_score = 0.0

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
            tasks=tasks,  # Store for visualization
        )
