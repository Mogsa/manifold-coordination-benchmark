# ChaosBench v4: Metacognitive Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an LLM agent that reasons about chaotic systems with persistent learnings, producing Φ(t) curves and traces for dissertation analysis.

**Architecture:** Session runner loops through tasks, calling LLM with observations + learnings. Agent responds with JSON actions (PREDICT/WRITE/DELETE/MOVE_ON). Traces logged for analysis.

**Tech Stack:** Python 3.13, LiteLLM (Gemini API), pytest, existing ChaosBench v3 systems.

---

## Task 1: Data Classes

**Files:**
- Create: `chaosbench/agents/metacognitive_types.py`
- Test: `chaosbench/tests/test_metacognitive_types.py`

**Step 1: Write the failing test**

```python
# chaosbench/tests/test_metacognitive_types.py
"""Tests for metacognitive agent data types."""
import pytest
import numpy as np
from chaosbench.agents.metacognitive_types import (
    Feedback,
    AgentObservation,
    AgentAction,
    parse_action,
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
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_metacognitive_types.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chaosbench.agents.metacognitive_types'"

**Step 3: Write minimal implementation**

```python
# chaosbench/agents/metacognitive_types.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_metacognitive_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chaosbench/agents/metacognitive_types.py chaosbench/tests/test_metacognitive_types.py
git commit -m "feat(chaosbench): add metacognitive agent data types"
```

---

## Task 2: Learnings Manager

**Files:**
- Create: `chaosbench/agents/learnings.py`
- Test: `chaosbench/tests/test_learnings.py`

**Step 1: Write the failing test**

```python
# chaosbench/tests/test_learnings.py
"""Tests for learnings manager."""
import pytest
from chaosbench.agents.learnings import LearningsManager


class TestLearningsManager:
    def test_initial_state_empty(self):
        lm = LearningsManager()
        assert lm.content == "# My Learnings\n"

    def test_write_appends(self):
        lm = LearningsManager()
        lm.write("First insight")
        assert "First insight" in lm.content
        lm.write("Second insight")
        assert "First insight" in lm.content
        assert "Second insight" in lm.content

    def test_delete_section(self):
        lm = LearningsManager()
        lm.write("## Section A\nContent A")
        lm.write("## Section B\nContent B")
        assert "Section A" in lm.content
        assert "Section B" in lm.content

        lm.delete("## Section A")
        assert "Section A" not in lm.content
        assert "Section B" in lm.content

    def test_delete_nonexistent_section_no_error(self):
        lm = LearningsManager()
        lm.write("Some content")
        lm.delete("## Nonexistent")  # Should not raise
        assert "Some content" in lm.content

    def test_reset(self):
        lm = LearningsManager()
        lm.write("Some content")
        lm.reset()
        assert lm.content == "# My Learnings\n"

    def test_token_count(self):
        lm = LearningsManager()
        assert lm.token_count() > 0  # Header has tokens
        lm.write("A" * 1000)
        assert lm.token_count() > 100  # Rough estimate
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_learnings.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# chaosbench/agents/learnings.py
"""Manages persistent learnings for metacognitive agent."""
import re


class LearningsManager:
    """Manages the agent's persistent learnings notepad."""

    HEADER = "# My Learnings\n"

    def __init__(self):
        self._content = self.HEADER

    @property
    def content(self) -> str:
        return self._content

    def write(self, text: str) -> None:
        """Append text to learnings."""
        self._content += f"\n{text}\n"

    def delete(self, section: str) -> None:
        """Delete a section by its header.

        Removes from the header to the next header of same or higher level,
        or to end of content.
        """
        # Escape regex special chars in section header
        escaped = re.escape(section)
        # Match section header to next same-or-higher level header or end
        pattern = rf'{escaped}.*?(?=\n#{{1,2}} |\Z)'
        self._content = re.sub(pattern, '', self._content, flags=re.DOTALL)

    def reset(self) -> None:
        """Reset to initial state."""
        self._content = self.HEADER

    def token_count(self) -> int:
        """Rough token estimate (chars / 4)."""
        return len(self._content) // 4
```

**Step 4: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_learnings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chaosbench/agents/learnings.py chaosbench/tests/test_learnings.py
git commit -m "feat(chaosbench): add learnings manager"
```

---

## Task 3: Trace Logger

**Files:**
- Create: `chaosbench/experiments/trace.py`
- Test: `chaosbench/tests/test_trace.py`

**Step 1: Write the failing test**

```python
# chaosbench/tests/test_trace.py
"""Tests for trace logging."""
import pytest
from chaosbench.experiments.trace import TraceLogger, Turn, TaskTrace
from chaosbench.agents.metacognitive_types import AgentAction, Feedback


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
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_trace.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# chaosbench/experiments/trace.py
"""Trace logging for metacognitive agent sessions."""
from dataclasses import dataclass, field
from typing import List, Optional
import time

from chaosbench.agents.metacognitive_types import AgentAction, Feedback


@dataclass
class Turn:
    """A single turn in a task."""
    turn_number: int
    reasoning: str
    action: AgentAction
    feedback: Optional[Feedback]
    timestamp: float


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
    ) -> None:
        if self.current_task is None:
            raise RuntimeError("No active task")

        turn = Turn(
            turn_number=len(self.current_task.turns) + 1,
            reasoning=reasoning,
            action=action,
            feedback=feedback,
            timestamp=time.time(),
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

            lines.append("---\n")

        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_trace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chaosbench/experiments/trace.py chaosbench/tests/test_trace.py
git commit -m "feat(chaosbench): add trace logger for session analysis"
```

---

## Task 4: Session Runner

**Files:**
- Create: `chaosbench/experiments/session.py`
- Test: `chaosbench/tests/test_session.py`

**Step 1: Write the failing test**

```python
# chaosbench/tests/test_session.py
"""Tests for session runner."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from chaosbench.experiments.session import SessionRunner, SessionConfig, SessionResult
from chaosbench.agents.metacognitive_types import AgentAction


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
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_session.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# chaosbench/experiments/session.py
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
    parse_action,
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
    """Point on the Φ(t) curve."""
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
            family=task.family if self.config.conditional else None,
            learnings=self.learnings.content,
            last_feedback=last_feedback,
        )

    def run(self, agent: Callable) -> SessionResult:
        """Run the session.

        Args:
            agent: Callable that takes AgentObservation and returns (reasoning, AgentAction)

        Returns:
            SessionResult with Φ(t) curve, trace, and final learnings.
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
```

**Step 4: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_session.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chaosbench/experiments/session.py chaosbench/tests/test_session.py
git commit -m "feat(chaosbench): add session runner for metacognitive agent"
```

---

## Task 5: LLM Agent

**Files:**
- Create: `chaosbench/agents/metacognitive_agent.py`
- Modify: `chaosbench/prompts/` (create directory and prompt file)
- Test: `chaosbench/tests/test_metacognitive_agent.py`

**Step 1: Create system prompt file**

```bash
mkdir -p chaosbench/prompts
```

```
# chaosbench/prompts/metacognitive_system.txt
You are a scientist studying unknown dynamical systems. Your goal is to
predict future states from past observations.

## Each Task

You observe a time series and must predict the next value:
- Observations: x_0, x_1, ..., x_49 (50 values)
- Your job: predict x_50

After each PREDICT, you see:
- Actual value
- Your score (0 = far off, 1 = perfect)

You may predict multiple times per task before moving on.

## Actions

First write your reasoning, then output ONE action as JSON.

PREDICT — Make or revise prediction
{"action": "PREDICT", "value": 0.42}

WRITE — Add to your learnings (persists across tasks)
{"action": "WRITE", "text": "Your note here"}

DELETE — Remove from learnings by section header
{"action": "DELETE", "section": "## Section Header"}

MOVE_ON — Accept current score, proceed to next task
{"action": "MOVE_ON"}

## Your Learnings

You have a persistent notepad. It appears below each task. Use it to
record patterns, mistakes, and insights that may help future tasks.

## Constraints

- Session has a time limit
- Some systems may be fundamentally hard to predict
- Knowing when to move on is valuable

Begin.
```

**Step 2: Write the failing test**

```python
# chaosbench/tests/test_metacognitive_agent.py
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
```

**Step 3: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_metacognitive_agent.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# chaosbench/agents/metacognitive_agent.py
"""LLM-based metacognitive agent for ChaosBench."""
from pathlib import Path
from typing import Tuple
import numpy as np

from shared.llm_utils import call_llm
from chaosbench.agents.metacognitive_types import (
    AgentObservation,
    AgentAction,
    parse_action,
)


class MetacognitiveAgent:
    """LLM agent with persistent learnings."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = self._load_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "metacognitive_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        # Fallback inline prompt
        return """You are a scientist studying unknown dynamical systems.
Predict future states from observations. Output JSON actions."""

    def _format_observation(self, obs: AgentObservation) -> str:
        """Format observation as user message."""
        lines = []

        lines.append(f"## Task {obs.task_id}")
        if obs.family:
            lines.append(f"**System family:** {obs.family}")

        # Format observations compactly
        obs_str = ", ".join(f"{x:.3f}" for x in obs.observations[:10])
        if len(obs.observations) > 10:
            obs_str += f", ... ({len(obs.observations)} total)"
        lines.append(f"**Observations:** [{obs_str}]")
        lines.append(f"**Predict:** x_{len(obs.observations)}")

        # Include feedback if present
        if obs.last_feedback:
            lines.append("")
            lines.append("**Last attempt:**")
            lines.append(f"- Your prediction: {obs.last_feedback.prediction:.3f}")
            lines.append(f"- Actual value: {obs.last_feedback.actual:.3f}")
            lines.append(f"- Score: {obs.last_feedback.score:.2f}")

        # Include learnings
        lines.append("")
        lines.append("---")
        lines.append("**Your Learnings:**")
        lines.append(obs.learnings)

        return "\n".join(lines)

    def __call__(self, obs: AgentObservation) -> Tuple[str, AgentAction]:
        """Get action from LLM.

        Returns:
            Tuple of (reasoning, action)
        """
        user_message = self._format_observation(obs)

        # Build messages for this turn
        messages = self.messages + [{"role": "user", "content": user_message}]

        # Call LLM
        response = call_llm(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse action from response
        action = parse_action(response)

        # Extract reasoning (everything before the JSON)
        reasoning = response.split('{"action"')[0].strip()

        return reasoning, action

    def reset(self) -> None:
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
```

**Step 5: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_metacognitive_agent.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add chaosbench/agents/metacognitive_agent.py chaosbench/prompts/metacognitive_system.txt chaosbench/tests/test_metacognitive_agent.py
git commit -m "feat(chaosbench): add LLM-based metacognitive agent"
```

---

## Task 6: Main Runner Script

**Files:**
- Create: `chaosbench/run_metacognitive.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
# chaosbench/run_metacognitive.py
"""Run a metacognitive agent session on ChaosBench."""
import argparse
from pathlib import Path
import json

from chaosbench.experiments.session import SessionRunner, SessionConfig
from chaosbench.agents.metacognitive_agent import MetacognitiveAgent


def main():
    parser = argparse.ArgumentParser(description="Run metacognitive agent on ChaosBench")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--timeout", type=int, default=300, help="Session timeout in seconds")
    parser.add_argument("--output", default="session_output", help="Output directory")
    parser.add_argument("--conditional", action="store_true", help="Reveal system family")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Create agent and runner
    agent = MetacognitiveAgent(model=args.model)
    config = SessionConfig(
        n_tasks=args.n_tasks,
        timeout_seconds=args.timeout,
        conditional=args.conditional,
    )
    runner = SessionRunner(config)

    print(f"Running {args.n_tasks} tasks with {args.model}...")
    print(f"Timeout: {args.timeout}s, Conditional: {args.conditional}")
    print("-" * 50)

    # Run session
    result = runner.run(agent)

    # Print summary
    print("-" * 50)
    print(f"Tasks completed: {result.tasks_completed}")
    print(f"Final Φ: {result.final_phi:.2f}")
    print(f"Total time: {result.total_time:.1f}s")
    print(f"Tasks/second: {result.tasks_completed / result.total_time:.2f}")

    # Save outputs
    trace_path = output_dir / "trace.md"
    trace_path.write_text(result.trace.to_markdown())
    print(f"Trace saved to: {trace_path}")

    learnings_path = output_dir / "learnings.md"
    learnings_path.write_text(result.final_learnings)
    print(f"Learnings saved to: {learnings_path}")

    # Save Φ(t) curve as JSON
    phi_path = output_dir / "phi_curve.json"
    phi_data = [
        {"time": p.wall_time, "phi": p.cumulative_phi, "tasks": p.tasks_completed}
        for p in result.phi_curve
    ]
    phi_path.write_text(json.dumps(phi_data, indent=2))
    print(f"Φ(t) curve saved to: {phi_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable and test**

Run: `python -m chaosbench.run_metacognitive --n-tasks 2 --timeout 60`
Expected: Runs 2 tasks with Gemini, prints summary, saves outputs

**Step 3: Commit**

```bash
git add chaosbench/run_metacognitive.py
git commit -m "feat(chaosbench): add main runner script for metacognitive agent"
```

---

## Task 7: Integration Test

**Files:**
- Create: `chaosbench/tests/test_integration.py`

**Step 1: Write integration test**

```python
# chaosbench/tests/test_integration.py
"""Integration test for full metacognitive session."""
import pytest
from unittest.mock import patch

from chaosbench.experiments.session import SessionRunner, SessionConfig
from chaosbench.agents.metacognitive_agent import MetacognitiveAgent


@pytest.mark.integration
class TestFullSession:
    @patch('chaosbench.agents.metacognitive_agent.call_llm')
    def test_full_session_with_mock_llm(self, mock_llm):
        """Run a full session with mocked LLM responses."""
        # Simulate agent behavior: predict, learn, move on
        responses = [
            'Analyzing pattern...\n{"action": "PREDICT", "value": 0.5}',
            'Score was low, let me try again\n{"action": "PREDICT", "value": 0.3}',
            'Recording insight\n{"action": "WRITE", "text": "## Logistic Maps\\nThey oscillate"}',
            'Moving on\n{"action": "MOVE_ON"}',
        ] * 10  # Repeat for multiple tasks

        mock_llm.side_effect = responses

        agent = MetacognitiveAgent(model="gemini/gemini-2.0-flash")
        config = SessionConfig(n_tasks=3, timeout_seconds=60)
        runner = SessionRunner(config)

        result = runner.run(agent)

        assert result.tasks_completed == 3
        assert result.final_phi > 0
        assert "Logistic Maps" in result.final_learnings
        assert len(result.trace.tasks) == 3
```

**Step 2: Run integration test**

Run: `pytest chaosbench/tests/test_integration.py -v -m integration`
Expected: PASS

**Step 3: Commit**

```bash
git add chaosbench/tests/test_integration.py
git commit -m "test(chaosbench): add integration test for metacognitive session"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Data classes + parse_action | `metacognitive_types.py` |
| 2 | Learnings manager | `learnings.py` |
| 3 | Trace logger | `trace.py` |
| 4 | Session runner | `session.py` |
| 5 | LLM agent | `metacognitive_agent.py` |
| 6 | Main runner script | `run_metacognitive.py` |
| 7 | Integration test | `test_integration.py` |

After completing all tasks, run:
```bash
python -m chaosbench.run_metacognitive --n-tasks 10 --conditional
```

This will run 10 tasks with Gemini and produce:
- `session_output/trace.md` — Full reasoning trace
- `session_output/learnings.md` — What the agent learned
- `session_output/phi_curve.json` — Φ(t) data for plotting
