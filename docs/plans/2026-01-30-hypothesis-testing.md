# Hypothesis-Driven ChaosBench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HYPOTHESIZE and FIT actions so agents can test models against observations before predicting.

**Architecture:** Agent proposes model+params → system backtests on x_0...x_49 → returns MAE + predicted x_50 → agent refines or commits. x_50 hidden until PREDICT.

**Tech Stack:** Python 3.13, numpy, scipy.optimize, pytest, existing ChaosBench model classes.

---

## Task 1: Model Factory

**Files:**
- Create: `chaosbench/core/models.py`
- Test: `chaosbench/tests/test_models.py`

**Step 1: Write the failing test**

Create `chaosbench/tests/test_models.py`:

```python
"""Tests for model factory."""
import pytest
import numpy as np

from chaosbench.core.models import create_model, MODEL_PARAMS


class TestCreateModel:
    def test_create_logistic(self):
        """Create logistic map with r parameter."""
        model = create_model("logistic", {"r": 3.9})
        assert model.family == "logistic"
        assert model.r == 3.9

    def test_create_tent(self):
        """Create tent map with mu parameter."""
        model = create_model("tent", {"mu": 1.8})
        assert model.family == "tent"
        assert model.mu == 1.8

    def test_unknown_family_raises(self):
        """Unknown family raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model family"):
            create_model("unknown", {})

    def test_model_params_lists_families(self):
        """MODEL_PARAMS contains all supported families."""
        assert "logistic" in MODEL_PARAMS
        assert "tent" in MODEL_PARAMS
        assert MODEL_PARAMS["logistic"] == ["r"]
        assert MODEL_PARAMS["tent"] == ["mu"]
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_models.py -v`

Expected: FAIL with "No module named 'chaosbench.core.models'"

**Step 3: Write minimal implementation**

Create `chaosbench/core/models.py`:

```python
"""Model factory for creating chaotic systems from family name + params."""
from typing import Dict, Any, List

from .Chaosbench_v3 import (
    ChaoticSystem,
    LogisticMap,
    TentMap,
    HenonMap,
    StandardMap,
    LorenzDisc,
)


# Maps family name -> list of parameter names
MODEL_PARAMS: Dict[str, List[str]] = {
    "logistic": ["r"],
    "tent": ["mu"],
    "henon": ["a", "b"],
    "standard": ["K"],
    "lorenz": ["sigma", "rho", "beta"],
}


def create_model(family: str, params: Dict[str, Any]) -> ChaoticSystem:
    """Create a chaotic system from family name and parameters.

    Args:
        family: One of "logistic", "tent", "henon", "standard", "lorenz"
        params: Dict of parameter values (e.g., {"r": 3.9} for logistic)

    Returns:
        ChaoticSystem instance

    Raises:
        ValueError: If family unknown or params invalid
    """
    if family == "logistic":
        return LogisticMap(r=params.get("r", 4.0))
    elif family == "tent":
        return TentMap(mu=params.get("mu", 2.0))
    elif family == "henon":
        return HenonMap(a=params.get("a", 1.4), b=params.get("b", 0.3))
    elif family == "standard":
        return StandardMap(K=params.get("K", 1.0))
    elif family == "lorenz":
        return LorenzDisc(
            sigma=params.get("sigma", 10.0),
            rho=params.get("rho", 28.0),
            beta=params.get("beta", 8/3),
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_models.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/core/models.py chaosbench/tests/test_models.py
git commit -m "feat(chaosbench): add model factory for creating systems from family+params"
```

---

## Task 2: Backtest Function

**Files:**
- Create: `chaosbench/core/backtest.py`
- Test: `chaosbench/tests/test_backtest.py`

**Step 1: Write the failing test**

Create `chaosbench/tests/test_backtest.py`:

```python
"""Tests for backtest function."""
import pytest
import numpy as np

from chaosbench.core.backtest import backtest_model, BacktestResult


class TestBacktestModel:
    def test_perfect_logistic_fit(self):
        """Perfect model should have near-zero MAE."""
        # Generate observations from logistic r=3.9
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.9}, x)

        assert isinstance(result, BacktestResult)
        assert result.mae < 0.01  # Near-perfect fit
        assert result.predicted_next is not None

    def test_wrong_params_high_mae(self):
        """Wrong parameters should have high MAE."""
        # Generate from r=3.9, test with r=3.5
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.5}, x)

        assert result.mae > 0.1  # Poor fit

    def test_predicts_next_value(self):
        """Result includes prediction for x_50."""
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.9}, x)

        # Predicted x_50 should be r * x_49 * (1 - x_49)
        expected = r * x[-1] * (1 - x[-1])
        assert abs(result.predicted_next - expected) < 0.001

    def test_tent_map_backtest(self):
        """Backtest works for tent map."""
        mu = 1.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = mu * x[i-1] if x[i-1] < 0.5 else mu * (1 - x[i-1])

        result = backtest_model("tent", {"mu": 1.9}, x)

        assert result.mae < 0.01
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_backtest.py -v`

Expected: FAIL with "No module named 'chaosbench.core.backtest'"

**Step 3: Write minimal implementation**

Create `chaosbench/core/backtest.py`:

```python
"""Backtest models against observations."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .models import create_model


@dataclass
class BacktestResult:
    """Result of backtesting a model against observations."""
    mae: float  # Mean absolute error on one-step predictions
    predicted_next: float  # Model's prediction for x_50


def backtest_model(
    family: str,
    params: Dict[str, Any],
    observations: np.ndarray,
) -> BacktestResult:
    """Test a model against observations.

    Computes one-step prediction error: for each x_i, predict x_{i+1}
    using the model, compare to actual x_{i+1}.

    Args:
        family: Model family ("logistic", "tent", etc.)
        params: Model parameters (e.g., {"r": 3.9})
        observations: Array of x_0, x_1, ..., x_49

    Returns:
        BacktestResult with MAE and predicted x_50
    """
    model = create_model(family, params)
    obs = observations.flatten()

    # One-step predictions: predict x_{i+1} from x_i
    errors = []
    for i in range(len(obs) - 1):
        x_i = obs[i:i+1] if model.dim == 1 else obs[i]
        predicted = model.step(x_i)
        if model.dim == 1:
            predicted = float(predicted)
        else:
            predicted = float(predicted[0])  # Take first component for comparison
        actual = float(obs[i + 1])
        errors.append(abs(predicted - actual))

    mae = float(np.mean(errors))

    # Predict x_50
    x_last = obs[-1:] if model.dim == 1 else obs[-model.dim:]
    next_pred = model.step(x_last)
    if model.dim == 1:
        predicted_next = float(next_pred)
    else:
        predicted_next = float(next_pred[0])

    return BacktestResult(mae=mae, predicted_next=predicted_next)
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_backtest.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/core/backtest.py chaosbench/tests/test_backtest.py
git commit -m "feat(chaosbench): add backtest function to test models against observations"
```

---

## Task 3: Parameter Fitting

**Files:**
- Create: `chaosbench/core/fitting.py`
- Test: `chaosbench/tests/test_fitting.py`

**Step 1: Write the failing test**

Create `chaosbench/tests/test_fitting.py`:

```python
"""Tests for parameter fitting."""
import pytest
import numpy as np

from chaosbench.core.fitting import fit_model, FitResult


class TestFitModel:
    def test_fit_logistic_recovers_r(self):
        """Fitting logistic data should recover r parameter."""
        # Generate observations from logistic r=3.85
        true_r = 3.85
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_r * x[i-1] * (1 - x[i-1])

        result = fit_model("logistic", x)

        assert isinstance(result, FitResult)
        assert abs(result.params["r"] - true_r) < 0.1  # Within 0.1 of true
        assert result.mae < 0.05

    def test_fit_tent_recovers_mu(self):
        """Fitting tent data should recover mu parameter."""
        true_mu = 1.85
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_mu * x[i-1] if x[i-1] < 0.5 else true_mu * (1 - x[i-1])

        result = fit_model("tent", x)

        assert abs(result.params["mu"] - true_mu) < 0.1
        assert result.mae < 0.05

    def test_fit_includes_prediction(self):
        """Fit result includes predicted x_50."""
        true_r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_r * x[i-1] * (1 - x[i-1])

        result = fit_model("logistic", x)

        assert result.predicted_next is not None
        # Prediction should be reasonable (in [0, 1] for logistic)
        assert 0 <= result.predicted_next <= 1
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_fitting.py -v`

Expected: FAIL with "No module named 'chaosbench.core.fitting'"

**Step 3: Write minimal implementation**

Create `chaosbench/core/fitting.py`:

```python
"""Fit model parameters to observations."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize_scalar, minimize

from .backtest import backtest_model


@dataclass
class FitResult:
    """Result of fitting a model to observations."""
    params: Dict[str, float]
    mae: float
    predicted_next: float


def fit_model(family: str, observations: np.ndarray) -> FitResult:
    """Fit model parameters to observations.

    Uses scipy.optimize to find parameters that minimize
    one-step prediction error.

    Args:
        family: Model family ("logistic", "tent", etc.)
        observations: Array of x_0, x_1, ..., x_49

    Returns:
        FitResult with estimated params, MAE, and predicted x_50
    """
    obs = observations.flatten()

    if family == "logistic":
        # Fit r in [3.5, 4.0] (chaotic regime)
        def loss(r):
            result = backtest_model("logistic", {"r": r}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(3.5, 4.0), method='bounded')
        best_r = opt.x
        best_result = backtest_model("logistic", {"r": best_r}, obs)
        return FitResult(
            params={"r": float(best_r)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "tent":
        # Fit mu in [1.0, 2.0]
        def loss(mu):
            result = backtest_model("tent", {"mu": mu}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(1.0, 2.0), method='bounded')
        best_mu = opt.x
        best_result = backtest_model("tent", {"mu": best_mu}, obs)
        return FitResult(
            params={"mu": float(best_mu)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "henon":
        # Fit a, b
        def loss(params):
            a, b = params
            result = backtest_model("henon", {"a": a, "b": b}, obs)
            return result.mae

        opt = minimize(loss, x0=[1.4, 0.3], bounds=[(1.0, 1.5), (0.1, 0.5)], method='L-BFGS-B')
        best_a, best_b = opt.x
        best_result = backtest_model("henon", {"a": best_a, "b": best_b}, obs)
        return FitResult(
            params={"a": float(best_a), "b": float(best_b)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "standard":
        # Fit K
        def loss(K):
            result = backtest_model("standard", {"K": K}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(0.5, 2.0), method='bounded')
        best_K = opt.x
        best_result = backtest_model("standard", {"K": best_K}, obs)
        return FitResult(
            params={"K": float(best_K)},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    elif family == "lorenz":
        # Use standard Lorenz params, just fit rho
        def loss(rho):
            result = backtest_model("lorenz", {"sigma": 10.0, "rho": rho, "beta": 8/3}, obs)
            return result.mae

        opt = minimize_scalar(loss, bounds=(20.0, 35.0), method='bounded')
        best_rho = opt.x
        best_result = backtest_model("lorenz", {"sigma": 10.0, "rho": best_rho, "beta": 8/3}, obs)
        return FitResult(
            params={"sigma": 10.0, "rho": float(best_rho), "beta": 8/3},
            mae=best_result.mae,
            predicted_next=best_result.predicted_next,
        )

    else:
        raise ValueError(f"Unknown model family: {family}")
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_fitting.py -v`

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/core/fitting.py chaosbench/tests/test_fitting.py
git commit -m "feat(chaosbench): add parameter fitting using scipy.optimize"
```

---

## Task 4: New Action Types

**Files:**
- Modify: `chaosbench/agents/metacognitive_types.py`
- Test: `chaosbench/tests/test_metacognitive_types.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_metacognitive_types.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_metacognitive_types.py::TestParseHypothesisActions -v`

Expected: FAIL with AttributeError (action has no 'model' attribute)

**Step 3: Write minimal implementation**

Modify `chaosbench/agents/metacognitive_types.py`:

Replace the AgentAction dataclass and parse_action function:

```python
@dataclass
class AgentAction:
    """What agent can do."""
    action: Literal["PREDICT", "WRITE", "DELETE", "MOVE_ON", "HYPOTHESIZE", "FIT"]
    value: float | None = None
    text: str | None = None
    section: str | None = None
    model: str | None = None  # For HYPOTHESIZE/FIT
    params: dict | None = None  # For HYPOTHESIZE


def parse_action(response: str) -> AgentAction:
    """Parse an AgentAction from LLM response.

    Extracts JSON from response text (agent may write reasoning before JSON).

    Raises:
        ValueError: If no valid action JSON found.
    """
    # Find JSON object in response - handle nested objects for params
    json_match = re.search(r'\{[^{}]*"action"[^{}]*(?:\{[^{}]*\})?[^{}]*\}', response)
    if not json_match:
        # Try simpler pattern
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response)
    if not json_match:
        raise ValueError(f"No JSON action found in response: {response[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    action_type = data.get("action")
    valid_actions = ("PREDICT", "WRITE", "DELETE", "MOVE_ON", "HYPOTHESIZE", "FIT")
    if action_type not in valid_actions:
        raise ValueError(f"Invalid action type: {action_type}")

    return AgentAction(
        action=action_type,
        value=data.get("value"),
        text=data.get("text"),
        section=data.get("section"),
        model=data.get("model"),
        params=data.get("params"),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_metacognitive_types.py -v`

Expected: PASS (all tests including new ones)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/agents/metacognitive_types.py chaosbench/tests/test_metacognitive_types.py
git commit -m "feat(chaosbench): add HYPOTHESIZE and FIT action types"
```

---

## Task 5: Backtest Feedback Dataclass

**Files:**
- Modify: `chaosbench/agents/metacognitive_types.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_metacognitive_types.py`:

```python
from chaosbench.agents.metacognitive_types import BacktestFeedback


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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_metacognitive_types.py::TestBacktestFeedback -v`

Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `chaosbench/agents/metacognitive_types.py`:

```python
@dataclass
class BacktestFeedback:
    """Feedback from testing a hypothesis."""
    model: str
    params: dict
    mae: float
    predicted_next: float

    def format(self) -> str:
        """Format as human-readable feedback."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        quality = "fits well" if self.mae < 0.05 else "doesn't reproduce the observations well"

        return f"""Model: {self.model} ({params_str})

Backtest (fitting x_0 → x_49):
  MAE: {self.mae:.3f}
  Your model {quality}.

If you trust this model, it predicts x_50 = {self.predicted_next:.4f}"""
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_metacognitive_types.py::TestBacktestFeedback -v`

Expected: PASS

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/agents/metacognitive_types.py chaosbench/tests/test_metacognitive_types.py
git commit -m "feat(chaosbench): add BacktestFeedback dataclass for hypothesis results"
```

---

## Task 6: Session Runner - HYPOTHESIZE Handler

**Files:**
- Modify: `chaosbench/experiments/session.py`
- Test: `chaosbench/tests/test_session.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_session.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_session.py::TestHypothesizeHandler -v`

Expected: FAIL (HYPOTHESIZE not handled)

**Step 3: Write minimal implementation**

In `chaosbench/experiments/session.py`, add import at top:

```python
from chaosbench.core.backtest import backtest_model
from chaosbench.agents.metacognitive_types import (
    AgentObservation,
    AgentAction,
    Feedback,
    BacktestFeedback,
)
```

Then add handler in the action dispatch (after PREDICT handler, around line 143):

```python
                elif action.action == "HYPOTHESIZE":
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
                    self.trace.log_turn(reasoning, action, feedback=None, backtest=backtest_fb)
```

Also update `_build_observation` to include backtest feedback (add parameter):

```python
    def _build_observation(
        self,
        task: Task,
        last_feedback: Optional[Feedback],
        last_backtest: Optional[BacktestFeedback] = None,
    ) -> AgentObservation:
```

And update `AgentObservation` in types.py to include:

```python
    last_backtest: BacktestFeedback | None = None
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_session.py::TestHypothesizeHandler -v`

Expected: PASS

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/experiments/session.py chaosbench/agents/metacognitive_types.py chaosbench/tests/test_session.py
git commit -m "feat(chaosbench): add HYPOTHESIZE handler to session runner"
```

---

## Task 7: Session Runner - FIT Handler

**Files:**
- Modify: `chaosbench/experiments/session.py`
- Test: `chaosbench/tests/test_session.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_session.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_session.py::TestFitHandler -v`

Expected: FAIL (FIT not handled)

**Step 3: Write minimal implementation**

In `chaosbench/experiments/session.py`, add import:

```python
from chaosbench.core.fitting import fit_model
```

Add handler after HYPOTHESIZE:

```python
                elif action.action == "FIT":
                    # Fit parameters for the model family
                    result = fit_model(action.model, task.observations.flatten())
                    backtest_fb = BacktestFeedback(
                        model=action.model,
                        params=result.params,
                        mae=result.mae,
                        predicted_next=result.predicted_next,
                    )
                    self.trace.log_turn(reasoning, action, feedback=None, backtest=backtest_fb)
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_session.py::TestFitHandler -v`

Expected: PASS

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/experiments/session.py chaosbench/tests/test_session.py
git commit -m "feat(chaosbench): add FIT handler to session runner"
```

---

## Task 8: Update System Prompt

**Files:**
- Create: `chaosbench/prompts/hypothesis_system.txt`

**Step 1: Create the prompt file**

Create `chaosbench/prompts/hypothesis_system.txt`:

```
You are a scientist studying unknown dynamical systems. Your goal is to
identify the underlying system and use that understanding to predict
future states.

## Each Task

You observe a time series: x_0, x_1, ..., x_49
Your job: predict x_50

But prediction requires understanding. You should:
1. Form hypotheses about what system generated this data
2. Test your hypotheses against the observations
3. Refine until you have a good model
4. Use your model to predict

## Actions

You MUST end every response with exactly one JSON action.

HYPOTHESIZE — Test a specific model against observations
{"action": "HYPOTHESIZE", "model": "logistic", "params": {"r": 3.85}}

You will see:
- Backtest MAE (how well model fits x_0...x_49)
- Quality message (good fit or poor fit)
- Predicted x_50 (if you want to use it)

FIT — Auto-estimate parameters for a model family
{"action": "FIT", "model": "logistic"}

You will see:
- Estimated parameters
- Fit quality (MAE)
- Predicted x_50

PREDICT — Commit your final prediction
{"action": "PREDICT", "value": 0.42}

This is your answer. Only submit when confident.

WRITE — Record learnings for future tasks
{"action": "WRITE", "text": "## Logistic Maps\nThey have period-doubling..."}

DELETE — Remove a section from learnings
{"action": "DELETE", "section": "## Old Section"}

MOVE_ON — Accept score and proceed to next task
{"action": "MOVE_ON"}

## Model Families

You may hypothesize these model types:
- logistic (params: r) — x_{n+1} = r * x_n * (1 - x_n)
- tent (params: mu) — piecewise linear map
- henon (params: a, b) — 2D quadratic map
- standard (params: K) — Hamiltonian chaos
- lorenz (params: sigma, rho, beta) — 3D continuous flow

## Scoring

Score ranges from 0 (far off) to 1 (perfect):
- Score 1.00 = exact match
- Score 0.61 = off by 0.1
- Score 0.37 = off by 0.2

## Strategy

1. Look at the data — is it bounded? Oscillating? Chaotic?
2. HYPOTHESIZE a model family and rough parameters
3. Check the MAE — if poor, adjust parameters or try different family
4. When MAE is low, trust the model's prediction
5. PREDICT to commit your answer
6. After seeing result, WRITE what you learned

## Your Learnings

You have a persistent notepad shown below each task. Use it to record:
- Which model families you've seen
- Parameter ranges that work
- Patterns that help identify system type

Good learnings help you on future tasks.

Begin.
```

**Step 2: Verify file created**

Run: `cat "/Users/morgan/Desktop/Year 3/Diss/DISS/chaosbench/prompts/hypothesis_system.txt" | head -20`

Expected: First 20 lines of prompt

**Step 3: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/prompts/hypothesis_system.txt
git commit -m "feat(chaosbench): add hypothesis-driven system prompt"
```

---

## Task 9: Integration Test

**Files:**
- Create: `chaosbench/tests/test_hypothesis_integration.py`

**Step 1: Write the integration test**

Create `chaosbench/tests/test_hypothesis_integration.py`:

```python
"""Integration test for hypothesis-driven session."""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from chaosbench.experiments.session import SessionRunner, SessionConfig
from chaosbench.agents.metacognitive_types import AgentAction


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
```

**Step 2: Run integration test**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_hypothesis_integration.py -v`

Expected: PASS (2 tests)

**Step 3: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/tests/test_hypothesis_integration.py
git commit -m "test(chaosbench): add integration tests for hypothesis-driven flow"
```

---

## Task 10: Run All Tests

**Step 1: Run full test suite**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/ -v`

Expected: All tests PASS

**Step 2: Final commit if needed**

If any fixes were required, commit them.

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Model Factory | `models.py`, `test_models.py` |
| 2 | Backtest Function | `backtest.py`, `test_backtest.py` |
| 3 | Parameter Fitting | `fitting.py`, `test_fitting.py` |
| 4 | New Action Types | `metacognitive_types.py` |
| 5 | Backtest Feedback | `metacognitive_types.py` |
| 6 | HYPOTHESIZE Handler | `session.py` |
| 7 | FIT Handler | `session.py` |
| 8 | System Prompt | `hypothesis_system.txt` |
| 9 | Integration Test | `test_hypothesis_integration.py` |
| 10 | Final Verification | Run all tests |

**Total: ~10 commits, ~500 lines of new code**
