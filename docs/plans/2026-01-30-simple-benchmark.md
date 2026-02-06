# Simple ChaosBench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a simple benchmark for testing LLM prediction on chaotic time series, with configurable difficulty (prediction horizon + noise), visualization, and pasteable output for manual testing.

**Architecture:** Task generator creates observations from chaotic systems, adds noise, and formats output. Visualizer plots time series with ground truth. No scaffolding tools (HYPOTHESIZE/FIT) — raw prediction only.

**Tech Stack:** Python 3.13, numpy, matplotlib, existing ChaosBench chaotic systems.

---

## Task 1: Task Generator

**Files:**
- Create: `chaosbench/simple/generator.py`
- Test: `chaosbench/tests/test_simple_generator.py`

**Step 1: Write the failing test**

Create `chaosbench/tests/test_simple_generator.py`:

```python
"""Tests for simple benchmark task generator."""
import pytest
import numpy as np

from chaosbench.simple.generator import generate_task, TaskDifficulty


class TestGenerateTask:
    def test_easy_task_one_step_no_noise(self):
        """Easy task: 1-step prediction, no noise."""
        task = generate_task(
            difficulty=TaskDifficulty(horizon=1, noise_std=0.0),
            system="logistic",
            seed=42,
        )

        assert len(task.observations) == 50
        assert task.target_step == 50  # x_50
        assert task.noise_std == 0.0
        assert task.target_value is not None
        assert 0 < task.target_value < 1  # Logistic bounded

    def test_hard_task_twenty_step_noisy(self):
        """Hard task: 20-step prediction, noisy."""
        task = generate_task(
            difficulty=TaskDifficulty(horizon=20, noise_std=0.05),
            system="logistic",
            seed=42,
        )

        assert len(task.observations) == 50
        assert task.target_step == 69  # x_69
        assert task.noise_std == 0.05

    def test_reproducible_with_seed(self):
        """Same seed produces same task."""
        task1 = generate_task(TaskDifficulty(1, 0.0), "logistic", seed=123)
        task2 = generate_task(TaskDifficulty(1, 0.0), "logistic", seed=123)

        np.testing.assert_array_equal(task1.observations, task2.observations)
        assert task1.target_value == task2.target_value
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_generator.py -v`

Expected: FAIL with "No module named 'chaosbench.simple'"

**Step 3: Create module structure**

Create `chaosbench/simple/__init__.py`:

```python
"""Simple benchmark for raw LLM prediction testing."""
```

**Step 4: Write minimal implementation**

Create `chaosbench/simple/generator.py`:

```python
"""Task generator for simple benchmark."""
from dataclasses import dataclass
from typing import Literal
import numpy as np

from chaosbench.core.Chaosbench_v3 import LogisticMap, TentMap, HenonMap


@dataclass
class TaskDifficulty:
    """Difficulty settings for a task."""
    horizon: int  # Steps ahead to predict (1=easy, 20=hard)
    noise_std: float  # Observation noise (0=clean, 0.05=noisy)


@dataclass
class Task:
    """A prediction task."""
    observations: np.ndarray  # x_0 to x_49
    target_step: int  # Which step to predict (e.g., 50, 55, 69)
    target_value: float  # Ground truth
    noise_std: float  # Noise level used
    system_name: str  # For logging only
    system_params: dict  # For logging only


def generate_task(
    difficulty: TaskDifficulty,
    system: Literal["logistic", "tent", "henon"] = "logistic",
    seed: int | None = None,
    n_observations: int = 50,
) -> Task:
    """Generate a prediction task.

    Args:
        difficulty: Horizon and noise settings
        system: Which chaotic system to use
        seed: Random seed for reproducibility
        n_observations: Number of observation points

    Returns:
        Task with observations and ground truth
    """
    if seed is not None:
        np.random.seed(seed)

    # Create system with random parameters in chaotic regime
    if system == "logistic":
        r = np.random.uniform(3.7, 4.0)
        sys = LogisticMap(r=r)
        x0 = np.array([np.random.uniform(0.1, 0.9)])
        params = {"r": r}
    elif system == "tent":
        mu = np.random.uniform(1.5, 2.0)
        sys = TentMap(mu=mu)
        x0 = np.array([np.random.uniform(0.1, 0.9)])
        params = {"mu": mu}
    elif system == "henon":
        a = np.random.uniform(1.2, 1.4)
        b = 0.3
        sys = HenonMap(a=a, b=b)
        x0 = np.array([0.1, 0.1])
        params = {"a": a, "b": b}
    else:
        raise ValueError(f"Unknown system: {system}")

    # Generate trajectory
    total_steps = n_observations + difficulty.horizon
    trajectory = sys.trajectory(x0, total_steps)

    # Extract observations (first component for 2D systems)
    if sys.dim == 1:
        obs_clean = trajectory[:n_observations, 0]
        target = float(trajectory[n_observations + difficulty.horizon - 1, 0])
    else:
        obs_clean = trajectory[:n_observations, 0]  # Only x component
        target = float(trajectory[n_observations + difficulty.horizon - 1, 0])

    # Add noise
    if difficulty.noise_std > 0:
        noise = np.random.normal(0, difficulty.noise_std, n_observations)
        observations = obs_clean + noise
        # Clip to valid range for logistic/tent
        if system in ("logistic", "tent"):
            observations = np.clip(observations, 0.001, 0.999)
    else:
        observations = obs_clean

    return Task(
        observations=observations,
        target_step=n_observations + difficulty.horizon - 1,
        target_value=target,
        noise_std=difficulty.noise_std,
        system_name=system,
        system_params=params,
    )
```

**Step 5: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_generator.py -v`

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/simple/__init__.py chaosbench/simple/generator.py chaosbench/tests/test_simple_generator.py
git commit -m "feat(chaosbench): add simple benchmark task generator"
```

---

## Task 2: Task Formatter (Pasteable Output)

**Files:**
- Modify: `chaosbench/simple/generator.py`
- Test: `chaosbench/tests/test_simple_generator.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_simple_generator.py`:

```python
from chaosbench.simple.generator import format_task_for_llm, format_system_prompt


class TestFormatTask:
    def test_format_includes_observations(self):
        """Formatted task includes all observations."""
        task = generate_task(TaskDifficulty(1, 0.0), "logistic", seed=42)
        text = format_task_for_llm(task)

        assert "x_0 to x_49" in text
        assert "Predict: x_50" in text
        # Should have numbers
        assert "0." in text

    def test_format_system_prompt(self):
        """System prompt is minimal, no hints."""
        prompt = format_system_prompt()

        assert "PREDICTION:" in prompt
        # Should NOT mention model families
        assert "logistic" not in prompt.lower()
        assert "tent" not in prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_generator.py::TestFormatTask -v`

Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `chaosbench/simple/generator.py`:

```python
def format_system_prompt() -> str:
    """Generate the system prompt (no model hints)."""
    return """You are analyzing time series data from unknown dynamical systems.

## Task
Given observations x_0, x_1, ..., x_n, predict a future value x_m.

## Rules
- Analyze the pattern in the data
- Reason about what process might generate it
- Make your best prediction

## Response Format
Think through your analysis, then end with:
PREDICTION: [your number]"""


def format_task_for_llm(task: Task) -> str:
    """Format task as pasteable text for LLM."""
    obs_str = ", ".join(f"{x:.4f}" for x in task.observations)
    n_obs = len(task.observations)

    return f"""## Task

**Observations (x_0 to x_{n_obs - 1}):**
[{obs_str}]

**Predict:** x_{task.target_step}"""
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_generator.py::TestFormatTask -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/simple/generator.py chaosbench/tests/test_simple_generator.py
git commit -m "feat(chaosbench): add LLM-pasteable task formatter"
```

---

## Task 3: Visualizer

**Files:**
- Create: `chaosbench/simple/visualizer.py`
- Test: `chaosbench/tests/test_simple_visualizer.py`

**Step 1: Write the failing test**

Create `chaosbench/tests/test_simple_visualizer.py`:

```python
"""Tests for simple benchmark visualizer."""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from chaosbench.simple.generator import generate_task, TaskDifficulty
from chaosbench.simple.visualizer import plot_task, plot_difficulty_grid


class TestPlotTask:
    def test_plot_creates_file(self):
        """plot_task creates a PNG file."""
        task = generate_task(TaskDifficulty(1, 0.0), "logistic", seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            plot_task(task, save_path=path)
            assert path.exists()
            assert path.stat().st_size > 0


class TestPlotDifficultyGrid:
    def test_grid_creates_file(self):
        """plot_difficulty_grid creates a PNG with multiple subplots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grid.png"
            plot_difficulty_grid(save_path=path, seed=42)
            assert path.exists()
            assert path.stat().st_size > 1000  # Should be non-trivial
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_visualizer.py -v`

Expected: FAIL with "No module named 'chaosbench.simple.visualizer'"

**Step 3: Write minimal implementation**

Create `chaosbench/simple/visualizer.py`:

```python
"""Visualization for simple benchmark tasks."""
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .generator import Task, TaskDifficulty, generate_task


def plot_task(
    task: Task,
    save_path: Optional[Path] = None,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a task's observations and target.

    Args:
        task: The task to visualize
        save_path: Path to save PNG (optional)
        show: Whether to display interactively
        ax: Existing axes to plot on (optional)

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    n_obs = len(task.observations)

    # Plot observations
    ax.plot(range(n_obs), task.observations, 'b.-',
            label='Observations', markersize=3, linewidth=0.5)

    # Mark prediction point
    ax.axvline(n_obs - 0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark target
    ax.scatter([task.target_step], [task.target_value],
               color='red', s=100, zorder=5,
               label=f'Target x_{task.target_step} = {task.target_value:.4f}')

    ax.set_xlabel('Time step')
    ax.set_ylabel('x')
    ax.set_title(f'{task.system_name} | horizon={task.target_step - n_obs + 1} | noise={task.noise_std}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_difficulty_grid(
    save_path: Optional[Path] = None,
    show: bool = False,
    seed: int = 42,
) -> plt.Figure:
    """Plot a 3x3 grid of tasks at different difficulties.

    Rows: horizon (1, 5, 20)
    Cols: noise (0, 0.02, 0.05)
    """
    horizons = [1, 5, 20]
    noises = [0.0, 0.02, 0.05]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    for i, horizon in enumerate(horizons):
        for j, noise in enumerate(noises):
            ax = axes[i, j]
            difficulty = TaskDifficulty(horizon=horizon, noise_std=noise)
            task = generate_task(difficulty, "logistic", seed=seed + i * 10 + j)

            n_obs = len(task.observations)
            ax.plot(range(n_obs), task.observations, 'b.-', markersize=2, linewidth=0.5)
            ax.scatter([task.target_step], [task.target_value], color='red', s=50, zorder=5)
            ax.axvline(n_obs - 0.5, color='gray', linestyle=':', alpha=0.5)

            if i == 0:
                ax.set_title(f'noise σ={noise}')
            if j == 0:
                ax.set_ylabel(f'{horizon}-step\nx')
            ax.set_xlim(-2, task.target_step + 5)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Difficulty Grid: Prediction Horizon × Observation Noise', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple_visualizer.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/simple/visualizer.py chaosbench/tests/test_simple_visualizer.py
git commit -m "feat(chaosbench): add task visualizer with difficulty grid"
```

---

## Task 4: CLI Tool

**Files:**
- Create: `chaosbench/simple/cli.py`
- Modify: `chaosbench/simple/__init__.py`

**Step 1: Write the CLI**

Create `chaosbench/simple/cli.py`:

```python
#!/usr/bin/env python3
"""CLI for generating simple benchmark tasks."""
import argparse
from pathlib import Path

from .generator import generate_task, TaskDifficulty, format_system_prompt, format_task_for_llm
from .visualizer import plot_task, plot_difficulty_grid


def main():
    parser = argparse.ArgumentParser(description="Generate simple benchmark tasks")
    parser.add_argument("--horizon", type=int, default=1, help="Steps ahead to predict (1, 5, 20)")
    parser.add_argument("--noise", type=float, default=0.0, help="Observation noise std (0, 0.02, 0.05)")
    parser.add_argument("--system", default="logistic", choices=["logistic", "tent", "henon"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Save visualization to PNG")
    parser.add_argument("--grid", action="store_true", help="Generate difficulty grid instead")
    parser.add_argument("--show-prompt", action="store_true", help="Print system prompt")
    args = parser.parse_args()

    if args.show_prompt:
        print("=" * 60)
        print("SYSTEM PROMPT")
        print("=" * 60)
        print(format_system_prompt())
        print("=" * 60)
        print()

    if args.grid:
        print("Generating difficulty grid...")
        save_path = args.output or Path("difficulty_grid.png")
        plot_difficulty_grid(save_path=save_path, seed=args.seed or 42)
        print(f"Saved: {save_path}")
        return

    # Generate single task
    difficulty = TaskDifficulty(horizon=args.horizon, noise_std=args.noise)
    task = generate_task(difficulty, args.system, seed=args.seed)

    # Print pasteable format
    print("=" * 60)
    print("USER MESSAGE (paste to Claude)")
    print("=" * 60)
    print(format_task_for_llm(task))
    print()
    print("=" * 60)
    print(f"GROUND TRUTH: x_{task.target_step} = {task.target_value:.4f}")
    print(f"System: {task.system_name}, params: {task.system_params}")
    print("=" * 60)

    # Visualize if requested
    if args.output:
        plot_task(task, save_path=args.output)
        print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Update __init__.py for easy imports**

Replace `chaosbench/simple/__init__.py`:

```python
"""Simple benchmark for raw LLM prediction testing."""
from .generator import (
    Task,
    TaskDifficulty,
    generate_task,
    format_system_prompt,
    format_task_for_llm,
)
from .visualizer import plot_task, plot_difficulty_grid

__all__ = [
    "Task",
    "TaskDifficulty",
    "generate_task",
    "format_system_prompt",
    "format_task_for_llm",
    "plot_task",
    "plot_difficulty_grid",
]
```

**Step 3: Test CLI manually**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && python -m chaosbench.simple.cli --horizon 5 --noise 0.02 --seed 42 --show-prompt`

Expected: Prints system prompt + formatted task + ground truth

**Step 4: Commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add chaosbench/simple/cli.py chaosbench/simple/__init__.py
git commit -m "feat(chaosbench): add CLI for simple benchmark generation"
```

---

## Task 5: Run All Tests

**Step 1: Run full test suite**

Run: `cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate && pytest chaosbench/tests/test_simple*.py -v`

Expected: All tests PASS

**Step 2: Generate example outputs**

Run:
```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS" && source venv/bin/activate
python -m chaosbench.simple.cli --grid --output difficulty_grid.png
python -m chaosbench.simple.cli --horizon 1 --noise 0 --seed 100 --show-prompt
python -m chaosbench.simple.cli --horizon 20 --noise 0.05 --seed 100
```

**Step 3: Final commit**

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
git add difficulty_grid.png
git commit -m "docs(chaosbench): add example difficulty grid visualization"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Task Generator | `generator.py`, `test_simple_generator.py` |
| 2 | Task Formatter | `generator.py` (additions) |
| 3 | Visualizer | `visualizer.py`, `test_simple_visualizer.py` |
| 4 | CLI Tool | `cli.py`, `__init__.py` |
| 5 | Final Verification | Run all tests |

**Total: ~5 commits, ~350 lines of new code**

**Usage after implementation:**

```bash
# Show system prompt + easy task
python -m chaosbench.simple.cli --show-prompt --horizon 1 --noise 0

# Hard task (20-step, noisy)
python -m chaosbench.simple.cli --horizon 20 --noise 0.05 --seed 42

# Generate difficulty grid visualization
python -m chaosbench.simple.cli --grid --output grid.png
```
