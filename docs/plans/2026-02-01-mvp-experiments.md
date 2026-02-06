# MVP Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run actual LLM experiments on raw chaotic prediction (no scaffolding) and visualize results.

**Architecture:** Add `scaffolded=False` mode to existing session runner, create MVP prompt, add visualization module for task plots and Φ(n) curves.

**Tech Stack:** Python, matplotlib, existing chaosbench infrastructure

---

## What Already Exists

| Component | Status | Location |
|-----------|--------|----------|
| 5 chaotic systems | ✅ | `core/Chaosbench_v3.py` |
| Task generation | ✅ | `core/Chaosbench_v3.py` |
| Session runner | ✅ | `experiments/session.py` |
| Agent framework | ✅ | `agents/metacognitive_agent.py` |
| CLI runner | ✅ | `run_metacognitive.py` |
| Trace logging | ✅ | `experiments/trace.py` |

## What We Need to Build

| Component | Purpose |
|-----------|---------|
| MVP prompt | No HYPOTHESIZE/FIT hints |
| `scaffolded` flag | Disable scaffolding actions |
| Visualization | Task plots + Φ(n) curves |
| Updated CLI | `--scaffolded` flag |

---

## Task 1: Create MVP System Prompt

**Files:**
- Create: `chaosbench/prompts/mvp_system.txt`

**Step 1: Write the MVP prompt file**

```text
You are predicting the next value in a time series from an unknown dynamical system.

## Each Task

You observe a sequence: x_0, x_1, ..., x_49
Your job: predict x_50

## Actions

You MUST end every response with exactly one JSON action.

PREDICT — Submit your prediction
{"action": "PREDICT", "value": 0.42, "uncertainty": 0.1}

The "uncertainty" field is optional. If provided, it represents your confidence interval (±).

MOVE_ON — Proceed to next task (after seeing your score)
{"action": "MOVE_ON"}

## What You See

After PREDICT, you will see:
- Your prediction
- The actual value
- Your score (0 to 1, higher is better)

Use this feedback to improve on future tasks.

## Tips

- Look for patterns: bounded values, oscillations, trends
- The systems are deterministic but chaotic
- Learning from earlier tasks may help later ones

Begin.
```

**Step 2: Verify file was created**

Run: `cat chaosbench/prompts/mvp_system.txt | head -5`
Expected: First 5 lines of the prompt

**Step 3: Commit**

```bash
git add chaosbench/prompts/mvp_system.txt
git commit -m "feat(chaosbench): add MVP system prompt (no scaffolding)"
```

---

## Task 2: Add Scaffolded Flag to SessionConfig

**Files:**
- Modify: `chaosbench/experiments/session.py`

**Step 1: Add `scaffolded` field to SessionConfig**

In `session.py`, find the `SessionConfig` dataclass (~line 26) and add:

```python
@dataclass
class SessionConfig:
    """Configuration for a session."""
    n_tasks: int = 50
    timeout_seconds: float = 600  # 10 minutes
    conditional: bool = True
    weighting: Callable[[float], float] = DifficultyWeighting.linear
    max_turns_per_task: int = 20  # Safety limit
    scaffolded: bool = True  # If False, disable HYPOTHESIZE/FIT/WRITE/DELETE
```

**Step 2: Modify SessionRunner to reject scaffolding actions when disabled**

In the `run()` method, after handling PREDICT (~line 148), add validation:

```python
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
                    # ... existing code ...

                elif action.action == "FIT":
                    if not self.config.scaffolded:
                        self.trace.log_turn(reasoning, action, feedback=None)
                        continue
                    # ... existing code ...
```

**Step 3: Run existing tests to ensure no regression**

Run: `pytest chaosbench/tests/test_session.py -v`
Expected: All 5 tests pass

**Step 4: Commit**

```bash
git add chaosbench/experiments/session.py
git commit -m "feat(chaosbench): add scaffolded flag to SessionConfig"
```

---

## Task 3: Update MetacognitiveAgent to Support MVP Prompt

**Files:**
- Modify: `chaosbench/agents/metacognitive_agent.py`

**Step 1: Add prompt selection based on scaffolded flag**

Modify `__init__` to accept a `scaffolded` parameter:

```python
class MetacognitiveAgent:
    """LLM agent for metacognitive experiments."""

    def __init__(self, model: str = "gemini/gemini-2.0-flash", scaffolded: bool = True):
        self.model = model
        self.scaffolded = scaffolded
        self.system_prompt = self._load_system_prompt()
        self.messages = []
```

**Step 2: Update `_load_system_prompt()` to select correct prompt**

```python
    def _load_system_prompt(self) -> str:
        """Load the appropriate system prompt."""
        prompt_dir = Path(__file__).parent.parent / "prompts"
        if self.scaffolded:
            prompt_file = prompt_dir / "hypothesis_system.txt"
        else:
            prompt_file = prompt_dir / "mvp_system.txt"
        return prompt_file.read_text()
```

**Step 3: Run agent tests**

Run: `pytest chaosbench/tests/test_metacognitive_agent.py -v`
Expected: All 3 tests pass

**Step 4: Commit**

```bash
git add chaosbench/agents/metacognitive_agent.py
git commit -m "feat(chaosbench): add scaffolded flag to agent for prompt selection"
```

---

## Task 4: Update CLI Runner

**Files:**
- Modify: `chaosbench/run_metacognitive.py`

**Step 1: Add --scaffolded flag**

```python
def main():
    parser = argparse.ArgumentParser(description="Run metacognitive agent on ChaosBench")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--timeout", type=int, default=300, help="Session timeout in seconds")
    parser.add_argument("--output", default="session_output", help="Output directory")
    parser.add_argument("--conditional", action="store_true", help="Reveal system family")
    parser.add_argument("--scaffolded", action="store_true", help="Enable HYPOTHESIZE/FIT actions")
    args = parser.parse_args()
```

**Step 2: Pass flag to agent and config**

```python
    # Create agent and runner
    agent = MetacognitiveAgent(model=args.model, scaffolded=args.scaffolded)
    config = SessionConfig(
        n_tasks=args.n_tasks,
        timeout_seconds=args.timeout,
        conditional=args.conditional,
        scaffolded=args.scaffolded,
    )
```

**Step 3: Update print statement**

```python
    print(f"Running {args.n_tasks} tasks with {args.model}...")
    print(f"Timeout: {args.timeout}s, Conditional: {args.conditional}, Scaffolded: {args.scaffolded}")
```

**Step 4: Commit**

```bash
git add chaosbench/run_metacognitive.py
git commit -m "feat(chaosbench): add --scaffolded flag to CLI"
```

---

## Task 5: Create Visualization Module

**Files:**
- Create: `chaosbench/visualization/__init__.py`
- Create: `chaosbench/visualization/plots.py`

**Step 1: Create visualization package**

Create `chaosbench/visualization/__init__.py`:

```python
"""Visualization tools for ChaosBench experiments."""
from .plots import plot_task, plot_phi_curve, plot_session_summary
```

**Step 2: Create plots.py with task visualization**

```python
"""Plotting functions for ChaosBench experiments."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List


def plot_task(
    observations: np.ndarray,
    actual: float,
    prediction: Optional[float] = None,
    uncertainty: Optional[float] = None,
    title: str = "Task",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a single task: observations + prediction vs actual.

    Args:
        observations: Array of observed values [x_0, ..., x_49]
        actual: True value to predict (x_50)
        prediction: Agent's prediction (optional)
        uncertainty: Agent's uncertainty ± (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    n_obs = len(observations)
    t_obs = np.arange(n_obs)
    t_target = n_obs

    # Plot observations
    ax.plot(t_obs, observations, 'b.-', label='Observations', markersize=4, linewidth=1)

    # Plot actual value
    ax.scatter([t_target], [actual], color='green', s=100, marker='o',
               label=f'Actual: {actual:.3f}', zorder=5)

    # Plot prediction if provided
    if prediction is not None:
        ax.scatter([t_target], [prediction], color='red', s=100, marker='x',
                   label=f'Prediction: {prediction:.3f}', zorder=5)

        # Add uncertainty bar if provided
        if uncertainty is not None:
            ax.errorbar([t_target], [prediction], yerr=uncertainty,
                       color='red', capsize=5, capthick=2, linewidth=2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add vertical line separating observations from prediction
    ax.axvline(x=n_obs - 0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_phi_curve(
    phi_points: List[dict],
    title: str = "Learning Curve Φ(n)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the Φ(n) learning curve.

    Args:
        phi_points: List of dicts with 'tasks' and 'phi' keys
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = [p['tasks'] for p in phi_points]
    phi = [p['phi'] for p in phi_points]

    # Plot actual curve
    ax.plot(tasks, phi, 'b.-', linewidth=2, markersize=8, label='Φ(n)')

    # Plot linear reference (no learning)
    if len(tasks) > 1:
        avg_increment = phi[-1] / tasks[-1]
        linear_ref = [avg_increment * t for t in tasks]
        ax.plot(tasks, linear_ref, 'k--', alpha=0.5, label='Linear (no learning)')

    ax.set_xlabel('Tasks completed (n)')
    ax.set_ylabel('Cumulative Φ(n)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_session_summary(
    phi_points: List[dict],
    scores: List[float],
    title: str = "Session Summary",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot session summary: Φ(n) curve + per-task scores.

    Args:
        phi_points: List of dicts with 'tasks' and 'phi' keys
        scores: List of per-task scores
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Φ(n) curve
    tasks = [p['tasks'] for p in phi_points]
    phi = [p['phi'] for p in phi_points]

    ax1.plot(tasks, phi, 'b.-', linewidth=2, markersize=8, label='Φ(n)')

    if len(tasks) > 1:
        avg_increment = phi[-1] / tasks[-1]
        linear_ref = [avg_increment * t for t in tasks]
        ax1.plot(tasks, linear_ref, 'k--', alpha=0.5, label='Linear (no learning)')

    ax1.set_xlabel('Tasks completed (n)')
    ax1.set_ylabel('Cumulative Φ(n)')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Per-task scores
    task_nums = list(range(1, len(scores) + 1))
    colors = ['green' if s > 0.5 else 'orange' if s > 0.2 else 'red' for s in scores]
    ax2.bar(task_nums, scores, color=colors, alpha=0.7)
    ax2.axhline(y=np.mean(scores), color='blue', linestyle='--',
                label=f'Mean: {np.mean(scores):.2f}')
    ax2.set_xlabel('Task number')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Task Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
```

**Step 3: Commit**

```bash
git add chaosbench/visualization/
git commit -m "feat(chaosbench): add visualization module for tasks and Φ(n) curves"
```

---

## Task 6: Add Visualization to CLI Output

**Files:**
- Modify: `chaosbench/run_metacognitive.py`

**Step 1: Import visualization**

Add at top of file:
```python
from chaosbench.visualization import plot_phi_curve, plot_session_summary
```

**Step 2: Add visualization after running session**

After saving phi_curve.json, add:

```python
    # Generate visualizations
    print("Generating visualizations...")

    # Plot Φ(n) curve
    phi_plot_path = output_dir / "phi_curve.png"
    plot_phi_curve(
        phi_data,
        title=f"Learning Curve - {args.model}",
        save_path=str(phi_plot_path),
        show=False,
    )
    print(f"Φ(n) plot saved to: {phi_plot_path}")

    # Extract per-task scores from trace
    scores = [task.final_score for task in result.trace.tasks]

    # Plot session summary
    summary_plot_path = output_dir / "session_summary.png"
    plot_session_summary(
        phi_data,
        scores,
        title=f"Session Summary - {args.model} ({'Scaffolded' if args.scaffolded else 'MVP'})",
        save_path=str(summary_plot_path),
        show=False,
    )
    print(f"Summary plot saved to: {summary_plot_path}")
```

**Step 3: Commit**

```bash
git add chaosbench/run_metacognitive.py
git commit -m "feat(chaosbench): add visualization output to CLI"
```

---

## Task 7: Run First MVP Experiment

**Files:**
- None (execution only)

**Step 1: Verify all tests still pass**

Run: `pytest chaosbench/tests/ -v`
Expected: All 76+ tests pass

**Step 2: Run MVP experiment (5 tasks, quick test)**

```bash
cd /Users/morgan/Desktop/Year\ 3/Diss/DISS
python -m chaosbench.run_metacognitive --n-tasks 5 --output mvp_test_run
```

Expected:
- Session completes
- `mvp_test_run/trace.md` created
- `mvp_test_run/phi_curve.json` created
- `mvp_test_run/phi_curve.png` created
- `mvp_test_run/session_summary.png` created

**Step 3: Run scaffolded experiment for comparison**

```bash
python -m chaosbench.run_metacognitive --n-tasks 5 --scaffolded --output scaffolded_test_run
```

**Step 4: Compare the two outputs visually**

Open both `phi_curve.png` files and compare the learning curves.

---

## Task 8: Run Full Experiment (20 tasks)

**Files:**
- None (execution only)

**Step 1: Run full MVP experiment**

```bash
python -m chaosbench.run_metacognitive --n-tasks 20 --timeout 600 --output mvp_full_run
```

**Step 2: Run full scaffolded experiment**

```bash
python -m chaosbench.run_metacognitive --n-tasks 20 --timeout 600 --scaffolded --output scaffolded_full_run
```

**Step 3: Compare results**

- Open both summary plots
- Check if Φ(n) curves show superlinear growth
- Compare MVP vs scaffolded performance

---

## Summary

| Task | Description | Estimated Effort |
|------|-------------|------------------|
| 1 | MVP prompt | 5 min |
| 2 | Scaffolded flag in SessionConfig | 15 min |
| 3 | Agent prompt selection | 10 min |
| 4 | CLI flag | 5 min |
| 5 | Visualization module | 30 min |
| 6 | CLI visualization | 10 min |
| 7 | Test run (5 tasks) | 5 min |
| 8 | Full run (20 tasks) | 15 min |

**Total: ~1.5 hours to first experiment results**
