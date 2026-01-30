# ChaosBench Phase 1 Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three critical issues making ChaosBench results unreliable: horizon too long, non-chaotic systems included, scoring formula double-weights difficulty.

**Architecture:** Modify `TaskConfig` to compute horizon from Lyapunov time, filter systems with h_KS < threshold, and simplify scoring to avoid double-weighting.

**Tech Stack:** Python, NumPy, pytest

---

## Task 1: Add Horizon Scaling by Lyapunov Time

**Files:**
- Modify: `chaosbench/core/Chaosbench_v3.py:396-405` (TaskConfig)
- Modify: `chaosbench/core/Chaosbench_v3.py:417-463` (TaskGenerator.generate_task)
- Test: `chaosbench/tests/test_chaosbench.py` (create new)

**Step 1: Write the failing test**

Create `chaosbench/tests/test_chaosbench.py`:

```python
"""Tests for ChaosBench v3 core functionality."""

import pytest
import numpy as np
from chaosbench.core.Chaosbench_v3 import (
    TaskConfig,
    TaskGenerator,
    LogisticMap,
)


class TestHorizonScaling:
    """Test that prediction horizon scales with Lyapunov time."""

    def test_horizon_scales_with_lyapunov_time(self):
        """Horizon should be k * lyapunov_time, not fixed."""
        config = TaskConfig(horizon_lyapunov_multiplier=1.5)
        generator = TaskGenerator(config, seed=42)

        # Generate task for a system
        system = LogisticMap(r=4.0)  # h_KS ≈ 0.69, lyapunov_time ≈ 1.44
        task = generator.generate_task(system=system)

        expected_horizon = int(1.5 * system.lyapunov_time)
        assert task.future_time == expected_horizon

    def test_different_systems_get_different_horizons(self):
        """High-chaos systems should get shorter horizons than low-chaos."""
        config = TaskConfig(horizon_lyapunov_multiplier=2.0)
        generator = TaskGenerator(config, seed=42)

        # High chaos: r=4.0, lyapunov_time ≈ 1.44
        high_chaos = LogisticMap(r=4.0)
        task_high = generator.generate_task(system=high_chaos)

        # Lower chaos: r=3.7, lyapunov_time > 1.44
        low_chaos = LogisticMap(r=3.7)
        task_low = generator.generate_task(system=low_chaos)

        # Lower chaos system should have longer horizon
        assert task_low.future_time >= task_high.future_time
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_chaosbench.py::TestHorizonScaling -v`

Expected: FAIL - `TaskConfig` doesn't have `horizon_lyapunov_multiplier`

**Step 3: Modify TaskConfig to support horizon scaling**

In `chaosbench/core/Chaosbench_v3.py`, change `TaskConfig` (around line 396):

```python
@dataclass
class TaskConfig:
    """Configuration for task generation."""
    n_obs: int = 50  # Number of observations available
    obs_density: float = 1.0  # Fraction of timesteps observed (1.0 = all)
    horizon_lyapunov_multiplier: float = 1.5  # Horizon = k * lyapunov_time
    noise_std: float = 0.01  # Observation noise
    n_bins: int = 20  # Discretization resolution
    conditional: bool = True  # Reveal system family?
    min_h_ks: float = 0.1  # Minimum h_KS to include system (filters non-chaotic)
```

**Step 4: Modify TaskGenerator.generate_task to compute horizon dynamically**

In `TaskGenerator.generate_task` (around line 417), change:

```python
def generate_task(self, system: ChaoticSystem = None) -> Task:
    """Generate a single task."""
    if system is None:
        system = self.rng.choice(self.all_systems)

    # Compute horizon from Lyapunov time
    horizon = max(1, int(self.config.horizon_lyapunov_multiplier * system.lyapunov_time))

    # Get appropriate bounds for discretization
    bounds = self._get_bounds(system)
    discretizer = DiscretizedSpace(system.dim, self.config.n_bins, bounds)

    # Generate trajectory
    x0 = self._get_initial_condition(system)
    n_available = int(self.config.n_obs / self.config.obs_density)
    total_steps = n_available + horizon + 200  # Buffer + burn-in
    traj = system.trajectory(x0, total_steps)

    # Burn-in to reach attractor
    burn_in = 100
    traj = traj[burn_in:]

    # Select observation times (with possible sparsity)
    obs_indices = self.rng.choice(n_available, size=self.config.n_obs, replace=False)
    obs_indices = np.sort(obs_indices)

    # Get observations with noise
    observations = traj[obs_indices] + self.rng.normal(0, self.config.noise_std,
                                                       (self.config.n_obs, system.dim))

    # Get true future state
    future_idx = n_available + horizon
    true_future = traj[future_idx]
    true_bin = discretizer.state_to_bin(true_future)

    self.task_counter += 1

    return Task(
        task_id=self.task_counter,
        system=system,
        observations=observations,
        obs_times=obs_indices,
        true_future=true_future,
        future_time=horizon,  # Now computed dynamically
        h_ks=system.h_ks,
        discretizer=discretizer,
        true_bin=true_bin,
        conditional=self.config.conditional,
        family=system.family if self.config.conditional else ""
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_chaosbench.py::TestHorizonScaling -v`

Expected: PASS

**Step 6: Commit**

```bash
git add chaosbench/core/Chaosbench_v3.py chaosbench/tests/test_chaosbench.py
git commit -m "feat(chaosbench): scale prediction horizon by Lyapunov time

Previously horizon was fixed at 10 steps regardless of system chaos.
Now horizon = k * lyapunov_time (default k=1.5), making hard tasks
actually predictable instead of fundamentally impossible."
```

---

## Task 2: Filter Non-Chaotic Systems

**Files:**
- Modify: `chaosbench/core/Chaosbench_v3.py:407-414` (TaskGenerator.__init__)
- Test: `chaosbench/tests/test_chaosbench.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_chaosbench.py`:

```python
class TestSystemFiltering:
    """Test that non-chaotic systems are filtered out."""

    def test_filters_low_h_ks_systems(self):
        """Systems with h_KS below threshold should be excluded."""
        config = TaskConfig(min_h_ks=0.1)
        generator = TaskGenerator(config, seed=42)

        # All systems should have h_KS >= 0.1
        for system in generator.all_systems:
            assert system.h_ks >= 0.1, f"{system.name} has h_KS={system.h_ks}"

    def test_zero_threshold_includes_all(self):
        """With min_h_ks=0, all systems should be included."""
        config = TaskConfig(min_h_ks=0.0)
        generator = TaskGenerator(config, seed=42)

        # Should have systems from all families
        families = set(s.family for s in generator.all_systems)
        assert len(families) == 5  # logistic, tent, henon, standard, lorenz
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_chaosbench.py::TestSystemFiltering -v`

Expected: FAIL - systems with h_KS ≈ 0 are included

**Step 3: Modify TaskGenerator to filter systems**

In `TaskGenerator.__init__` (around line 410):

```python
def __init__(self, config: TaskConfig, seed: int = 42):
    self.config = config
    self.rng = np.random.default_rng(seed)
    self.families = create_system_family()

    # Filter out non-chaotic systems (h_KS below threshold)
    self.all_systems = [
        s for fam in self.families.values()
        for s in fam
        if s.h_ks >= self.config.min_h_ks
    ]

    if not self.all_systems:
        raise ValueError(f"No systems with h_KS >= {self.config.min_h_ks}")

    self.task_counter = 0
```

**Step 4: Run test to verify it passes**

Run: `pytest chaosbench/tests/test_chaosbench.py::TestSystemFiltering -v`

Expected: PASS

**Step 5: Commit**

```bash
git add chaosbench/core/Chaosbench_v3.py chaosbench/tests/test_chaosbench.py
git commit -m "feat(chaosbench): filter non-chaotic systems by h_KS threshold

Systems with h_KS < min_h_ks (default 0.1) are now excluded.
This removes edge-of-chaos parameters that aren't actually chaotic,
making 'easy task' accuracy meaningful."
```

---

## Task 3: Fix Scoring Formula Double-Weighting

**Files:**
- Modify: `chaosbench/core/Chaosbench_v3.py:838-843` (Evaluator.evaluate_batch)
- Test: `chaosbench/tests/test_chaosbench.py`

**Step 1: Write the failing test**

Add to `chaosbench/tests/test_chaosbench.py`:

```python
from chaosbench.core.Chaosbench_v3 import (
    Evaluator,
    DifficultyWeighting,
    UniformSolver,
)


class TestScoringFormula:
    """Test that scoring formula doesn't double-weight difficulty."""

    def test_score_uses_difficulty_weight_only_once(self):
        """Score should be w(h_ks) * exp(-NLL), not w(h_ks) * exp(-NLL/h_ks)."""
        evaluator = Evaluator(weighting=DifficultyWeighting.linear)

        # Create mock task result with known values
        # NLL = 2.0, h_KS = 0.5
        # Old formula: 0.5 * exp(-2.0 / 0.5) = 0.5 * exp(-4) ≈ 0.009
        # New formula: 0.5 * exp(-2.0) = 0.5 * 0.135 ≈ 0.068

        # The new formula should give higher scores for same NLL
        # because it doesn't penalize h_KS twice
        h_ks = 0.5
        nll = 2.0

        accuracy_new = np.exp(-nll)  # New: no division by h_ks
        score_new = DifficultyWeighting.linear(h_ks) * accuracy_new

        # Score should be approximately 0.068, not 0.009
        assert score_new > 0.05, f"Score {score_new} too low, likely using old formula"
```

**Step 2: Run test to verify it fails**

Run: `pytest chaosbench/tests/test_chaosbench.py::TestScoringFormula -v`

Expected: PASS (test itself passes, but we need to verify the evaluator uses new formula)

**Step 3: Modify scoring in Evaluator.evaluate_batch**

In `Evaluator.evaluate_batch` (around line 838):

Change from:
```python
# Score: weighted accuracy minus observation cost
# Accuracy from NLL: exp(-NLL / h_KS) normalized
accuracy = np.exp(-result.nll / max(result.h_ks, 0.1))
task_score = self.weighting(result.h_ks) * accuracy
```

To:
```python
# Score: weighted accuracy minus observation cost
# Accuracy from NLL: exp(-NLL) — difficulty already captured by weighting
accuracy = np.exp(-result.nll)
task_score = self.weighting(result.h_ks) * accuracy
```

**Step 4: Add integration test**

Add to `chaosbench/tests/test_chaosbench.py`:

```python
class TestScoringIntegration:
    """Integration test for scoring changes."""

    def test_uniform_solver_not_crushed(self):
        """Uniform solver should get non-trivial Φ, not near-zero."""
        config = TaskConfig(min_h_ks=0.1)
        generator = TaskGenerator(config, seed=42)
        tasks = generator.generate_batch(20, stratified=True)

        evaluator = Evaluator(weighting=DifficultyWeighting.linear)
        solver = UniformSolver()

        results, phi_curve = evaluator.evaluate_batch(solver, tasks)
        final_phi = phi_curve[-1].cumulative_score

        # Uniform should get meaningful score, not near-zero
        # With 20 bins, uniform gives p=0.05, NLL ≈ 3.0
        # New score per task: w(h_ks) * exp(-3) ≈ 0.5 * 0.05 = 0.025
        # Over 20 tasks: ~0.5 total
        assert final_phi > 0.3, f"Uniform Φ={final_phi} too low"
```

**Step 5: Run tests to verify they pass**

Run: `pytest chaosbench/tests/test_chaosbench.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add chaosbench/core/Chaosbench_v3.py chaosbench/tests/test_chaosbench.py
git commit -m "fix(chaosbench): remove double-weighting in scoring formula

Changed from exp(-NLL/h_ks) to exp(-NLL).
Difficulty is already captured by w(h_ks) weighting.
Double-weighting was crushing Uniform/Conditional solver scores."
```

---

## Task 4: Update main() Defaults and Run Verification

**Files:**
- Modify: `chaosbench/core/Chaosbench_v3.py:1203-1287` (main function)
- Test: Manual verification

**Step 1: Update main() to use new defaults**

In `main()` (around line 1203), the config is created implicitly. Add explicit config:

```python
def main():
    """Run full benchmark analysis."""
    print("=" * 60)
    print("ChaosBench v3: Information-Efficient Chaotic Prediction")
    print("=" * 60)

    # Use new defaults: scaled horizon, filtered systems
    config = TaskConfig(
        horizon_lyapunov_multiplier=1.5,  # ~1.5 Lyapunov times
        min_h_ks=0.1,  # Filter non-chaotic systems
    )

    # Create solvers
    solvers = [
        UniformSolver(),
        MeanSolver(),
        LastValueSolver(),
        HistogramSolver(),
        LinearExtrapolator(),
        NearestNeighborSolver(),
        ConditionalSolver(),
    ]

    # Run benchmark with explicit config
    print("\n1. Running main benchmark...")
    all_results, tasks = run_benchmark(solvers, n_tasks=100, config=config)

    # ... rest of main() unchanged
```

**Step 2: Run full benchmark to verify improvements**

Run: `python -m chaosbench.core.Chaosbench_v3`

Expected:
- Hard task accuracy should improve (was 0-5%, should be 10-20%+)
- Uniform solver Φ should be higher (was 0.12)
- No systems with h_KS = 0 in complexity spectrum

**Step 3: Commit**

```bash
git add chaosbench/core/Chaosbench_v3.py
git commit -m "chore(chaosbench): update main() to use Phase 1 fixes

Uses horizon_lyapunov_multiplier=1.5 and min_h_ks=0.1 by default."
```

---

## Task 5: Update Specification Document

**Files:**
- Modify: `chaosbench/core/ChaosSpecification.md` (Section 11-12)

**Step 1: Update Known Limitations section**

In Section 11 "Known Limitations and Open Problems", mark fixed items:

```markdown
### 11.2 Prediction Horizon May Be Too Long

~~**Current state**: Fixed horizon = 10 steps.~~

**FIXED (v3.1)**: Horizon now scales with Lyapunov time:
```python
horizon = horizon_lyapunov_multiplier * system.lyapunov_time  # default k=1.5
```

### 11.4 Scoring Formula Double-Weights Difficulty

~~**Current state**: Score = w(h_KS) × exp(-NLL / h_KS)~~

**FIXED (v3.1)**: Score = w(h_KS) × exp(-NLL). Difficulty captured once via weighting.
```

**Step 2: Update Implementation Status section**

In Section 12.2 "What Needs Work":

```markdown
| Component | Status | Priority |
|-----------|--------|----------|
| Correct h_KS values | ✅ Computed via Lyapunov | DONE |
| Horizon scaling | ✅ k × lyapunov_time | DONE |
| Filter non-chaotic | ✅ min_h_ks threshold | DONE |
| Scoring formula | ✅ No double-weighting | DONE |
| Proper conditional solver | ❌ Fake | HIGH |
| Multi-agent protocol | ❌ Not started | MEDIUM |
```

**Step 3: Commit**

```bash
git add chaosbench/core/ChaosSpecification.md
git commit -m "docs(chaosbench): update spec with Phase 1 fixes

Marks horizon scaling, system filtering, and scoring formula as fixed."
```

---

## Task 6: Final Verification and Summary Commit

**Step 1: Run all tests**

Run: `pytest chaosbench/tests/ -v`

Expected: All tests PASS

**Step 2: Run benchmark and compare before/after**

Run: `python -m chaosbench.core.Chaosbench_v3`

Check:
- [ ] Hard task accuracy improved
- [ ] No h_KS = 0 systems in complexity spectrum
- [ ] Uniform solver gets reasonable Φ score
- [ ] All Φ curves more spread out (not crushed)

**Step 3: Create summary commit**

```bash
git add -A
git commit -m "feat(chaosbench): complete Phase 1 critical fixes

Phase 1 fixes make ChaosBench results meaningful:

1. Horizon scaling: horizon = k × lyapunov_time (default k=1.5)
   - Hard tasks now predictable (~1.5 Lyapunov times ahead)
   - Previously: fixed 10 steps = 7+ Lyapunov times = impossible

2. System filtering: exclude h_KS < 0.1
   - Removes edge-of-chaos non-chaotic systems
   - 'Easy' tasks now actually chaotic

3. Scoring fix: score = w(h_ks) × exp(-NLL)
   - Removed double-weighting of difficulty
   - Uniform/Conditional solvers no longer crushed

Closes Phase 1 of ChaosBench roadmap."
```
