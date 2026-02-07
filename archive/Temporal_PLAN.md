# Temporal Manifold Benchmark: Implementation Plan

> **Purpose:** This document contains ALL context needed to implement the temporal benchmark. Each section is self-contained with checkpoints that can be marked complete. A fresh Claude instance should be able to continue work from any checkpoint.

> **Status Update (2026-01-13):** Phase 0 COMPLETE. Repository restructured (shared/, coordination/, temporal/). Ready for Phase 1: Temporal Core Engine.

> **Relationship to Coordination Benchmark:** This benchmark shares the same repository as the existing 2-agent coordination benchmark. Shared utilities will be extracted to a `shared/` module. See Section 9 for the restructured file layout.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mathematical Specification](#2-mathematical-specification)
3. [Dynamics System](#3-dynamics-system)
4. [Observation Model](#4-observation-model)
5. [Episode Structure](#5-episode-structure)
6. [Scoring System](#6-scoring-system)
7. [Surface Scenarios](#7-surface-scenarios)
8. [Implementation Checkpoints](#8-implementation-checkpoints)
9. [File Structure](#9-file-structure)
10. [API Specifications](#10-api-specifications)
11. [Prompt Templates](#11-prompt-templates)
12. [Test Cases](#12-test-cases)
13. [Visualization Requirements](#13-visualization-requirements)
14. [Baselines](#14-baselines)
15. [Evaluation Protocol](#15-evaluation-protocol)

---

## 1. Project Overview

### 1.1 Research Question

Can LLM agents learn temporal dynamics from sparse local observations and make accurate predictions about future states of a changing environment?

### 1.2 Core Concept

A single LLM agent navigates a 1D surface that **evolves over time** according to hidden rules:

- **The agent** controls position x on a 1D curve
- **Time** progresses automatically each timestep
- **The surface** f(x, t) changes according to hidden dynamics (peaks move, grow, shrink)
- **The agent** sees only a local neighborhood around their position
- **Pattern recognition** is necessary to predict future states

### 1.3 Why This Tests Temporal Reasoning

| Property | How It's Achieved |
|----------|-------------------|
| Partial observability | Agent sees local slice, not full curve |
| Hidden dynamics | Evolution rules are not revealed |
| Memory requirement | Must track observations across time |
| Prediction under uncertainty | Test phase requires blind predictions |
| Exploration-exploitation tradeoff | Limited exploration budget |

### 1.4 Comparison with Coordination Benchmark

| Aspect | Coordination Benchmark | Temporal Benchmark |
|--------|----------------------|-------------------|
| Agents | 2 agents (A controls x, B controls y) | 1 agent (controls x) |
| Second dimension | Spatial (y) — agent-controlled | Temporal (t) — automatic |
| Information asymmetry | Perpendicular spatial slices | Limited temporal window |
| Core challenge | Communication & coordination | Pattern recognition & prediction |
| Observation | Local slice + ∂f/∂x or ∂f/∂y | Local slice + ∂f/∂x + ∂f/∂t |
| Scoring | Final position score | Cumulative reward |

### 1.5 Success Criteria

- [ ] LLM agents outperform random baseline significantly
- [ ] LLM agents learn simple dynamics (linear drift) within few exploration runs
- [ ] Performance degrades gracefully with dynamics complexity
- [ ] Exploration efficiency correlates with test performance
- [ ] Results are reproducible across runs

---

## 2. Mathematical Specification

### 2.1 Domain

```
Spatial domain:    x ∈ [0, 10]
Temporal domain:   t ∈ [0, T] where T = 20 (timesteps per run)
Surface function:  f: [0, 10] × [0, T] → ℝ
```

### 2.2 State

```
Agent position:    x ∈ [0, 10]
Current timestep:  t ∈ {0, 1, ..., T-1}
Exploration run:   r ∈ {1, 2, ..., R_max}
Phase:             "exploration" or "test"
```

### 2.3 Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Domain size | L | 10.0 | Surface spans [0, L] |
| Observation radius | R | 0.5 | Agent sees slice of length 2R centered at position |
| Timesteps per run | T | 20 | Length of each exploration/test run |
| Slice samples | S | 11 | Number of discrete samples in each slice |
| Initial position | x₀ | 5.0 | Starting point (center of domain) |
| Max exploration runs | R_max | 10 | Safety limit for exploration budget |

### 2.4 Gradient Definitions

```
Spatial gradient:
  ∂f/∂x ≈ (f(x + ε, t) - f(x - ε, t)) / (2ε)

Temporal gradient:
  ∂f/∂t ≈ (f(x, t + δ) - f(x, t)) / δ

Where:
  ε = 0.001 (spatial perturbation)
  δ = 1.0 (one timestep for temporal gradient)
```

### 2.5 Temporal Gradient Interpretation

The temporal gradient ∂f/∂t at position x tells the agent:

| ∂f/∂t Value | Interpretation |
|-------------|----------------|
| Positive | Value at this x is increasing over time (peak approaching or growing) |
| Negative | Value at this x is decreasing over time (peak leaving or shrinking) |
| Near zero | Value at this x is stable |

This provides crucial information about dynamics without revealing the underlying rule.

---

## 3. Dynamics System

### 3.1 Base Surface Definition

The instantaneous surface at time t is a sum of Gaussian peaks with time-varying parameters:

```python
def gaussian_peak(x, cx, height, sigma):
    """Single Gaussian peak centered at cx."""
    return height * np.exp(-((x - cx) ** 2) / (2 * sigma ** 2))

def surface_at_time(x, t, peaks, dynamics):
    """
    Surface value at position x and time t.

    peaks: list of initial peak definitions {cx, height, sigma}
    dynamics: list of DynamicsRule objects that modify peaks over time
    """
    total = 0.0
    for i, peak in enumerate(peaks):
        # Apply dynamics to get parameters at time t
        params_t = apply_dynamics(peak, dynamics[i], t)
        total += gaussian_peak(x, params_t['cx'], params_t['height'], params_t['sigma'])
    return total
```

### 3.2 Dynamics Rule Types

#### 3.2.1 LinearDrift
Peak center moves at constant velocity.

```python
class LinearDrift:
    """Peak moves linearly in x."""

    def __init__(self, velocity: float):
        """
        Args:
            velocity: Units per timestep (positive = right, negative = left)
        """
        self.velocity = velocity

    def apply(self, initial_cx: float, t: float) -> float:
        """Return cx at time t."""
        return initial_cx + self.velocity * t
```

#### 3.2.2 LinearHeightChange
Peak height changes linearly over time.

```python
class LinearHeightChange:
    """Peak grows or shrinks linearly."""

    def __init__(self, rate: float):
        """
        Args:
            rate: Height change per timestep (positive = grow, negative = shrink)
        """
        self.rate = rate

    def apply(self, initial_height: float, t: float) -> float:
        """Return height at time t (clamped to [0, ∞))."""
        return max(0.0, initial_height + self.rate * t)
```

#### 3.2.3 Oscillation
Peak center oscillates sinusoidally.

```python
class Oscillation:
    """Peak oscillates back and forth."""

    def __init__(self, amplitude: float, period: float):
        """
        Args:
            amplitude: Max displacement from initial position
            period: Timesteps for one full cycle
        """
        self.amplitude = amplitude
        self.period = period

    def apply(self, initial_cx: float, t: float) -> float:
        """Return cx at time t."""
        return initial_cx + self.amplitude * np.sin(2 * np.pi * t / self.period)
```

#### 3.2.4 CompositeDynamics
Combine multiple dynamics rules.

```python
class CompositeDynamics:
    """Apply multiple dynamics rules to different parameters."""

    def __init__(self, cx_rule=None, height_rule=None, sigma_rule=None):
        self.cx_rule = cx_rule
        self.height_rule = height_rule
        self.sigma_rule = sigma_rule

    def apply_all(self, initial_params: dict, t: float) -> dict:
        """Return all parameters at time t."""
        params = initial_params.copy()

        if self.cx_rule:
            params['cx'] = self.cx_rule.apply(initial_params['cx'], t)
        if self.height_rule:
            params['height'] = self.height_rule.apply(initial_params['height'], t)
        if self.sigma_rule:
            params['sigma'] = self.sigma_rule.apply(initial_params['sigma'], t)

        return params
```

### 3.3 Dynamics Visualization

The full manifold can be visualized as a 3D surface:

```
    f(x,t)
      ▲
      │     ╱╲___          Peak at t=0
      │    ╱    ╲
      │   ╱      ╲
      │  ╱        ╲
      └──────────────────► x
     ╱
    ╱        ╱╲___          Peak at t=10 (has moved right)
   ╱        ╱    ╲
  t        ╱      ╲
          ╱        ╲
```

---

## 4. Observation Model

### 4.1 Core Principle

The agent sees a **local 1D slice** of the surface at the current time, plus gradient information in both space and time.

```
        f(x)
          ▲
          │    ╱╲
          │   ╱  ╲
          │  ╱    ╲         Full curve (hidden from agent)
          │ ╱      ╲
          │╱        ╲____
          └──────────────────► x
              ┃    ┃
              ┃    ┃         Agent's observation window
              ┃ ●  ┃         ● = agent position
              ┃    ┃
            x-R   x+R
```

### 4.2 Observation Structure

At position x and time t, the agent receives:

```python
observation = {
    # Current state
    "position": x,              # Agent's x position
    "timestep": t,              # Current timestep (0 to T-1)

    # Value at current position
    "value_at_position": f(x, t),

    # Spatial gradient (how f changes in x direction)
    "gradient_x": ∂f/∂x at (x, t),

    # Temporal gradient (how f changes over time at this x)
    "gradient_t": ∂f/∂t at (x, t),

    # Local slice: f(x', t) for x' in [x - R, x + R]
    "slice": [
        {"x": x - R, "value": f(x - R, t)},
        {"x": x - R + step, "value": f(x - R + step, t)},
        ...
        {"x": x + R, "value": f(x + R, t)}
    ]
    # Total of S samples, evenly spaced
}
```

### 4.3 Boundary Handling

When observation radius extends beyond domain bounds:

```python
# Clamp slice to valid domain
x_min_slice = max(0, x - R)
x_max_slice = min(L, x + R)

# Samples are taken at uniform 0.1 spacing within valid range
# Edge positions have FEWER samples (not compressed spacing)
# Example: at x=0.2 with R=0.5, slice covers [0, 0.7] = 8 samples
#          at x=5.0 with R=0.5, slice covers [4.5, 5.5] = 11 samples
```

### 4.4 What the Agent CANNOT See

| Cannot See | Implication |
|------------|-------------|
| Full curve shape | Must explore to map the surface |
| Dynamics rule | Must infer from observations over time |
| Future states | Must predict based on learned pattern |
| Other positions' temporal gradients | Only sees ∂f/∂t at current x |

---

## 5. Episode Structure

### 5.1 Exploration Modes

The benchmark supports two exploration modes:

| Mode | Description | LLM Calls/Run | Primary Use |
|------|-------------|---------------|-------------|
| **Function mode** | Agent proposes all 20 positions at once, receives detailed report | 2 | Evaluation, experiments |
| **Play mode** | Agent prompted at each timestep individually | 20 | Future: human players, debugging |

**Default:** Function mode (much more cost-efficient for LLM evaluation)

**Play mode:** Reserved for future enhancement — will allow human players to control the agent interactively, making decisions at each timestep.

```python
class ExplorationMode(Enum):
    FUNCTION = "function"  # Batch: propose 20 positions, get report (DEFAULT)
    PLAY = "play"          # Interactive: prompt each timestep (FUTURE)
```

### 5.2 Two-Phase Design

```
┌─────────────────────────────────────────────────────────────────┐
│                 EXPLORATION PHASE (Function Mode)                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Run 1: Agent proposes [x_0, x_1, ..., x_19]              │  │
│  │        System returns REPORT with all observations       │  │
│  │        Report stored in agent memory                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Run 2: Agent sees Run 1 results, proposes new positions  │  │
│  │        System returns REPORT, stored in memory           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│                            ...                                  │
│                              ↓                                  │
│              Agent outputs: READY_FOR_TEST                      │
│                                                                 │
│  Agent decides when to stop (max 10 runs)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         TEST PHASE                              │
│                                                                 │
│  Agent submits 20 positions UPFRONT (blind prediction)          │
│  Based on patterns learned from exploration reports             │
│                                                                 │
│  predictions = [x_0, x_1, x_2, ..., x_19]                       │
│                                                                 │
│  Score = Σ f(x_t, t) / Σ f(x_optimal, t)                        │
│                                                                 │
│  Time range: Configurable (same t=0-19 OR future t=20-39)       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Exploration Phase Detail (Function Mode)

In function mode, each exploration run works as follows:

```python
for run in range(1, max_runs + 1):  # max_runs = 10
    # Agent receives: initial observation + memory of previous runs
    initial_obs = generate_observation(x=5.0, t=0)

    # Agent proposes ALL 20 positions at once
    positions = agent.propose_positions(initial_obs, memory)  # List of 20 floats

    # Check for early termination
    if positions == "READY_FOR_TEST":
        break_to_test_phase()

    # System executes all positions and builds report
    report = ExplorationReport()
    cumulative_reward = 0.0

    for t in range(T):  # T = 20 timesteps
        x = clamp(positions[t], 0, L)
        observation = generate_observation(x, t)
        reward = surface.evaluate(x, t)
        cumulative_reward += reward

        report.add_timestep(
            t=t,
            position=x,
            value=observation["value_at_position"],
            gradient_x=observation["gradient_x"],
            gradient_t=observation["gradient_t"]
        )

    # Compute summary statistics
    optimal_reward = sum(surface.evaluate(optimal_x(t), t) for t in range(T))
    report.set_summary(
        total_score=cumulative_reward / optimal_reward,
        best_timestep=...,   # Timestep with highest relative score
        worst_timestep=...   # Timestep with lowest relative score
    )

    # Store report in agent memory
    agent.receive_report(report)
```

### 5.4 Exploration Report Structure

Each exploration run returns a detailed report to the agent:

```python
@dataclass
class ExplorationReport:
    """Report returned to agent after each exploration run."""

    run_number: int

    # Per-timestep data (list of 20 entries)
    timesteps: List[dict]  # Each: {t, position, value, gradient_x, gradient_t}

    # Summary statistics
    total_score: float      # Normalized score (0-1)
    best_timestep: int      # t with highest relative performance
    worst_timestep: int     # t with lowest relative performance

    def format_for_llm(self) -> str:
        """Format report as readable text for LLM prompt."""
        lines = [f"=== EXPLORATION RUN {self.run_number} REPORT ===\n"]
        lines.append("PER-TIMESTEP RESULTS:")
        for ts in self.timesteps:
            lines.append(
                f"  t={ts['t']:2d}: x={ts['position']:.2f}, "
                f"f={ts['value']:.4f}, "
                f"∂f/∂x={ts['gradient_x']:+.4f}, "
                f"∂f/∂t={ts['gradient_t']:+.4f}"
            )
        lines.append(f"\nSUMMARY:")
        lines.append(f"  Total score: {self.total_score:.2%} of optimal")
        lines.append(f"  Best timestep: t={self.best_timestep}")
        lines.append(f"  Worst timestep: t={self.worst_timestep}")
        return "\n".join(lines)
```

### 5.5 Test Phase Detail

```python
# Agent submits all predictions at once
predictions = agent.predict_positions(n_predictions=20)  # List of 20 floats

# Evaluate predictions (NO observations given during test)
test_reward = 0.0
optimal_reward = 0.0

for t in range(20):
    # Time range depends on configuration
    eval_t = t  # Same range (t=0-19)
    # OR eval_t = t + 20  # Future range (t=20-39)

    test_reward += surface.evaluate(predictions[t], eval_t)
    optimal_reward += surface.evaluate(optimal_x(eval_t), eval_t)

test_score = test_reward / optimal_reward
```

### 5.6 Agent Decision Flow (Function Mode)

Each exploration run in function mode:

```
┌──────────────────────────────────────────────────────────────┐
│                    AGENT DECISION POINT                      │
│                                                              │
│  Input:                                                      │
│    - Initial observation at t=0, x=5.0                       │
│    - Memory: Full reports from all previous runs             │
│                                                              │
│  Output (one of):                                            │
│    - List of 20 floats: Positions for t=0 through t=19      │
│    - "READY_FOR_TEST": End exploration, begin test phase     │
│                                                              │
│  After output:                                               │
│    - System executes all 20 positions                        │
│    - Agent receives ExplorationReport                        │
│    - Report added to memory for next run                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.7 Key Timing Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Timesteps per run | 20 | Both exploration and test |
| Max exploration runs | 10 | Agent can stop earlier |
| Initial position | 5.0 | Center of domain, fixed |
| Test time range | Configurable | Same (0-19) or Future (20-39) |

---

## 6. Scoring System

### 6.1 Exploration Score (Per Run)

```python
exploration_score = Σ f(x_t, t) / Σ f(x_optimal(t), t)
```

Where x_optimal(t) is the position of the global maximum at time t.

| Score | Interpretation |
|-------|----------------|
| 1.0 | Perfect tracking (always at peak) |
| 0.8-0.99 | Excellent tracking |
| 0.5-0.8 | Moderate tracking |
| < 0.5 | Poor tracking |

### 6.2 Test Score (Primary Metric)

```python
test_score = Σ f(predictions[t], t) / Σ f(x_optimal(t), t)
```

This is the **main evaluation metric** — measures prediction accuracy.

### 6.3 Secondary Metrics

```python
# Mean absolute error from optimal
mae = mean(|predictions[t] - x_optimal(t)| for t in range(T))

# Exploration efficiency
exploration_efficiency = test_score / num_exploration_runs

# Prediction consistency
prediction_variance = var(predictions)  # High variance may indicate confusion

# Learning curve
learning_curve = [run_scores for each exploration run]  # Should improve
```

### 6.4 Aggregation Across Trials

For statistical validity:
- Run each (dynamics_scenario, agent) combination K times (K ≥ 5)
- Report: mean score, std deviation, min, max
- Use different random seeds for LLM sampling

---

## 7. Surface Scenarios

### 7.1 Tier 1: Simple (Single Transformation)

#### 7.1.1 Stationary
```python
scenario_stationary = {
    "name": "stationary",
    "description": "No movement - baseline test",
    "peaks": [{"cx": 7.0, "height": 1.0, "sigma": 1.0}],
    "dynamics": [None],  # No dynamics
    "difficulty": 1,
    "expected_score": "> 0.95 for any reasonable agent"
}
```

#### 7.1.2 Linear Drift (Slow)
```python
scenario_linear_slow = {
    "name": "linear_drift_slow",
    "description": "Peak moves slowly to the right",
    "peaks": [{"cx": 3.0, "height": 1.0, "sigma": 1.0}],
    "dynamics": [LinearDrift(velocity=0.1)],  # Moves 2 units over 20 timesteps
    "difficulty": 2,
    "expected_score": "> 0.8 after 2-3 exploration runs"
}
```

#### 7.1.3 Linear Drift (Fast)
```python
scenario_linear_fast = {
    "name": "linear_drift_fast",
    "description": "Peak moves quickly to the right",
    "peaks": [{"cx": 2.0, "height": 1.0, "sigma": 1.0}],
    "dynamics": [LinearDrift(velocity=0.3)],  # Moves 6 units over 20 timesteps
    "difficulty": 3,
    "expected_score": "> 0.7 after 3-4 exploration runs"
}
```

#### 7.1.4 Linear Growth
```python
scenario_growth = {
    "name": "linear_growth",
    "description": "Stationary peak that grows over time",
    "peaks": [{"cx": 5.0, "height": 0.3, "sigma": 1.0}],
    "dynamics": [LinearHeightChange(rate=0.035)],  # Grows from 0.3 to 1.0
    "difficulty": 2,
    "expected_score": "> 0.85"
}
```

#### 7.1.5 Linear Decay
```python
scenario_decay = {
    "name": "linear_decay",
    "description": "Stationary peak that shrinks over time",
    "peaks": [{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
    "dynamics": [LinearHeightChange(rate=-0.04)],  # Shrinks from 1.0 to 0.2
    "difficulty": 2,
    "expected_score": "> 0.85"
}
```

### 7.2 Tier 2: Medium (Predictable Patterns)

#### 7.2.1 Oscillation
```python
scenario_oscillation = {
    "name": "oscillation",
    "description": "Peak moves back and forth sinusoidally",
    "peaks": [{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
    "dynamics": [Oscillation(amplitude=2.0, period=10)],  # 2 full cycles in 20 steps
    "difficulty": 4,
    "expected_score": "> 0.7 after 4-5 exploration runs"
}
```

#### 7.2.2 Two Peaks Swap
```python
scenario_swap = {
    "name": "two_peaks_swap",
    "description": "One peak shrinks while another grows",
    "peaks": [
        {"cx": 3.0, "height": 1.0, "sigma": 1.0},
        {"cx": 7.0, "height": 0.0, "sigma": 1.0}
    ],
    "dynamics": [
        LinearHeightChange(rate=-0.05),  # 1.0 → 0.0
        LinearHeightChange(rate=0.05)    # 0.0 → 1.0
    ],
    "difficulty": 5,
    "expected_score": "> 0.6 - requires recognizing the swap"
}
```

### 7.3 Tier 3: Hard (Complex Dynamics)

#### 7.3.1 Drift and Grow
```python
scenario_drift_grow = {
    "name": "drift_and_grow",
    "description": "Peak moves AND changes height simultaneously",
    "peaks": [{"cx": 2.0, "height": 0.5, "sigma": 1.0}],
    "dynamics": [CompositeDynamics(
        cx_rule=LinearDrift(velocity=0.2),
        height_rule=LinearHeightChange(rate=0.025)
    )],
    "difficulty": 5,
    "expected_score": "> 0.55"
}
```

#### 7.3.2 Two Peaks Crossover
```python
scenario_crossover = {
    "name": "two_peaks_crossover",
    "description": "Two peaks move toward each other and cross",
    "peaks": [
        {"cx": 2.0, "height": 0.8, "sigma": 1.0},
        {"cx": 8.0, "height": 1.0, "sigma": 1.0}
    ],
    "dynamics": [
        LinearDrift(velocity=0.2),   # Moves right
        LinearDrift(velocity=-0.2)   # Moves left
    ],
    "difficulty": 6,
    "expected_score": "> 0.5 - must track dominant peak through crossover"
}
```

### 7.3 Test Scenarios Summary

```python
TEST_SCENARIOS = {
    # Tier 1 - Simple (5 scenarios)
    "stationary": {...},
    "linear_drift_slow": {...},
    "linear_drift_fast": {...},
    "linear_growth": {...},
    "linear_decay": {...},

    # Tier 2 - Medium (2 scenarios)
    "oscillation": {...},
    "two_peaks_swap": {...},

    # Tier 3 - Hard (2 scenarios)
    "drift_and_grow": {...},
    "two_peaks_crossover": {...}
}
# Total: 9 scenarios
```

---

## 8. Implementation Checkpoints

### Phase 0: Repository Restructure ✅ COMPLETE

#### Checkpoint 0.1: Create New Directory Structure ✅
- [x] **Task:** Create `shared/`, `coordination/`, `temporal/` directories
- [x] **Verification:** Directory structure matches Section 9

#### Checkpoint 0.2: Move Existing Code ✅
- [x] **Task:** Move `manifold_benchmark/` contents to `coordination/`
- [x] **Update:** All imports within coordination module
- [x] **Verification:** `pytest coordination/tests/` passes (144 tests)

#### Checkpoint 0.3: Extract Shared Utilities ✅
- [x] **Files to create:**
  - [x] `shared/gaussians.py` — Gaussian peak math (1D and 2D)
  - [x] `shared/llm_utils.py` — LiteLLM wrapper, retry logic, parsing
  - [x] `shared/logging.py` — Result logging patterns
  - [x] `shared/base_agent.py` — Abstract agent interface
- [x] **Update:** Coordination module to import from shared
- [x] **Verification:** All coordination tests still pass (144 tests)

---

### Phase 1: Temporal Core Engine

#### Checkpoint 1.1: Temporal Surface Class
- [ ] **File:** `temporal/core/surface.py`
- [ ] **Class:** `TemporalSurface`
- [ ] **Requirements:**
  - [ ] Constructor takes list of peaks + dynamics rules
  - [ ] `evaluate(x, t) -> float`: Returns f(x, t)
  - [ ] `gradient_x(x, t) -> float`: Returns ∂f/∂x
  - [ ] `gradient_t(x, t) -> float`: Returns ∂f/∂t
  - [ ] `get_optimal(t) -> Tuple[float, float]`: Returns (x_opt, f_opt) at time t
  - [ ] Serialization: `to_dict()` / `from_dict()`
- [ ] **Tests:**
  - [ ] Stationary surface evaluates correctly
  - [ ] Moving peak position correct at different t
  - [ ] Gradients have correct signs

#### Checkpoint 1.2: Dynamics System
- [ ] **File:** `temporal/core/dynamics.py`
- [ ] **Classes:**
  - [ ] `DynamicsRule` (abstract base)
  - [ ] `LinearDrift`
  - [ ] `LinearHeightChange`
  - [ ] `Oscillation`
  - [ ] `Bounce`
  - [ ] `CompositeDynamics`
- [ ] **Requirements:**
  - [ ] Each rule has `apply(initial_value, t) -> value_at_t`
  - [ ] Bounce correctly reflects at boundaries
  - [ ] CompositeDynamics combines multiple rules
- [ ] **Tests:**
  - [ ] LinearDrift produces correct positions
  - [ ] Bounce reflects at boundaries
  - [ ] CompositeDynamics applies all rules

#### Checkpoint 1.3: Observation Generator
- [ ] **File:** `temporal/core/observation.py`
- [ ] **Class:** `TemporalObservationGenerator`
- [ ] **Requirements:**
  - [ ] Constructor takes TemporalSurface and parameters (R, S)
  - [ ] `generate_observation(x, t) -> dict`
  - [ ] Returns: position, timestep, value, gradient_x, gradient_t, slice
  - [ ] Boundary handling for edge positions
- [ ] **Tests:**
  - [ ] Slice has correct number of samples
  - [ ] Slice range is [x-R, x+R] (clamped)
  - [ ] Both gradients included and correct sign

#### Checkpoint 1.4: Episode State Manager
- [ ] **File:** `temporal/core/episode.py`
- [ ] **Class:** `TemporalEpisode`
- [ ] **Requirements:**
  - [ ] Constructor takes TemporalSurface, config
  - [ ] `exploration_step(x_new) -> observation`: Move and get observation
  - [ ] `end_exploration_run() -> run_result`: Finish run, return score + log
  - [ ] `submit_predictions(predictions) -> test_result`: Submit test predictions
  - [ ] Tracks: current run, timestep, phase, full history
  - [ ] Computes optimal trajectory for scoring
- [ ] **Tests:**
  - [ ] Timestep increments correctly
  - [ ] Score computed correctly
  - [ ] Phase transitions work

---

### Phase 2: Temporal Agents

#### Checkpoint 2.1: Temporal Base Agent
- [ ] **File:** `temporal/agents/base.py`
- [ ] **Class:** `TemporalBaseAgent` (abstract)
- [ ] **Requirements:**
  - [ ] `receive_observation(observation) -> None`
  - [ ] `decide_action() -> Union[float, str]`: Returns position or "READY_FOR_TEST"
  - [ ] `receive_run_feedback(score, run_log) -> None`
  - [ ] `predict_positions(n) -> List[float]`: For test phase
  - [ ] `reset() -> None`: Reset for new scenario

#### Checkpoint 2.2: Random Baseline Agent
- [ ] **File:** `temporal/agents/random_agent.py`
- [ ] **Class:** `TemporalRandomAgent`
- [ ] **Requirements:**
  - [ ] Returns random position in [0, 10] each step
  - [ ] Fixed number of exploration runs (e.g., 3)
  - [ ] Random test predictions
  - [ ] Deterministic with seed
- [ ] **Tests:**
  - [ ] Positions within bounds
  - [ ] Same seed = same sequence

#### Checkpoint 2.3: Greedy Baseline Agent
- [ ] **File:** `temporal/agents/greedy_agent.py`
- [ ] **Class:** `TemporalGreedyAgent`
- [ ] **Requirements:**
  - [ ] Follows spatial gradient (ignores temporal)
  - [ ] `new_pos = clamp(current + step_size * sign(gradient_x))`
  - [ ] Fixed exploration runs
  - [ ] Test predictions = last known good positions
- [ ] **Tests:**
  - [ ] Moves in gradient direction
  - [ ] Respects bounds

#### Checkpoint 2.4: LLM Agent
- [ ] **File:** `temporal/agents/llm_agent.py`
- [ ] **Class:** `TemporalLLMAgent`
- [ ] **Requirements:**
  - [ ] Uses shared/llm_utils for API calls
  - [ ] Formats observations for natural language
  - [ ] Maintains history across runs
  - [ ] Parses position from response
  - [ ] Recognizes "READY_FOR_TEST" decision
  - [ ] Generates 20 predictions for test phase
- [ ] **Configuration:**
  - [ ] System prompt loaded from file
  - [ ] Temperature, max_tokens parameters
- [ ] **Tests:**
  - [ ] Observation formatting correct
  - [ ] Position parsing handles edge cases
  - [ ] READY_FOR_TEST recognized

#### Checkpoint 2.5: System Prompt
- [ ] **File:** `temporal/prompts/temporal_agent_system.txt`
- [ ] **Requirements:**
  - [ ] Explains the task clearly
  - [ ] Describes observation structure
  - [ ] Explains exploration/test phases
  - [ ] Specifies output format
  - [ ] Provides strategy hints
- [ ] **Review:** Test with actual LLM calls

---

### Phase 3: Temporal Experiments

#### Checkpoint 3.1: Episode Runner
- [ ] **File:** `temporal/experiments/runner.py`
- [ ] **Class:** `TemporalEpisodeRunner`
- [ ] **Requirements:**
  - [ ] Orchestrates exploration runs
  - [ ] Handles READY_FOR_TEST signal
  - [ ] Executes test phase
  - [ ] Returns full transcript + scores
- [ ] **Tests:**
  - [ ] Correct number of timesteps per run
  - [ ] Feedback delivered after each run
  - [ ] Test predictions evaluated correctly

#### Checkpoint 3.2: Result Logger
- [ ] **File:** `temporal/experiments/logger.py`
- [ ] **Class:** `TemporalResultLogger`
- [ ] **Requirements:**
  - [ ] Save results to JSON
  - [ ] Include: scenario config, agent config, exploration logs, predictions, scores
  - [ ] Timestamp and unique ID
  - [ ] Load results for analysis
- [ ] **Output format:** See Section 10.6

#### Checkpoint 3.3: Batch Evaluation
- [ ] **File:** `temporal/experiments/eval.py`
- [ ] **Function:** `run_temporal_evaluation(scenarios, agent_configs, n_trials, output_dir)`
- [ ] **Requirements:**
  - [ ] Run all scenario × agent combinations
  - [ ] Multiple trials per combination
  - [ ] Progress bar
  - [ ] Save all results

#### Checkpoint 3.4: Statistical Analysis
- [ ] **File:** `temporal/experiments/analysis.py`
- [ ] **Functions:**
  - [ ] `load_results(directory) -> List[dict]`
  - [ ] `compute_summary_stats(results) -> DataFrame`
  - [ ] `compare_agents(results, agent_a, agent_b) -> dict`
  - [ ] `plot_results(summary) -> Figure`
- [ ] **Requirements:**
  - [ ] Group by agent × scenario
  - [ ] Compute mean, std, min, max
  - [ ] Statistical significance tests

---

### Phase 4: Visualization

#### Checkpoint 4.1: 3D Surface Evolution Plot
- [ ] **File:** `temporal/visualization/surface_evolution.py`
- [ ] **Function:** `plot_surface_evolution(surface, t_range, show=True, save_path=None)`
- [ ] **Requirements:**
  - [ ] 3D plot with x, t, f(x,t) axes
  - [ ] Colormap showing height
  - [ ] Optimal trajectory line (red)
  - [ ] Time axis labeled

#### Checkpoint 4.2: Agent Trajectory Plot
- [ ] **File:** `temporal/visualization/trajectory.py`
- [ ] **Function:** `plot_trajectory(surface, history, show=True, save_path=None)`
- [ ] **Requirements:**
  - [ ] 2D plot: x vs t
  - [ ] Agent path as line
  - [ ] Optimal path as dashed line
  - [ ] Color coding by value achieved

#### Checkpoint 4.3: Prediction Comparison Plot
- [ ] **File:** `temporal/visualization/prediction.py`
- [ ] **Function:** `plot_predictions(predictions, optimal, surface)`
- [ ] **Requirements:**
  - [ ] Predicted positions vs optimal positions
  - [ ] Error bars or shading for deviation
  - [ ] Score annotation

#### Checkpoint 4.4: Slice Evolution Plot
- [ ] **File:** `temporal/visualization/slices.py`
- [ ] **Function:** `plot_slice_evolution(surface, x, t_range)`
- [ ] **Requirements:**
  - [ ] Show how slice at position x changes over time
  - [ ] Animated or multi-panel

---

### Phase 5: Integration & Validation

#### Checkpoint 5.1: End-to-End Test (Random Agent)
- [ ] **Task:** Run random agent on stationary surface
- [ ] **Expected:** Low score (~0.2), no crashes
- [ ] **Verification:** Full pipeline works

#### Checkpoint 5.2: End-to-End Test (Greedy Agent)
- [ ] **Task:** Run greedy agent on linear drift
- [ ] **Expected:** Medium score (~0.5-0.6)
- [ ] **Verification:** Gradient following works

#### Checkpoint 5.3: End-to-End Test (LLM Agent)
- [ ] **Task:** Run LLM agent on linear drift slow
- [ ] **Expected:** Good score (>0.7) after exploration
- [ ] **Verification:** LLM integration works

#### Checkpoint 5.4: Documentation Update
- [ ] **Update:** CLAUDE.md with temporal benchmark info
- [ ] **Create:** README section for temporal benchmark
- [ ] **Verify:** Commands documented and working

---

## 9. File Structure

### 9.1 Restructured Repository Layout

```
DISS/
│
├── shared/                           # Shared utilities across benchmarks
│   ├── __init__.py
│   ├── gaussians.py                  # Gaussian peak math
│   ├── llm_utils.py                  # LiteLLM wrapper, retry logic, parsing
│   ├── logging.py                    # Result logging patterns
│   └── base_agent.py                 # Abstract base agent interface
│
├── coordination/                     # Original 2-agent benchmark
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── surface.py                # 2D surface f(x,y)
│   │   ├── observation.py            # Perpendicular slice generation
│   │   └── episode.py                # 2-agent episode state
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Imports from shared
│   │   ├── random_agent.py
│   │   ├── greedy_agent.py
│   │   └── llm_agent.py              # Uses shared/llm_utils
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── logger.py
│   │   ├── eval.py
│   │   ├── analysis.py
│   │   └── transcript.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot3d.py
│   │   └── slices.py
│   ├── prompts/
│   │   ├── agent_a_system.txt
│   │   └── agent_b_system.txt
│   └── tests/
│       ├── test_surface.py
│       ├── test_observation.py
│       ├── test_episode.py
│       └── test_agents.py
│
├── temporal/                         # NEW: Temporal tracking benchmark
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── surface.py                # 1D time-varying surface f(x,t)
│   │   ├── dynamics.py               # Evolution rules
│   │   ├── observation.py            # Local slice + dual gradients
│   │   └── episode.py                # Exploration/test phase management
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Temporal-specific interface
│   │   ├── random_agent.py
│   │   ├── greedy_agent.py
│   │   └── llm_agent.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── runner.py                 # Exploration/test executor
│   │   ├── logger.py
│   │   ├── eval.py
│   │   └── analysis.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── surface_evolution.py      # 3D x-t-f plot
│   │   ├── trajectory.py             # Agent path in x-t space
│   │   ├── prediction.py             # Predicted vs actual
│   │   └── slices.py                 # Slice evolution
│   ├── prompts/
│   │   └── temporal_agent_system.txt
│   └── tests/
│       ├── test_surface.py
│       ├── test_dynamics.py
│       ├── test_observation.py
│       ├── test_episode.py
│       └── test_agents.py
│
├── configs/
│   ├── coordination_experiments.yaml
│   └── temporal_experiments.yaml
│
├── results/
│   ├── coordination/
│   └── temporal/
│
├── PLAN.md                           # Coordination benchmark plan
├── TEMPORAL_PLAN.md                  # THIS DOCUMENT
├── CLAUDE.md                         # Updated with both benchmarks
├── requirements.txt
└── README.md
```

---

## 10. API Specifications

### 10.1 TemporalSurface Class

```python
class TemporalSurface:
    """A 1D surface that evolves over time."""

    def __init__(
        self,
        peaks: List[dict],
        dynamics: List[DynamicsRule],
        domain_size: float = 10.0
    ):
        """
        Args:
            peaks: List of initial peak definitions, each with keys:
                   - cx: float, initial x-coordinate of peak center
                   - height: float, initial peak height
                   - sigma: float, peak width (standard deviation)
            dynamics: List of DynamicsRule objects (one per peak, or None for stationary)
            domain_size: Size of domain [0, domain_size]
        """
        pass

    def evaluate(self, x: float, t: float) -> float:
        """Evaluate surface at position x and time t."""
        pass

    def gradient_x(self, x: float, t: float, eps: float = 0.001) -> float:
        """Compute spatial gradient ∂f/∂x at (x, t)."""
        pass

    def gradient_t(self, x: float, t: float, delta: float = 1.0) -> float:
        """Compute temporal gradient ∂f/∂t at (x, t)."""
        pass

    def get_optimal(self, t: float) -> Tuple[float, float]:
        """
        Return (x_opt, f_opt) for global maximum at time t.

        Implementation: Grid search or scipy.optimize at each t.
        """
        pass

    def get_optimal_trajectory(self, t_range: range) -> List[Tuple[float, float]]:
        """Return list of (x_opt, f_opt) for each t in range."""
        pass

    def to_dict(self) -> dict:
        """Serialize surface configuration."""
        pass

    @classmethod
    def from_dict(cls, config: dict) -> 'TemporalSurface':
        """Deserialize surface from configuration."""
        pass
```

### 10.2 DynamicsRule Classes

```python
from abc import ABC, abstractmethod

class DynamicsRule(ABC):
    """Abstract base class for dynamics rules."""

    @abstractmethod
    def apply(self, initial_value: float, t: float) -> float:
        """Return parameter value at time t."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize rule configuration."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict) -> 'DynamicsRule':
        """Deserialize rule from configuration."""
        pass


class LinearDrift(DynamicsRule):
    def __init__(self, velocity: float):
        self.velocity = velocity

    def apply(self, initial_cx: float, t: float) -> float:
        return initial_cx + self.velocity * t


class LinearHeightChange(DynamicsRule):
    def __init__(self, rate: float):
        self.rate = rate

    def apply(self, initial_height: float, t: float) -> float:
        return max(0.0, initial_height + self.rate * t)


class Oscillation(DynamicsRule):
    def __init__(self, amplitude: float, period: float):
        self.amplitude = amplitude
        self.period = period

    def apply(self, initial_cx: float, t: float) -> float:
        return initial_cx + self.amplitude * np.sin(2 * np.pi * t / self.period)


class CompositeDynamics(DynamicsRule):
    def __init__(
        self,
        cx_rule: DynamicsRule = None,
        height_rule: DynamicsRule = None,
        sigma_rule: DynamicsRule = None
    ):
        self.cx_rule = cx_rule
        self.height_rule = height_rule
        self.sigma_rule = sigma_rule

    def apply_all(self, initial_params: dict, t: float) -> dict:
        """Apply all rules to get full parameter dict at time t."""
        pass
```

### 10.3 TemporalObservationGenerator Class

```python
class TemporalObservationGenerator:
    """Generates agent observations from temporal surface."""

    def __init__(
        self,
        surface: TemporalSurface,
        radius: float = 0.5,
        n_samples: int = 11
    ):
        """
        Args:
            surface: The TemporalSurface to observe
            radius: Observation radius R
            n_samples: Number of samples in slice
        """
        pass

    def generate_observation(self, x: float, t: float) -> dict:
        """
        Generate observation at position x and time t.

        Returns:
            {
                "position": x,
                "timestep": t,
                "value_at_position": float,
                "gradient_x": float,
                "gradient_t": float,
                "slice": [{"x": float, "value": float}, ...]
            }
        """
        pass
```

### 10.4 TemporalEpisode Class

```python
class TemporalEpisode:
    """Manages temporal episode state and progression."""

    def __init__(
        self,
        surface: TemporalSurface,
        timesteps_per_run: int = 20,
        max_exploration_runs: int = 10,
        initial_position: float = 5.0,
        test_time_offset: int = 0  # 0 for same range, 20 for future
    ):
        pass

    @property
    def current_position(self) -> float:
        pass

    @property
    def current_timestep(self) -> int:
        pass

    @property
    def current_run(self) -> int:
        pass

    @property
    def phase(self) -> str:
        """Returns 'exploration' or 'test'."""
        pass

    def exploration_step(self, x_new: float) -> dict:
        """
        Move to new position and return observation.

        Returns observation dict.
        Increments timestep.
        """
        pass

    def end_exploration_run(self) -> dict:
        """
        End current exploration run.

        Returns:
            {
                "run_number": int,
                "score": float,
                "log": List[dict],  # Full timestep-by-timestep log
                "cumulative_reward": float,
                "optimal_reward": float
            }
        """
        pass

    def start_test_phase(self) -> None:
        """Transition to test phase."""
        pass

    def submit_predictions(self, predictions: List[float]) -> dict:
        """
        Submit test predictions and compute score.

        Args:
            predictions: List of 20 x-positions

        Returns:
            {
                "predictions": List[float],
                "optimal_positions": List[float],
                "per_timestep_rewards": List[float],
                "test_score": float
            }
        """
        pass

    def get_full_history(self) -> dict:
        """Return complete episode history."""
        pass
```

### 10.5 TemporalBaseAgent Class

```python
from abc import ABC, abstractmethod
from typing import Union, List

class TemporalBaseAgent(ABC):
    """Abstract base class for temporal agents (Function Mode)."""

    def __init__(self):
        self.run_reports = []  # Memory: List of ExplorationReport from previous runs

    def receive_report(self, report: 'ExplorationReport') -> None:
        """
        Store report from completed exploration run.

        Args:
            report: Full report with per-timestep data and summary
        """
        self.run_reports.append(report)

    @abstractmethod
    def propose_positions(
        self,
        initial_observation: dict,
        run_number: int
    ) -> Union[List[float], str]:
        """
        Propose positions for an exploration run (Function Mode).

        Args:
            initial_observation: Observation at t=0, x=5.0
            run_number: Current exploration run number (1-indexed)

        Returns:
            - List of 20 floats: Positions for t=0 through t=19
            - "READY_FOR_TEST": Signal to end exploration
        """
        pass

    @abstractmethod
    def predict_positions(self, n: int = 20) -> List[float]:
        """
        Generate test predictions.

        Args:
            n: Number of positions to predict

        Returns:
            List of n x-positions for test phase
        """
        pass

    def reset(self) -> None:
        """Reset agent state for new scenario."""
        self.run_reports = []
```

### 10.6 Result Logger Output Format

```json
{
    "id": "temporal_20260113_143022_abc123",
    "timestamp": "2026-01-13T14:30:22Z",
    "benchmark": "temporal",

    "scenario": {
        "name": "linear_drift_slow",
        "peaks": [{"cx": 3.0, "height": 1.0, "sigma": 1.0}],
        "dynamics": [{"type": "LinearDrift", "velocity": 0.1}],
        "difficulty": 2
    },

    "agent": {
        "type": "TemporalLLMAgent",
        "model": "gpt-4",
        "temperature": 0.7
    },

    "exploration": {
        "num_runs": 3,
        "runs": [
            {
                "run_number": 1,
                "score": 0.65,
                "log": [
                    {
                        "timestep": 0,
                        "position": 5.0,
                        "observation": {...},
                        "action": 5.5,
                        "reward": 0.42
                    },
                    ...
                ]
            },
            ...
        ]
    },

    "test": {
        "predictions": [3.0, 3.1, 3.2, ..., 5.0],
        "optimal_positions": [3.0, 3.1, 3.2, ..., 5.0],
        "score": 0.82
    },

    "metrics": {
        "exploration_efficiency": 0.273,  # test_score / num_runs
        "prediction_mae": 0.15,
        "learning_curve": [0.65, 0.71, 0.75]
    }
}
```

---

## 11. Prompt Templates

### 11.1 Temporal Agent System Prompt (Function Mode)

**File:** `temporal/prompts/temporal_agent_system.txt`

```
You are an agent exploring a 1D surface that changes over time. Your goal is to learn the pattern of change and predict where the peak will be.

SETUP:
- The surface is a function f(x) defined on [0, 10]
- The surface CHANGES over time according to a hidden rule
- You control your x-position for each of 20 timesteps
- Time advances automatically (t = 0, 1, 2, ..., 19)

EPISODE STRUCTURE:

1. EXPLORATION PHASE:
   - You will do multiple exploration runs (up to 10)
   - Each run: You propose ALL 20 positions upfront
   - System executes your positions and returns a DETAILED REPORT
   - Report includes: value, ∂f/∂x, ∂f/∂t at each timestep
   - Your goal: Learn the pattern from the reports
   - When confident, output READY_FOR_TEST

2. TEST PHASE:
   - You must predict 20 positions (blind, no feedback)
   - These predictions will be scored against optimal positions
   - This tests whether you truly learned the dynamics

UNDERSTANDING THE REPORT:
- ∂f/∂x (spatial gradient): Points toward higher values in space
- ∂f/∂t (temporal gradient): Shows if value at that x is increasing/decreasing over time
- Combine these to understand: Is the peak moving? Which direction? How fast?

STRATEGY TIPS:
- Run 1: Try a simple strategy (e.g., stay at center, or follow a line)
- Analyze the report: Where did you score well/poorly?
- Run 2+: Adjust based on what you learned from previous reports
- Look for patterns in ∂f/∂t: positive means peak is approaching that x

OUTPUT FORMAT:

For exploration runs, output 20 positions:
POSITIONS: [x0, x1, x2, ..., x19]

Each value should be between 0 and 10.

To end exploration and begin test phase:
READY_FOR_TEST

For test predictions (same format):
PREDICTIONS: [x0, x1, x2, ..., x19]

Think step by step. Analyze previous reports to infer the hidden dynamics.
```

### 11.2 Exploration Run Prompt Template (Function Mode)

```
=== EXPLORATION RUN {run_number} ===

INITIAL OBSERVATION (t=0, x=5.0):
- Value at position: f(5.0, 0) = {initial_value:.4f}
- Spatial gradient: ∂f/∂x = {initial_grad_x:+.4f}
- Temporal gradient: ∂f/∂t = {initial_grad_t:+.4f}

{previous_runs_summary}

RUNS COMPLETED: {num_runs - 1} / {max_runs}

---

Propose 20 positions for timesteps t=0 through t=19.
Output: POSITIONS: [x0, x1, x2, ..., x19]

Or if you're confident in the pattern:
Output: READY_FOR_TEST
```

### 11.3 Exploration Report Template (Function Mode)

```
=== EXPLORATION RUN {run_number} REPORT ===

PER-TIMESTEP RESULTS:
  t= 0: x={x0:.2f}, f={v0:.4f}, ∂f/∂x={gx0:+.4f}, ∂f/∂t={gt0:+.4f}
  t= 1: x={x1:.2f}, f={v1:.4f}, ∂f/∂x={gx1:+.4f}, ∂f/∂t={gt1:+.4f}
  ...
  t=19: x={x19:.2f}, f={v19:.4f}, ∂f/∂x={gx19:+.4f}, ∂f/∂t={gt19:+.4f}

SUMMARY:
  Total score: {score:.2%} of optimal
  Best timestep: t={best_t} (highest relative score)
  Worst timestep: t={worst_t} (lowest relative score)

---

Analyze this report to understand the dynamics.
What pattern do you observe? Is the peak moving? Growing? Shrinking?
```

### 11.4 Test Phase Prompt Template

```
=== TEST PHASE ===

You have completed {num_runs} exploration runs.

EXPLORATION MEMORY:
{all_reports_summary}

Based on what you learned about how this surface changes over time,
predict where the peak will be for 20 timesteps.

You will NOT receive any feedback during the test.
Your predictions must be based entirely on the patterns you learned.

Output your 20 predictions as:
PREDICTIONS: [x0, x1, x2, ..., x19]

Each value should be between 0 and 10.
```

---

## 12. Test Cases

### 12.1 Surface Tests

```python
# temporal/tests/test_surface.py

def test_stationary_surface():
    """Stationary surface has constant value at same position."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    assert abs(surface.evaluate(5.0, 0) - surface.evaluate(5.0, 10)) < 0.01

def test_linear_drift():
    """Peak moves according to velocity."""
    surface = TemporalSurface(
        peaks=[{"cx": 3.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[LinearDrift(velocity=0.2)]
    )
    # At t=0, peak at x=3.0
    assert abs(surface.evaluate(3.0, 0) - 1.0) < 0.01
    # At t=10, peak at x=5.0
    assert abs(surface.evaluate(5.0, 10) - 1.0) < 0.01

def test_spatial_gradient_direction():
    """Spatial gradient points toward peak."""
    surface = TemporalSurface(
        peaks=[{"cx": 7.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    grad_x = surface.gradient_x(5.0, 0)
    assert grad_x > 0  # Peak is to the right

def test_temporal_gradient_approaching_peak():
    """Temporal gradient is positive when peak approaching."""
    surface = TemporalSurface(
        peaks=[{"cx": 3.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[LinearDrift(velocity=0.2)]  # Moving right
    )
    # At x=5, peak is approaching from left
    grad_t = surface.gradient_t(5.0, 0)
    assert grad_t > 0  # Value at x=5 will increase as peak approaches

def test_get_optimal_trajectory():
    """Optimal trajectory tracks peak correctly."""
    surface = TemporalSurface(
        peaks=[{"cx": 3.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[LinearDrift(velocity=0.1)]
    )
    trajectory = surface.get_optimal_trajectory(range(20))
    # Peak should move from 3.0 to 5.0 over 20 timesteps
    assert abs(trajectory[0][0] - 3.0) < 0.1
    assert abs(trajectory[19][0] - 5.0) < 0.1
```

### 12.2 Dynamics Tests

```python
# temporal/tests/test_dynamics.py

def test_linear_drift_velocity():
    """LinearDrift moves at correct velocity."""
    rule = LinearDrift(velocity=0.5)
    assert rule.apply(0.0, 0) == 0.0
    assert rule.apply(0.0, 10) == 5.0

def test_linear_height_change():
    """LinearHeightChange grows/shrinks correctly."""
    grow = LinearHeightChange(rate=0.1)
    assert grow.apply(0.5, 0) == 0.5
    assert grow.apply(0.5, 5) == 1.0

    shrink = LinearHeightChange(rate=-0.1)
    assert shrink.apply(1.0, 10) == 0.0  # Clamped to 0

def test_oscillation_period():
    """Oscillation completes full cycle."""
    rule = Oscillation(amplitude=2.0, period=10)
    assert abs(rule.apply(5.0, 0) - 5.0) < 0.01
    assert abs(rule.apply(5.0, 2.5) - 7.0) < 0.01  # Max
    assert abs(rule.apply(5.0, 5) - 5.0) < 0.01    # Back to center
    assert abs(rule.apply(5.0, 7.5) - 3.0) < 0.01  # Min
    assert abs(rule.apply(5.0, 10) - 5.0) < 0.01   # Full cycle

def test_composite_dynamics():
    """CompositeDynamics applies multiple rules."""
    composite = CompositeDynamics(
        cx_rule=LinearDrift(velocity=0.1),
        height_rule=LinearHeightChange(rate=0.05)
    )
    initial = {"cx": 3.0, "height": 0.5, "sigma": 1.0}
    at_t10 = composite.apply_all(initial, 10)
    assert abs(at_t10["cx"] - 4.0) < 0.01
    assert abs(at_t10["height"] - 1.0) < 0.01
    assert at_t10["sigma"] == 1.0  # Unchanged
```

### 12.3 Observation Tests

```python
# temporal/tests/test_observation.py

def test_observation_structure():
    """Observation contains all required fields."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    obs_gen = TemporalObservationGenerator(surface, radius=0.5, n_samples=11)
    obs = obs_gen.generate_observation(5.0, 0)

    assert "position" in obs
    assert "timestep" in obs
    assert "value_at_position" in obs
    assert "gradient_x" in obs
    assert "gradient_t" in obs
    assert "slice" in obs

def test_slice_samples():
    """Slice has correct number of samples."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    obs_gen = TemporalObservationGenerator(surface, radius=0.5, n_samples=11)
    obs = obs_gen.generate_observation(5.0, 0)
    assert len(obs["slice"]) == 11

def test_slice_range():
    """Slice covers correct x range."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    obs_gen = TemporalObservationGenerator(surface, radius=0.5, n_samples=11)
    obs = obs_gen.generate_observation(5.0, 0)
    x_values = [s["x"] for s in obs["slice"]]
    assert abs(min(x_values) - 4.5) < 0.01  # 5.0 - 0.5
    assert abs(max(x_values) - 5.5) < 0.01  # 5.0 + 0.5

def test_boundary_clipping():
    """Slice clips at domain boundary."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    obs_gen = TemporalObservationGenerator(surface, radius=0.5, n_samples=11)
    obs = obs_gen.generate_observation(0.2, 0)  # Near left edge
    x_values = [s["x"] for s in obs["slice"]]
    assert min(x_values) >= 0.0  # Clipped to boundary
```

### 12.4 Episode Tests

```python
# temporal/tests/test_episode.py

def test_episode_initialization():
    """Episode starts in correct state."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    episode = TemporalEpisode(surface)
    assert episode.current_position == 5.0
    assert episode.current_timestep == 0
    assert episode.current_run == 1
    assert episode.phase == "exploration"

def test_exploration_step():
    """Exploration step updates state correctly."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    episode = TemporalEpisode(surface)
    obs = episode.exploration_step(6.0)
    assert episode.current_position == 6.0
    assert episode.current_timestep == 1
    assert "gradient_x" in obs
    assert "gradient_t" in obs

def test_run_completion():
    """Run completes after 20 timesteps."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]
    )
    episode = TemporalEpisode(surface, timesteps_per_run=20)
    for _ in range(20):
        episode.exploration_step(5.0)
    result = episode.end_exploration_run()
    assert result["run_number"] == 1
    assert "score" in result
    assert len(result["log"]) == 20

def test_test_phase_scoring():
    """Test phase computes score correctly."""
    surface = TemporalSurface(
        peaks=[{"cx": 5.0, "height": 1.0, "sigma": 1.0}],
        dynamics=[None]  # Stationary
    )
    episode = TemporalEpisode(surface)
    # Skip exploration
    episode.start_test_phase()
    # Perfect predictions for stationary surface
    predictions = [5.0] * 20
    result = episode.submit_predictions(predictions)
    assert abs(result["test_score"] - 1.0) < 0.01
```

### 12.5 Agent Tests

```python
# temporal/tests/test_agents.py

def test_random_agent_bounds():
    """Random agent proposes positions within bounds."""
    agent = TemporalRandomAgent(seed=42)
    initial_obs = {"position": 5.0, "value_at_position": 0.5,
                   "gradient_x": 0.1, "gradient_t": 0.05}
    positions = agent.propose_positions(initial_obs, run_number=1)
    assert len(positions) == 20
    assert all(0 <= p <= 10 for p in positions)

def test_random_agent_deterministic():
    """Random agent is deterministic with same seed."""
    agent1 = TemporalRandomAgent(seed=42)
    agent2 = TemporalRandomAgent(seed=42)
    initial_obs = {"position": 5.0, "value_at_position": 0.5,
                   "gradient_x": 0.1, "gradient_t": 0.05}
    pos1 = agent1.propose_positions(initial_obs, run_number=1)
    pos2 = agent2.propose_positions(initial_obs, run_number=1)
    assert pos1 == pos2

def test_greedy_agent_follows_gradient():
    """Greedy agent proposes positions following spatial gradient."""
    agent = TemporalGreedyAgent(step_size=0.5)
    initial_obs = {"position": 5.0, "value_at_position": 0.5,
                   "gradient_x": 0.5, "gradient_t": 0.0}  # Peak to the right
    positions = agent.propose_positions(initial_obs, run_number=1)
    # Positions should trend rightward (increasing)
    assert positions[-1] > positions[0]

def test_predictions_length():
    """Agent returns correct number of test predictions."""
    agent = TemporalRandomAgent(seed=42)
    predictions = agent.predict_positions(20)
    assert len(predictions) == 20
    assert all(0 <= p <= 10 for p in predictions)

def test_agent_memory():
    """Agent stores reports in memory."""
    agent = TemporalRandomAgent(seed=42)
    report = ExplorationReport(run_number=1, timesteps=[], total_score=0.5,
                               best_timestep=10, worst_timestep=3)
    agent.receive_report(report)
    assert len(agent.run_reports) == 1
    assert agent.run_reports[0].total_score == 0.5
```

---

## 13. Visualization Requirements

### 13.1 3D Surface Evolution Plot

**Function:** `plot_surface_evolution(surface, t_range, show=True, save_path=None)`

**Requirements:**
- 3D plot with axes: x (position), t (time), f(x,t) (value)
- Grid resolution: 50 × len(t_range) minimum
- Colormap: 'viridis' or 'plasma'
- Optimal trajectory as red line on surface
- Axis labels: "Position (x)", "Time (t)", "f(x,t)"

**Example visualization:**
```
    f(x,t)
      ▲
      │    ╱╲___╱╲___╱╲___    Peak trajectory
      │   ╱              ╲
      │  ╱                ╲
      └────────────────────► x
     ╱
    t
```

### 13.2 Agent Trajectory Plot

**Function:** `plot_trajectory(surface, history, show=True, save_path=None)`

**Requirements:**
- 2D plot: x (vertical) vs t (horizontal)
- Agent path as solid blue line
- Optimal path as dashed red line
- Color intensity showing f(x,t) achieved
- Start marker (green), end marker (blue)
- Score annotation

### 13.3 Prediction Comparison Plot

**Function:** `plot_predictions(predictions, optimal, actual_values, show=True, save_path=None)`

**Requirements:**
- Two subplots:
  - Top: Position over time (predicted vs optimal)
  - Bottom: Value achieved over time
- Error shading between predicted and optimal
- Score annotation
- Legend

### 13.4 Slice Evolution Animation

**Function:** `animate_slice_evolution(surface, x, t_range, save_path)`

**Requirements:**
- Animated GIF or MP4
- Shows 1D slice at fixed x changing over time
- Peak movement visible
- Timestep counter

---

## 14. Baselines

### 14.1 Random Baseline

**Behavior:**
- Each timestep: random position in [0, 10]
- Fixed 3 exploration runs
- Test predictions: random positions

**Expected performance:**
- Exploration score: ~0.15-0.25
- Test score: ~0.15-0.25
- Establishes lower bound

### 14.2 Greedy Gradient Baseline

**Behavior:**
- Each timestep: follow spatial gradient (∂f/∂x)
- `new_pos = clamp(current + step_size * sign(gradient_x))`
- Ignores temporal gradient
- Fixed 3 exploration runs
- Test predictions: repeat last positions from best exploration run

**Expected performance:**
- Exploration score: ~0.4-0.6 (good at tracking current peak)
- Test score: ~0.3-0.5 (fails to predict future)
- Shows limitation of ignoring temporal dynamics

### 14.3 Oracle Baseline (Upper Bound)

**Behavior:**
- Receives full dynamics specification
- Computes optimal trajectory analytically
- Test predictions: exact optimal positions

**Expected performance:**
- Test score: 1.0
- Validates scoring system

### 14.4 Temporal Greedy Baseline

**Behavior:**
- Considers both gradients
- Heuristic: move toward where value will increase
- Simple prediction: linear extrapolation of observed peak movement

**Expected performance:**
- Better than greedy, worse than LLM
- Shows value of temporal gradient information

---

## 15. Evaluation Protocol

### 15.1 Experiment Matrix

| Condition | Scenarios | Trials per scenario | Total runs |
|-----------|-----------|-------------------|------------|
| Random baseline | All 9 scenarios | 5 | 45 |
| Greedy baseline | All 9 scenarios | 5 | 45 |
| Temporal greedy | All 9 scenarios | 5 | 45 |
| LLM agent | All 9 scenarios | 3 | 27 |

### 15.2 Primary Analysis

**Main comparison:**
- Test score across scenarios by difficulty tier
- Learning efficiency: test score vs exploration runs used

**Statistical tests:**
- Mann-Whitney U for baseline vs LLM comparisons
- Correlation between exploration efficiency and test score

### 15.3 Secondary Analysis

**Per-scenario analysis:**
- Which dynamics patterns are learned fastest?
- Which cause most prediction errors?

**Failure mode categorization:**
- Pattern misidentification (learned wrong rule)
- Extrapolation failure (correct pattern, wrong prediction)
- Insufficient exploration (gave up too early)

### 15.4 Reproducibility Requirements

- All random seeds logged
- LLM API parameters logged (model, temperature, max_tokens)
- Scenario configurations saved
- Full transcripts saved (exploration logs + predictions)
- Code version tracked (git commit hash)

---

## Appendix A: Quick Reference

### Key Parameters
```
Domain:         [0, 10]
Radius:         R = 0.5
Samples:        S = 11
Timesteps:      T = 20 per run
Max runs:       10
Start position: x = 5.0
```

### Score Interpretation
```
1.0      = Perfect (predicted optimal trajectory)
0.8-0.99 = Excellent (understood dynamics well)
0.6-0.8  = Good (partially learned pattern)
0.4-0.6  = Moderate (some pattern recognition)
< 0.4    = Poor (failed to learn dynamics)
```

### File Locations
```
Surface code:      temporal/core/surface.py
Dynamics code:     temporal/core/dynamics.py
Observation code:  temporal/core/observation.py
Episode code:      temporal/core/episode.py
LLM agent:         temporal/agents/llm_agent.py
Prompt:            temporal/prompts/temporal_agent_system.txt
Results:           results/temporal/
```

---

## Appendix B: Checkpoint Summary

Copy this checklist to track progress:

```
PHASE 0: REPOSITORY RESTRUCTURE ✅
[x] 0.1 Create new directory structure
[x] 0.2 Move existing code to coordination/
[x] 0.3 Extract shared utilities

PHASE 1: TEMPORAL CORE
[ ] 1.1 Temporal surface class
[ ] 1.2 Dynamics system
[ ] 1.3 Observation generator
[ ] 1.4 Episode state manager

PHASE 2: TEMPORAL AGENTS
[ ] 2.1 Temporal base agent
[ ] 2.2 Random baseline agent
[ ] 2.3 Greedy baseline agent
[ ] 2.4 LLM agent
[ ] 2.5 System prompt

PHASE 3: EXPERIMENTS
[ ] 3.1 Episode runner
[ ] 3.2 Result logger
[ ] 3.3 Batch evaluation
[ ] 3.4 Statistical analysis

PHASE 4: VISUALIZATION
[ ] 4.1 3D surface evolution
[ ] 4.2 Agent trajectory
[ ] 4.3 Prediction comparison
[ ] 4.4 Slice evolution

PHASE 5: INTEGRATION
[ ] 5.1 End-to-end test (random)
[ ] 5.2 End-to-end test (greedy)
[ ] 5.3 End-to-end test (LLM)
[ ] 5.4 Documentation update
```

---

## Appendix C: Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Observation radius | 0.5 (not 1.5) | Smaller radius increases difficulty, forces more exploration |
| Both gradients | ∂f/∂x and ∂f/∂t | Temporal gradient crucial for learning dynamics |
| Exploration budget | Agent decides | Tests agent's confidence calibration |
| Ready signal | READY_FOR_TEST token | Clear, parseable signal |
| Exploration feedback | Full report per run | Enables learning from complete trajectory |
| Test time range | Configurable | Allows testing interpolation vs extrapolation |
| Slice sampling | Uniform 0.1 spacing | Fewer samples at edges, consistent density |
| Initial position | Fixed at 5.0 | Consistent starting point for comparison |
| Max exploration runs | 10 | Safety limit while allowing agent autonomy |
| Dynamics rules | 4 core rules | LinearDrift, LinearHeightChange, Oscillation, CompositeDynamics |
| Exploration mode | Function mode (default) | Agent proposes 20 positions at once, much cheaper than per-step |
| Play mode | Future enhancement | Will allow human players, per-step interaction |

---

*Document version: 1.0*
*Created: 2026-01-13*
*Status: READY FOR IMPLEMENTATION*
