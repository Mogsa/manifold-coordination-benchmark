# ChaosBench v1: Minimal Benchmark Design

**Date:** 2026-01-31
**Status:** Design Complete — Ready for Implementation
**Origin:** Brainstorming session returning to first principles

---

## Research Question

> Does an LLM, when solving sequential chaotic prediction tasks, exhibit evidence of in-context learning of task structure — measured by superlinear Φ(n)?

### What We're Testing

| If we observe... | It suggests... |
|------------------|----------------|
| Superlinear Φ(n) + Decreasing T(n) | **Strong learning**: harder tasks, less effort |
| Linear Φ(n) + Constant T(n) | **No transfer**: each task treated independently |
| Superlinear Φ(n) + Constant T(n) | Learning *what* to attempt, not *how* efficiently |
| Linear Φ(n) + Decreasing T(n) | Getting faster but not tackling harder problems |

---

## MVP Design

### The Simplest Possible Benchmarkl

```
Agent sees: [x_0, x_1, ..., x_49]
Agent outputs: x_50 ± σ
System scores: NLL over discretized bins
Repeat for n tasks
Measure: Φ(n), T(n)
```

### Actions (MVP)

| Action | Purpose | Required |
|--------|---------|----------|
| **PREDICT** | Final answer (value ± σ) | Yes |
| **MOVE_ON** | Proceed to next task | Yes |

That's it. Two actions. The agent reasons in natural language, then predicts.

### Why No Scaffolding

The research question is "does the agent learn structure?" — not "can the agent use scientific tools?"

If the agent learns patterns (e.g., "bounded [0,1] + parabolic = logistic"), it should predict better on later tasks through in-context learning alone. The conversation history carries the learning.

---

## Evaluation Framework

### Primary Metrics

```
Φ(n) = Σ w(h_KS) × Score    (cumulative after n tasks)
T(n) = turns for task n      (efficiency per task)
```

For MVP, T(n) = 1 always (single prediction per task). Future versions with retry logic would vary T(n).

### Scoring

Agent outputs: `prediction ± σ` (e.g., "0.67 ± 0.05")

Conversion to NLL:
1. If σ given → truncated Gaussian centered at prediction
2. If no σ → default σ = 0.1 (one bin width)
3. Discretize to 20 bins over [0, 1]
4. NLL = -log(p[true_bin])

Score per task:
```python
score = exp(-NLL)
weighted_score = w(h_KS) * score
```

### Why NLL (Not MSE)

MSE is gameable by mean reversion. Predicting the attractor center gives low MSE because you're always "close."

NLL punishes confident wrong predictions. Mean reversion confidently predicts the center bin, but that bin is only correct ~5% of the time. Terrible NLL.

---

## Agent Interface

### Input (per task)

```
Task 7 of 20

Observations (x_0 to x_49):
[0.312, 0.834, 0.481, 0.867, 0.401, ...]

Predict: x_50

Format: {"action": "PREDICT", "value": <number>, "uncertainty": <number>}
```

### Output (from agent)

```
Looking at the data, values oscillate between near 0 and near 1,
suggesting a chaotic map like logistic. The last value is 0.401.
If this is logistic with r≈4, next value ≈ 4 × 0.401 × 0.599 ≈ 0.96.

{"action": "PREDICT", "value": 0.96, "uncertainty": 0.08}
```

### Feedback (after PREDICT)

```
Actual: 0.847
Score: 0.72

{"action": "MOVE_ON"}
```

Then next task appears.

---

## Systems (v1)

### Logistic Map
- Equation: x_{n+1} = r × x_n × (1 - x_n)
- Parameter: r ∈ [3.57, 4.0] (chaotic regime)
- h_KS: ln(r) for r=4, approximately ln(2) ≈ 0.693
- Range: [0, 1]

### Tent Map
- Equation: x_{n+1} = μ × min(x_n, 1 - x_n)
- Parameter: μ ∈ [1.0, 2.0]
- h_KS: ln(μ)
- Range: [0, 1]

Both maps share [0, 1] range → same 20-bin discretization.

---

## Parameter Space

### Primary Difficulty Knobs

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `horizon` | int | 1 | 1, 5, 10, 20 |
| `noise_std` | float | 0.0 | 0.0, 0.01, 0.02, 0.05 |
| `n_obs` | int | 50 | 10, 25, 50, 100 |
| `family` | str | "logistic" | "logistic", "tent" |

### Fixed for v1

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_bins` | 20 | Uniform over [0, 1] |
| `weighting` | linear | w(h) = h |
| `conditional` | False | Family not revealed |
| `n_tasks` | 20 | Per session |

### Config Structure

```python
@dataclass
class TaskConfig:
    family: str = "logistic"
    params: dict = None  # None = sample randomly
    n_obs: int = 50
    noise_std: float = 0.0
    horizon: int = 1
    n_bins: int = 20

@dataclass
class SessionConfig:
    n_tasks: int = 20
    families: tuple = ("logistic", "tent")
    task_order: str = "random"
    difficulty: str = "mixed"  # or "easy", "hard", "sweep"
```

---

## Implementation Plan

### Phase 1: Core Types
- [ ] `TaskConfig` dataclass
- [ ] `SessionConfig` dataclass
- [ ] `TaskResult` dataclass (prediction, actual, score, h_ks)
- [ ] `SessionResult` dataclass (Φ curve, T curve, all results)

### Phase 2: Task Generator
- [ ] Sample (family, params) with known h_KS
- [ ] Generate trajectory
- [ ] Add observation noise
- [ ] Select prediction target at horizon

### Phase 3: Scoring
- [ ] Parse "value ± σ" from agent output
- [ ] Convert to Gaussian → discretize → NLL
- [ ] Compute weighted score
- [ ] Track Φ(n), T(n)

### Phase 4: Session Runner
- [ ] Format task prompt
- [ ] Call LLM agent
- [ ] Parse response
- [ ] Provide feedback
- [ ] Loop for n_tasks

### Phase 5: Analysis
- [ ] Plot Φ(n) curve
- [ ] Plot T(n) trend (if applicable)
- [ ] Detect superlinearity
- [ ] Export results

---

## Future Extensions

### v1.1: Retry Logic
- Agent can PREDICT multiple times before MOVE_ON
- T(n) becomes meaningful (turns per task)
- Adds: how quickly does agent converge?

### v1.2: Scaffolded Condition
Add optional scientific tools:

| Action | Purpose |
|--------|---------|
| HYPOTHESIZE | Test model against observations → get MAE |
| FIT | Auto-estimate parameters for a family |

Compare Φ(n) with vs without scaffolding. The gap measures value of explicit scientific reasoning tools.

### v1.3: Learnings Notepad
Add optional memory:

| Action | Purpose |
|--------|---------|
| WRITE | Record insight to persistent notepad |
| DELETE | Remove section from notepad |

Compare with vs without. Does explicit reflection help?

### v2: Extended Systems
- Hénon map (2D, predict x only)
- Standard map (2D)
- Lorenz (3D via Poincaré section)

### v3: Observation Costs
- Agent can request more observations
- Cost per observation
- Enables multi-agent advantage analysis

---

## Success Criteria

1. **Functional**: Run 20-task session with LLM, get predictions
2. **Measurable**: Φ(n) curve generated and plottable
3. **Diagnostic**: Can distinguish linear vs superlinear growth
4. **Reproducible**: Same config → comparable results
5. **Extensible**: Easy to add scaffolding/learnings as experimental conditions

---

## Difficulty Grid (Reference)

| | Clean (σ=0) | Noisy (σ=0.02) | Very Noisy (σ=0.05) |
|---|---|---|---|
| **1-step** | Easy | Medium | Medium |
| **5-step** | Medium | Hard | Hard |
| **20-step** | Hard | Very Hard | Extreme |

Use this to validate that difficulty knobs work as expected.
