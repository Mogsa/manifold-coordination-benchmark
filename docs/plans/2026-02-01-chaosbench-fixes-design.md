# ChaosBench Fixes Design

**Date:** 2026-02-01
**Status:** Ready for Implementation
**Context:** MVP experiments showed perfect scores because (1) systems are periodic not chaotic, (2) scoring is too lenient

---

## Three Issues to Fix

### 1. NLL Bin-Based Scoring (Priority: High)

**Current (broken):**
```python
error = abs(prediction - actual)
score = np.exp(-error * 5)  # Too lenient
```

**Designed (from MVP spec):**
- Agent outputs: `value ± σ` (default σ=0.1)
- Convert to truncated Gaussian over [0, 1]
- Discretize into 20 bins
- NLL = -log(probability_mass_in_true_bin)
- Score = exp(-NLL) for Φ(n) accumulation

**Why it matters:** Current scoring lets rough guesses score perfectly. NLL punishes confident wrong predictions.

**Files to modify:**
- `chaosbench/experiments/session.py` — replace `_compute_score()`
- `chaosbench/core/scoring.py` — new file with NLL logic

---

### 2. Truly Chaotic Systems (Priority: High)

**Current (broken):**
- Henon map has period-2 attractor (alternates -10, -3, -10, -3...)
- Agent learns "it alternates" → perfect prediction at any horizon
- Not actually chaotic despite positive Lyapunov exponent

**Fix options:**

**Option A (Recommended): Restrict to 1D maps at chaotic parameters**
- Logistic map: r = 4.0 (fully chaotic, h_KS = ln(2) ≈ 0.693)
- Tent map: μ = 2.0 (fully chaotic, h_KS = ln(2) ≈ 0.693)
- These are PROVABLY chaotic with no periodic windows

**Option B: Add periodicity detection**
- Run trajectory, check if it repeats within N steps
- Reject systems with detected periods
- More complex, still might miss quasi-periodic

**Option C: Use only high-h_KS regime**
- Filter for h_KS > 0.5
- Doesn't guarantee aperiodic

**Recommendation:** Option A — simplest, mathematically guaranteed chaotic.

**Files to modify:**
- `chaosbench/experiments/session.py` — add `families` config option
- `chaosbench/core/Chaosbench_v3.py` — add `create_chaotic_system()` with fixed params

---

### 3. Task Visualization (Priority: Medium)

**What to show:**
1. **Time series plot:** x_0, x_1, ..., x_49 (what agent sees)
2. **Prediction target:** x_50 (hidden, then revealed)
3. **Agent's guess:** prediction ± uncertainty
4. **Bin overlay:** Show the 20 bins and which one is correct

**When to generate:**
- Save one plot per task to output directory
- Or: save only for first N tasks (avoid clutter)

**Files to modify:**
- `chaosbench/visualization/plots.py` — already has `plot_task()`, enhance with bins
- `chaosbench/run_metacognitive.py` — add `--save-task-plots` flag

---

## Implementation Order

1. **Scoring (NLL)** — creates `chaosbench/core/scoring.py`, updates `session.py`
2. **Chaotic systems** — restricts to logistic r=4, tent μ=2
3. **Visualization** — enhances `plot_task()` with bins

---

## Test Cases

### Scoring
- Prediction exactly in true bin → low NLL, high score
- Prediction confident but wrong bin → high NLL, low score
- Prediction with high σ (uncertain) → moderate score regardless of bin

### Chaotic Systems
- Logistic r=4 trajectory should NOT repeat within 1000 steps
- h_KS should be ≈ 0.693 for both logistic r=4 and tent μ=2

### Visualization
- Plot shows 50 observations + prediction point + actual point
- Bins visible as background shading or grid lines

---

## After Clearing Context

Run:
```
/superpowers:execute-plan docs/plans/2026-02-01-chaosbench-fixes-design.md
```

Or reference this doc when starting implementation.
