# Findings: ChaosBench v4 Design

## Design Decisions Made

### Passive First (v1.0)
- Start with 50 observations given upfront
- Probing deferred to v1.1
- Rationale: Test base case (can LLM reason about chaos?) before adding complexity

### Simplified Actions
- PREDICT, WRITE, DELETE, MOVE_ON (no EDIT)
- JSON format for reliable parsing
- Agent writes reasoning before JSON action

### Minimal Feedback
- Agent sees only: prediction, actual value, score
- Must infer *why* it was wrong — that inference is part of what we measure

### Score Counting: Last
- Last prediction before MOVE_ON is banked
- Matches scientific process — final understanding matters

---

## Hypothesis-Driven Redesign (2026-01-30)

### Why Blind Prediction Failed
- Agent kept predicting 20x without committing (no MOVE_ON)
- Without feedback, multiple predictions are pointless
- Blind guessing isn't scientific reasoning

### The Fix: Backtest on Known Data
**Insight (Morgan):** Real science tests hypotheses against existing data before extrapolating.

**New feedback loop:**
1. Agent proposes model + parameters
2. System runs model on x_0...x_49, computes MAE
3. Agent sees: "Your model has MAE 0.147 — doesn't fit well"
4. Agent refines model
5. When satisfied, commits prediction
6. Only then x_50 revealed

### Feedback Level: Minimal
**Decided:** MAE + simple quality message + predicted x_50

**Format:**
```
Model: logistic (r=3.7)

Backtest (fitting x_0 → x_49):
  MAE: 0.147
  Your model doesn't reproduce the observations well.

If you trust this model, it predicts x_50 = 0.394
```

**Rationale:** Agent must interpret WHY model fails. Rich diagnostics would test less reasoning.

### Experimental Design
- **Phase A:** Scaffolded (HYPOTHESIZE, FIT actions provided) — **Ready to run**
- **Phase B:** CODE only (just Python executor, no hints) — Not yet implemented
- **Phase C:** Compare — did agents discover the fit-and-test strategy?

The gap between A and B is the finding.

### Implementation Status (2026-01-30)
**Phase A infrastructure complete:**
- Model factory supports: logistic, tent, henon, standard, lorenz
- Backtest computes one-step MAE on observations
- Fitting uses scipy.optimize with bounded search
- Session runner handles HYPOTHESIZE/FIT actions
- Feedback format: MAE + quality message + predicted x_50
- 76 tests pass

### Model Families (Already Exist)
From `Chaosbench_v3.py`:
- LogisticMap(r) — 1D, r ∈ [3.57, 4.0]
- TentMap(μ) — 1D, μ ∈ [1.0, 2.0]
- HenonMap(a, b) — 2D
- StandardMap(K) — 2D
- LorenzDisc(σ, ρ, β) — 3D

---

---

## Simple Benchmark Design (2026-01-31)

### Why Scaffolded Was Too Easy
- Claude one-shot logistic map with HYPOTHESIZE/FIT available
- Even Henon (hidden state problem) was one-shot
- LLMs may have memorized these systems from training data

### The Fix: Raw Prediction Benchmark
**No scaffolding at all:**
- No model family hints in prompt
- No HYPOTHESIZE/FIT actions
- Just: "Here's data, predict x_m"

**Two difficulty dimensions:**
1. **Prediction horizon** — 1-step (easy) to 20-step (hard)
2. **Observation noise** — clean (σ=0) to noisy (σ=0.05)

**Experimental conditions:**
- **Condition A:** Raw (simple benchmark, no tools)
- **Condition B:** Scaffolded (HYPOTHESIZE/FIT available)
- **Gap A→B:** Value of cognitive scaffolding

### Future: Reasoning Instrumentation
Brainstormed tools for understanding AI cognition:
- Pattern detection (DETECT_PATTERN → "bounded [0,1]", "parabolic return map")
- Hypothesis tracking (REGISTER_HYPOTHESIS / REJECT_HYPOTHESIS)
- Evidence gathering (link observations to hypotheses)
- Uncertainty estimation (ESTIMATE_CONFIDENCE)

Decision: Start with raw baseline first, add instrumentation later.

---

## Back to Basics Redesign (2026-01-31)

### The Problem: Scope Creep

Original research question:
> "Do LLMs learn transferable structure across chaotic prediction tasks?"

What we were actually testing:
> "Can LLMs use scientific tools (HYPOTHESIZE, FIT) effectively?"

These are different questions. The scaffolding obscured the core signal.

### What Got Stripped (Everything Non-Essential)

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| HYPOTHESIZE | Core action | **Removed (future v1.2)** | Not needed for base measurement |
| FIT | Core action | **Removed (future v1.2)** | Scaffolding, not baseline |
| WRITE/DELETE | Core actions | **Removed (future v1.3)** | Context window is memory |
| Learnings notepad | Required | **Removed** | Conversation history suffices |
| Systems | 5 families | **2 (logistic, tent)** | 1D only, clean h_KS |
| Metrics | Wall-clock Φ(t) | **Task-index Φ(n)** | Wall-clock too noisy |

### What Remains (True MVP)

| Component | Description |
|-----------|-------------|
| **PREDICT** | Agent outputs value ± σ |
| **MOVE_ON** | Proceed to next task |
| **Φ(n)** | Cumulative h_KS-weighted score |
| **Sequential conversation** | Context carries learning |

### The Key Insight

The research question is "does the agent learn structure?" — not "can the agent use scientific tools?"

HYPOTHESIZE/FIT test tool use. The MVP tests raw learning.

If an agent learns "bounded [0,1] + parabolic = logistic", it should predict better on later tasks through in-context learning alone. No scaffolding required.

Scaffolding becomes an experimental condition:
- **Condition A (MVP):** Raw prediction only
- **Condition B (v1.2):** Add HYPOTHESIZE/FIT
- **Gap A→B** = value of explicit scientific tools

### New Evaluation Framework

**Two curves, orthogonal phenomena:**

| Curve | Measures |
|-------|----------|
| Φ(n) | Cumulative capability (h_KS-weighted) |
| T(n) | Per-task efficiency (turns) |

**Interpretation matrix:**

| Φ(n) | T(n) | Meaning |
|------|------|---------|
| Superlinear | Decreasing | **Strong learning** |
| Linear | Constant | No transfer |
| Superlinear | Constant | Learning what to attempt |
| Linear | Decreasing | Getting faster, not harder |

### Scoring Decision

Agent outputs: `value ± σ` (e.g., "0.67 ± 0.05")

Conversion:
1. σ given → truncated Gaussian
2. No σ → default σ = 0.1
3. Discretize to 20 bins over [0,1]
4. NLL = -log(p[true_bin])

Why NLL over MSE: Mean reversion gaming. Predicting attractor center gives good MSE but terrible NLL (confident + wrong in specific bin).

### Parameter Space Defined

**Primary difficulty knobs:**
- `horizon`: 1, 5, 10, 20 steps
- `noise_std`: 0.0, 0.01, 0.02, 0.05
- `n_obs`: 10, 25, 50, 100
- `family`: logistic, tent

**Secondary (fixed for v1):**
- `n_bins`: 20
- `conditional`: False
- `weighting`: linear

### Design Doc

Full specification: `docs/plans/2026-01-31-chaosbench-v1-minimal-design.md`

---

## MVP Implementation Findings (2026-02-03)

### NLL Scoring Implementation

**Why NLL over simple error:**
- MSE gameable by mean reversion (predicting center gives "low" error)
- NLL punishes confident wrong predictions via bin-based probability

**Implementation:**
```python
def compute_score(prediction, actual, sigma=0.1, n_bins=20, bounds=(0,1)):
    # Truncated Gaussian centered at prediction
    # Score = P(true value's bin | prediction, sigma)
```

Agent outputs point estimate → system treats as Gaussian with σ=0.1 → compute probability mass in bin containing actual value.

### Tent Map Removal

**Problem:** Tent map with μ=2 degenerates to fixed point x=0 due to floating-point precision.
- Exact formula: x_{n+1} = μ * min(x, 1-x)
- At x=0.5, next = 1.0 (exactly)
- At x=1.0, next = 0.0 (exactly)
- At x=0.0, stays at 0 forever

**Verification:** Ran 1000 iterations, tent map produced only 3 unique values; logistic r=4 produced 1000 unique values.

**Decision:** Removed tent map. MVP uses logistic r=4 only.

### Protocol: Auto-Advance After PREDICT

**Original design:** Agent must call MOVE_ON to advance to next task.
**Problem:** Prompt said "after PREDICT you'll see your score" but code only showed score after MOVE_ON.
**Result:** Agent predicted once, got no feedback, kept predicting the same value.

**Fix:** PREDICT now auto-advances. Simpler workflow:
1. See data
2. PREDICT
3. See feedback
4. Next task

### LLM Anchoring on Examples

**Observation:** Agent predicted 0.42 for every single task (5/5 tasks).
**Root cause:** Prompt example was `{"action": "PREDICT", "value": 0.42}`.
**Fix:** Changed to `{"action": "PREDICT", "value": <your_prediction>}`.
**Result:** Agent now produces varied predictions based on data analysis.

**Lesson:** Never put concrete numeric examples in prompts where the model should reason about values. Use placeholders.

---

## Open Questions

### Transfer Detection
- How do we measure if learnings actually helped?
- Possible: ablation (run with vs without learnings)
- Possible: correlate learnings content with score improvement

### Task Ordering
- Currently stratified by h_KS
- Should we group by family to make transfer easier to observe?
- Or random to test generalization?

---

## References

- Design doc: `docs/plans/2026-01-30-chaosbench-metacognitive-agent-design.md`
- ChaosBench spec: `chaosbench/core/ChaosSpecification.md`
- Existing implementation: `chaosbench/core/Chaosbench_v3.py`
