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
- **Phase A:** Scaffolded (HYPOTHESIZE, FIT actions provided)
- **Phase B:** CODE only (just Python executor, no hints)
- **Phase C:** Compare — did agents discover the fit-and-test strategy?

The gap between A and B is the finding.

### Model Families (Already Exist)
From `Chaosbench_v3.py`:
- LogisticMap(r) — 1D, r ∈ [3.57, 4.0]
- TentMap(μ) — 1D, μ ∈ [1.0, 2.0]
- HenonMap(a, b) — 2D
- StandardMap(K) — 2D
- LorenzDisc(σ, ρ, β) — 3D

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
