# Progress Log: ChaosBench v4

## 2026-02-03 (NLL Scoring & Protocol Fixes)

### Session: MVP Implementation Complete

**Context:** Implementing fixes from `docs/plans/2026-02-01-chaosbench-fixes-design.md`

**Work Completed:**

1. **NLL Bin-Based Scoring** (`chaosbench/core/scoring.py`)
   - Created truncated Gaussian scoring over 20 bins
   - Punishes confident wrong predictions, rewards calibration
   - Score = probability mass in bin containing true value

2. **Chaotic Systems Factory** (`chaosbench/core/Chaosbench_v3.py`)
   - Added `create_chaotic_system()` and `get_chaotic_systems()`
   - Restricted to logistic r=4 only (h_KS = ln(2) ≈ 0.693)
   - Removed tent map due to numerical instability (degenerates to x=0)

3. **Task Visualization** (`chaosbench/visualization/plots.py`)
   - Enhanced `plot_task()` with 20-bin overlay
   - Highlights bin containing actual value

4. **Protocol Fix: Auto-Advance After PREDICT**
   - Problem: Agent never called MOVE_ON because prompt said feedback comes after PREDICT but code showed feedback after MOVE_ON
   - Fix: PREDICT now automatically ends task and shows feedback

5. **Anchoring Fix**
   - Problem: Agent always predicted 0.42 (example value in prompt)
   - Fix: Changed `{"action": "PREDICT", "value": 0.42}` to `{"action": "PREDICT", "value": <your_prediction>}`

**Test Results:**
- 5 tasks with Gemini 2.0 Flash
- Final Phi: 0.08 (non-zero!)
- Agent now produces varied predictions (0.859, 0.650, 0.120, 0.150, 0.600)
- Scores still low (0.00-0.11) but protocol works

**Commits on `feature/hypothesis-testing`:**
```
1fa0445 fix(prompt): remove example value to prevent anchoring
f152247 fix: auto-advance after PREDICT, remove MOVE_ON requirement
036de78 fix: remove tent map due to numerical instability
a669559 feat: add NLL scoring, chaotic systems, task visualization
```

**Key Insight:** LLMs anchor strongly on example values in prompts. Using placeholder `<your_prediction>` eliminated the constant 0.42 predictions.

---

## 2026-01-31 (Back to Basics Session)

### Session: Fundamental Redesign → True MVP

**Context:** Realized the benchmark had drifted from original research question.

**Original question:** "Do LLMs learn transferable structure?" (superlinear Φ)
**What we were testing:** "Can LLMs use scientific tools?" (different question)

**Key Insight (Morgan):**
> "We need to go back to basics... the agent has a shared context window so in theory we want to see how if it sees one task then the next the information is in the context window so possibly it could help with solving"

### Initial Decisions

**1. Metrics: Φ(n) and T(n) by task index**
- Wall-clock is too noisy for cognitive claims
- Φ(n) = cumulative h_KS-weighted score after n tasks
- T(n) = turns required for task n
- Interpretation matrix: superlinear + decreasing = strong learning

**2. Output format: Point ± σ**
- Agent outputs "0.67 ± 0.05" or just "0.67"
- Convert to truncated Gaussian → 20 bins → NLL
- Default σ = 0.1 if not given

**3. Systems: 1D only for v1**
- Logistic (r ∈ [3.57, 4]) and Tent (μ ∈ [1, 2])
- Clean h_KS computation
- No multi-dimensional binning complexity

**4. Parameter space defined**
- Primary: horizon, noise_std, n_obs, family
- All adjustable for difficulty sweeps

### Final Push: True MVP

**Morgan's challenge:** "Why do we still have HYPOTHESIZE and FIT? We're making the MVP."

**Resolution:** Strip everything. The true MVP is:

| Action | Purpose |
|--------|---------|
| PREDICT | Output value ± σ |
| MOVE_ON | Next task |

No HYPOTHESIZE. No FIT. No WRITE. No DELETE.

**Rationale:** The research question is "does the agent learn?" — not "can the agent use tools?" If learning happens, it should show in raw prediction accuracy improving over tasks. Scaffolding becomes a future experimental condition (v1.2).

### Why NLL (from first principles)

MSE is gameable by mean reversion — predicting attractor center gives low MSE because you're "close to everything."

NLL punishes confident wrong predictions. Mean reversion predicts center bin confidently, but that bin is only correct ~5% of the time.

### Version Plan

| Version | Features |
|---------|----------|
| **v1 (MVP)** | PREDICT, MOVE_ON only |
| v1.1 | Add retry logic (multiple predictions per task) |
| v1.2 | Add HYPOTHESIZE, FIT (scaffolding condition) |
| v1.3 | Add WRITE, DELETE (learnings condition) |

### Design Document

`docs/plans/2026-01-31-chaosbench-v1-minimal-design.md`

---

## 2026-01-31 (Earlier: Simple Benchmark)

### Session: Simple Benchmark Design

**Context:** Claude one-shot both logistic and Henon tasks. Too easy with current setup.

**Evolved into:** The back-to-basics redesign above.

**Difficulty Grid (still valid):**

| | Clean (σ=0) | Noisy (σ=0.02) | Very Noisy (σ=0.05) |
|---|---|---|---|
| **1-step** | Easy | Medium | Medium |
| **5-step** | Medium | Hard | Hard |
| **20-step** | Hard | Very Hard | Extreme |

---

## 2026-01-30 (Verification Session)

### Session: Implementation Verification

**Verified Phase 6 fully implemented:**

| Task | Component | Status |
|------|-----------|--------|
| 1 | Model Factory (`models.py`) | ✅ 4 tests pass |
| 2 | Backtest Function (`backtest.py`) | ✅ 4 tests pass |
| 3 | Parameter Fitting (`fitting.py`) | ✅ 5 tests pass |
| 4 | New Action Types | ✅ HYPOTHESIZE, FIT in types |
| 5 | BacktestFeedback | ✅ Dataclass + format() |
| 6 | HYPOTHESIZE Handler | ✅ session.py:159 |
| 7 | FIT Handler | ✅ session.py:175 |
| 8 | System Prompt | ✅ hypothesis_system.txt |
| 9 | Integration Test | ✅ 2 tests pass |
| 10 | Full Test Suite | ✅ **76 tests pass** |

**Updated planning docs** to reflect true status — Phase 6 complete.

**Next:** Phase 7 — Run experiments with hypothesis-driven agent.

---

## 2026-01-30 (Evening Session)

### Session: Hypothesis-Driven Redesign

**Key Insight (Morgan):**
> "In real life we don't know the answer, we do a model, we test it on existing data and extrapolate"

Blind prediction isn't scientific reasoning. Real science involves testing hypotheses against known data.

**Design Completed:**
- [x] HYPOTHESIZE action — test model against observations, get MAE
- [x] FIT action — auto-fit parameters for model family
- [x] Feedback format — minimal (MAE + quality message + predicted x_50)
- [x] Experimental design — scaffolded vs CODE-only conditions

**New Actions:**
```
HYPOTHESIZE: {"action": "HYPOTHESIZE", "model": "logistic", "params": {"r": 3.85}}
FIT: {"action": "FIT", "model": "logistic"}
```

**Feedback Format:**
```
Model: logistic (r=3.7)

Backtest (fitting x_0 → x_49):
  MAE: 0.147
  Your model doesn't reproduce the observations well.

If you trust this model, it predicts x_50 = 0.394
```

**Implementation estimate:** ~530 lines total
- backtest.py: ~60 lines
- fitting.py: ~80 lines
- Session integration: ~90 lines
- Prompts: ~110 lines
- Tests: ~160 lines

**Next:**
- [ ] Create feature branch
- [ ] Implement backtest.py
- [ ] Implement fitting.py
- [ ] Wire into session runner
- [ ] Update prompt
- [ ] Test

**Implementation Plan:** `docs/plans/2026-01-30-hypothesis-testing.md`

---

## 2026-01-30 (Afternoon Session)

### Session: Debugging Blind Prediction

**Issue Found:**
- Original design gave feedback after each PREDICT
- Agent exploited this by iterating toward revealed answer ("hot/cold")
- Got perfect scores by just converging, not actually predicting

**Fix Applied:**
- Changed to two-phase design: blind prediction → reflection
- Agent predicts WITHOUT seeing answer
- Only sees result after MOVE_ON commit
- Can then WRITE learnings before next task

**Current Bug:**
- Agent never says MOVE_ON (hits 20-turn limit)
- Still getting perfect scores (suspicious - needs debugging)
- Agent not using WRITE at all

**Files Modified:**
- `chaosbench/experiments/session.py` - blind prediction logic
- `chaosbench/prompts/metacognitive_system.txt` - two-phase instructions
- `shared/llm_utils.py` - added dotenv loading

**Next:**
- [ ] Debug why agent never commits
- [ ] Verify scoring is truly blind
- [ ] Consider single-shot prediction

---

## 2026-01-30 (Morning Session)

### Session: Implementation

**Completed:**
- [x] Task 1: Data types (metacognitive_types.py)
- [x] Task 2: Learnings manager (learnings.py)
- [x] Task 3: Trace logger (trace.py)
- [x] Task 4: Session runner (session.py)
- [x] Task 5: LLM agent (metacognitive_agent.py)
- [x] Task 6: CLI runner (run_metacognitive.py)
- [x] Task 7: Integration test

**Test Results:**
- All 55 tests pass
- Ran 15 tasks with Gemini 2.0 Flash
- Agent got perfect scores on 14/15 tasks (but was exploiting feedback)

---

## 2026-01-30 (First Session)

### Session: Design Brainstorm

**Completed:**
- [x] Reviewed ChaosBench v3 spec and implementation
- [x] Defined research hypothesis (superlinear Φ(t) = transfer)
- [x] Designed metacognitive agent protocol
- [x] Decided: passive observations for v1.0, probing in v1.1
- [x] Decided: PREDICT, WRITE, DELETE, MOVE_ON actions
- [x] Decided: JSON action format, minimal feedback
- [x] Decided: last score counts, wall-clock time pressure
- [x] Drafted system prompt
- [x] Defined Φ(t) calculation
- [x] Created design doc

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Gemini 3 Flash JSON output | Thinking tokens consumed budget | Use Gemini 2.0 Flash instead |
| Agent exploiting feedback | Iterating toward revealed answer | Changed to blind prediction |
| Agent never MOVE_ON | Keeps predicting, hits turn limit | Root cause: no feedback loop to learn from |
| Blind prediction meaningless | Multiple guesses without feedback | Redesigned: hypothesis testing with backtest |
| Tent map degenerates to x=0 | Floating-point rounds to 0, stays there | Removed tent map, logistic r=4 only |
| Agent never calls MOVE_ON | Prompt/code mismatch (feedback timing) | Auto-advance after PREDICT |
| Agent always predicts 0.42 | LLM anchoring on prompt example | Changed example to `<your_prediction>` |
