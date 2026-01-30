# Progress Log: ChaosBench v4

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
