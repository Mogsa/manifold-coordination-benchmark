# Task Plan: ChaosBench v4 Metacognitive Agent

**Goal**: Implement an LLM agent protocol that tests scientific reasoning via explicit metacognition on chaotic dynamical systems.

**Hypothesis**: Superlinear Φ(t) = transfer learning; Linear Φ(t) = per-task heuristics

---

## Phases

### Phase 1: Core Protocol ✅ COMPLETE
- [x] AgentObservation / AgentAction dataclasses
- [x] Session runner (task loop with PREDICT/WRITE/DELETE/MOVE_ON)
- [x] LEARNINGS.md read/write/delete mechanics
- [x] JSON action parsing from LLM output
- [x] Trace logging (every turn, every task)

### Phase 2: LLM Integration ✅ COMPLETE
- [x] Gemini API wrapper (via LiteLLM)
- [x] System prompt injection
- [x] Message formatting (observations + learnings + feedback)
- [x] Response parsing (reasoning + JSON action)

### Phase 3: Evaluation ✅ COMPLETE
- [x] Φ(t) curve tracking
- [x] Per-task metrics (attempts, time, final score)
- [x] Session summary (total Φ, tasks completed, learnings size)

### Phase 4: Analysis `pending`
- [ ] Trace viewer / pretty-printer
- [ ] Learnings evolution over session
- [ ] Transfer detection (did learnings help?)

### Phase 5: Design Fix ✅ COMPLETE
- [x] Identified feedback exploit (agent iterating toward revealed answer)
- [x] Implemented blind prediction (no feedback until MOVE_ON)
- [x] Added reflection phase (WRITE after seeing result)
- [x] Debugged: Scores not perfect (0.22 avg), agent just never commits
- [x] Root cause: Blind guessing isn't scientific reasoning

### Phase 6: Hypothesis-Driven Redesign ✅ COMPLETE
**Insight:** Real science involves testing hypotheses against data, not blind guessing.

New actions:
- [x] Design HYPOTHESIZE action (test model against observations)
- [x] Design FIT action (auto-fit parameters)
- [x] Implement `chaosbench/core/models.py` (model factory)
- [x] Implement `chaosbench/core/backtest.py` (~60 lines)
- [x] Implement `chaosbench/core/fitting.py` (~80 lines)
- [x] Add new action types to `metacognitive_types.py`
- [x] Add handlers to `session.py`
- [x] Update system prompt (`hypothesis_system.txt`)
- [x] Unit tests for models/backtest/fitting (13 tests)
- [x] Integration test (2 tests)
- [x] All 76 chaosbench tests pass

**Feedback format (minimal):**
```
Model: logistic (r=3.7)

Backtest (fitting x_0 → x_49):
  MAE: 0.147
  Your model doesn't reproduce the observations well.

If you trust this model, it predicts x_50 = 0.394
```

**Experimental design:**
- Phase A: Scaffolded (HYPOTHESIZE, FIT provided)
- Phase B: CODE only (see if agents discover strategy)
- Phase C: Analyze the gap

---

## Implementation Status

### Existing (Scaffolded Version — Deferred to v1.2)
| Component | File | Status |
|-----------|------|--------|
| Data Types | `chaosbench/agents/metacognitive_types.py` | ✅ Done |
| Learnings | `chaosbench/agents/learnings.py` | ✅ Done |
| Session Runner | `chaosbench/experiments/session.py` | ✅ Done |
| Model Factory | `chaosbench/core/models.py` | ✅ Done |
| Backtest | `chaosbench/core/backtest.py` | ✅ Done |
| Fitting | `chaosbench/core/fitting.py` | ✅ Done |

### MVP v1 ✅ IMPLEMENTED
| Component | File | Status |
|-----------|------|--------|
| Scoring (NLL) | `chaosbench/core/scoring.py` | ✅ Done |
| Chaotic Systems | `chaosbench/core/Chaosbench_v3.py` | ✅ Done (logistic r=4 only) |
| Session Runner | `chaosbench/experiments/session.py` | ✅ Done (auto-advance) |
| MVP Prompt | `chaosbench/prompts/mvp_system.txt` | ✅ Done |
| Task Visualization | `chaosbench/visualization/plots.py` | ✅ Done (bins overlay) |
| CLI | `chaosbench/run_metacognitive.py` | ✅ Done (--save-task-plots) |

---

## Key Decisions

### MVP (v1)
| Decision | Choice | Status |
|----------|--------|--------|
| Actions | **PREDICT, MOVE_ON only** | ✅ |
| Systems | Logistic, Tent (1D) | ✅ |
| Output | Point ± σ → Gaussian → NLL | ✅ |
| Metrics | Φ(n), T(n) by task index | ✅ |
| Bins | 20 uniform over [0, 1] | ✅ |
| Feedback | Actual value after PREDICT | ✅ |

### Future (v1.2+)
| Decision | Choice | Status |
|----------|--------|--------|
| Scaffolding | HYPOTHESIZE, FIT | Planned |
| Learnings | WRITE, DELETE | Planned |
| Retry logic | Multiple PREDICT per task | Planned |

---

## Current Work: MVP Implementation

**Phase 6 Complete.** Hypothesis-driven framework exists but is deferred to v1.2.

**MVP Focus (v1):**
1. Agent sees observations [x_0, ..., x_49]
2. Agent outputs prediction (value ± σ)
3. System scores via NLL
4. Agent sees actual value
5. Next task
6. Measure Φ(n) for superlinearity

### Phase 7: Minimal Benchmark v1 (MVP) ✅ COMPLETE
**Goal:** Simplest possible benchmark — observations in, prediction out, measure Φ(n).

**MVP Scope:**
- **Actions:** PREDICT only (auto-advances after prediction)
- **Systems:** Logistic r=4 only (tent removed due to numerical instability)
- **Output:** Point estimate → Gaussian σ=0.1 → NLL over 20 bins
- **Metrics:** Φ(n) by task index
- **No scaffolding:** No HYPOTHESIZE, no FIT, no WRITE

**Implementation Complete:**
- [x] NLL Scoring: `chaosbench/core/scoring.py` (truncated Gaussian over 20 bins)
- [x] Chaotic Systems: `create_chaotic_system()`, `get_chaotic_systems()` in Chaosbench_v3.py
- [x] Session Runner: Auto-advance after PREDICT, NLL integration
- [x] MVP Prompt: `chaosbench/prompts/mvp_system.txt`
- [x] Visualization: `plot_task()` with bins overlay

**Bugs Fixed:**
- Tent map numerical instability → Removed
- MOVE_ON never called → Auto-advance after PREDICT
- Anchoring on example 0.42 → Use `<your_prediction>` placeholder

**Test Results (Gemini 2.0 Flash, 5 tasks):**
- Final Phi: 0.08
- Varied predictions (not anchored)
- Protocol working correctly

**Future extensions (not MVP):**
- v1.1: Retry logic (multiple PREDICT before MOVE_ON)
- v1.2: Scaffolding (HYPOTHESIZE, FIT)
- v1.3: Learnings notepad (WRITE, DELETE)

### Phase 8: Run MVP Experiments 🔧 NEXT
- [x] Initial test run (Gemini 2.0 Flash, 5 tasks) — Working
- [ ] Run larger sessions (50+ tasks) to see Φ(n) curve shape
- [ ] Test with different LLMs (Claude, GPT-4)
- [ ] Vary difficulty: horizon × noise grid
- [ ] Analyze Φ(n) shape for superlinearity
- [ ] Baseline comparison (random, mean reversion)

### Phase 9: Scaffolding Comparison `future`
- [ ] Implement v1.2 (add HYPOTHESIZE/FIT)
- [ ] Run same tasks with scaffolding
- [ ] Compare Φ(n) curves: MVP vs Scaffolded
- [ ] Measure value of explicit scientific tools

---

## Design Docs

| Doc | Purpose |
|-----|---------|
| **`2026-01-31-chaosbench-v1-minimal-design.md`** | MVP specification (current) |
| `2026-01-30-chaosbench-metacognitive-agent-design.md` | Scaffolded version (v1.2) |
| `2026-01-30-hypothesis-testing.md` | HYPOTHESIZE/FIT implementation |
| `chaosbench-v4-findings.md` | Research decisions and rationale |
| `chaosbench-v4-progress.md` | Session-by-session log |
