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

### Phase 6: Hypothesis-Driven Redesign 🔧 IN PROGRESS
**Insight:** Real science involves testing hypotheses against data, not blind guessing.

New actions:
- [x] Design HYPOTHESIZE action (test model against observations)
- [x] Design FIT action (auto-fit parameters)
- [ ] Implement `chaosbench/core/backtest.py` (~60 lines)
- [ ] Implement `chaosbench/core/fitting.py` (~80 lines)
- [ ] Add new action types to `metacognitive_types.py`
- [ ] Add handlers to `session.py`
- [ ] Update system prompt
- [ ] Unit tests for backtest/fitting
- [ ] Integration test

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

| Component | File | Status |
|-----------|------|--------|
| Data Types | `chaosbench/agents/metacognitive_types.py` | ✅ Done (needs HYPOTHESIZE/FIT) |
| Learnings | `chaosbench/agents/learnings.py` | ✅ Done |
| Trace Logger | `chaosbench/experiments/trace.py` | ✅ Done |
| Session Runner | `chaosbench/experiments/session.py` | ✅ Done (needs new handlers) |
| LLM Agent | `chaosbench/agents/metacognitive_agent.py` | ✅ Done |
| CLI Runner | `chaosbench/run_metacognitive.py` | ✅ Done |
| System Prompt | `chaosbench/prompts/metacognitive_system.txt` | ✅ Done (needs hypothesis update) |
| Integration Test | `chaosbench/tests/test_integration.py` | ✅ Done |
| **Backtest** | `chaosbench/core/backtest.py` | 🆕 Not started |
| **Fitting** | `chaosbench/core/fitting.py` | 🆕 Not started |
| **Hypothesis Prompt** | `chaosbench/prompts/hypothesis_system.txt` | 🆕 Not started |

---

## Key Decisions

| Decision | Choice | Status |
|----------|--------|--------|
| Retry policy | Agent decides, time cost | ✅ |
| Score counting | Last prediction before MOVE_ON | ✅ |
| Feedback | ~~After each PREDICT~~ → Backtest on known data | 🔧 Redesigned |
| Observations | Passive — 50 given upfront (v1.0) | ✅ |
| Actions | PREDICT, WRITE, DELETE, MOVE_ON, **HYPOTHESIZE, FIT** | 🔧 Expanded |
| Action format | JSON | ✅ |
| x_50 visibility | Hidden until PREDICT commit | ✅ |
| Backtest feedback | MAE + quality message + predicted x_50 | 🆕 |

---

## Current Work: Hypothesis-Driven Redesign

**Problem solved:** Blind guessing isn't scientific reasoning. Agent can't learn without feedback loop.

**Solution:** Let agent test models against KNOWN data (x_0...x_49), then extrapolate to x_50.

**The loop:**
1. Agent proposes model (e.g., "logistic r=3.85")
2. System backtests against observations, reports MAE
3. Agent refines or tries different model
4. When satisfied, agent commits prediction
5. Only then x_50 revealed

This mirrors real science: fit model to existing data, extrapolate to unknown.

---

## Design Doc

Full specification: `docs/plans/2026-01-30-chaosbench-metacognitive-agent-design.md`
Status doc: `docs/plans/2026-01-30-metacognitive-agent-status.md`
