# Task Plan: ChaosBench v4 Metacognitive Agent

**Goal**: Implement an LLM agent protocol that tests scientific reasoning via explicit metacognition on chaotic dynamical systems.

**Hypothesis**: Superlinear Φ(t) = transfer learning; Linear Φ(t) = per-task heuristics

---

## Phases

### Phase 1: Core Protocol `pending`
- [ ] AgentObservation / AgentAction dataclasses
- [ ] Session runner (task loop with PREDICT/WRITE/DELETE/MOVE_ON)
- [ ] LEARNINGS.md read/write/delete mechanics
- [ ] JSON action parsing from LLM output
- [ ] Trace logging (every turn, every task)

### Phase 2: LLM Integration `pending`
- [ ] Gemini API wrapper
- [ ] System prompt injection
- [ ] Message formatting (observations + learnings + feedback)
- [ ] Response parsing (reasoning + JSON action)

### Phase 3: Evaluation `pending`
- [ ] Φ(t) curve tracking
- [ ] Per-task metrics (attempts, time, final score)
- [ ] Session summary (total Φ, tasks completed, learnings size)

### Phase 4: Analysis `pending`
- [ ] Trace viewer / pretty-printer
- [ ] Learnings evolution over session
- [ ] Transfer detection (did learnings help?)

---

## Key Decisions

| Decision | Choice |
|----------|--------|
| Retry policy | Agent decides, time cost |
| Score counting | Last prediction before MOVE_ON |
| Feedback | Minimal (prediction, actual, score) |
| Observations | Passive — 50 given upfront (v1.0) |
| Actions | PREDICT, WRITE, DELETE, MOVE_ON |
| Action format | JSON |

---

## Files to Create

- `chaosbench/agents/metacognitive.py` — LLM agent implementation
- `chaosbench/experiments/session_runner.py` — Session loop
- `chaosbench/experiments/trace.py` — Trace logging
- `chaosbench/prompts/metacognitive_system.txt` — System prompt

---

## Design Doc

Full specification: `docs/plans/2026-01-30-chaosbench-metacognitive-agent-design.md`
