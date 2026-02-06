# Progress Log: ChaosBench v2

## 2026-02-06 Session 2: Hybrid Plan + Repo Refactor

### Context
Brainstorming session with Morgan. Agreed on hybrid build path: static spine + proposition sandbox.

### Key Decisions Made

1. **Morgan's core excitement**: IMO-style proposition — can LLMs create layered, deceptively simple problems?
   - This is v1.5 (tournament) in the PRD, but we're pulling it forward
   - Static spine keeps NeurIPS transfer experiment alive

2. **Hybrid path chosen**:
   - Phase 1-2: Atoms + verification + mini-bank (12-18 tasks)
   - Phase 3-4: Agent interface + proposition sandbox (Gemini propose/solve)
   - Phase 5: Blocked vs shuffled experiment
   - Phase 6: Expand + write

3. **Repo refactor done**:
   - v1 agent/experiment/scoring code → `chaosbench/legacy_v0/`
   - New module directories created: `grammar/`, `problems/`, `validation/`, `scoring/`, `agents/`, `experiment/`, `sandbox/`, `storage/`, `analysis/`
   - Kept: `core/Chaosbench_v3.py`, `core/lyapunov.py`, `core/models.py`
   - New branch: `feature/chaosbench-v2`

4. **Code triage**:
   - KEEP: lyapunov.py (all 3 methods), system classes from Chaosbench_v3.py
   - KILL (moved to legacy): metacognitive agent, session loop, NLL scoring, backtest/fitting, v1 prompts
   - Model access: Gemini only for now

### Work Completed
- [x] Created `feature/chaosbench-v2` branch
- [x] Moved v1 code to `legacy_v0/`
- [x] Created v2 module directory structure
- [x] Wrote hybrid task plan (6 phases)
- [ ] Phase 1 implementation (next)

---

## 2026-02-06 Session 1: Direction Reset

### Session: Cleanup + New Direction

**Context:** Switching from v1 metacognitive agent prototype to PRD v2 implementation.

**Work Completed:**

1. **Committed all uncommitted changes** from v1 prototyping
2. **Removed old experiment outputs**
3. **Removed superseded docs** (10,102 lines of bloat)
4. **Updated planning files** to align with PRD v2

**New PRD:** `docs/Chaos_IMO` — 1050-line comprehensive spec targeting NeurIPS 2026

---

## Previous Sessions (v1 Prototype — Archived)

Summary of v1 prototype journey:
1. Jan 30 AM: Built metacognitive agent (PREDICT/WRITE/DELETE/MOVE_ON)
2. Jan 30 PM: Found feedback exploit → blind prediction
3. Jan 30 EVE: Blind prediction meaningless → hypothesis-driven redesign (HYPOTHESIZE/FIT)
4. Jan 31: Scope creep recognized → stripped to MVP (PREDICT only)
5. Feb 1-3: NLL scoring, tent map removal, anchoring fix, auto-advance
6. Feb 6: Direction reset → PRD v2

---

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| Agent always predicts 0.42 | Prompt example anchoring | Use `<placeholder>` not concrete values |
| Tent map degenerates to x=0 | mu=2 floating-point | Non-standard params + stability gate |
| NLL scoring 1.00 everywhere | Bins [0,1] for non-[0,1] systems | Use k_eff for PREDICT, exact match for CLASSIFY |
| Agent never calls MOVE_ON | Prompt/code protocol mismatch | Test with real LLM, not just unit tests |
| Scaffolding too easy | Claude one-shots with HYPOTHESIZE | Raw mode first, scaffolding as condition |
