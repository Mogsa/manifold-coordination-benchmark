# Progress Log: ChaosBench v2

## 2026-02-06 (Direction Reset)

### Session: Cleanup + New Direction

**Context:** Switching from v1 metacognitive agent prototype to PRD v2 implementation.

**Work Completed:**

1. **Committed all uncommitted changes** from v1 prototyping
   - Prompt anchoring fix, auto-advance, tent map removal, planning docs
   - `41d0ac8`, `48b0701`

2. **Removed old experiment outputs**
   - `hard_run/`, `mvp_test_run/`, `hypothesis_test_log.txt`
   - `session_output/`, `session_output_15/`, `session_output_fixed/`, `test_debug/`
   - `54abf39`, `2954fd1`

3. **Removed superseded docs** (10,102 lines of bloat)
   - 8 old planning docs (phase1-fixes, metacognitive-agent-design, hypothesis-testing, etc.)
   - `FUTURE_IDEAS.md` (coordination benchmark, not chaosbench)
   - `.ralph/` directory (unrelated agent framework)

4. **Reviewed trace.md from hard_run** — identified scoring bug
   - All 5 tasks scored 1.00 despite clearly wrong predictions
   - Root cause: NLL bin scoring [0,1] applied to systems with different ranges
   - Agent reasoning patterns documented (period-2 detection on henon, mean reversion fallback on chaotic)

5. **Updated planning files** to align with PRD v2 (`docs/Chaos_IMO`)
   - `task_plan.md` — 6 new phases matching PRD build sequence
   - `findings.md` — Distilled lessons from v1, framed as rules for v2
   - `progress.md` — This file, fresh session log

**New PRD:** `docs/Chaos_IMO` — 1050-line comprehensive spec targeting NeurIPS 2026
- Multiple question types (CLASSIFY, IDENTIFY, PREDICT + 3 deferred)
- Grammar system (atoms + connectives)
- Blocked vs shuffled vs independent conditions
- Validation pipeline (hard gates + baseline battery)
- SQLite storage with full reproducibility

**Key Decision**: Starting Phase 1 (Mathematical Core) from PRD v2 spec. Old `chaosbench/` code exists but may need restructuring into new module layout (`grammar/`, `problems/`, `validation/`, etc.).

**Open Question for Morgan**: Reuse existing `chaosbench/` code and refactor, or build fresh in the PRD v2 module structure?

---

## Previous Sessions (v1 Prototype — Archived)

Full session logs from Jan 30 – Feb 3 were in the old version of this file.
Key events preserved in `findings.md` as distilled lessons.

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
| Tent map degenerates to x=0 | μ=2 floating-point | Non-standard params + stability gate |
| NLL scoring 1.00 everywhere | Bins [0,1] for non-[0,1] systems | Use k_eff for PREDICT, exact match for CLASSIFY |
| Agent never calls MOVE_ON | Prompt/code protocol mismatch | Test with real LLM, not just unit tests |
| Scaffolding too easy | Claude one-shots with HYPOTHESIZE | Raw mode first, scaffolding as condition |
