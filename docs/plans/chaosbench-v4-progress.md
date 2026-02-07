# Progress Log: ChaosBench v2

## Project Roadmap

- [x] Phase 1-2: Core grammar, problems, validation, scoring (213 tests)
- [x] Phase 3: Agent interface + Gemini testing (36% CLASSIFY, 55% IDENTIFY, 79% PREDICT)
- [ ] **Phase 4: Adversarial arena** (propose/solve/review/consensus) <-- CURRENT
- [ ] Phase 5: Static experiment (blocked vs shuffled, Phi(n) curves)
- [ ] Phase 6: Expand + write dissertation

---

## Session: 2026-02-06 (Session 5) — Phase 4 Arena Implementation

### Work Done
- Implemented all 6 arena source files: protocol.py, prompts.py, parsing.py, consensus.py, runner.py
- Created comprehensive test suite: 41 tests across 5 test classes
- **254 tests total, all passing, zero regressions**

### Files Created
- `chaosbench/arena/__init__.py` — module init
- `chaosbench/arena/protocol.py` — 5 dataclasses (Proposal, SolveResult, Review, Reputation, RoundResult)
- `chaosbench/arena/prompts.py` — 3 role-specific prompt formatters (proposer, solver, reviewer)
- `chaosbench/arena/parsing.py` — fail-safe parsers (JSON proposals, Likert reviews, solve results)
- `chaosbench/arena/consensus.py` — consensus aggregation, discrimination scoring, reputation updates
- `chaosbench/arena/runner.py` — 4-phase round loop + CLI entry point
- `chaosbench/tests/test_arena.py` — 41 tests with mocked LLM

### Architecture Decisions
- **Solver prompt reuses agents/prompts.py** — no duplication, same format as Phase 3
- **Parsing delegates to agents/parsing.py** — solve result parsing reuses existing parse_response()
- **Fail-safe pattern everywhere** — proposal/review parsers never crash, return defaults on garbage
- **Reputation is three-axis** — propose (validation+discrimination), solve (accuracy), review (correlation with math truth)
- **Consensus = highest-rated solver's answer** — simple, reviewers weight it indirectly

### Remaining
- [ ] Live run with Gemini (10 rounds, ~$0.30-0.50)

---

## Session: 2026-02-06 (Session 4) — PRD Alignment + Vision Update

### Work Done
- Ran full alignment audit: PRD vs implementation (identified 7 divergences)
- Fixed PRD §1.1 scoring formula (was `raw × (1+h_KS)`, now points to §7.2 composite)
- Fixed PRD Appendix A.2 tent map range ([1.0, 1.95], not [1.0, 2.0])
- Added PRD §8.4: Tool-Augmented Mode (v2) — minimal tool set (COMPUTE_STATS, ITERATE, LYAPUNOV_ESTIMATE, RETURN_MAP, COMPARE)
- Strengthened PRD §3.1: explicit IMO property (difficulty from chaining, not exotic knowledge)
- Expanded PRD §5.2: round lifecycle with problem backlog (discriminating problems persist)
- Updated PRD §14: committee review (v2), web platform (v3), 100+ LLM scale (v3)
- Archived CHAOSBENCH_REPORT.md (v1 prototype doc, superseded)

### Key Design Decisions (with Morgan)
- **Tools are minimal**: enough for educated guesses, not enough for brute force. Reasoning is the differentiator.
- **Solver knows the registry** but not which atoms were used — like knowing it's a geometry problem but not which theorems apply
- **Committee review** is v2 (future) — v1.5 uses automated validation only
- **Problem backlog** is the benchmark — self-improving over time
- **Web platform + 100+ LLMs** is v3 vision — anyone can participate without API cost

---

## Session: 2026-02-06 (Session 3) — Consolidation

### Work Done
- Consolidated 6 planning files into 3 (task plan, findings, progress)
- Removed root-level duplicates and superseded Phase 1-2 plan
- Updated CLAUDE.md with architecture documentation
- Ready to begin Phase 4 implementation

---

## Session: 2026-02-06 (Session 2) — Phase 3 + Phase 4 Planning

### Phase 3 Implementation (Complete)
- Created 5 source files: agents/protocol.py, prompts.py, parsing.py, llm_agent.py, experiment/runner.py
- Created 2 test files: test_parsing.py, test_runner.py
- Fixed "quasiperiodic" substring bug (sort labels longest-first)
- Fixed confidence regex not matching negatives
- **44 new tests, 213 total, all passing**
- Ran Gemini on all 27 problems: CLASSIFY=36%, IDENTIFY=55%, PREDICT=79%
- Diagnosed CLASSIFY failures: Gemini can't distinguish chaotic/quasiperiodic without analytical tools
- Diagnosed IDENTIFY failures: "logistic" default bias from training data

### Phase 4 Planning (Complete)
- Designed adversarial arena with Morgan
- Four-phase round: PROPOSE -> BLIND SOLVE -> PEER REVIEW -> CONSENSUS
- Key decisions: Gemini-only, both ground truths tracked, chaosbench/arena/ module
- NeurIPS-style Likert scales defined (1-6 question quality, 1-6 answer correctness, 1-5 confidence)
- Three-axis reputation: proposing, solving, reviewing

### Files Created
- `chaosbench/agents/protocol.py`
- `chaosbench/agents/prompts.py`
- `chaosbench/agents/parsing.py`
- `chaosbench/agents/llm_agent.py`
- `chaosbench/experiment/runner.py`
- `chaosbench/tests/test_parsing.py`
- `chaosbench/tests/test_runner.py`

---

## Session: 2026-02-06 (Session 1) — Phase 1-2 + Repo Refactor

### Phase 1-2 Implementation (Complete)
- 4 atoms implemented (logistic, tent, damped_linear, rotation)
- 27 validated problems in frozen bank (mini_bank.json)
- 7 quality gates + 5 baselines
- 3 verifiers (classify, identify, predict)
- 107 tests passing

### Repo Refactor
- Moved v1 code to `legacy_v0/`
- Created v2 module structure (grammar/, problems/, validation/, scoring/, agents/, experiment/)
- Created `feature/chaosbench-v2` branch
- Agreed on hybrid build path: static spine + proposition sandbox

### Key Decisions
- Morgan's core excitement: IMO-style proposition (can LLMs create layered problems?)
- This is v1.5 (tournament) in the PRD, pulled forward
- Static spine keeps NeurIPS transfer experiment alive
- Kept: lyapunov.py (all 3 methods), system classes from Chaosbench_v3.py
- Killed (moved to legacy): metacognitive agent, session loop, NLL scoring, backtest/fitting, v1 prompts

---

## Earlier Sessions: v1 Prototype (Archived)

Summary of v1 prototype journey:
1. Jan 30 AM: Built metacognitive agent (PREDICT/WRITE/DELETE/MOVE_ON)
2. Jan 30 PM: Found feedback exploit -> blind prediction
3. Jan 30 EVE: Blind prediction meaningless -> hypothesis-driven redesign (HYPOTHESIZE/FIT)
4. Jan 31: Scope creep recognized -> stripped to MVP (PREDICT only)
5. Feb 1-3: NLL scoring, tent map removal, anchoring fix, auto-advance
6. Feb 6: Direction reset -> PRD v2

---

## Errors Encountered

| Error | Context | Resolution |
|-------|---------|------------|
| Agent always predicts 0.42 | Prompt example anchoring | Use `<placeholder>` not concrete values |
| Tent map degenerates to x=0 | mu=2 floating-point | Non-standard params + stability gate |
| NLL scoring 1.00 everywhere | Bins [0,1] for non-[0,1] systems | Use k_eff for PREDICT, exact match |
| Agent never calls MOVE_ON | Prompt/code protocol mismatch | Test with real LLM, not just unit tests |
| Scaffolding too easy | Claude one-shots with HYPOTHESIZE | Raw mode first, scaffolding as condition |
| "quasiperiodic" substring bug | "periodic" matched first | Sort labels longest-first |
| Confidence regex misses negatives | Regex pattern issue | Fixed regex |

---

## Test Results

```
pytest chaosbench/tests/ -v
254 passed (213 core + 41 arena)
```
