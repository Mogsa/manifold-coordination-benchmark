# ChaosBench v4 Task Plan (Single Source of Truth)

**Last Updated:** 2026-02-07 (expanded atoms + conjugacy)
**Purpose:** This is the only live planning file for what is done, what is next, and what changed direction.

---

## Rules (Non-Negotiable)

1. Every research claim in this file must be tagged as one of:
   - `[BASELINE]` = implemented and reproducible now.
   - `[PLANNED]` = not implemented yet.
2. Every claim must have a stable `CLAIM_ID`.
3. Every `[BASELINE]` claim must include evidence and a verification command.
4. Every `[PLANNED]` claim must include an exit criterion.
5. Do not duplicate status in other plan files; this file is authoritative.

---

## Current Snapshot

- Tests: **316 passing** (`pytest --ignore=chaosbench/tests/test_benchmark_api.py chaosbench/tests/ -q`)
- Atom families: **7** (logistic, tent, damped_linear, rotation, sine, circle, henon)
- Bank: **84 raw problems** (7×3×3 depth-0 + 7×3 depth-1 conjugated)
- Connectives: **AffineConjugacy** (depth-1, preserves dynamics, shifts domain)
- Implemented benchmark question types: **CLASSIFY, IDENTIFY, PREDICT**
- Arena status: **implemented and test-backed** (single-file source under `chaosbench/arena/`)

---

## Phase Board (Done vs To Do)

- [x] **Phase 1-2:** Core grammar, problem factory, validation, scoring
- [x] **Phase 3:** Agent protocol + baseline run infrastructure
- [x] **Phase 4 (core):** Arena loop (propose/solve/review/consensus) + tests
- [ ] **Phase 5:** Static transfer experiment (blocked vs shuffled vs independent)
- [ ] **Phase 6:** Expansion + dissertation analysis pipeline

---

## Continuous To-Do List

### Done

- [x] Build 7 atom families (`logistic`, `tent`, `damped_linear`, `rotation`, `sine`, `circle`, `henon`)
- [x] Implement deterministic problem generation with seeds and metadata
- [x] Implement Stage 1 gates + Stage 2 baseline validation
- [x] Implement verifiers for classify/identify/predict
- [x] Freeze mini-bank and run baseline evaluations
- [x] Implement arena protocol/prompt/parsing/consensus/runner modules
- [x] Add arena test suite and maintain full-suite green
- [x] Add 3 new atom families: `sine` (confuser for logistic), `circle` (3-regime, 2-param), `henon` (2D→1D projection)
- [x] Implement AffineConjugacy connective (depth-1, anti-memorization)
- [x] Expand bank generation to 84 raw problems (depth-0 + depth-1 conjugated)
- [x] Update IDENTIFY prompt to list all 7 families

### Next (Highest Priority)

- [ ] Freeze expanded bank and run Gemini baseline comparison (expect IDENTIFY drop)
- [ ] Implement blocked/shuffled/independent condition generator (`experiment/conditions.py`)
- [ ] Implement replicated experiment runner with condition parity
- [ ] Add Phi(n) curve fitting + confidence intervals (`analysis/phi_curve.py`)
- [ ] Add publication-ready figures (`analysis/figures.py`)
- [ ] Add second model family for cross-vendor transfer comparison

### Later (After Phase 5)
- [ ] Expand beyond 3 question types (ESTIMATE, RECONSTRUCT, RELATE)
- [ ] Add tool-augmented solver mode (§8.4 in PRD)
- [ ] Add persistent problem backlog and archive policy in arena
- [ ] Add SQLite-backed experiment storage and reproducibility logging
- [ ] Review and refine NeurIPS vision document after Phase 5 results

---

## Claim Registry — Implemented Baseline (Truth Today)

| CLAIM_ID | State | Claim | Evidence | Verify |
|---|---|---|---|---|
| CB-BASE-001 | `[BASELINE]` | Seven atom families are implemented and parameter-bounded (logistic, tent, damped_linear, rotation, sine, circle, henon). | `chaosbench/grammar/atoms.py`, `chaosbench/grammar/registry.py` | `pytest chaosbench/tests/test_atoms.py -q` |
| CB-BASE-002 | `[BASELINE]` | Problem generation supports three question types only: classify/identify/predict. | `chaosbench/problems/factory.py` | `pytest chaosbench/tests/test_factory.py -q` |
| CB-BASE-003 | `[BASELINE]` | Validation pipeline is two-stage; strict gating is enforced for PREDICT, informational for CLASSIFY/IDENTIFY. | `chaosbench/problems/bank.py`, `chaosbench/validation/*` | `pytest chaosbench/tests/test_gates.py chaosbench/tests/test_baselines.py -q` |
| CB-BASE-004 | `[BASELINE]` | Frozen mini-bank currently contains 27 validated problems. | `chaosbench/data/mini_bank.json` | `python - <<'PY'\nimport json\nprint(json.load(open('chaosbench/data/mini_bank.json'))['n_problems'])\nPY` |
| CB-BASE-005 | `[BASELINE]` | Static agent mode is raw prompt solving (no tool calls). | `chaosbench/agents/llm_agent.py`, `chaosbench/agents/prompts.py` | `pytest chaosbench/tests/test_parsing.py chaosbench/tests/test_runner.py -q` |
| CB-BASE-006 | `[BASELINE]` | Arena round loop exists and runs propose → solve → review → consensus. | `chaosbench/arena/runner.py` | `pytest chaosbench/tests/test_arena.py -q` |
| CB-BASE-007 | `[BASELINE]` | Arena parsing is fail-safe for malformed LLM outputs. | `chaosbench/arena/parsing.py` | `pytest chaosbench/tests/test_arena.py::TestArenaParsing -q` |
| CB-BASE-008 | `[BASELINE]` | Arena tracks consensus answer and compares against mathematical ground truth. | `chaosbench/arena/runner.py`, `chaosbench/arena/consensus.py` | `pytest chaosbench/tests/test_arena.py::TestConsensus -q` |
| CB-BASE-009 | `[BASELINE]` | Full test suite currently passes end-to-end (316 tests). | Local test run on 2026-02-07 | `pytest --ignore=chaosbench/tests/test_benchmark_api.py chaosbench/tests/ -q` |
| CB-BASE-011 | `[BASELINE]` | AffineConjugacy (depth-1) connective preserves dynamics and transforms domain. | `chaosbench/grammar/connectives.py`, `chaosbench/tests/test_connectives.py` | `pytest chaosbench/tests/test_connectives.py -q` |
| CB-BASE-012 | `[BASELINE]` | Bank generation includes depth-0 (7×3×3=63) and depth-1 conjugated (7×3=21) = 84 raw problems. | `chaosbench/problems/bank.py` | `pytest chaosbench/tests/test_bank.py::TestGenerateMiniBank::test_count -q` |
| CB-BASE-010 | `[BASELINE]` | Composite difficulty formula is active: `(1+h_KS)(1+depth)(1+10σ)`; in current generation depth is effectively 0. | `chaosbench/scoring/difficulty.py`, `chaosbench/problems/factory.py` | `pytest chaosbench/tests/test_difficulty.py -q` |

---

## Claim Registry — Planned Extensions (Explicitly Not Implemented Yet)

| CLAIM_ID | State | Planned Extension | Exit Criterion |
|---|---|---|---|
| CB-PLAN-001 | `[PLANNED]` | Blocked/shuffled/independent condition engine for transfer experiments. | `experiment/conditions.py` exists + tests proving identical bank parity across conditions. |
| CB-PLAN-002 | `[PLANNED]` | Multi-replicate experiment runner and summary outputs. | Replicate loop implemented; reproducible seeds and aggregate outputs persisted. |
| CB-PLAN-003 | `[PLANNED]` | Phi(n) curve fitting with CI and blocked-vs-shuffled gap report. | `analysis/phi_curve.py` + bootstrap CI + regression tests. |
| CB-PLAN-004 | ~~`[PLANNED]`~~ **DONE** → CB-BASE-011 | Affine conjugacy (depth-1) connective integrated into generation/verification. | Implemented 2026-02-07. |
| CB-PLAN-005 | `[PLANNED]` | Expand question types beyond current 3 (starting with ESTIMATE). | End-to-end generation, verification, and tests for each new type. |
| CB-PLAN-006 | `[PLANNED]` | Tool-augmented mode per PRD §8.4 (minimal tool set). | Tool API + usage logging + comparative experiments vs raw mode. |
| CB-PLAN-007 | `[PLANNED]` | Persistent arena problem backlog and archive policy. | Discriminating problems persisted and reused in future rounds. |
| CB-PLAN-008 | `[PLANNED]` | SQLite storage + reproducibility metadata capture. | `storage/database.py` + schema + run metadata persisted automatically. |
| CB-PLAN-009 | `[PLANNED]` | Multi-model roster for cross-vendor transfer comparison. | At least 2 model families run through identical condition/replicate pipeline. |

---

## Design Choice Log

### What Worked (Keep)

| DECISION_ID | Choice | Outcome | Evidence |
|---|---|---|---|
| DC-OK-001 | Non-textbook parameters to reduce memorization | Effective; generated meaningful variation while staying valid | `docs/research/chaosbench-v4-findings.md` |
| DC-OK-002 | k_eff-style prediction scoring | Prevented mean-reversion gaming; cleaner than MSE/NLL in this setup | `docs/research/chaosbench-v4-findings.md` |
| DC-OK-003 | Fail-safe parser strategy (“never crash”) | Improved robustness in both agent and arena loops | `chaosbench/agents/parsing.py`, `chaosbench/arena/parsing.py` |
| DC-OK-004 | Separate arena from static benchmark track | Preserved ability to answer two different research questions | `docs/Chaos_IMO` §5.1 vs §5.2 |

### What Did Not Work (Retire or Constrain)

| DECISION_ID | Choice | Why It Failed | Action |
|---|---|---|---|
| DC-NO-001 | Concrete numeric examples in prompts | Strong anchoring behavior (e.g., repeated 0.42 outputs) | Keep placeholder-only examples |
| DC-NO-002 | Tent parameter at mu=2.0 | Numerical degeneration to fixed point under finite precision | Keep hard cap mu<=1.95 |
| DC-NO-003 | NLL bins fixed to [0,1] for all systems | Invalid for non-[0,1] domains; scores collapsed | Use current verifier stack |
| DC-NO-004 | Assuming protocol consistency without live LLM checks | Prompt/code mismatch went undetected in earlier iteration | Require live smoke tests for protocol changes |

---

## Alignment Notes (Reality vs Vision)

- `[BASELINE]` Current implementation is a strong MVP benchmark spine with a functioning arena.
- `[PLANNED]` PRD-level goals requiring depth/connectives, full transfer-condition experiments, tool-augmentation, and persistent backlog are not yet complete.
- `[BASELINE]` Dissertation claims must currently be framed around implemented components unless explicitly marked as future work.

---

## Update Procedure (Each Session)

1. Update `Last Updated` date.
2. Move completed items from “Next/Later” to “Done”.
3. Add/adjust `CLAIM_ID` entries with correct state tags.
4. If a planned claim becomes implemented, move it from planned table to baseline table and add evidence + verify command.
5. Append any design reversal to “What Did Not Work”.

