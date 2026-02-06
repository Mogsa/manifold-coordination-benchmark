# Task Plan: ChaosBench v2 — Hybrid Build (Static Spine + Proposition Sandbox)

**Goal**: Build two things in parallel:
1. **Static spine**: atoms + verifier + scoring + frozen 12-18 task mini-bank (keeps NeurIPS path alive)
2. **Proposition sandbox**: Gemini proposes + solves problems to test IMO chaining thesis early

**Master Spec**: `docs/Chaos_IMO` (PRD v2)
**Branch**: `feature/chaosbench-v2`
**Testing model**: Gemini (via LiteLLM)

---

## Repo Layout (after refactor)

```
chaosbench/
├── legacy_v0/          # v1 metacognitive agent code (deprecated, preserved)
│
├── core/               # KEPT: math foundations
│   ├── Chaosbench_v3.py   # System classes (LogisticMap, TentMap, HenonMap, etc.)
│   ├── lyapunov.py        # Lyapunov computation (1D, spectrum, continuous, h_KS)
│   └── models.py          # Model factory
│
├── grammar/            # NEW: v2 atom/connective system
│   ├── atoms.py           # 4 MVP atoms with standard interface
│   ├── connectives.py     # Affine conjugacy (stub for depth-0 MVP)
│   ├── system.py          # DynamicalSystem: trajectory + metadata
│   └── registry.py        # Atom registry, parameter ranges, valid configs
│
├── problems/           # NEW: problem generation
│   ├── factory.py         # system_spec → Problem with ground truth
│   ├── bank.py            # Generate + freeze problem bank
│   └── verification.py    # CLASSIFY/IDENTIFY/PREDICT verify functions
│
├── validation/         # NEW: quality gates
│   ├── gates.py           # Stage 1: bounds, stability, periodicity, structure
│   └── baselines.py       # Stage 2: persistence, mean reversion, AR(5), etc.
│
├── scoring/            # NEW: v2 scoring
│   └── difficulty.py      # Composite difficulty + weighted score
│
├── agents/             # NEW: v2 agent interface
│   ├── protocol.py        # Agent protocol (solve(problem, history) → Solution)
│   ├── prompts.py         # System prompts + question instructions
│   ├── parsing.py         # Parse LLM output → structured answers
│   └── llm_agent.py       # Gemini/Claude/GPT wrapper via LiteLLM
│
├── experiment/         # NEW: experiment runner
│   ├── conditions.py      # Blocked/shuffled ordering generation
│   └── runner.py          # Run experiment loop
│
├── sandbox/            # NEW: proposition sandbox
│   ├── proposer.py        # LLM proposes system_spec + question
│   ├── validator.py       # Validate proposals (gates + baselines)
│   └── arena.py           # Propose → validate → solve → score loop
│
├── storage/            # NEW: persistence
│   └── database.py        # SQLite schema + read/write
│
├── analysis/           # NEW: results analysis
│   ├── phi_curve.py       # Phi(n) computation + exponent fitting
│   └── figures.py         # Plots
│
├── tests/              # Updated for v2
└── visualization/      # Existing (keep)
```

---

## Phases

### Phase 1: Mathematical Core (Atoms + Verification) `pending`
**No LLM, no API. Pure math + tests. Reuse lyapunov.py as-is.**

- [ ] `grammar/atoms.py` — 4 MVP atoms with unified interface:
  - `LogisticAtom(r)`: r in [2.5, 4.0], f(x) = rx(1-x), f'(x) = r(1-2x)
  - `TentAtom(mu)`: mu in [1.0, 2.0), f(x) = mu*min(x, 1-x), f'(x) = ±mu
  - `DampedLinearAtom(lam)`: lam in (0, 0.99), f(x) = lam*x, f'(x) = lam
  - `RotationAtom(omega)`: omega in (0, 1), f(x) = (x + omega) mod 1, f'(x) = 1
  - Each: `iterate(x)`, `derivative(x)`, `trajectory(x0, n)`, `lyapunov()`, `regime()`
  - Non-standard params only (no r=4.0, mu=2.0)
  - Reuse `lyapunov.py` for exponent computation
- [ ] `grammar/system.py` — `DynamicalSystem` wrapping an atom:
  - `generate_trajectory(x0, n_points, noise_std, seed)` → observations
  - `compute_metadata()` → {h_ks, lambda_max, tau_lambda, regime, family}
  - Observation model: trajectory + Gaussian noise
- [ ] `grammar/registry.py` — Registry of atoms + valid parameter ranges
- [ ] `grammar/connectives.py` — Stub (depth-0 only for MVP)
- [ ] `problems/verification.py` — 3 verify functions:
  - `verify_classify(answer, ground_truth)` → {0, 1} exact match
  - `verify_identify(answer, ground_truth)` → {0, 1} exact match
  - `verify_predict(predictions, actuals, attractor_diameter)` → k_eff/K
- [ ] `scoring/difficulty.py` — `composite_difficulty(h_ks, depth, noise_std)`
- [ ] Tests for every atom, every verify function

**Success gate**: All atoms generate stable trajectories with correct Lyapunov exponents. All verify functions score correctly on known inputs.

### Phase 2: Problem Bank + Validation `pending`
**Depends on Phase 1. Still no LLM.**

- [ ] `problems/factory.py` — `make_problem(family, params, question_type, ...)` → Problem
  - Problem dataclass: observations, question_type, question_params, ground_truth, metadata
- [ ] `validation/gates.py` — Stage 1 hard gates:
  - Parameter bounds, trajectory stability (no NaN/divergence)
  - Periodicity screen (reject periodic orbits in chaotic tasks)
  - Permutation entropy > threshold (data has structure)
- [ ] `validation/baselines.py` — Stage 2 baseline battery:
  - PREDICT: persistence, mean reversion, AR(5)
  - CLASSIFY: statistical moments heuristic
  - IDENTIFY: return map nearest-neighbor
  - Must pass: at least one baseline > random. Must fail: no baseline > ceiling.
- [ ] `problems/bank.py` — Generate mini-bank:
  - 4 families x 3 param settings x 3 question types = 36 candidates
  - Filter through gates + baselines → freeze 12-18 valid tasks
  - Serialize to JSON (SQLite deferred)

**Success gate**: 12-18 validated problems with known ground truth, baseline scores, oracle scores.

### Phase 3: Agent Interface + Solver `pending`
**First LLM calls. Gemini via LiteLLM.**

- [ ] `agents/protocol.py` — Agent interface:
  ```python
  class Agent(Protocol):
      def solve(self, problem: Problem, history: list[TaskResult]) -> Solution
  ```
- [ ] `agents/prompts.py` — Prompt templates from PRD §11:
  - System prompt (scientist analyzing dynamical system data)
  - Question-specific instructions (CLASSIFY, IDENTIFY, PREDICT)
  - Task history format (sequential conditions)
- [ ] `agents/parsing.py` — Parse LLM responses:
  - CLASSIFY: extract regime label
  - IDENTIFY: extract family name
  - PREDICT: extract comma-separated numbers
  - Regex fallback for messy outputs, score 0 on parse failure
- [ ] `agents/llm_agent.py` — LiteLLM wrapper (reuse `shared/llm_utils.py` patterns)

**Success gate**: Gemini answers all 3 question types with parseable responses. Score > random floor on at least one slice.

### Phase 4: Proposition Sandbox `pending`
**The exciting part. Gemini as both proposer and solver.**

- [ ] `sandbox/proposer.py` — Proposition interface:
  - Gemini receives: atom registry, parameter ranges, question types, grammar
  - Gemini outputs: system_spec (family + params) + question_type + "why this is hard"
  - Parse into structured Problem
- [ ] `sandbox/validator.py` — Validate proposals:
  - Run Stage 1 gates (is this a valid system?)
  - Run Stage 2 baselines (is this non-trivial?)
  - Compute difficulty score
- [ ] `sandbox/arena.py` — Propose-validate-solve loop:
  - Gemini proposes N problems
  - Validate each
  - Gemini (or another solver) attempts each
  - Report: proposal validity rate, difficulty distribution, solve rate
  - Log reasoning traces

**Success gate**: Gemini proposes valid problems (>50% pass validation). At least some proposals are harder than random parameter selection. Solve rate < 100% (problems aren't trivial).

### Phase 5: Static Experiment (Blocked vs Shuffled) `pending`
**The NeurIPS path. Uses mini-bank from Phase 2.**

- [ ] `experiment/conditions.py` — Generate orderings:
  - Blocked: group by family, sort by question type within
  - Shuffled: random permutation (same problems)
- [ ] `experiment/runner.py` — Full experiment loop:
  - For each condition: present problems in order, accumulate history
  - Score each response, compute weighted scores
  - 3 replicates per condition
- [ ] `analysis/phi_curve.py` — Phi(n) computation:
  - Cumulative weighted scores
  - Fit a*n^b via log-log regression
  - Bootstrap 95% CI on b
- [ ] `analysis/figures.py` — Plot Phi(n) curves with CIs

**Success gate**: Phi(n) curves are non-degenerate. Blocked vs shuffled produces different b values.

### Phase 6: Expand + Write `pending`
- [ ] Scale bank to 36+ tasks
- [ ] Add independent condition
- [ ] Add second model (if budget allows)
- [ ] Reasoning trace analysis
- [ ] Draft results section

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build order | Phases 1-2 (static spine) then 3-4 (sandbox) in parallel with Phase 5 | Gets to both research questions fast |
| v1 code | Preserved in `legacy_v0/`, not deleted | Fallback + test reference |
| Math reuse | `lyapunov.py` as-is, system classes refactored into atoms | Tested, correct code |
| Storage | JSON first, SQLite when needed | Simplicity until scale demands it |
| Model | Gemini via LiteLLM for all testing | Budget-friendly, Morgan's current access |
| Agent mode | Raw (no tools) for static benchmark | v1 lesson: scaffolding obscures signal |
| Proposition | Separate sandbox, not integrated into static benchmark | Different research question, different protocol |

---

## Relationship to PRD v2

| PRD Section | This Plan | Status |
|-------------|-----------|--------|
| §1.1 MVP Scope | Phase 1-2 (mini-bank, 12-18 tasks not 36) | Smaller first |
| §3 Grammar | Phase 1 (depth-0 atoms only) | MVP faithful |
| §4 Questions | Phase 1 (CLASSIFY, IDENTIFY, PREDICT) | MVP faithful |
| §5.1 Static | Phase 5 (blocked vs shuffled) | Deferred to after sandbox |
| §5.2 Tournament | Phase 4 (proposition sandbox, simplified) | Early exploration |
| §6 Validation | Phase 2 (gates + baselines) | MVP faithful |
| §7 Scoring | Phase 1 (difficulty) + Phase 5 (Phi curves) | Split across phases |
| §8 Agent | Phase 3 (raw mode) | MVP faithful |
| §10 Storage | JSON first, SQLite later | Simplified |
| §11 Prompts | Phase 3 | MVP faithful |

---

## v1 Lessons Applied

| Lesson | How Applied |
|--------|------------|
| Prompt anchoring (0.42 bug) | No concrete numeric examples in prompts |
| Tent map instability | Non-standard params + stability gate in validation |
| NLL scoring breaks | k_eff for PREDICT, exact match for CLASSIFY/IDENTIFY |
| Scaffolding too easy | Raw mode first (no HYPOTHESIZE/FIT tools) |
| Protocol mismatch | Test with real LLM early (Phase 3 before Phase 5) |
| Mean reversion gaming | k_eff threshold scoring (fails on wrong step) |
