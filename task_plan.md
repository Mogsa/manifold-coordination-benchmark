# Task Plan: ChaosBench v2 — Phase 4 (Adversarial Arena)

**Goal:** Build a NeurIPS-style adversarial loop where agents propose problems from the grammar, solve each other's problems blind, and peer-review both question quality and answer correctness. Track both consensus ground truth and mathematical ground truth to test whether peer review is calibrated.

**Branch:** `feature/chaosbench-v2`
**Module:** `chaosbench/arena/`
**Model:** Gemini 2.0 Flash (single model, multiple roles via prompt/temperature)

---

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| 0. Documentation | `in_progress` | This file + findings + progress |
| 1. Protocol (dataclasses) | `pending` | arena/protocol.py |
| 2. Prompts (3 roles) | `pending` | arena/prompts.py |
| 3. Parsing (proposal + review) | `pending` | arena/parsing.py |
| 4. Consensus + Reputation | `pending` | arena/consensus.py |
| 5. Runner (main loop) | `pending` | arena/runner.py |
| 6. Tests (mocked LLM) | `pending` | tests/test_arena.py |
| 7. Live run + analysis | `pending` | Run with Gemini, interpret results |

---

## The Four-Phase Round

```
Round N:
  1. PROPOSE     — Agent suggests problem spec + mock solution + "why it's hard"
  2. BLIND SOLVE — Other agents solve (proposer excluded), answer + short explanation
  3. PEER REVIEW — Rate question quality (Likert 1-6) + answer correctness (Likert 1-6)
                   + confidence (1-5). Compact context: just answers, no full traces.
  4. CONSENSUS   — Aggregate reviews -> accepted answers -> reputation update
                   Also compare against mathematical ground truth (meta-analysis)
```

---

## Architecture

### What Exists (Reuse as-is)

| Component | File | Interface |
|-----------|------|-----------|
| Problem factory | `problems/factory.py` | `create_problem(family, params, question_type, ...) -> Problem` |
| Validation gates | `problems/bank.py` | `validate_problem(problem) -> (bool, dict)` |
| Math verifiers | `problems/verification.py` | `verify_classify/identify/predict(answer, truth) -> float` |
| LLM caller | `shared/llm_utils.py` | `call_llm(model, messages, temperature, max_tokens) -> str` |
| Solver parsing | `agents/parsing.py` | `parse_response(response, question_type, params) -> Any` |
| Atom registry | `grammar/registry.py` | `ATOM_REGISTRY`, parameter ranges, `create_atom()` |
| Difficulty scoring | `scoring/difficulty.py` | `weighted_score(raw, difficulty) -> float` |

### What We Build (New)

| File | Purpose |
|------|---------|
| `arena/__init__.py` | Module init |
| `arena/protocol.py` | Dataclasses: Proposal, SolveResult, Review, Reputation, RoundResult |
| `arena/prompts.py` | Role-specific prompt formatters (proposer, solver, reviewer) |
| `arena/parsing.py` | Parse proposal JSON + review Likert ratings from LLM output |
| `arena/consensus.py` | Aggregate reviews, update reputation, compute discrimination |
| `arena/runner.py` | Main arena loop + `__main__` entry point |
| `tests/test_arena.py` | Mocked LLM tests for all components |

---

## Dataclass Design

### Proposal
```
proposer_id: str
family: str              # From ATOM_REGISTRY (logistic, tent, damped_linear, rotation)
params: dict             # e.g. {"r": 3.72}
question_type: str       # classify | identify | predict
reasoning: str           # Why the proposer thinks this is hard
mock_answer: str         # Proposer's own answer
mock_confidence: float   # 0-1
```

### SolveResult
```
solver_id: str
answer: Any              # str or list[float]
explanation: str         # Short reasoning (<=500 chars stored)
confidence: float
```

### Review
```
reviewer_id: str
question_quality: int    # 1-6 Likert
question_reasoning: str  # Why this rating
answer_ratings: dict     # solver_id -> 1-6 correctness Likert
confidence: int          # 1-5
```

### Reputation (three-axis)
```
agent_id: str
propose_score: float     # Running average: validation pass + discrimination
solve_score: float       # Running average: accuracy vs consensus + math
review_score: float      # Running average: calibration vs math ground truth
rounds_participated: int
```

### RoundResult
```
round_number: int
proposal: Proposal
problem: Problem | None  # None if validation failed
validation_passed: bool
validation_details: dict
solves: list[SolveResult]
reviews: list[Review]
consensus_answer: Any
math_ground_truth: Any
consensus_matches_math: bool
reputation_updates: dict[str, Reputation]
```

---

## Agent Differentiation (Gemini-only)

| Role | Temperature | Prompt Focus |
|------|-------------|-------------|
| Proposer | 0.9 | Creative: explore grammar space, argue for difficulty |
| Solver | 0.5 | Analytical: systematic reasoning about data |
| Reviewer | 0.3 | Calibrated: careful assessment, NeurIPS-style ratings |

Proposer exclusion enforced by runner (never solves own problem).

---

## Likert Scales

### Question Quality (1-6)
| Score | Label | Meaning |
|-------|-------|---------|
| 1 | Reject | Trivial, invalid, or nonsensical |
| 2 | Weak Reject | Too easy or poorly constructed |
| 3 | Borderline Reject | Some merit but significant issues |
| 4 | Borderline Accept | Reasonable problem with minor issues |
| 5 | Weak Accept | Good problem, tests genuine reasoning |
| 6 | Accept | Excellent, discriminating, well-constructed |

### Answer Correctness (1-6)
| Score | Label | Meaning |
|-------|-------|---------|
| 1 | Definitely Wrong | Clear error, nonsensical |
| 2 | Likely Wrong | Reasoning flawed, answer implausible |
| 3 | Uncertain-Wrong | Could go either way, lean wrong |
| 4 | Uncertain-Right | Could go either way, lean right |
| 5 | Likely Right | Sound reasoning, plausible answer |
| 6 | Definitely Right | Clearly correct with good justification |

### Reviewer Confidence (1-5)
| Score | Label |
|-------|-------|
| 1 | Uninformed guess |
| 2 | Low confidence |
| 3 | Moderate confidence |
| 4 | High confidence |
| 5 | Expert confidence |

---

## Grammar Available to Proposer

| Family | Parameter | Range | Domain | Regimes |
|--------|-----------|-------|--------|---------|
| logistic | r | [2.5, 4.0] | [0, 1] | fixed_point, periodic, chaotic |
| tent | mu | [1.0, 1.95] | [0, 1] | fixed_point, chaotic |
| damped_linear | lam | (0, 0.99) | [-10, 10] | fixed_point only |
| rotation | omega | (0, 1) | [0, 1) | periodic, quasiperiodic |

Question types: CLASSIFY, IDENTIFY, PREDICT

Observation params the proposer doesn't control (fixed): n_points=200, noise_std=0.01, stride=1, burn_in=500

---

## Research Questions

1. Can Gemini propose meaningful problems? Does it understand what makes a chaotic system hard vs easy?
2. Is peer review calibrated? Do Likert ratings correlate with mathematical ground truth?
3. Does the proposer learn? Over 10 rounds, do proposals get more discriminating?
4. Does the consensus mechanism work? When reviewers disagree, who's right?
5. What grammar regions are interesting? Which families/params produce discriminating problems?

---

## OPEN QUESTIONS (for Morgan)

- **Grammar selection:** How do we decide which grammar combinations work best? Options: empirical (let arena discover), theory-driven (h_KS/bifurcation distance), agent-driven (reputation feedback), Morgan picks. The arena itself generates empirical data for this.
- **Reviewer sees observations?** Current plan: reviewer sees problem metadata + solver answers only (not the raw data). Alternative: show data so reviewer can re-reason. Trade-off: showing data makes review more expensive but more informed.
- **Multi-model later?** Current plan is Gemini-only. When budget allows, adding Claude/GPT as different agents would test whether model diversity improves discrimination.

---

## Success Gates

1. Proposer generates valid problems (>40% pass automated validation)
2. At least one CLASSIFY and one PREDICT problem proposed across 10 rounds
3. Reviewer ratings show non-trivial variance (not all 3s or all 6s)
4. Consensus matches math ground truth on >50% of accepted problems
5. Reputation scores diverge (not all agents rated equally)

---

## Verification Commands

```bash
# Unit tests (no API calls)
source venv/bin/activate
pytest chaosbench/tests/test_arena.py -v

# Existing tests still pass
pytest chaosbench/tests/ -v

# Live arena run (10 rounds, ~$0.30-0.50)
python -m chaosbench.arena.runner

# Quick 3-round test
python -m chaosbench.arena.runner --rounds 3
```

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
