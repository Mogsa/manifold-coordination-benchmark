# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to Work With Morgan

**Morgan is a researcher, not a coder.** Morgan makes research decisions, predicts agent behavior, interprets results, explains findings for dissertation. Claude writes code, runs experiments, builds infrastructure.

**Rules:**
- Don't just execute — engage. Ask research questions before major work.
- Surface assumptions. Explain WHY you're making architectural choices.
- Before implementing: "What are the inputs/outputs? What could go wrong?"
- After implementing: "Predict what happens. Now what if we change X?"
- Morgan owns the LOGIC, Claude handles the SYNTAX.

---

## Project: ChaosBench v2

Benchmark testing whether LLMs can reason about chaotic dynamical systems. Generates problems from a grammar of dynamical systems, validated through quality gates, difficulty-scored. Three types: CLASSIFY, IDENTIFY, PREDICT.

**Current state:** Phase 4 done. 7 atom families + AffineConjugacy. 316 tests, 84 raw problems. Phase 3 baseline (4 families): Gemini 36% CLASSIFY, 55% IDENTIFY, 79% PREDICT. Expanded bank needs re-freeze + re-run. Master spec: `docs/Chaos_IMO`.

**Archived benchmarks in `archive/` — do not touch unless Morgan asks.**

## Commands

```bash
source venv/bin/activate
pytest chaosbench/tests/ -v --ignore=chaosbench/tests/test_benchmark_api.py  # Full suite (316 tests)
pytest chaosbench/tests/test_atoms.py -v        # Single test file
pytest chaosbench/tests/ -k "test_logistic" -v  # Single test by name
python -m chaosbench.experiment.runner           # Run Gemini on frozen bank (~$0.30)
python -m chaosbench.arena.runner --rounds 3    # Phase 4 arena (when implemented)
python --version                                 # 3.13.7
```

No build step needed. No linter configured (use `ruff` if adding one). Dependencies in `requirements.txt`, installed in local `venv/`.

## Architecture

The pipeline flows left-to-right. No circular dependencies.

```
core/lyapunov → grammar/atoms → grammar/registry → grammar/system
                                                          ↓
                                            problems/factory → problems/bank
                                                  ↓                ↓
                                          scoring/difficulty   validation/{gates, baselines}
                                                  ↓
                                         problems/verification
                                                  ↓
                              agents/{protocol, prompts, parsing, llm_agent}
                                                  ↓
                                         experiment/runner
```

### Grammar Layer (`grammar/`)
- **Atoms** (`atoms.py`): 7 concrete `Atom` subclasses — `LogisticAtom`, `TentAtom`, `DampedLinearAtom`, `RotationAtom`, `SineAtom`, `CircleAtom`, `HenonAtom`. Each provides `iterate()`, `derivative()`, `lyapunov()`, `h_ks()`, `regime()`, `trajectory()`, `prepare()`.
- **Registry** (`registry.py`): `ATOM_REGISTRY` maps family name → `AtomSpec(cls, param_names, param_ranges)`. `MINI_BANK_PARAMS` holds 3 non-textbook parameter sets per family (anti-memorization). `CONJ_BANK_PARAMS` holds affine conjugacy settings. `create_atom(family, params)` is the factory.
- **System** (`system.py`): `DynamicalSystem` wraps an atom with `observe()` (reproducible noisy observations with burn-in/stride/seed, calls `atom.prepare()` for stateful atoms), `metadata()`, `future_trajectory()`, `prediction_horizon()`.
- **Connectives** (`connectives.py`): `AffineConjugacy` wraps any atom via `y = a·f((y-b)/a) + b`. Preserves dynamics (Lyapunov, regime, h_KS), transforms domain. Subclasses `Atom` so it's transparent to the rest of the pipeline.

### Problem Layer (`problems/`)
- **Factory** (`factory.py`): `create_problem(family, params, question_type, ..., conjugacy=None)` → `Problem` with deterministic ID, observations, `GroundTruth`, `SystemMetadata`, and `difficulty`. Optional `conjugacy={"a": float, "b": float}` wraps atom in `AffineConjugacy` at grammar_depth=1. `QuestionType` enum: CLASSIFY, IDENTIFY, PREDICT.
- **Bank** (`bank.py`): `generate_mini_bank()` → 84 raw problems (7 families × 3 params × 3 types depth-0 + 7 families × 1 param × 3 types depth-1 conjugated). `validate_problem()` runs 2-stage validation (gates then baselines). `freeze_bank()`/`load_bank()` serialize to `data/mini_bank.json`.
- **Verification** (`verification.py`): `verify_classify/identify()` → exact match (0 or 1). `verify_predict()` → `k_eff/K` (how many future steps were accurate within epsilon).

### Validation Layer (`validation/`)
- **Gates** (`gates.py`): Stage 1 hard filters — parameter bounds, trajectory stability, attractor bounds, plus chaotic-only: minimum Lyapunov (λ>0.01), periodicity check (period≤1000), autocorrelation (|ACF(1)|<0.95), permutation entropy (PE>0.5).
- **Baselines** (`baselines.py`): Stage 2 — persistence, mean-reversion, AR(5) for PREDICT; heuristic classifiers for CLASSIFY/IDENTIFY (informational, not rejection).

### Agent Layer (`agents/`)
- **Protocol** (`protocol.py`): `Agent` protocol (requires `agent_id` + `solve(problem, task_history) → Solution`). `Solution` and `TaskResult` dataclasses.
- **Prompts** (`prompts.py`): `format_system_prompt()` (no numeric examples — avoids anchoring). `format_problem_prompt()` includes 200 observations as CSV, domain, noise, type-specific question.
- **Parsing** (`parsing.py`): Robust extraction — never crashes on bad LLM output, returns default and scores 0. Classify/identify: check `ANSWER:` line first, fallback to last occurrence (longest-first label matching to avoid "periodic" eating "quasiperiodic"). Predict: tries `ANSWER:` line, bracketed list, then all floats.
- **LLMAgent** (`llm_agent.py`): Stateless wrapper around `shared/llm_utils.call_llm()`. Default: `gemini/gemini-2.0-flash`, temperature 0.7.

### Scoring (`scoring/difficulty.py`)
- `composite_difficulty(h_ks, depth, noise) = (1 + h_ks) × (1 + depth) × (1 + 10×noise)`
- `weighted_score(raw, difficulty) = raw × difficulty`

### Shared (`shared/`)
- `llm_utils.py`: `call_llm()` — LiteLLM wrapper with exponential backoff retry. Supports any provider via model string prefix (e.g., `gemini/...`, `anthropic/...`).
- API keys loaded from `.env` (GEMINI_API_KEY, etc.)

## Key Design Patterns

| Pattern | Detail |
|---------|--------|
| Anti-memorization | `MINI_BANK_PARAMS` uses non-textbook values (r=3.891 not r=4.0) + `AffineConjugacy` shifts domains |
| No anchoring | Prompts contain zero numeric examples |
| Fail-safe parsing | `agents/parsing.py` never crashes; bad output → score 0 |
| Deterministic IDs | Problem IDs are hashes of (family, params, type, seeds, config) |
| Layered validation | Stage 1 gates (hard filters) then Stage 2 baselines (difficulty checks) |
| Stateless agent | `LLMAgent.solve()` takes problem + history, no internal state |

## Testing

Tests in `chaosbench/tests/`. Class-based organization, no shared fixtures beyond `conftest.py` (which just sets up `sys.path` and registers `@pytest.mark.integration`). No mocking in core tests — components tested in isolation with real logic. The experiment runner (`python -m chaosbench.experiment.runner`) serves as the integration test. `pytest.ini` excludes `legacy_v0/` from test discovery.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Atoms | logistic, tent, damped_linear, rotation, sine, circle, henon |
| Connectives | AffineConjugacy (depth-1, domain shift) |
| Types | CLASSIFY, IDENTIFY, PREDICT |
| Validation | Quality gates + baseline checks |
| Scoring | Weighted accuracy by difficulty |
| Agent | LLM via LiteLLM (Gemini) |

## Workflow

**Active planning (in `docs/plans/`):**
- `chaosbench-v4-task-plan.md` — THE to-do list (current phase + roadmap)
- `chaosbench-v4-progress.md` — Session log

**Research docs (in `docs/research/`):**
- `chaosbench-neurips-vision.md` — NeurIPS vision / north star (beyond MVP)
- `chaosbench-v4-findings.md` — Experimental results + design rationale
- `epiplexity-integration.md` — Epiplexity paper analysis + theoretical grounding

**Hooks enforce planning:** PostToolUse reminds to update task plan after edits. Stop checks completion progress.

**For new features/phases:**
1. **Brainstorm** (`superpowers:brainstorming`) — Socratic design with Morgan → save to `docs/plans/`
2. **Plan** — Update `docs/plans/chaosbench-v4-task-plan.md` with checklist (`- [ ]` / `- [x]`)
3. **Implement** (`superpowers:subagent-driven-development` for multi-task) — check boxes as done, log to findings + progress, run tests after each task
4. **Verify** (`superpowers:verification-before-completion`) — all tests pass, all boxes checked

**Rules:** 2-Action Rule (save findings every 2 operations). 3-Strike Rule (escalate after 3 failures).

**Session recovery:** Read task-plan → findings → progress → resume from first unchecked task.
