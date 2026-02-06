# Task Plan: ChaosBench v2 — Scientific Reasoning Benchmark

**Goal**: Implement a benchmark measuring whether LLMs discover transferable structure across dynamical systems tasks, using multiple question types and blocked-vs-shuffled experimental design.

**Master Spec**: `docs/Chaos_IMO` (PRD v2)

**Core Hypothesis**: Superlinear Φ(n) in blocked condition vs linear in shuffled = structural transfer.

**Headline Statistical Test**: Blocked vs Shuffled exponent b comparison (p < 0.05).

---

## MVP Scope (Section 1.1 of PRD)

| Constraint | MVP | Full v1 |
|------------|-----|---------|
| Atoms | logistic, tent, damped_linear, rotation | + henon, sine, circle |
| Grammar depth | 0 only | 0 + 1 (affine conjugacy) |
| Question types | CLASSIFY, IDENTIFY, PREDICT | + ESTIMATE, RECONSTRUCT, RELATE |
| Bank size | 36 (4×3×3) | 60-100 |
| Conditions | blocked + shuffled | + independent |
| Replicates | 3 per model-condition | 5 |

**Success gate**: End-to-end pipeline with reproducible Φ(n) curves; blocked vs shuffled non-degenerate; at least one model beats best naive baseline on one family/question slice.

---

## Phases

### Phase 1: Mathematical Core `pending`
**No LLM, no API. Pure math + tests.**

- [ ] `grammar/atoms.py` — 4 MVP atoms: logistic, tent, damped_linear, rotation
  - Each: `iterate()`, `lyapunov()`, `derivative()`
  - Non-standard params mandatory (no textbook r=4.0, μ=2.0)
  - Hard gates: λ_max > 0.01 for chaotic, periodicity screen, bound check
- [ ] `grammar/connectives.py` — Affine conjugacy (stub for MVP, depth-0 only)
- [ ] `grammar/system.py` — DynamicalSystem: compose grammar, generate trajectory, compute metadata (h_KS, λ_max, τ_λ, regime)
- [ ] `grammar/registry.py` — Atom/connective registry, parameter ranges
- [ ] `problems/verification.py` — 3 MVP verify functions: CLASSIFY (exact match), IDENTIFY (exact match), PREDICT (k_eff threshold)
- [ ] `scoring/difficulty.py` — Composite difficulty: `(1 + h_KS) × (1 + depth) × (1 + 10σ)`
- [ ] Unit tests for every atom, every verify function

### Phase 2: Problem Bank & Validation `pending`
- [ ] `problems/factory.py` — system_spec → Problem with ground truth
- [ ] `problems/bank.py` — Generate 36-task bank (4 families × 3 params × 3 questions), freeze
- [ ] `validation/gates.py` — Stage 1: param bounds, trajectory stability, NaN/divergence, periodicity
- [ ] `validation/baselines.py` — Stage 2: persistence, mean reversion, AR(5), statistical moments, return map NN
- [ ] `validation/oracle.py` — Oracle solver + random floor computation
- [ ] `storage/database.py` — SQLite schema (experiments, problems, orderings, submissions, phi_curves)
- [ ] `storage/reproducibility.py` — Seed management, commit hash, metadata

### Phase 3: Experiment Infrastructure `pending`
- [ ] `experiment/conditions.py` — Blocked/shuffled ordering generation
- [ ] `agents/protocol.py` — Agent protocol (solve(problem, history) → Solution)
- [ ] `agents/prompts.py` — System prompt + question-specific instructions + task history format
- [ ] `agents/parsing.py` — Parse LLM output into structured answers (with regex fallback)
- [ ] `agents/llm_agent.py` — API wrappers (Anthropic/OpenAI/Google) with retry, rate limiting, cost tracking
- [ ] `experiment/runner.py` — Full experiment loop (mock mode first, then live)
- [ ] `experiment/replication.py` — Multi-replicate wrapper

### Phase 4: First Live Run `pending`
- [ ] Run with 2 models, 1 replicate, ~20 problems (subset)
- [ ] Validate: Φ(n) curves computable, scoring non-degenerate, parse rates acceptable
- [ ] Debug, iterate, fix edge cases
- [ ] Verify: oracle > agent > baseline > random (proper separation)

### Phase 5: Full Experiment `pending`
- [ ] Full 36-task bank
- [ ] 2+ models × 2 conditions × 3 replicates
- [ ] Store all results in SQLite
- [ ] Add independent condition if pipeline stable

### Phase 6: Analysis & Writing `pending`
- [ ] `scoring/phi_curve.py` — Φ(n) computation, exponent fitting (a·n^b), bootstrap CIs
- [ ] `analysis/transfer.py` — Blocked vs shuffled statistical comparison
- [ ] `analysis/figures.py` — Publication-quality Φ(n) plots with CIs
- [ ] Reasoning trace analysis
- [ ] Draft results section

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent interface | Raw mode (no tools, no DSL) | Performance = reasoning capability, not engineering |
| Scoring | Per-question verify → raw_score × difficulty | Composite difficulty handles non-chaotic systems (h_KS=0) |
| Transfer detection | Φ(n) = a·n^b, report b with 95% CI | Superlinear b > 1.05 = transfer |
| Anti-gaming | Non-standard params + noise + validation pipeline | Memorization fails on novel params |
| Storage | SQLite | Reproducible, queryable, single file |
| Conditions | Blocked vs Shuffled (headline), Independent (secondary) | Blocked-Shuffled gap = structural transfer |

---

## Relationship to Previous Work

The old metacognitive agent approach (Phases 1-8 in old plan) explored:
- Single-task prediction (PREDICT only)
- NLL bin scoring for [0,1] systems
- HYPOTHESIZE/FIT scaffolding

**Lessons carried forward:**
- LLMs anchor on example values in prompts → use placeholders
- Tent map μ=2 degenerates → use non-standard params + stability validation
- NLL bin scoring breaks for non-[0,1] systems → use k_eff + exact match instead
- Auto-advance after action → simpler protocol

**What changes:**
- Multiple question types (not just PREDICT)
- Grammar system (not just raw systems)
- Validation pipeline (baselines prove problems aren't trivial)
- Experimental conditions (blocked/shuffled/independent)
- SQLite storage (not JSON files)

---

## Design Docs

| Doc | Purpose |
|-----|---------|
| **`docs/Chaos_IMO`** | Master PRD v2 specification |
| `docs/plans/chaosbench-v4-findings.md` | Research decisions and lessons learned |
| `docs/plans/chaosbench-v4-progress.md` | Session-by-session log |
| `docs/plans/2026-01-31-chaosbench-v1-minimal-design.md` | Old MVP spec (historical reference) |
| `chaosbench/core/ChaosSpecification.md` | Old math spec (partial overlap with PRD) |
