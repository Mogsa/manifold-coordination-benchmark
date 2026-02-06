# Findings: ChaosBench v2

## Lessons from v1 Prototyping (Jan 30 – Feb 3)

These findings from the old metacognitive agent approach inform the v2 design.

### LLM Anchoring on Prompt Examples
- Agent predicted 0.42 for every task because the prompt example used 0.42
- **Rule**: Never put concrete numeric examples where the model should reason. Use `<placeholder>` syntax.
- Applies to: solver prompts, question-specific instructions

### Tent Map Numerical Instability
- Tent map with μ=2 degenerates to fixed point x=0 (floating-point rounds 0.5→1.0→0.0→stuck)
- **Rule**: Hard gate on trajectory stability. Reject if orbit has < N unique values.
- PRD v2 handles this via validation pipeline Stage 1 (trajectory stability gate)

### NLL Bin Scoring Breaks for Non-[0,1] Systems
- Old scorer used 20 bins over [0,1]. Henon/Lorenz/standard map values fall entirely outside.
- All tasks scored 1.00 regardless of prediction quality.
- **Rule**: Scoring must be domain-aware. PRD v2 uses k_eff (effective prediction horizon) for PREDICT, exact match for CLASSIFY/IDENTIFY.

### Protocol Mismatch (Prompt vs Code)
- Prompt said "feedback after PREDICT", code gave feedback after MOVE_ON → agent never advanced
- **Rule**: Prompt and code must agree on protocol. Test with actual LLM, not just unit tests.

### Scaffolding Obscures the Core Signal
- HYPOTHESIZE/FIT tested tool use, not transfer learning
- **Rule**: Start with raw capability measurement. Scaffolding is an experimental condition, not a baseline.
- PRD v2 embodies this: v1 is raw mode, tools are deferred to v2.

### Mean Reversion Gaming
- Predicting attractor center gives low MSE by being "close to everything"
- **Rule**: Use scoring that punishes confident wrong answers.
- PRD v2 handles this: PREDICT uses k_eff (threshold-based, fails on any wrong step), CLASSIFY/IDENTIFY use exact match.

---

## Key Design Decisions (PRD v2)

### Why Multiple Question Types
A single PREDICT question tests one cognitive skill. Multiple types test whether understanding transfers across tasks:
- CLASSIFY on logistic → learn what "chaotic" looks like
- Does that help IDENTIFY a tent map? (same universality class)
- Does IDENTIFY help PREDICT? (knowing the family enables model-based prediction)

Cross-question transfer is a stronger signal than within-question improvement.

### Why Blocked vs Shuffled (Not Just Sequential)
- **Sequential only**: Can't distinguish "learning structure" from "context accumulation"
- **Blocked vs Shuffled**: Same problems, same context size. Only ordering differs.
- Blocked groups by family → structural learning possible
- Shuffled randomizes → only generic context effects
- **Gap = structural transfer** (the headline result)

### Why Non-Standard Parameters
- LLMs have likely seen logistic r=4.0 in training data
- Using r=3.831 forces genuine reasoning, not recall
- PRD specifies: "avoid textbook values (r=4.0, r=3.57, a=1.4)"

### Why Composite Difficulty (Not Just h_KS)
- h_KS = 0 for non-chaotic systems (damped_linear, rotation)
- Raw h_KS weighting would make these tasks weightless
- Composite: `(1 + h_KS) × (1 + depth) × (1 + 10σ)` ensures all tasks contribute

### Why Validation Pipeline
- Without validation, problem bank may contain trivially solvable or impossible tasks
- Stage 1 (hard gates): reject degenerate trajectories
- Stage 2 (baselines): reject if naive strategies solve too well OR if data has no exploitable structure
- Valid problems live between random floor and oracle ceiling

---

## Open Questions for v2

### What Reusable Code Exists?
The old `chaosbench/` directory has code from v1 prototyping. Before building Phase 1, audit what's salvageable:
- `core/Chaosbench_v3.py` has logistic/tent/henon/standard/lorenz implementations
- `core/models.py` has model factory
- `agents/metacognitive_agent.py` has LiteLLM wrapper
- `shared/llm_utils.py` has API utilities

### Module Layout: Reuse vs Rebuild?
PRD v2 specifies a new module structure (`grammar/`, `problems/`, `validation/`, `experiment/`, `scoring/`, `agents/`, `storage/`, `analysis/`). This differs from the existing `chaosbench/core/`, `chaosbench/agents/`, `chaosbench/experiments/` layout. Decision needed: refactor existing code into new structure, or start fresh?

### Tent Map Revisited
PRD v2 includes tent map as an MVP atom despite our earlier removal. The fix is non-standard params (μ ≠ 2.0) + stability validation, not removal. Need to verify tent map works for μ ∈ [1.0, 2.0) with proper numerical handling.

---

## References

- Master spec: `docs/Chaos_IMO`
- Old MVP spec: `docs/plans/2026-01-31-chaosbench-v1-minimal-design.md`
- Old math spec: `chaosbench/core/ChaosSpecification.md`
