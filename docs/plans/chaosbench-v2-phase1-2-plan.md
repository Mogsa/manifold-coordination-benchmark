# Plan: ChaosBench v2 MVP — Phase 1 (Math Core) + Phase 2 (Problem Bank)

## Context

Morgan's dissertation benchmark measures whether LLMs can reason about dynamical systems. The PRD v2 (`docs/Chaos_IMO`) specifies a hybrid build: a static benchmark spine (for NeurIPS transfer experiments) and a proposition sandbox (for testing IMO-style problem generation).

This plan covers **Phases 1-2**: the mathematical core and problem bank that both tracks depend on. No LLM calls — pure math, tests, and validation. This is the foundation everything else builds on.

**Branch**: `feature/chaosbench-v2`
**What exists**: `core/lyapunov.py` (reuse as-is), `core/Chaosbench_v3.py` (reference patterns), v1 code in `legacy_v0/`, empty v2 module directories.

---

## Files to Create/Modify (in order)

### 1. `chaosbench/grammar/atoms.py` — 4 MVP Atoms

**Reuse**: `core/lyapunov.py` → `compute_lyapunov_1d()` for Lyapunov computation.
**Reference**: `core/Chaosbench_v3.py` → LogisticMap/TentMap patterns (but don't import from it).

`Atom` ABC with: `family`, `params`, `domain` (properties), `iterate(x)`, `derivative(x)`, `trajectory(x0, n)`, `lyapunov()`, `h_ks()`, `regime()`, `name`.

| Class | Formula | Domain | Params | Regime Logic |
|-------|---------|--------|--------|-------------|
| `LogisticAtom(r)` | rx(1-x) | [0,1] | r∈[2.5,4.0] | r<3→fixed_point, r<3.57→periodic, else compute λ: >0.01→chaotic else periodic |
| `TentAtom(mu)` | mu·min(x,1-x) | [0,1] | mu∈[1.0,1.95] | mu≤1→fixed_point, mu>1→chaotic. λ=ln(mu) exact. **Hard cap mu≤1.95** (v1 bug: mu=2.0 degenerates) |
| `DampedLinearAtom(lam)` | lam·x | [-10,10] | lam∈(0,0.99) | Always fixed_point. λ=ln(lam) exact. h_ks=0. |
| `RotationAtom(omega)` | (x+ω) mod 1 | [0,1) | ω∈(0,1) | Rational ω→periodic, irrational→quasiperiodic. λ=0 exact. h_ks=0. Use `Fraction.limit_denominator(1000)` for rationality test. |

Key: `trajectory()` on base class (loop calling `iterate()`). `lyapunov()` calls `compute_lyapunov_1d` by default; TentAtom, DampedLinearAtom, RotationAtom override with analytical values. Parameter validation in `__init__`.

### 2. `chaosbench/grammar/registry.py` — Atom Registry + Mini-Bank Params

`ATOM_REGISTRY: Dict[str, AtomSpec]` mapping family→(cls, param_names, param_ranges).
`create_atom(family, params) → Atom` factory.
`list_families() → List[str]`.

**Mini-bank parameter selections** (non-standard, verified chaotic where needed):

```python
MINI_BANK_PARAMS = {
    "logistic": [
        {"r": 2.73},    # fixed_point (r < 3)
        {"r": 3.236},   # periodic (r ∈ [3, 3.57))
        {"r": 3.891},   # chaotic (λ≈0.49, verified NOT in periodic window)
    ],
    "tent": [
        {"mu": 1.237},  # chaotic, λ=0.21
        {"mu": 1.574},  # chaotic, λ=0.45
        {"mu": 1.891},  # chaotic, λ=0.64
    ],
    "damped_linear": [
        {"lam": 0.312},  # fast decay
        {"lam": 0.673},  # medium decay
        {"lam": 0.941},  # slow decay (near marginal)
    ],
    "rotation": [
        {"omega": 0.25},          # periodic (period 4)
        {"omega": 0.381966011},   # quasiperiodic (golden-ratio related)
        {"omega": 0.723},         # quasiperiodic
    ],
}
```

**Critical**: r=3.831 falls in the period-3 window (λ=-0.37). Verified r=3.891 is genuinely chaotic (λ≈0.49).

### 3. `chaosbench/grammar/system.py` — DynamicalSystem Wrapper

`SystemMetadata` dataclass: family, params, regime, h_ks, lambda_max, tau_lambda, dim, grammar_depth, domain.
`Observation` dataclass: data (noisy), clean_data, n_points, noise_std, stride, x0, seeds.

`DynamicalSystem(atom, grammar_depth=0)`:
- `metadata()` → SystemMetadata (cached)
- `observe(n_points=200, noise_std=0.01, stride=1, ic_seed, noise_seed, burn_in=500)` → Observation
- `future_trajectory(observation, horizon)` → np.ndarray (continues from last clean value)
- `prediction_horizon()` → int (ceil(5·τ_λ) for chaotic, 20 for non-chaotic)

Observation model: generate clean trajectory with burn-in, add Gaussian noise. Seeds for reproducibility.

### 4. `chaosbench/grammar/connectives.py` — Stub

`AffineConjugacy.__init__` raises `NotImplementedError`. Placeholder for post-MVP depth-1.

### 5. `chaosbench/problems/verification.py` — 3 Verify Functions

```python
verify_classify(agent_answer: str, ground_truth: str) → float  # 0 or 1, case-insensitive exact match
verify_identify(agent_answer: str, ground_truth: str) → float   # 0 or 1, case-insensitive exact match
verify_predict(predictions, true_future, training_data) → dict  # {k_eff, K, raw_score, epsilon}
```

PREDICT: ε = 0.1 × attractor_diameter. k_eff = largest k where ALL |pred-true| < ε for i≤k. raw_score = k_eff/K. Epsilon floor at 0.01 for convergent systems. Pad short predictions with NaN.

### 6. `chaosbench/scoring/difficulty.py` — Composite Difficulty

```python
composite_difficulty(h_ks, grammar_depth=0, noise_std=0.01) → float
# = (1 + h_ks) × (1 + grammar_depth) × (1 + 10σ)

weighted_score(raw_score, difficulty) → float
# = raw_score × difficulty
```

### 7. `chaosbench/validation/gates.py` — Stage 1 Hard Gates

Individual gate functions (each returns `(passed: bool, reason: str)`):
- `check_parameter_bounds(family, params, valid_ranges)`
- `check_trajectory_stability(trajectory)` — no NaN/Inf, max |x| < 1e10
- `check_attractor_bounds(trajectory, domain, margin=0.1)`
- `check_minimum_lyapunov(lambda_max, threshold=0.01)` — chaotic only
- `check_periodicity(trajectory, max_period=1000)` — chaotic only
- `check_permutation_entropy(trajectory, order=5, threshold=0.5)` — chaotic + quasiperiodic only
- `check_autocorrelation(trajectory, lag=1, threshold=0.05)` — chaotic only

`validate_stage1(trajectory, family, params, valid_ranges, domain, regime, lambda_max)` → (all_passed, results)

**Key design decision**: PE and ACF gates apply only to chaotic regime. Fixed_point trajectories converge to zero (trivially low PE). Periodic rotations have ACF(1)=cos(2πω) which is 0 for ω=0.25. These aren't bugs — the parameter/stability/bounds gates are sufficient for non-chaotic systems.

### 8. `chaosbench/validation/baselines.py` — Stage 2 Baseline Battery

**PREDICT baselines**:
- `baseline_persistence(data, K)` → repeat last value
- `baseline_mean_reversion(data, K)` → repeat mean
- `baseline_ar5(data, K)` → fit AR(5) via `np.linalg.lstsq`, extrapolate recursively

**CLASSIFY baseline**:
- `baseline_classify_moments(data)` → heuristic: low variance→fixed_point, strong ACF peak→periodic, else chaotic

**IDENTIFY baseline**:
- `baseline_identify_return_map(data, families)` → heuristic: converging→damped_linear, linear return map→rotation, piecewise-linear→tent, curved→logistic

`validate_stage2_predict(results, tau_lambda)` — checks k_eff>0 (structure) and k_eff<3·τ_λ (not trivial).
CLASSIFY/IDENTIFY: record per-problem scores, aggregate check at bank level.

### 9. `chaosbench/problems/factory.py` — Problem Generation

`QuestionType` enum: CLASSIFY, IDENTIFY, PREDICT.
`GroundTruth` dataclass: regime, family, future_values, params.
`Problem` dataclass: problem_id, question_type, observations, question_params, metadata (shown to agent), ground_truth (hidden), system_metadata, observation_detail, difficulty, stage1_passed, stage2_results.

`create_problem(family, params, question_type, ic_seed, noise_seed, n_points=200, noise_std=0.01)` → Problem.

### 10. `chaosbench/problems/bank.py` — Mini-Bank Generation

`generate_mini_bank(ic_seed, noise_seed, n_points, noise_std)` → 36 Problems (4×3×3).
`validate_problem(problem)` → (passed, details) running gates + baselines.
`validate_and_filter_bank(problems)` → list of valid Problems.
`freeze_bank(problems, path)` → serialize to JSON.
`load_bank(path)` → dict.

---

## Tests (in `chaosbench/tests/`)

| Test File | What It Verifies | Key Assertions |
|-----------|-----------------|----------------|
| `test_atoms.py` | All 4 atoms: iterate, derivative, lyapunov, regime, domain, param validation | Logistic r=4 λ≈ln(2)±0.02; tent λ=ln(mu) exact; damped h_ks=0; rotation λ=0; r=3.891 regime=chaotic; r=2.73 regime=fixed_point |
| `test_system.py` | DynamicalSystem: observe, metadata, future_trajectory, prediction_horizon | Shape checks, seed reproducibility, future[0]=iterate(clean[-1]), noisy≠clean |
| `test_verification.py` | verify_classify, verify_identify, verify_predict | Exact match, case insensitivity, k_eff computation, epsilon floor, padding |
| `test_difficulty.py` | composite_difficulty, weighted_score | Minimum=1.0 (h_ks=0,depth=0,σ=0), chaos increases difficulty, non-chaotic≠zero |
| `test_gates.py` | All 7 gates + validate_stage1 | Stable traj passes, NaN fails, periodic orbit caught, chaotic has high PE |
| `test_baselines.py` | persistence, mean_reversion, AR(5), classify_moments, identify_return_map | Correct shapes, convergent detected, short data fallback |
| `test_factory.py` | create_problem for all 3 question types | PREDICT has future_values, CLASSIFY has regime, deterministic IDs |
| `test_bank.py` | generate_mini_bank, validate, freeze/load roundtrip | 36 problems generated, no textbook params, JSON roundtrip |

---

## Implementation Order

```
Step 1: grammar/atoms.py + test_atoms.py          (independent, core)
Step 2: grammar/registry.py                        (depends on atoms)
Step 3: grammar/system.py + test_system.py         (depends on atoms)
Step 4: grammar/connectives.py                     (stub, trivial)
Step 5: problems/verification.py + test_verification.py  (independent)
Step 6: scoring/difficulty.py + test_difficulty.py       (independent)
Step 7: validation/gates.py + test_gates.py        (needs atoms for test data)
Step 8: validation/baselines.py + test_baselines.py (needs atoms + verification)
Step 9: problems/factory.py + test_factory.py      (integration: atoms+system+registry+difficulty)
Step 10: problems/bank.py + test_bank.py           (integration: factory+gates+baselines)
```

Steps 1, 5, 6 can run in parallel. Steps 7-8 need atoms. Steps 9-10 are the integration layer.

---

## Verification

After all files are implemented:

```bash
# Run all new v2 tests
source venv/bin/activate
pytest chaosbench/tests/test_atoms.py -v
pytest chaosbench/tests/test_system.py -v
pytest chaosbench/tests/test_verification.py -v
pytest chaosbench/tests/test_difficulty.py -v
pytest chaosbench/tests/test_gates.py -v
pytest chaosbench/tests/test_baselines.py -v
pytest chaosbench/tests/test_factory.py -v
pytest chaosbench/tests/test_bank.py -v

# Integration: generate and validate mini-bank
python3 -c "
from chaosbench.problems.bank import generate_mini_bank, validate_and_filter_bank, freeze_bank
bank = generate_mini_bank()
print(f'Generated {len(bank)} problems')
valid = validate_and_filter_bank(bank)
print(f'Validated: {len(valid)} passed')
for p in valid:
    print(f'  {p.problem_id}: {p.question_type.value} difficulty={p.difficulty:.2f}')
freeze_bank(valid, 'chaosbench/data/mini_bank.json')
print('Bank frozen to JSON')
"

# Verify existing tests still pass (no regressions)
pytest chaosbench/tests/test_lyapunov.py -v
```

**Success gate**: All new tests pass. Mini-bank has 12-18+ validated problems. All 4 families × 3 question types represented. No textbook parameter values.
