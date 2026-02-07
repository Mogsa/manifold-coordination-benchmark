# Findings: ChaosBench v2

## v1 Prototype Lessons (Jan 30 – Feb 3)

These findings from the old metacognitive agent approach inform the v2 design.

### LLM Anchoring on Prompt Examples
- Agent predicted 0.42 for every task because the prompt example used 0.42
- **Rule**: Never put concrete numeric examples where the model should reason. Use `<placeholder>` syntax.

### Tent Map Numerical Instability
- Tent map with mu=2 degenerates to fixed point x=0 (floating-point rounds 0.5->1.0->0.0->stuck)
- **Rule**: Hard gate on trajectory stability. Hard cap mu<=1.95.

### NLL Bin Scoring Breaks for Non-[0,1] Systems
- Old scorer used 20 bins over [0,1]. Henon/Lorenz values fall entirely outside -> all tasks scored 1.00.
- **Rule**: Scoring must be domain-aware. v2 uses k_eff for PREDICT, exact match for CLASSIFY/IDENTIFY.

### Protocol Mismatch (Prompt vs Code)
- Prompt said "feedback after PREDICT", code gave feedback after MOVE_ON -> agent never advanced.
- **Rule**: Test with actual LLM, not just unit tests.

### Scaffolding Obscures the Core Signal
- HYPOTHESIZE/FIT tested tool use, not transfer learning.
- **Rule**: Start with raw capability measurement. Scaffolding is an experimental condition, not a baseline.

### Mean Reversion Gaming
- Predicting attractor center gives low MSE by being "close to everything."
- **Rule**: Use k_eff (threshold-based, fails on any wrong step).

---

## Design Rationale

### Why Multiple Question Types
CLASSIFY, IDENTIFY, PREDICT test different cognitive skills. Cross-question transfer (CLASSIFY logistic -> IDENTIFY tent -> PREDICT) is a stronger signal than within-question improvement.

### Why Blocked vs Shuffled (Not Just Sequential)
- Sequential only: can't distinguish "learning structure" from "context accumulation"
- Blocked groups by family -> structural learning possible
- Shuffled randomizes -> only generic context effects
- Gap = structural transfer (the headline result)

### Why Non-Standard Parameters
LLMs have likely seen logistic r=4.0 in training data. Using r=3.891 forces genuine reasoning, not recall. r=3.831 falls in period-3 window (lambda=-0.37) — verified r=3.891 is genuinely chaotic (lambda~0.49).

### Why Composite Difficulty (Not Just h_KS)
h_KS=0 for non-chaotic systems (damped_linear, rotation). Raw h_KS weighting would make these tasks weightless. Composite `(1 + h_KS)(1 + depth)(1 + 10σ)` ensures all tasks contribute.

### Why 2-Stage Validation
- Stage 1 (hard gates): reject degenerate trajectories
- Stage 2 (baselines): reject if naive strategies solve too well OR if data has no exploitable structure
- Valid problems live between random floor and oracle ceiling

---

## Phase 3 Results (Gemini 2.0 Flash, 27 problems)

### Scores by Question Type

| Type | Avg Score | Correct/Total | Notes |
|------|-----------|---------------|-------|
| CLASSIFY | 0.364 | 4/11 | Massive "quasiperiodic" bias |
| IDENTIFY | 0.545 | 6/11 | "logistic" default bias |
| PREDICT | 0.790 | 4/5 | Surprisingly strong |
| Overall | 0.517 | 14/27 | |

### CLASSIFY Failure Analysis

Gemini answered "quasiperiodic" for 6/7 wrong CLASSIFY answers. In a diagnostic re-run, it switched to "chaotic" for most things.

**Root cause:** Gemini cannot distinguish chaotic from quasiperiodic without analytical tools. It *knows* it should compute Lyapunov exponents (says so in reasoning) but can't execute the math. Temperature=0.7 adds noise to an already uncertain classification.

Specific failure modes:
- **Damped linear (fixed_point):** lam=0.941 decays very slowly. After burn-in, noise dominates -> looks "random" -> calls it chaotic.
- **Rotation (quasiperiodic):** Rotation maps on [0,1) look like random numbers. No tools to detect irrational frequencies.
- **Tent/logistic (chaotic):** Coin-flip between "chaotic" and "quasiperiodic."

**Implication for arena:** CLASSIFY is a meaningfully hard task. Good for the benchmark.

### IDENTIFY Failure Analysis

Gemini defaults to "logistic" for unknown systems. Gets logistic and damped_linear reliably, misidentifies tent and rotation as logistic.

**Implication:** "logistic" bias likely reflects training data frequency.

### PREDICT Success Analysis

79% average is surprisingly high. Gemini correctly predicted damped linear (trivial: exponential decay), tent map (short-term), and logistic (one scored 0.95 on chaotic system).

**Implication:** PREDICT tests pattern continuation — closer to next-token prediction, which LLMs are literally trained for.

### Key Insight

**PREDICT beating CLASSIFY is counterintuitive** — predicting chaos *should* be harder than classifying it. But it reflects:
- CLASSIFY: requires domain knowledge (what IS chaos?)
- PREDICT: requires pattern continuation (an LLM's core competency)

This is a genuine finding for the dissertation.

---

## Phase 4 Design Findings

### Why the Arena
The static benchmark showed difficulty varies hugely by type. A fixed bank can't adapt. The arena lets us:
1. Discover which problems are actually discriminating (not just hard)
2. Test whether the difficulty landscape is explored efficiently
3. Compare automated scoring with peer-review consensus

### Grammar Space Coverage
Current mini-bank has 12 parameter settings. The full grammar space is continuous. Interesting regions the proposer might discover:
- Logistic near r=3.57 (onset of chaos)
- Logistic near r=3.83 (period-3 window — looks chaotic but isn't)
- Tent near mu=1.0 (barely chaotic, low lambda)
- Rotation with simple rational omega (easily detected periodicity)

### Consensus vs Math Ground Truth
We track both because:
- Math ground truth is exact (regime/family are computable facts, future trajectory is deterministic)
- Consensus is what reviewers agree on from reasoning alone
- **The gap between them measures how well peer review works** — the most interesting Phase 4 research question

---

## Phase 4 Results (Gemini 2.0 Flash, 10-round Arena)

### Arena Summary

| Metric | Result | Gate | Status |
|--------|--------|------|--------|
| Validation pass rate | 90% (9/10) | >40% | PASS |
| Question types proposed | classify only | ≥1 CLASSIFY + ≥1 PREDICT | PARTIAL FAIL |
| Reviewer rating variance | Low (mostly 4s) | Non-trivial | PARTIAL FAIL |
| Consensus accuracy | 78% (7/9) | >50% | PASS |
| Reputation divergence | solver_0=0.67, solver_1=0.56, solver_2=0.56; reviewer_0=0.85, reviewer_1=0.93 | Scores diverge | PASS |

### Round-by-Round Results

| Round | Family | Params | Type | Valid? | Consensus | Truth | Match? | Discrimination |
|-------|--------|--------|------|--------|-----------|-------|--------|---------------|
| 1 | tent | mu=1.92 | classify | PASS | chaotic | chaotic | Yes | 0.22 |
| 2 | logistic | r=3.56 | classify | PASS | periodic | periodic | Yes | 0.22 |
| 3 | tent | mu=1.75 | classify | PASS | quasiperiodic | chaotic | No | 0.00 |
| 4 | logistic | r=3.99 | classify | PASS | chaotic | chaotic | Yes | 0.22 |
| 5 | logistic | r=3.7 | classify | FAIL (PE gate) | — | — | — | — |
| 6 | tent | mu=1.85 | classify | PASS | chaotic | chaotic | Yes | 0.00 |
| 7 | tent | mu=1.9 | classify | PASS | chaotic | chaotic | Yes | 0.00 |
| 8 | rotation | omega=0.382 | classify | PASS | quasiperiodic | quasiperiodic | Yes | 0.22 |
| 9 | logistic | r=3.56 | classify | PASS | quasiperiodic | periodic | No | 0.00 |
| 10 | logistic | r=3.99 | classify | PASS | chaotic | chaotic | Yes | 0.00 |

### Key Findings

**1. Proposer diversity is limited.**
Gemini only proposed CLASSIFY problems across all 10 rounds. It explored 3 families (logistic, tent, rotation — never damped_linear) and varied parameters meaningfully, but never tried IDENTIFY or PREDICT. The proposer prompt may need stronger nudges toward diversity, or we could rotate question types externally.

**2. High validation pass rate (90%).**
Only 1 proposal failed (logistic r=3.7, permutation_entropy gate). This suggests Gemini understands which parameter regions produce valid dynamics — r=3.7 falls in a periodic window within the chaotic regime, and the PE gate caught it.

**3. Consensus tracks ground truth well (78%).**
7/9 accepted problems had correct consensus. The 2 failures (rounds 3, 9) both involved Gemini calling something "quasiperiodic" when it's actually chaotic or periodic — the same confusion seen in Phase 3.

**4. Discrimination is low.**
Most rounds had 0 discrimination (all solvers agree). Only 4/9 rounds showed any variance (0.22). This makes sense: with the same model at temp=0.5, solvers tend to converge. Multi-model testing would increase discrimination.

**5. Reviewers are well-calibrated but lack variance.**
Both reviewers consistently gave quality=4 (borderline accept) and confidence=3-4. Their answer ratings correlated well with ground truth (reviewer_0=0.85, reviewer_1=0.93 accuracy). But the Likert distributions are narrow — the reviewers aren't using the full 1-6 scale.

**6. Reputation scores diverge meaningfully.**
solver_0 (0.67) outperformed solver_1 and solver_2 (both 0.56). Reviewer_1 (0.93) slightly outperformed reviewer_0 (0.85). The proposer scored 0.65, reflecting the one validation failure and low discrimination.

### Implications for Dissertation

- **CLASSIFY remains the hardest type** — consistent with Phase 3 findings
- **Single-model arena has limited discrimination** — need multi-model or multi-temperature to get more signal
- **Peer review works** — 78% consensus accuracy validates the approach
- **Proposer needs nudging** — future work: rotate question types, increase temperature, or add diversity bonuses to reputation

---

## References

- Master spec: `docs/Chaos_IMO`
- Phase 3 analysis plots: `results/chaosbench/phase3_analysis/`
