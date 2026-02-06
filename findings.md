# Findings: ChaosBench v2

## Phase 3 Results (Gemini 2.0 Flash, 27 problems)

### Scores by Question Type

| Type | Avg Score | Correct/Total | Notes |
|------|-----------|---------------|-------|
| CLASSIFY | 0.364 | 4/11 | Massive "quasiperiodic" bias |
| IDENTIFY | 0.545 | 6/11 | "logistic" default bias |
| PREDICT | 0.790 | 4/5 | Surprisingly strong |
| Overall | 0.517 | 14/27 | |

### CLASSIFY Failure Analysis

Gemini answered "quasiperiodic" for 6 out of 7 wrong CLASSIFY answers. In a diagnostic re-run, it switched to answering "chaotic" for most things.

**Root cause:** Gemini cannot distinguish chaotic from quasiperiodic without analytical tools. It *knows* it should compute Lyapunov exponents (says so in its reasoning) but can't execute the math in its head. Temperature=0.7 adds noise to an already uncertain classification.

Specific failure modes:
- **Damped linear (fixed_point):** lam=0.941 decays very slowly. After burn-in, values near zero but noise dominates. Looks "random" to Gemini -> calls it chaotic.
- **Rotation (quasiperiodic):** Rotation maps on [0,1) look like random numbers. No tools to detect irrational frequencies.
- **Tent/logistic (chaotic):** Sometimes correct, sometimes guesses "quasiperiodic." Basically coin-flip between the two.

**Implication for arena:** CLASSIFY is a meaningfully hard task that can't be solved by surface-level pattern recognition. Good for the benchmark.

### IDENTIFY Failure Analysis

Gemini defaults to "logistic" for unknown systems. Gets logistic and damped_linear reliably, but misidentifies tent and rotation as logistic.

**Implication:** The "logistic" bias likely reflects training data frequency. Logistic map is the most common textbook example.

### PREDICT Success Analysis

79% average is surprisingly high. Gemini correctly predicted:
- Damped linear (trivial: exponential decay to 0)
- Tent map trajectories (short-term)
- Logistic map trajectories (one scored 0.95 on a chaotic system)

**Implication:** PREDICT tests a different capability than CLASSIFY. The agent doesn't need to *name* the regime — it needs to *continue the pattern*. This is closer to next-token prediction, which LLMs are literally trained for.

### Key Insight

**PREDICT beating CLASSIFY is counterintuitive** — predicting future values of a chaotic system *should* be harder than classifying its regime. But it reflects the difference between:
- CLASSIFY: requires domain knowledge (what IS chaos? what IS quasiperiodicity?)
- PREDICT: requires pattern continuation (what comes next? — an LLM's core competency)

This is a genuine finding for the dissertation.

---

## Phase 4 Design Findings

### Why the Arena

The static benchmark (Phase 3) showed that question difficulty varies hugely by type. A fixed bank can't adapt. The arena lets us:
1. Discover which problems are actually discriminating (not just hard)
2. Test whether the difficulty landscape is explored efficiently
3. Compare automated scoring with peer-review consensus

### Grammar Space Coverage

Current mini-bank has 12 parameter settings (4 families x 3 each). The full grammar space is continuous:
- logistic: r in [2.5, 4.0] — includes bifurcation boundaries, period-3 windows
- tent: mu in [1.0, 1.95] — simpler but mu near 2.0 is degenerate
- damped_linear: lam in (0, 0.99) — entire range is fixed_point
- rotation: omega in (0, 1) — rationals = periodic, irrationals = quasiperiodic

**Interesting regions the proposer might discover:**
- Logistic near r=3.57 (onset of chaos)
- Logistic near r=3.83 (period-3 window — looks chaotic but isn't)
- Tent near mu=1.0 (barely chaotic, low lambda)
- Rotation with omega = simple rational (period easily detected vs complex rational)

### Consensus vs Math Ground Truth

We track both because:
- For CLASSIFY/IDENTIFY: math ground truth is exact (regime/family are computable facts)
- For PREDICT: math ground truth is the actual future trajectory
- Consensus is what reviewers agree on from reasoning alone
- **The gap between them measures how well peer review works** — the most interesting research question
