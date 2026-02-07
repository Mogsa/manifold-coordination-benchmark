# Epiplexity × ChaosBench: Integration Analysis

**Paper:** "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence" (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson — CMU/NYU, Jan 2026, arXiv:2601.03220v1)

**Source PDF:** `~/Downloads/2601.03220v1.pdf`

---

## 1. What the Paper Says (Core Ideas)

### 1.1 The Central Claim

Shannon entropy and Kolmogorov complexity assume **unbounded computation** by the observer. For computationally bounded observers (like LLMs, or any real system), information splits into two components:

- **Time-bounded entropy H_T(X):** The irreducibly random, unpredictable part — what you *can't* learn no matter how clever your model is, given compute budget T.
- **Epiplexity S_T(X):** The structural, learnable part — patterns, regularities, and circuits that a bounded observer can extract and store in its weights.

Together: `MDL_T(X) = S_T(X) + H_T(X)` (total time-bounded information = structure + noise).

### 1.2 Formal Definition (Definition 8)

Given random variable X on {0,1}^n and time bound T:

```
P* = argmin_{P in P_T} { |P| + E[log 1/P(X)] }
```

This is the program that minimizes the two-part MDL (model description + data given model) under a runtime constraint T. Then:

- **Epiplexity:** S_T(X) = |P*| (length of the optimal program — the "structure bits")
- **Time-bounded entropy:** H_T(X) = E[log 1/P*(X)] (expected residual loss — the "noise bits")

Key insight: **the same data can have different epiplexity depending on the observer's compute budget.** More compute → can extract more structure → higher epiplexity, lower entropy.

### 1.3 Time-Bounded Complexity: How It Differs from Classical Measures

This is the crucial point. Classical information theory says:

| Property | Shannon/Kolmogorov | Epiplexity (time-bounded) |
|---|---|---|
| Deterministic transforms | Can't increase information | **CAN increase information** (Theorem 12) |
| Factorization order | Information is order-invariant | **Order matters** (Theorem 13) |
| Distribution matching | Likelihood model = generating process | **Model can learn MORE than generating process** |
| CSPRNG output | Low Shannon entropy (= seed length k) | **Near-maximal time-bounded entropy** (= output length n) |
| Chaotic trajectory | Low Kolmogorov complexity (= short program) | **High time-bounded entropy + moderate epiplexity** |

The time bound T is what makes everything different. Without it:
- A logistic map trajectory has low Kolmogorov complexity (short program: "iterate r*x*(1-x)")
- But a polynomial-time observer **cannot** use that short program to predict — they'd need to iterate from the exact initial condition, and any precision error grows exponentially

With the time bound:
- The trajectory's **time-bounded entropy** is high (can't predict past Lyapunov time)
- But the trajectory's **epiplexity** is moderate (can still learn: regime, family, invariant measure, attractor shape)

### 1.4 Three Paradoxes Resolved

**Paradox 1: Information can't be created by deterministic processes.**
- Classical: K(f(x)) ≤ K(x) + K(f) + c. Applying a function can't increase info.
- Epiplexity: A CSPRNG stretches k random bits into n >> k bits of **time-bounded** randomness. The ECA Rule 54 creates **structural** information from simple rules. AlphaZero creates chess knowledge from game rules.
- **ChaosBench connection:** Our chaotic maps create both time-bounded entropy AND epiplexity from simple parameter choices. The iterate() function is trivial, but the resulting trajectory carries rich structure for bounded observers.

**Paradox 2: Information is independent of factorization order.**
- Classical: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y). Order doesn't matter.
- Epiplexity: Predicting chess moves→board is easy (forward computation). Predicting board→moves is hard (requires induction). The **reverse order has higher epiplexity** because the model must learn richer internal representations.
- **ChaosBench connection:** This is exactly why blocked ordering should beat shuffled — seeing related systems in sequence is the "easier direction" for accumulating structural knowledge. The Φ(n) curve measures this.

**Paradox 3: Likelihood modeling is just distribution matching.**
- Classical: The optimal model just matches the data-generating distribution.
- Epiplexity: A bounded model must learn to **induct** — infer hidden variables, recognize patterns that weren't explicit in the generating process. The model ends up with MORE structure (higher epiplexity) than the generating process.
- **ChaosBench connection:** When an LLM identifies a logistic map from observations, it's performing induction — the generating process was just "iterate r*x*(1-x)", but the LLM must learn return map shapes, Lyapunov estimation heuristics, family signatures. The LLM's solution program is LONGER than the generating program.

### 1.5 The Chaotic Systems Connection (Appendix F)

The paper explicitly discusses chaotic dynamical systems:

> "There is a precise sense in which entropy is created in [the Lorenz system] at a rate of λ₁ log₂(e) bits per second, formalized through Pesin's theorem."

For chaotic systems:
- **Time-bounded entropy** accumulates at rate h_KS (= sum of positive Lyapunov exponents)
- **Epiplexity** = the structural information an observer can extract: attractor shape, invariant measure, regime classification, family identity
- An LLM trained on Lorenz trajectories **can learn the butterfly attractor shape** (the SRB measure) even though it can't predict individual trajectories (Figure 11)

Key quote: "The epiplexity of the attractor for limited compute may be larger than a description of the dynamics: S_T(Φ^t(X)) > S_T(Φ, t)"

Translation: The learned model of the attractor is MORE complex than the generating equations. This is emergence.

### 1.6 Measuring Epiplexity in Practice

Two methods:

**Prequential coding (heuristic, cheap):**
- Train a model on the data, record the loss curve
- Epiplexity ≈ area under the loss curve above the final loss
- Intuition: if training produces a large, sustained reduction in loss, the model absorbed a lot of structural information

**Requential coding (rigorous, expensive):**
- Train a student model on synthetic data from a teacher model
- Epiplexity = cumulative KL divergence between teacher and student
- Provides an explicit code for the model weights

For practical purposes, the prequential estimate is ~2-10x larger but **correlates well** with requential coding for ranking datasets.

### 1.7 Epiplexity Predicts OOD Generalization

The paper's strongest empirical claim: **higher epiplexity data produces models that generalize better OOD.**

- OpenWebText (language) has the highest epiplexity → transfers broadly
- Chess data has moderate epiplexity → transfers to chess puzzles
- CIFAR-5M (images) has the lowest epiplexity despite highest total information → transfers poorly
- Reverse-order chess has higher epiplexity than forward-order → better OOD performance on chess puzzles

This directly supports ChaosBench's Φ(n) transfer hypothesis: if blocked ordering produces higher epiplexity in the model, it should show better transfer (superlinear Φ(n)).

---

## 2. How Time-Bounded Complexity Maps onto ChaosBench

### 2.1 Current ChaosBench Measures vs Epiplexity Decomposition

| ChaosBench Concept | Epiplexity Analog | How It Maps |
|---|---|---|
| h_KS (Kolmogorov-Sinai entropy) | Rate of time-bounded entropy creation | Identical concept. h_KS = rate at which bounded observers lose predictive ability. Pesin's theorem: h_KS = Σ positive Lyapunov exponents. |
| Lyapunov time τ_λ = 1/λ_max | Time horizon of predictability | After τ_λ, time-bounded entropy dominates. Before τ_λ, epiplexity (structure) is still usable for prediction. |
| PREDICT score = k_eff/K | Measures boundary between structure and noise | k_eff is exactly where the model's structural knowledge (epiplexity) runs out and time-bounded entropy takes over. |
| CLASSIFY/IDENTIFY accuracy | Measures extracted epiplexity | These tasks ask about STRUCTURE (regime, family) not trajectories. Success = the model has extracted epiplexity. |
| composite_difficulty | Proxy for total MDL_T | (1+h_KS)×(1+depth)×(1+10σ) combines entropy rate, structural complexity, and noise — roughly tracks total information content. |
| Φ(n) transfer curve | Epiplexity accumulation over tasks | Superlinear Φ(n) means the model is accumulating reusable structural information (epiplexity) across tasks. |
| Blocked vs shuffled ordering | Factorization order effect (Paradox 2) | Blocked = coherent factorization (easier to extract structure). Shuffled = incoherent (less epiplexity extracted). |
| Non-standard parameters | Ensures high time-bounded entropy | Textbook params might have low time-bounded entropy for LLMs (memorized). Non-standard params ensure the trajectory is genuinely unpredictable. |

### 2.2 What's NOT Represented in ChaosBench Yet

**The key gap: ChaosBench doesn't explicitly measure epiplexity as distinct from entropy.**

Currently:
- `h_KS` measures entropy rate (how fast randomness accumulates)
- `composite_difficulty` conflates structural complexity with randomness
- There's no metric for "how much learnable structure does this problem contain?"

A problem with high h_KS and simple structure (tent map, μ=1.9) is DIFFERENT from a problem with moderate h_KS and rich structure (logistic near bifurcation boundary, r≈3.57). The first has high entropy but low epiplexity. The second has moderate entropy but high epiplexity.

**This matters for:**
- Problem bank quality: high-epiplexity problems are more discriminating
- Arena proposer scoring: prefer high-epiplexity proposals
- Transfer analysis: epiplexity predicts which problems benefit from sequential context
- Difficulty calibration: separate "hard because random" from "hard because structurally complex"

---

## 3. Integration Ideas (Concrete)

### 3.1 Epiplexity Proxy as Problem Quality Metric

**Value:** HIGH | **Effort:** LOW-MEDIUM | **Where:** `scoring/difficulty.py` or new `scoring/epiplexity.py`

**What:** For each problem's observation sequence, estimate epiplexity via prequential coding proxy.

**How:**
- Train a tiny sequence model (even just AR(p) with increasing p, or a small transformer) on the observations
- Measure loss curve: L(t) for t = 1, ..., T
- Epiplexity proxy = Σ (L(t) - L(T)) = area under loss curve above final loss
- Normalize by sequence length

**Why it matters:**
- Gives a second difficulty axis: "structural complexity" separate from "entropy rate"
- Problems with high epiplexity proxy are better discriminators (more structure to learn)
- Can weight Φ(n) by epiplexity instead of (or in addition to) composite difficulty

**Simpler alternative:** Use existing statistics as proxy:
- Permutation entropy alone doesn't work (it tracks total complexity, not structure)
- But: `permutation_entropy × (1 - |ACF(1)|)` might work — high PE (complex) + low autocorrelation (not trivially predictable) = high structural content
- Or: variance of the return map residuals after polynomial fit — high residual variance = complex dynamics

### 3.2 Theoretical Framing for Dissertation

**Value:** HIGH | **Effort:** ZERO (writing only)

Frame ChaosBench results using epiplexity vocabulary:

1. **PREDICT tests the entropy boundary.** The effective prediction horizon k_eff marks where time-bounded entropy overwhelms the model's structural knowledge. Scoring by k_eff × h_KS literally measures "bits of information correctly predicted before chaos wins."

2. **CLASSIFY/IDENTIFY test epiplexity extraction.** Success on these tasks means the LLM has extracted structural information (regime, family) that persists despite chaotic trajectories. This is exactly what epiplexity measures — learnable structure visible to bounded observers.

3. **Blocked > shuffled = factorization order effect.** The paper proves (Theorem 13) that factorization order affects extractable information for bounded observers. Blocked ordering is the "coherent factorization" where related structure appears together, enabling more efficient extraction. This gives theoretical backing to the Φ(n) hypothesis.

4. **Anti-memorization = high time-bounded entropy.** Non-standard parameters ensure trajectories have near-maximal time-bounded entropy — the exact trajectory is unpredictable even if the family is known. But the *structural* properties (regime, family, attractor shape) remain learnable. This is the entropy/epiplexity split in action.

5. **Arena discrimination ≈ epiplexity variation.** A discriminating problem is one where solvers differ in their ability to extract structure. High-epiplexity problems naturally discriminate because there's more structure to find or miss.

### 3.3 PREDICT-LONG: Invariant Measure Question Type

**Value:** MEDIUM-HIGH | **Effort:** MEDIUM | **Where:** New question type in `problems/factory.py`

The paper (Appendix F, Figure 11) shows LLMs can learn the invariant measure (SRB measure) of chaotic systems even when they can't predict trajectories. This is pure epiplexity — structural information that survives chaos.

**Implementation:**
- Generate long trajectory (>>τ_λ, e.g., 1000+ points)
- Define bins covering the attractor: [lo, lo+Δ), [lo+Δ, lo+2Δ), ...
- Ask: "Estimate the probability that the next value falls in each bin"
- Ground truth: empirical distribution from a very long trajectory (100K+ points)
- Verify: KS-test or chi-squared between predicted and true distributions
- Score: 1 - KS_statistic (or similar)

**Why it matters:**
- Tests whether LLMs extract the *invariant measure* — the deepest structural property of chaotic systems
- Directly measures epiplexity (the invariant measure IS the structural information)
- Complements PREDICT (which tests the entropy boundary) with a test of the *structure* boundary
- The paper proves this is learnable even when point prediction is impossible

**Already in the PRD:** Section 4.4 lists "PREDICT-LONG: Characterise the invariant distribution (not point prediction). Verified via KS test against true invariant measure." So this was already planned — the paper gives theoretical backing.

### 3.4 Emergence Detection in the Arena

**Value:** MEDIUM | **Effort:** LOW | **Where:** `arena/consensus.py` or new analysis module

The paper's Definition 14 (Epiplexity Emergent) formalizes when a system exhibits emergence: two observers with different compute budgets see asymptotically different structural complexity for multi-step dynamics, but not for single-step.

**For ChaosBench:** This manifests as problems where:
- CLASSIFY is easy (single-step structure: "is it chaotic?")
- PREDICT is hard beyond τ_λ (multi-step entropy dominates)
- But PREDICT-LONG (invariant measure) is learnable (multi-step structure survives)

Track which problems show this "emergence signature" — easy to classify, impossible to point-predict, but possible to characterize statistically. These are the highest-epiplexity problems and should be the most discriminating in the arena.

### 3.5 Arena Proposer: Prefer High-Epiplexity Proposals

**Value:** MEDIUM | **Effort:** LOW | **Where:** `arena/runner.py` Phase 1 selection

Currently we select the best proposal by highest `composite_difficulty`. We could also factor in an epiplexity proxy:

```python
# Rank proposals by epiplexity potential, not just difficulty
# Systems near bifurcation boundaries have higher epiplexity
# (complex structure + moderate entropy > simple structure + high entropy)
score = difficulty * epiplexity_proxy
```

Simple proxy for proposal ranking (no model training needed):
- Compute permutation entropy PE and autocorrelation ACF(1) from a short trajectory
- `structural_score = PE * (1 - |ACF(1)|)`
- High PE + low ACF = rich, non-trivial structure = high epiplexity
- Prefer proposals with high structural_score

### 3.6 Data Selection for Experiments (Section 6.4 of Paper)

The paper shows that "Adaptive Data Optimization" (selecting training data by loss curve slope) inadvertently selects high-epiplexity data and improves OOD performance.

**For ChaosBench v1 experiments:** When constructing the problem bank, prefer problems whose observation sequences have high estimated epiplexity. This should produce a bank that is:
- More discriminating (more structure to find)
- Better for measuring transfer (more reusable structure across problems)
- More interesting scientifically (tests genuine reasoning, not just noise tolerance)

---

## 4. Open Questions for Discussion

1. **How to estimate epiplexity cheaply enough for real-time use?** The prequential coding method requires training a model — too expensive for arena-time proposal ranking. Need a fast proxy. Permutation entropy + ACF is one option. Return map polynomial fit residuals is another. What's the simplest thing that correlates with epiplexity?

2. **Does epiplexity predict Φ(n) transfer rate?** The paper claims high-epiplexity data enables better OOD transfer. Our Φ(n) curves measure transfer. If we compute per-problem epiplexity estimates, does epiplexity × blocked_ordering predict superlinear Φ(n)?

3. **Should we weight Φ(n) by epiplexity instead of composite_difficulty?** Currently: Φ(n) = Σ difficulty_i × raw_score_i. Alternative: Φ(n) = Σ epiplexity_i × raw_score_i. This would weight "structurally rich" problems higher than "noisy but simple" ones.

4. **PREDICT-LONG vs PREDICT: when to add?** The PRD defers PREDICT-LONG. The paper gives strong motivation to add it sooner. It's the purest test of epiplexity extraction. But it needs: binning scheme, KS-test verification, prompt engineering for distributional answers. Medium effort.

5. **Can we use the paper's "reverse ordering" trick?** The paper shows reverse-order chess has higher epiplexity. For ChaosBench: what if we show the *question first* and *data second* (reverse of current)? Would this force the model to develop richer internal representations? This is a free experimental condition to test.

6. **Time bound T — what is it for LLMs?** The paper parameterizes everything by compute budget T. For an LLM with fixed weights, T ≈ context_length × model_size. For ChaosBench, the "observer" is the LLM at inference time. Its compute budget is fixed by the model architecture. Different models have different effective T, which should produce different epiplexity extraction. This connects to why larger/better models should show more transfer.

---

## 5. Key Quotes for Citation

> "Chaotic dynamical systems produce both apparently random behavior and structure: the state of the system cannot be predicted precisely over long time-scales, but such observers may still learn meaningful predictive distributions, as shown by the invariant measure." (Section 2.2)

> "There is a precise sense in which entropy is created in [the Lorenz] system at a rate of λ₁ log₂(e) bits per second, formalized through Pesin's theorem." (Appendix F)

> "The epiplexity of the attractor for limited compute may be larger than a description of the dynamics: S_T(Φ^t(X)) > S_T(Φ, t)." (Appendix F)

> "We explore other kinds of emergence, such as in chaotic dynamical systems... clear evidence that in pursuit of the best probability distribution to explain the data, observers with limited compute will require models with greater description length than the minimal data generating process." (Section 5.3.2)

> "Epiplexity measures the structural information learned by the model... the amount of structural information a model extracts, while being agnostic to whether these structures are relevant to a specific downstream task." (Section 6.1)

---

## 6. Session Recovery

**Files modified in this session (SPIRE improvements):**
- `arena/protocol.py` — Added `correct_answer: Any = None` to Review
- `arena/prompts.py` — Multi-proposal prompt, reviewer correct_answer prompt
- `arena/parsing.py` — New `parse_proposals()`, `parse_review()` extracts correct_answer
- `arena/consensus.py` — `REC_SCALE`/`CONF_SCALE`, `compute_weighted_rating()`, reputation-weighted consensus with reviewer answer fusion
- `arena/runner.py` — Multi-proposal selection in Phase 1, pass reputations to consensus
- `tests/test_arena.py` — 13 new tests (269 total, all passing)

**Next steps depend on Morgan's priorities.** Read this file to resume.
