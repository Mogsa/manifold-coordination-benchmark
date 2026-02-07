# ChaosBench NeurIPS Vision

**Status:** Aspirational — ideas beyond the current MVP scope defined in the PRD (`docs/Chaos_IMO` §1.1).
**Last Updated:** 2026-02-07

---

## §0 Relationship to Other Documents

| Document | Role |
|----------|------|
| `docs/Chaos_IMO` (PRD) | Implementation spec for the current MVP: 1D maps, static benchmark, Φ(n) transfer curves. Defines Phases 1–6. |
| `docs/plans/epiplexity-integration.md` | Deep analysis of the epiplexity paper (Finzi et al., 2026) and how it connects to ChaosBench. Theoretical grounding lives there; this document references it, doesn't duplicate it. |
| `docs/plans/chaosbench-v4-task-plan.md` | The live to-do list for what's being built *now*. |
| **This document** | The "north star" — where ChaosBench is heading beyond the MVP. Captures ideas from research discussions that fundamentally change the project's identity: new atoms, new metrics, new agent protocol. Maps onto the dissertation's future-work chapter. |

**Rule:** If an idea is in this document but not in the PRD or task-plan, it is *not* being implemented yet.

---

## §1 Compositionality Gap — The Headline Metric

### What It Is

```
Δ(D) = (performance on depth-D compositions) / (performance on isolated atoms)
```

Where D is composition depth: D=1 means isolated atoms, D=2 means two composed systems, D=3 means three, etc.

### Why It's Novel

- **One-sentence contribution:** "LLMs can identify isolated dynamical systems but fail when you compose them."
- Sharper than "LLMs struggle with chaotic systems" — it isolates *compositional reasoning* as the failure mode.
- First benchmark to systematically measure compositional failure in dynamical systems reasoning.
- Distinct from standard compositional generalization tests (COGS, SCAN) because it involves time-series dynamics, not static symbolic structures.

### Why It Matters

The compositionality gap should decrease with composition depth D but NOT necessarily with model scale. If Δ(D) stays low even for frontier models, composition is a fundamental reasoning barrier — not just a scaling problem.

### Measurement

1. Measure per-atom accuracy (baseline): Can models identify isolated Lorenz, Rossler, etc. from trajectory data?
2. Measure composed system accuracy at D=2, then D=3.
3. Plot Δ(D) vs. composition depth, coupling strength, and h_KS.

### Three Predictions

1. If Δ(D) decreases with D but NOT with model scale → composition is a fundamental barrier.
2. If Δ(D) also decreases with model scale → models are learning compositional reasoning (interesting positive result).
3. If Δ(1) is already low → different paper entirely ("LLMs can't even do spectral matching").

### Relationship to Current PRD

The PRD measures Φ(n) transfer curves (blocked vs. shuffled ordering). Δ(D) replaces Φ(n) as the primary metric in the evolved vision. The compositionality gap subsumes transfer: if models can't compose, transfer is moot.

---

## §2 Continuous Systems — The dysts Library

### What Changes

The current MVP uses 4 discrete 1D maps (logistic, tent, damped_linear, rotation). The evolved vision expands to ~20 continuous dynamical systems from the [dysts library](https://github.com/williamgilpin/dysts) (~200 known systems).

### Selection Strategy: Spanning the Complexity Spectrum

**Low chaos** (~4–5 systems):
- SprottTorus, Rossler, Thomas, ForcedVanDerPol, Duffing
- Clear spectral signatures, near-periodic behavior, recognizable attractor geometry
- Sanity-check baseline

**Moderate chaos** (~5–6 systems):
- Lorenz, Chua, HastingsPowell, Halvorsen, NoseHoover
- Broadband spectra but distinctive topology
- Core difficulty range for discriminating models

**Strong chaos** (~3–4 systems):
- Chen, BurkeShaw, QiChen, Dadras
- Short prediction horizons, statistics-only identification
- Tests limits of structural extraction

**Special structure** (~3–4 systems):
- DoublePendulum or HenonHeiles (Hamiltonian — conserved energy is an exploitable signal)
- MackeyGlass (delay signature in autocorrelation)
- Lorenz96 (spatiotemporal, scalable dimension)
- Each has a unique structural invariant that survives composition (critical for anti-hash)

**Discrete maps** (5–6 for debugging):
- Logistic, Tent, Henon, Ikeda, Tinkerbell
- Simpler, faster, good for initial development

### Selection Rationale

- Span the complexity-entropy plane (Cμ, hμ) from low to high h_KS.
- Each has distinctive signatures (power spectrum, attractor shape, Lyapunov spectrum).
- Mix of well-known (Lorenz) and less-memorizable systems.
- Special structures ensure exploitable invariants survive composition.

### Precomputed Metadata Per Atom

Stored as JSON:
- Lyapunov spectrum (dysts provides this)
- h_KS via Pesin's identity (sum of positive Lyapunov exponents)
- Attractor dimension
- Power spectrum
- Invariant measure statistics

---

## §3 Composition Engine

### Core Class: `ComposedSystem`

Takes two dysts atoms, a coupling map, and coupling strength ε. Integrates the combined ODE via `scipy.solve_ivp`. Hides internal structure — exposes only the black-box query interface (§4).

### Three Coupling Types

**1. Unidirectional drive:**
- A's state enters B's RHS, not vice versa.
- `dB/dt = f_B(B) + ε · g(A)`
- Tests one-way causal inference.

**2. Bidirectional coupling:**
- Symmetric or asymmetric cross-terms.
- `dA/dt = f_A(A) + ε · g(B, A)`, `dB/dt = f_B(B) + ε · h(A, B)`
- Tests mutual influence detection.

**3. Parameter modulation (hardest):**
- A's output slowly varies a parameter of B.
- Can push B through bifurcations dynamically.
- Creates emergent regimes not present in either isolated system.
- Hardest to identify because coupling is indirect.
- *Deferred to post-NeurIPS* — drive and bidirectional are sufficient for Δ(D).

### Implementation Decision

Write custom `ComposedSystem` on top of `scipy.solve_ivp` from day one. Use dysts only for atom definitions (equations + known Lyapunov spectra), NOT for integration. This avoids dysts SkewProduct integration issues (stiff ODEs, chaotic trajectory divergence at strong coupling) and gives full control over numerics.

### Ground-Truth Composition Metadata

For each composed problem:
- Component identities
- Coupling type and direction
- Coupling strength ε
- Combined Lyapunov spectrum
- Combined h_KS
- Statistical fidelity metrics for verification

---

## §4 Black-Box Active Querying

### Interface

```python
class BlackBox:
    def query(self, x0: np.array, T: float) -> np.array:
        """Returns trajectory from initial condition x0 for duration T."""
```

- Wraps a `ComposedSystem`.
- Exposes ONLY the query function.
- Hides all internal structure (equations, parameters, components).
- Counts queries for efficiency metrics.

### Why Black-Box Matters

- Tests *reasoning*, not pattern matching.
- Agents must strategically choose initial conditions to distinguish hypotheses.
- Fundamentally different from "here are 200 observations, what is it?"
- Where tool-calling LLMs should shine.
- Where the compositionality gap will bite hardest.

### Tool-Calling Agent Protocol

- Model gets a `query_system(x0, T)` tool.
- System prompt provides: atom vocabulary, composition operations, answer schema.
- Agent iteratively queries, hypothesizes, refines.
- Must decide WHEN to stop querying and submit.

### Query Budget Tiers

| Budget | Purpose |
|--------|---------|
| 10 | Can you identify with minimal exploration? |
| 50 | Standard difficulty |
| 200 | Generous budget for complex compositions |
| 1000 | Near-saturation — tests query efficiency ceiling |

The budget tiers trace the time-bounded complexity curve: more budget → higher effective T → more extractable epiplexity. This is the cleanest empirical connection to the epiplexity paper's time bound T.

### Relationship to Current PRD

The current PRD (§8.4) describes a tool-augmented mode with tools like COMPUTE_STATS, ITERATE, LYAPUNOV_ESTIMATE. The black-box vision simplifies this to a single primitive (`query`) and lets the agent decide what statistics to compute. More principled, more general.

---

## §5 Anti-Hash Principle

### Core Principle

> "Every problem must preserve at least one exploitable structural invariant."

### Why It Matters

Prevents the benchmark from being trivially impossible. Ensures problems test reasoning about *structure*, not tolerance for pure noise. A composed system whose output looks like white noise is a hash function, not a reasoning challenge.

### Exploitable Invariants (Examples)

- Conserved quantities (energy in Hamiltonian systems)
- Symmetries (rotational, time-reversal)
- Distinct frequency bands in power spectrum
- Transfer entropy revealing coupling direction
- Regime boundaries (periodic vs. chaotic)
- Return map structure (polynomial-fittable despite noise)

### Validation Pipeline

For every candidate problem (atom pair + coupling type + ε):

1. **Compute h_KS** of composed system (from numerical Lyapunov spectrum).

2. **Run baseline identifiers:**
   - **SINDy** (Sparse Identification of Nonlinear Dynamics): sparse regression to recover equations.
   - **Echo State Network (ESN)**: small reservoir computer trained on output.
   - **Pass if:** Baseline recovers partial structure (any parameter within 10% of true value, OR correct component count).
   - **Reject if:** Baseline gets literally nothing → it's a hash, discard.

3. **Compute mutual information:**
   - MI between component-A's trajectory and combined output.
   - MI between component-B's trajectory and combined output.
   - **Reject if:** MI ≈ 0 for both → components indistinguishable from noise.

4. **Record surviving invariants** — these become the "deductive ladder" a reasoning agent should climb.

### Deliverable

Validation script: `ComposedSystem` → (pass/fail, difficulty coordinates (h_KS, estimated Cμ), list of exploitable invariants). Curated problem set: ~50–100 validated problems spanning difficulty range. Each problem tagged with its exploitable invariants for analysis.

---

## §6 Theoretical Framing — Three Pillars

The NeurIPS paper connects three theoretical frameworks. For a deep treatment of epiplexity, see `docs/plans/epiplexity-integration.md`.

### Pillar 1: Epiplexity (Observer-Dependent Extractable Structure)

From Finzi et al. (2026, arXiv:2601.03220v1):

- For computationally bounded observers, information splits into **time-bounded entropy** H_T(X) (irreducibly random) and **epiplexity** S_T(X) (learnable structure).
- `MDL_T(X) = S_T(X) + H_T(X)`
- Key insight: the same data can have different epiplexity depending on the observer's compute budget T.

**How ChaosBench operationalizes this:**
- ChaosBench doesn't compute epiplexity directly (too expensive).
- Query efficiency and compositionality gap are *proxies* for epiplexity.
- CLASSIFY/IDENTIFY success = model extracted structural information (epiplexity).
- PREDICT horizon k_eff = boundary where time-bounded entropy overwhelms structure.
- Query-efficiency curves ≈ empirical V-entropy estimates (V = model class).

### Pillar 2: Computational Mechanics (Ground-Truth Difficulty)

- **Statistical complexity Cμ:** Minimum memory to optimally predict.
- **Entropy rate hμ (≈ h_KS for dynamical systems):** Rate of unpredictability.
- Each problem has known h_KS via Pesin's identity → objective difficulty baseline.
- Problems span the complexity-entropy plane (Cμ, hμ).

**Practical note:** Sidestep full ε-machine computation for high-dimensional continuous systems (computationally expensive). Use Lyapunov-based proxies for h_KS. For Q* (theoretical minimum queries), use statistical estimation bounds. Save full ε-machine analysis for the journal version.

### Pillar 3: Compositionality Gap Δ(D) (Primary Empirical Claim)

Connects the two theoretical frameworks:
- **Problem difficulty** = position in (Cμ, hμ, D) space (ground truth from Pillar 2).
- **Agent performance** = query efficiency relative to Q* (operationalizes Pillar 1).
- **Δ(D)** is the diagnostic that links them: do bounded observers lose more structure when problems compose?

### Paper Structure

- **Intro:** Motivate with epiplexity (observer-dependent complexity).
- **Background:** Computational mechanics (h_KS, Cμ).
- **Methods:** Compositionality gap Δ(D) as primary metric.
- **Results:** Measure gap across models, systems, depths.
- **Discussion:** Connect to epiplexity theory, future ε-machine analysis.

---

## §7 Scoring Evolution

### 7.1 Query Efficiency

```
Q_agent / Q*
```

- Q_agent = number of queries the agent used.
- Q* = theoretical minimum queries needed (statistical estimation bound).
- Directly measures time-bounded complexity.
- Lower ratio = more efficient exploration → higher effective compute budget T.

### 7.2 Model Parsimony

```
L_agent / L_true
```

- L_agent = description length of agent's proposed model (answer JSON length).
- L_true = description length of true system.
- Ratio > 1 → agent overparameterized. Ratio < 1 → underspecified. Ratio ≈ 1 → efficient representation.
- Motivated by epiplexity paper: bounded observers learn models MORE complex than the generating process.

### 7.3 Transfer Ratio (Deferred)

The current PRD's Φ(n) transfer curves are replaced by Δ(D) as the primary metric. Transfer ratio could return in a journal version for multi-task training experiments, potentially weighted by epiplexity instead of composite difficulty.

### Relationship to Current PRD

The current scoring uses `composite_difficulty = (1 + h_KS) × (1 + depth) × (1 + 10σ)` and `weighted_score = raw × difficulty`. The evolved vision adds query efficiency and model parsimony as orthogonal scoring axes, creating a richer picture of agent capability.

---

## §8 Web Platform

### Vision

Avoid massive API costs by leveraging users' existing LLM subscriptions. A user with a Claude or ChatGPT subscription visits the ChaosBench website. Their LLM, via tool-calling, interacts with the website: reads the problem, calls the query API for trajectories, submits an answer. The website verifies and records results.

### Architecture

- Hosted on Morgan's Linux server.
- **Query endpoint:** `POST /query { problem_id, x0, T }` → trajectory.
- **Submit endpoint:** `POST /submit { problem_id, answer }` → verification result.
- **Frontend:** Problem browser, leaderboard, per-problem stats, model comparison.
- **Backend:** Problem bank, ComposedSystem simulator, verification logic.

### Advantages

- Costs distributed across users (no central API bill).
- Tests models beyond the 2–3 we can afford to run ourselves.
- Community engagement and iterative improvement.

### Dependencies

Requires working ComposedSystem, validation pipeline, and black-box interface before building. Not for the initial NeurIPS submission — after baseline experiments prove the concept.

---

## §9 Answer Schema

### Structure

```json
{
  "num_components": 2,
  "components": [
    {
      "name": "Lorenz",
      "params": { "sigma": 10.0, "rho": 28.0, "beta": 2.667 }
    },
    {
      "name": "Rossler",
      "params": { "a": 0.2, "b": 0.2, "c": 5.7 }
    }
  ],
  "coupling": {
    "type": "drive",
    "direction": "Lorenz -> Rossler",
    "strength": 0.05,
    "coupling_variables": ["x"]
  }
}
```

### Fields

- **`num_components`:** Integer. First-order check — did the agent recognize composition depth?
- **`components`:** List. `name` from closed vocabulary of ~20 atoms. `params` as parameter estimates.
- **`coupling`:** `type` ∈ {drive, bidirectional, modulate}. `direction` for asymmetric coupling. `strength` (ε). `coupling_variables` identifies which state variables are coupled.

### Verification Metrics (Automated)

| Metric | What It Checks |
|--------|----------------|
| Component identification accuracy | Correct atom names (exact match) |
| Parameter error | RMSE on parameter estimates |
| Coupling architecture accuracy | Correct type and direction (exact match) |
| Coupling strength error | \|ε_agent − ε_true\| |
| Statistical fidelity | Simulate agent's proposed model, compare invariant measure / Lyapunov spectrum / power spectrum vs. true system (Wasserstein distance or KS-test) |

### Design Decision: Closed Vocabulary

The agent picks from a known list of ~20 atoms rather than proposing equations freely. This enables automated verification and prevents unbounded output. Coupling identification (type, direction, strength) adds genuine difficulty that compensates for the constrained component search.

---

## §10 Build Sequence

### Phase 1: Atom Library + Composition Engine (Weeks 1–3)

**Deliverables:** Curate 15–20 atoms from dysts. Build `ComposedSystem` class. Precompute metadata JSON per atom.

**Test:** `ComposedSystem(Lorenz(), Rossler(), coupling='drive', epsilon=0.05)` → trajectories + metadata.

**Risk:** Numerical integration stability (stiff ODEs, divergence). If Phase 1 takes >2 weeks, timeline is blown.

### Phase 2: Problem Validation Pipeline (Weeks 3–4)

**Dependencies:** Phase 1.

**Deliverables:** Anti-hash validation script (§5). Curated problem set (~50–100 validated). Per-problem: pass/fail, h_KS, Cμ estimate, exploitable invariants.

### Phase 3: Black Box + Agent Protocol (Weeks 4–5)

**Dependencies:** Phases 1 and 2.

**Deliverables:** `BlackBox` class with `query(x0, T)`. Agent system prompt. Answer schema + automated verification. Query budget tiers.

**Test:** Point an LLM at the black box → it queries → submits → gets automated scores.

### Phase 4: Baseline Experiments (Weeks 5–7)

**Dependencies:** Phase 3.

**Deliverables:** Run Claude Sonnet, GPT-4o, Gemini. Start with single atoms (sanity check), then D=2, then D=3. Measure Δ(D). Plot performance vs. h_KS, depth, coupling strength. Plot query efficiency vs. budget.

**Risk mitigation:** Pilot with 5–6 atoms, 2–3 compositions, ONE model first. If Δ(D) shows up in pilot → commit to full build. If not → rethink before investing weeks.

### Phase 5: Paper (Concurrent with Phase 4)

**Deliverables:** Draft using the three-pillar framing (§6). Frame results: compositionality gap as primary diagnostic. Discussion connecting to epiplexity and computational mechanics.

---

## §11 Scope Cuts for NeurIPS

The following are explicitly deferred:

| Item | Reason |
|------|--------|
| Cellular automata / discrete state machines | Adds complexity without strengthening Δ(D) story |
| Tier 3 mixed discrete-continuous problems | Engineering cost too high |
| Full ε-machine computation for Cμ | Use Lyapunov-based proxies; ε-machines for journal version |
| Cryptographic hardness guarantees | Related work mention only |
| Φ(n) transfer curves | Replaced by Δ(D) as primary metric |
| Parameter modulation coupling | Add after drive and bidirectional working |
| PREDICT-LONG (invariant measure) | Strong motivation from epiplexity paper but not essential for Δ(D) story |
| Discrete maps beyond debugging | Focus on continuous dysts systems |
| Web platform | After baseline experiments prove concept |

**Scope for NeurIPS submission:** ~20 continuous atoms, drive + bidirectional coupling, anti-hash validation, black box + answer schema + automated verification, 3 frontier models on ~50–100 problems, compositionality gap Δ(D) as primary result.

---

## §12 Open Questions

### 12.1 Cheap Epiplexity Estimation

Prequential coding is too expensive for real-time use. Possible proxies:
- Permutation entropy × (1 − |ACF(1)|)
- Return map smoothness (R² from polynomial fit)
- For NeurIPS: precompute on frozen bank, speed doesn't matter. Only critical for live arena.

### 12.2 Does Epiplexity Predict Transfer?

If per-problem epiplexity correlates with per-problem contribution to superlinear Φ(n), that's a novel empirical result connecting Finzi (2026) and ChaosBench. Deferred — interesting but Δ(D) is the primary story.

### 12.3 Weighting Δ(D) by Epiplexity vs. h_KS

Unweighted Δ(D) is the clean baseline. Weighted variants are follow-up analysis.

### 12.4 When to Add PREDICT-LONG?

Purest test of epiplexity extraction. LLMs can learn invariant measures even when point prediction fails (Finzi Figure 11). Already in PRD §4.4. Strong candidate for post-NeurIPS.

### 12.5 Reverse Ordering Experiment

Show question first, data second (vs. current data-first). Free experimental condition from the epiplexity paper's factorization-order result (Theorem 13). Include if time permits.

### 12.6 What Is T for LLMs?

The epiplexity framework parameterizes everything by compute budget T. For LLMs at inference, T is determined by architecture (fixed weights, context length). Treat as latent variable — ChaosBench measures the observable consequences via query efficiency. Discussion chapter material, no implementation needed.

### 12.7 Closed vs. Open Vocabulary (Decided)

Closed vocabulary for NeurIPS: agent picks from ~20 known atoms. Coupling identification compensates for constrained search. Needs pilot data to validate that single-atom ID isn't trivial (>90% accuracy with 50 queries would mean baseline is too easy, but Δ(D) story is clean).

### 12.8 Computing Q* (Decided)

Use statistical estimation bounds for NeurIPS ("queries needed for MLE to distinguish most confusable atom pairs at coupling strength ε"). Defer ε-machine approach to journal.

### 12.9 Relationship to v2 Results

Gemini results (36% CLASSIFY, 55% IDENTIFY, 79% PREDICT on 1D maps) could serve as motivation for NeurIPS paper ("1D maps partially solvable, need harder problems"). Morgan decides whether v2 results appear in the paper.

### 12.10 dysts Reliability (Decided)

Write custom `ComposedSystem` on `scipy.solve_ivp`. Use dysts only for atom definitions, not integration. De-risks dysts dependency.

### 12.11 Cost Management (Decided)

Pilot with 5–6 atoms, 2–3 compositions, one model first. If Δ(D) shows up → commit to full build. Web platform solves scalability long-term.
