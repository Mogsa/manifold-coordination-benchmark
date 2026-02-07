# ChaosBench NeurIPS Vision

**Status:** Aspirational — ideas beyond the current MVP scope defined in the PRD (`docs/Chaos_IMO` §1.1).
**Last Updated:** 2026-02-07

---

## The Core Idea

ChaosBench gives an agent a black box hiding a composed dynamical system. The agent queries it strategically, then identifies the generating structure. Difficulty is calibrated by where the system sits in the complexity-entropy plane and scaled by composition depth. Success is measured by query efficiency relative to information-theoretic bounds. The headline result is the **compositionality gap**: models identify isolated systems but fail compositions, and this doesn't improve with scale.

Reasoning is the process of finding the simple generating structure underneath complex observations. In dynamical systems terms: the attractor is low-dimensional, the time series looks high-dimensional. The act of reasoning is compressing the observation into the generator. That's not a metaphor — it's literally what Kolmogorov complexity and ε-machines formalize. ChaosBench measures whether LLMs can do this compression, and where they fail.

---

## §0 Relationship to Other Documents

| Document | Role |
|----------|------|
| `docs/Chaos_IMO` (PRD) | Implementation spec for the current MVP: 1D maps, static benchmark, Φ(n) transfer curves. Defines Phases 1–6. |
| `docs/research/epiplexity-integration.md` | Deep analysis of the epiplexity paper (Finzi et al., 2026) and how it connects to ChaosBench. Theoretical grounding lives there; this document references it, doesn't duplicate it. |
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

### Why It's the Right Metric

Of the ~80 papers surveyed in our literature review, the compositionality gap is the most robust empirical finding: Press et al. showed models solve sub-problems but fail compositions and this doesn't improve with scale. OMEGA confirmed it for scientific reasoning. The inequality composition paper confirmed it for formal provers. Potosnak & Challu (arXiv:2502.06037) formally defined compositional reasoning in time series and showed probe transfer error increases 2-5x on composite data. The PNAS subprocess decomposition paper (DOI:10.1073/pnas.2408134121) showed even humans need specialized mechanisms for compositional transfer in dynamical systems.

- **One-sentence contribution:** "LLMs can identify isolated dynamical systems but fail when you compose them."
- Sharper than "LLMs struggle with chaotic systems" — it isolates *compositional reasoning* as the failure mode.
- First benchmark to systematically measure compositional failure in dynamical systems reasoning.
- Distinct from standard compositional generalization tests (COGS, SCAN) because it involves time-series dynamics, not static symbolic structures.

### Why It Matters

The compositionality gap should decrease with composition depth D but NOT necessarily with model scale. If Δ(D) stays low even for frontier models, composition is a fundamental reasoning barrier — not just a scaling problem. This gives ChaosBench longevity that saturating benchmarks (MMLU, AIME) lack — you just increase composition depth.

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

The current MVP uses 4 discrete 1D maps (logistic, tent, damped_linear, rotation). The evolved vision expands to ~20 continuous dynamical systems from the [dysts library](https://github.com/williamgilpin/dysts) (~200 known systems, 135 continuous flows + 26 discrete maps, each with precomputed Lyapunov spectra, Pesin entropy, KY dimension, and correlation dimension).

### Why Continuous Systems Over Automata

Four arguments for focusing on continuous ODEs as the primary problem type:

1. **Practical:** dysts exists. 135 systems with precomputed metadata. The infrastructure is ready. Automata would need ε-machine pipelines, algebraic testing frameworks, and composition semantics built from scratch.

2. **Theoretical:** Continuous systems give a continuous difficulty dial. Coupling strength ε goes from 0 (trivial decomposition) to large (fully mixed → hash). Parameters slide through bifurcations. Automata give a discrete space — 256 ECA rules, finite compositions. The continuous parameter space is what makes the benchmark inexhaustible and contamination-proof, and guessproof in the FrontierMath sense: "what is the coupling strength?" has answer ε=0.0847... to arbitrary precision.

3. **Reasoning-test:** Continuous systems force quantitative scientific reasoning — estimation, error analysis, hypothesis refinement through measurement. This is closer to actual scientific reasoning than the algebraic/combinatorial reasoning automata test. When a scientist encounters an unknown system, they measure things, form hypotheses, design targeted experiments.

4. **Composition:** Lorenz driving Rossler through weak coupling is a composition. The agent must identify component count (dimension analysis), component identities (spectral analysis, attractor geometry), coupling architecture (transfer entropy), and coupling strength (perturbation analysis). Each step is a deductive link in a chain. The deductive ladder works perfectly with continuous systems.

### Selection Strategy: Spanning the Complexity Spectrum

**Low chaos** (~4–5 systems):
- SprottTorus (λ_max ≈ 0.02), Rossler (λ_max ≈ 0.07), Thomas (λ_max ≈ 0.03), ForcedVanDerPol, Duffing
- Clear spectral signatures, near-periodic behavior, recognizable attractor geometry
- Sanity-check baseline

**Moderate chaos** (~5–6 systems):
- Lorenz (λ_max ≈ 0.9), Chua (λ_max ≈ 0.4), HastingsPowell (λ_max ≈ 0.1), Halvorsen, NoseHoover
- Broadband spectra but distinctive topology
- Core difficulty range for discriminating models

**Strong chaos** (~3–4 systems):
- Chen (λ_max ≈ 2.0), BurkeShaw (λ_max ≈ 2.3), QiChen (λ_max ≈ 3.9), Dadras
- Short prediction horizons, statistics-only identification
- Tests limits of structural extraction

**Special structure** (~3–4 systems):
- DoublePendulum or HenonHeiles (Hamiltonian — conserved energy is an exploitable signal that prevents hash degradation)
- MackeyGlass (delay signature in autocorrelation)
- Lorenz96 (spatiotemporal, scalable dimension)
- Each has a unique structural invariant that survives composition (critical for anti-hash)

**Discrete maps** (5–6 for debugging):
- Logistic, Tent, Henon, Ikeda, Tinkerbell
- Simpler, faster, good for initial development and sanity checks

### Selection Rationale

- Span the complexity-entropy plane (Cμ, hμ) from low to high h_KS.
- Each has distinctive signatures (power spectrum, attractor shape, Lyapunov spectrum).
- Mix of well-known (Lorenz) and less-memorizable systems.
- Special structures ensure exploitable invariants survive composition.

### Precomputed Metadata Per Atom

Stored as JSON:
- Lyapunov spectrum (dysts provides this)
- h_KS via Pesin's identity (sum of positive Lyapunov exponents)
- Attractor dimension (Kaplan-Yorke + correlation dimension)
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

Write custom `ComposedSystem` on top of `scipy.solve_ivp` from day one. Use dysts only for atom definitions (equations + known Lyapunov spectra), NOT for integration. This avoids dysts SkewProduct integration issues (stiff ODEs, chaotic trajectory divergence at strong coupling) and gives full control over numerics. dysts's SkewProduct class exists but has a bug (returns None for trajectories) — bypassing it entirely de-risks the dependency.

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

### What an Agent Actually Does (The Deductive Ladder)

For a composed continuous system (e.g. Lorenz driving Rossler at ε=0.1), a reasoning agent should follow a chain of deductions where each step uses the result of the previous one:

1. **Timescale separation:** Power spectrum shows two distinct frequency bands → "there are two oscillators."
2. **Dimension decomposition:** Attractor dimension ≈ sum of component dimensions for weak coupling → "dim ≈ 4.1 suggests ~2.06-dimensional (Lorenz-like?) + ~2-dimensional component."
3. **Directional information flow:** Transfer entropy from subsystem A to B is high but B→A is low → "A drives B, not vice versa."
4. **Lyapunov spectrum decomposition:** Spectrum is approximately the union of component spectra with perturbative corrections → identify how many positive exponents each subsystem contributes.
5. **Periodic orbit skeleton:** Unstable periodic orbits have periods and stability eigenvalues relating to component orbits.
6. **Candidate testing:** Query specific initial conditions to confirm/reject hypotheses. Check if estimated parameters reproduce the observed statistics.

This chain is what makes the benchmark test genuine multi-step reasoning, not just "query a lot and pattern match." Each step mathematically depends on the previous one.

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

### Connection to AlphaProof / Test-Time Search

The relevant idea from AlphaProof isn't RL (that's for training) — it's test-time search: generating millions of variants and searching over strategies. ChaosBench should be designed so that brute-force search over parameter space is expensive (guessproof — dynamical systems give this for free) but structured search guided by understanding is dramatically more efficient. The ratio of structured-search-cost to brute-force-cost is itself a measure of how much reasoning helps.

### Relationship to Current PRD

The current PRD (§8.4) describes a tool-augmented mode with tools like COMPUTE_STATS, ITERATE, LYAPUNOV_ESTIMATE. The black-box vision simplifies this to a single primitive (`query`) and lets the agent decide what statistics to compute. More principled, more general.

### Differentiation from PhysGym

PhysGym (Chen et al., COLM 2025, arXiv:2507.15550) is the closest existing benchmark: interactive physics discovery from black-box environments with query budgets and 4 levels of prior knowledge. ChaosBench differs in three ways: (1) composition focus — PhysGym discovers single equations, ChaosBench identifies composed systems from a known vocabulary; (2) information-theoretic calibration — ChaosBench has ground-truth complexity measures (h_KS, Cμ); (3) the compositionality gap Δ(D) as primary metric, which PhysGym doesn't measure.

---

## §5 Anti-Hash Principle

### Core Principle

> **Good problems have heterogeneous structure.** Components with different complexity levels, different algebraic properties, or different timescales create information gradients that a reasoning agent can exploit sequentially. Homogeneous compositions (all components similar and fully chaotic) degrade toward hashes. It's the diversity of the components, not the depth of the composition, that makes a problem rich.

Every problem must preserve at least one exploitable structural invariant.

### Why It Matters

Prevents the benchmark from being trivially impossible. Ensures problems test reasoning about *structure*, not tolerance for pure noise. A composed system whose output looks like white noise is a hash function, not a reasoning challenge. A problem with zero epiplexity (no extractable structure for any bounded observer) is a hash. The anti-hash principle is just: every problem must have positive epiplexity. The exploitable invariants ARE the epiplexity.

### When Composition Degrades to Hash

- **Continuous systems:** Composing multiple high-entropy systems with strong coupling. Output converges to the maximum entropy measure on the attractor. If autocorrelation drops to zero within 1-2 timesteps and invariant measure is approximately uniform → washed out all structure.
- **Automata:** Composing multiple Class III rules. Output converges to i.i.d. uniform bits. Cμ → 0, hμ → log 2.
- **Detection:** MI between composition tree and output statistics ≈ 0 → it's a hash.

### Concrete Examples

**Bad (hash-like):** Rule 30 ∘ Rule 30 ∘ Rule 30 applied sequentially, 100 timesteps each. Output is effectively pseudorandom. No algebraic structure survives. The agent can't do better than enumeration. Tests nothing.

**Good (structured, automata):** Rule 90 (linear) applied to Rule 110 (Class IV, Turing-complete). Rule 90's linearity is partially preserved — superposition tests show structured violations that encode information about Rule 110's glider dynamics. The agent can: (1) detect linearity violation → "nonlinear component feeding the linear one," (2) characterize violation pattern → "nonlinearity produces persistent structures on these timescales," (3) test candidate rules → "Rule 110 produces exactly these glider collisions." Each step uses the previous result.

**Good (structured, continuous):** Logistic map at r=3.56 (near period-3 window) driving Hénon map through x-coupling, ε=0.05. The logistic near the bifurcation has intermittent behavior — long laminar phases interrupted by bursts. This temporal structure survives the coupling and modulates the Hénon dynamics. Agent can: (1) detect intermittency → "component near a bifurcation," (2) during laminar phases, characterize Hénon in isolation (driving nearly constant) → "driven component looks like Hénon with these parameters," (3) during bursts, estimate coupling strength from Hénon perturbation. The bifurcation creates a natural experiment within the data.

### The Inverse Principle (Non-Negotiable Design Constraint)

> **You cannot add a composition type to the benchmark until you can name its inverse.** Every coupling mechanism must be paired with a concrete decomposition technique that a reasoning agent could, in principle, apply to recover the components. If no inverse exists, the composition is a hash and has no place in the benchmark.

This is the operational rule that keeps the benchmark in check as the atom vocabulary scales from 4 to 20+. The atom count doesn't matter — what matters is that every *composition* has a documented deductive path back to its building blocks.

### Composition–Decomposition Pairing Table

| Composition Type | Inverse (Decomposition Technique) | What It Recovers |
|---|---|---|
| Unidirectional drive (A→B) | **Transfer entropy** — information flows A→B but not B→A | Coupling direction |
| Two coupled oscillators | **Power spectrum separation** — distinct frequency bands from different natural frequencies | Component count, approximate identities |
| Weak coupling (small ε) | **Lyapunov spectrum decomposition** — combined spectrum ≈ union of component spectra | Component count (count positive exponents), individual entropy estimates |
| Component near bifurcation | **Intermittency analysis** — laminar phases where driving is nearly constant, so driven system is characterizable in isolation | Driven component identity and parameters |
| Hamiltonian component | **Conservation law detection** — energy or other invariant stays constant, survives composition | Hamiltonian component identity |
| Different-dimensional components | **Embedding dimension analysis** — false nearest neighbors reveals total attractor dimension, which roughly adds for weak coupling | Component count, approximate dimensions |
| Parameter modulation (A varies B's params) | **Sliding-window bifurcation detection** — B's local statistics shift over time as A's output changes | Coupling type, modulation timescale |

If a new coupling type is proposed and no row can be added to this table, the coupling type is not ready for the benchmark.

### Difficulty = Chain Length, Not Signal Destruction

The pairing table clarifies what "hard" means in ChaosBench. A depth-2 problem with three clean inverses is harder than a depth-0 problem — but it's hard because the agent must execute a multi-step deductive chain (detect frequency separation → estimate dimensions → match against vocabulary → estimate coupling direction → estimate coupling strength), not because the signal has been destroyed. Each step in the chain is individually doable. The composition is where models break.

This is the core distinction: **difficulty scales with the number of deductive steps, not with entropy.** A high-entropy problem with no exploitable structure is trivially impossible — it's a hash. A moderate-entropy problem with a 5-step deductive ladder is genuinely hard. The benchmark must produce the latter, never the former.

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
   - **TEPC** (Topology-Enabled Predictions from Chaos, arXiv:2503.14956): topological methods that achieve predictions where traditional methods fail. Validated on Lorenz/Rossler. Potentially stronger baseline than SINDy/ESN.
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

### Why Only Three

The literature survey covered ~80 papers. After brutal distillation, only three ideas do real theoretical work for ChaosBench. Everything else — cryptographic hardness, scaling laws, edge-of-chaos, neural cryptanalysis — is related work or discussion section decoration. Cryptographic hardness is theoretically beautiful but practically unnecessary (we don't need to *prove* problems are hard; we need to show models fail in informative ways). Scaling laws are interesting for understanding *why* models fail but not actionable for benchmark design. The edge-of-chaos finding is about pretraining data, not evaluation.

For a deep treatment of epiplexity, see `docs/research/epiplexity-integration.md`.

### Pillar 1: Epiplexity (Observer-Dependent Extractable Structure)

From Finzi et al. (2026, arXiv:2601.03220v1):

- For computationally bounded observers, information splits into **time-bounded entropy** H_T(X) (irreducibly random) and **epiplexity** S_T(X) (learnable structure).
- `MDL_T(X) = S_T(X) + H_T(X)`
- Key insight: the same data can have different epiplexity depending on the observer's compute budget T.

**How epiplexity concretely helps ChaosBench (three ways, and only three):**

1. **Justifies query efficiency as the right metric.** More queries = higher effective T = more extractable epiplexity. The query-efficiency curve IS the empirical epiplexity curve for that model class. We don't need to compute epiplexity directly — we measure it through the benchmark.

2. **Explains why composition is fundamentally hard.** Epiplexity of a composed system for a bounded observer isn't additive — the observer may extract less structure from A+B than from A and B separately. Δ(D) measures exactly this non-additivity.

3. **Gives theoretical backing for anti-hash.** A problem with zero epiplexity is a hash. The exploitable invariants ARE the epiplexity. The anti-hash principle is just: every problem must have positive epiplexity.

The other connections in the epiplexity paper (factorization order → reverse ordering experiment, PREDICT-LONG → invariant measure) are interesting follow-ups but not load-bearing for the NeurIPS story.

### Pillar 2: Computational Mechanics (Ground-Truth Difficulty)

- **Statistical complexity Cμ:** Minimum memory to optimally predict.
- **Entropy rate hμ (≈ h_KS for dynamical systems):** Rate of unpredictability.
- Each problem has known h_KS via Pesin's identity → objective difficulty baseline.
- Problems span the complexity-entropy plane (Cμ, hμ).

Marzen, Riechers & Crutchfield (Scientific Reports, 2024) is the template: they generated processes at known (Cμ, hμ) coordinates and showed reservoir computers fail systematically in specific regions. ChaosBench does the same with LLM agents and composed dynamical systems.

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

## §7 Complexity Measurement and Scoring

### The Three-Number Problem Characterization

Since we construct every system ourselves, we know the ground truth. Every ChaosBench problem is characterized by three numbers:

| Symbol | Name | What It Measures | How Computed |
|--------|------|-----------------|--------------|
| **L** | Description length | Structural complexity of the true composition tree | Count of nodes/edges/parameters in our vocabulary. E.g., `drive(Lorenz(σ=10,ρ=28,β=2.667), Rossler(a=0.2,b=0.2,c=5.7), ε=0.05)` ≈ 10 parameters. |
| **h** | KS entropy | Noise floor (irreducible unpredictability) | Pesin's identity: h_KS = Σ positive Lyapunov exponents. Computed numerically. |
| **Q*(δ)** | Minimum queries | Theoretical minimum queries for optimal identifier to achieve fidelity δ | Statistical estimation bound: "queries needed for MLE to distinguish most confusable atom pairs at coupling strength ε." |

### Per-Agent Metrics

For each agent on each problem:

| Metric | Definition | What It Tests |
|--------|-----------|---------------|
| **Q_agent(δ)** | Queries the agent actually used | Raw cost |
| **L_agent** | Description length of agent's proposed model | Model complexity |
| **Q_agent / Q*** | Query efficiency | How close to optimal identification? Directly operationalizes time-bounded complexity. |
| **L_agent / L** | Model parsimony | Did it find the minimal description or overfit? Motivated by epiplexity: bounded observers learn models MORE complex than generators. |
| **Gap_agent** | cost(agent) − synchronization lower bound | How far from a perfect reasoner? Grows with D for brute-forcers, stays bounded for genuine reasoners. |

### Connection to Formal Theory

Think of the agent's query sequence as a "program" running on the black box ("universal machine"). Total cost = queries × cost-per-query + computation between queries = time bound t. The agent's proposed model is program p. We're measuring:

```
Kt_agent(system) = min over strategies {cost(strategy) : strategy produces model M with d(M, true) < δ}
```

This is directly analogous to time-bounded Kolmogorov complexity Kt — shortest program within time t — but fully computable because we control both the systems and the budget.

### Computable Complexity Proxies (Ranked by Practicality)

1. **Lempel-Ziv complexity** of observation sequence: computable upper bound on Kolmogorov complexity. Fast, well-understood. But conflates signal and noise — a simple system with high h_KS produces incompressible output.

2. **Statistical complexity Cμ** from ε-machines: measures the structural part only. Computable via CSSR algorithm. Best single proxy. The complexity-entropy plane (Cμ vs. hμ) gives a 2D decomposition. Cleanest for low-dimensional systems; harder for 6D+ compositions.

3. **MDL-based compression via neural networks** (Blier & Ollivier style): train a small network to predict the system, measure total codelength. Different model sizes N give different time bounds → trace the Kt curve as function of t by varying N.

### Scoring Evolution from Current PRD

The current scoring uses `composite_difficulty = (1 + h_KS) × (1 + depth) × (1 + 10σ)` and `weighted_score = raw × difficulty`. The evolved vision replaces this with the three-number framework above, adding query efficiency and model parsimony as orthogonal axes.

### Transfer Ratio (Deferred)

The current PRD's Φ(n) transfer curves are replaced by Δ(D) as the primary metric. Transfer ratio could return in a journal version for multi-task training experiments, potentially weighted by epiplexity instead of composite difficulty.

---

## §8 Web Platform (Building Separately)

### Vision

Avoid massive API costs by leveraging users' existing LLM subscriptions. A user with a Claude or ChatGPT subscription visits the ChaosBench website. Their LLM, via tool-calling, interacts with the website: reads the problem, calls the query API for trajectories, submits an answer. The website verifies and records results. Crowdsourced leaderboard of model performance.

**Status:** Morgan is building this as a separate project. The web platform is not part of the core benchmark codebase but depends on it.

### Architecture

- Hosted on Morgan's Linux server.
- **Problem browser:** Select from validated problem bank.
- **Query endpoint:** `POST /query { problem_id, x0, T }` → trajectory.
- **Submit endpoint:** `POST /submit { problem_id, answer }` → verification result.
- **Frontend:** Leaderboard, per-problem stats, model comparison.
- **Backend:** Problem bank, ComposedSystem simulator, verification logic.

### How It Works for Users

1. User goes to ChaosBench website.
2. Selects a problem from the bank.
3. Copies the system prompt + tool spec into their ChatGPT/Claude/etc. chat.
4. Their LLM uses tool-calling to query the API for trajectories.
5. LLM submits structured answer.
6. Website verifies and records to leaderboard.

### Advantages

- Costs distributed across users (no central API bill).
- Tests models beyond the 2–3 we can afford to run ourselves.
- Community engagement and iterative improvement.
- Enables long-term tracking of model improvements.
- Natural extension of the black-box querying paradigm.

### Challenges

- Authentication (prevent spam/gaming).
- Rate limiting (prevent DoS).
- Don't leak ground truth through query patterns.
- Ensure queries are legitimate, not adversarial.

### Dependencies

Requires working ComposedSystem, validation pipeline, and black-box interface. Not for the initial NeurIPS submission — after baseline experiments prove the concept. Inspired by LiveBench's automatic refresh principle: chaotic systems give automatic contamination resistance (new parameters, new compositions, new initial conditions, all with computable ground truth — no human in the loop).

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

- **`num_components`:** Integer (2 or 3 for MVP). First-order check — did the agent recognize composition depth?
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

Prediction: with 20-system vocabulary and 50 queries, single-atom ID accuracy for Sonnet should be >90%. If so, baseline is clean and Δ(D) story works. If <50%, different paper.

---

## §10 Future: Automata & Tiered Problem Structure

### Why Automata Are Interesting (Post-NeurIPS)

Automata sit at the exact intersection of ChaosBench's core ideas. An ECA rule is 8 bits that can produce outputs spanning the full complexity spectrum from trivial (Rule 0) to provably computationally irreducible (Rule 110). ε-machines were literally developed to analyze automata — the theory is strongest here. Zhang et al. (ICLR 2025) showed GPT-2 pretrained on Class IV CAs developed better downstream reasoning than Class I/II or Class III.

### What Automata Test vs. Continuous Systems

| Aspect | Continuous Systems | Automata |
|--------|-------------------|----------|
| Primary reasoning mode | Estimation and decomposition (quantitative) | Hypothesis testing and algebraic deduction (logical) |
| Exploitable structure | Geometry (smoothness, curvature, derivatives, invariant measures) | Algebra (linearity over GF(2), symmetries, light cone structure) |
| Difficulty dial | Continuous (coupling ε, parameter values) | Discrete (256 rules, finite compositions) |
| Ground truth | Real-valued, approximate | Exact, discrete |
| Numerical issues | Stiff ODEs, precision loss | None |

### Concrete Example: What an Agent Sees

For composed automata (Rule 90 on Rule 30): Agent queries with initial configurations, gets binary spacetime diagrams. Can test:
- **Algebraic residues:** XOR two initial conditions. If outputs XOR → linear component. Structured violations → nonlinear component partially breaking linearity.
- **Light cone geometry:** Information propagation speed reveals the rule's local structure.
- **Statistical fingerprints:** Rule 30's known biases transformed by the outer rule in rule-specific ways.

### Tiered Problem Structure (Long-Term Vision)

**Tier 1 (discrete, exact):** Composed automata. ECA rules composed through driving/coupling/boundary conditions. All complexity measures exact. Tests pure structural reasoning without numerical noise.

**Tier 2 (continuous, approximate):** Composed continuous systems from dysts. Tests the same reasoning but with measurement noise, finite precision, and continuous parameter identification.

**Tier 3 (mixed):** An automaton driving a continuous system, or a continuous system's symbolic dynamics generating an automaton-like structure. Tests whether the agent can reason across representations. If an agent succeeds on Tier 1 but fails on Tier 2, the bottleneck is numerical/continuous reasoning, not compositional reasoning. If it fails on both, the bottleneck is composition itself.

### What We Lose by Deferring Automata

Clean algebraic structure (linearity tests, exact answers), connection to Zhang et al.'s edge-of-chaos result, and the Turing-completeness angle (Rule 110). These are real losses. But some continuous systems have exact symmetries or conserved quantities that play the same role. The continuous systems story is more complete for NeurIPS; automata are a natural extension.

---

## §11 Build Sequence

### Phase 1: Atom Library + Composition Engine (Weeks 1–3)

**Deliverables:** Curate 15–20 atoms from dysts. Build `ComposedSystem` class. Precompute metadata JSON per atom.

**Test:** `ComposedSystem(Lorenz(), Rossler(), coupling='drive', epsilon=0.05)` → trajectories + metadata.

**Risk:** Numerical integration stability (stiff ODEs, divergence). If Phase 1 takes >2 weeks, timeline is blown. Pilot with 5–6 atoms FIRST to validate numerical stability.

### Phase 2: Problem Validation Pipeline (Weeks 3–4)

**Dependencies:** Phase 1.

**Deliverables:** Anti-hash validation script (§5). Curated problem set (~50–100 validated). Per-problem: pass/fail, h_KS, Cμ estimate, exploitable invariants.

### Phase 3: Black Box + Agent Protocol (Weeks 4–5)

**Dependencies:** Phases 1 and 2.

**Deliverables:** `BlackBox` class with `query(x0, T)`. Agent system prompt. Answer schema + automated verification. Query budget tiers.

**Test:** Point an LLM at the black box → it queries → submits → gets automated scores. Cost estimate: ~$1-5 per problem per model.

### Phase 4: Baseline Experiments (Weeks 5–7)

**Dependencies:** Phase 3.

**Deliverables:** Run Claude Sonnet, GPT-4o, Gemini. Start with single atoms (sanity check), then D=2, then D=3. Measure Δ(D). Plot performance vs. h_KS, depth, coupling strength. Plot query efficiency vs. budget.

**Risk mitigation:** Pilot with 5–6 atoms, 2–3 compositions, ONE model first. If Δ(D) shows up in pilot → commit to full build. If not → rethink before investing weeks.

### Phase 5: Paper (Concurrent with Phase 4)

**Deliverables:** Draft using the three-pillar framing (§6). Frame results: compositionality gap as primary diagnostic. Discussion connecting to epiplexity and computational mechanics.

---

## §12 Scope Cuts for NeurIPS

The following are explicitly deferred:

| Item | Reason |
|------|--------|
| Automata / cellular automata / Tier 1 | Post-NeurIPS extension (see §10) |
| Tier 3 mixed discrete-continuous | Engineering cost too high |
| Full ε-machine computation for Cμ | Use Lyapunov-based proxies; ε-machines for journal version |
| Cryptographic hardness guarantees | Related work mention only |
| Φ(n) transfer curves | Replaced by Δ(D) as primary metric |
| Parameter modulation coupling | Add after drive and bidirectional working |
| PREDICT-LONG (invariant measure) | Strong motivation from epiplexity paper but not essential for Δ(D) story |
| Discrete maps beyond debugging | Focus on continuous dysts systems |
| Web platform | Building separately; after baseline experiments prove concept |
| Epiplexity computation | Use query efficiency as proxy; direct computation for journal version |

**Scope for NeurIPS submission:** ~20 continuous atoms, drive + bidirectional coupling, anti-hash validation, black box + answer schema + automated verification, 3 frontier models on ~50–100 problems, compositionality gap Δ(D) as primary result.

---

## §13 Key Related Work

Papers that directly inform ChaosBench's design (beyond the three pillars in §6).

### Must-Cite (Differentiate From)

| Paper | Why It Matters |
|-------|---------------|
| **PhysGym** (Chen et al., COLM 2025, arXiv:2507.15550) | Closest existing benchmark. Interactive physics discovery from black-box environments with query budgets. ChaosBench differs: composition focus, information-theoretic calibration, Δ(D) metric. |
| **Potosnak & Challu** (arXiv:2502.06037, Feb 2025) | First formal definition of compositional reasoning in time series. Shows TSFMs struggle to disentangle mixtures. Validates that Δ(D) is measuring a real phenomenon. |
| **Gilpin, dysts** (NeurIPS 2021) | Our atom library. 135 systems with precomputed metadata. |
| **Press et al.** (EMNLP Findings 2023) | Defined the compositionality gap; showed it doesn't improve with scale. |
| **Marzen et al.** (Sci. Reports 2024) | Complexity-calibrated benchmarks using ε-machines. Our template for difficulty calibration. |

### Strong Supporting Evidence

| Paper | Why It Matters |
|-------|---------------|
| **PNAS subprocess decomposition** (DOI:10.1073/pnas.2408134121, Nov 2024) | Mechanistic evidence that humans extract and transfer subprocess knowledge compositionally in dynamical systems. Validates the compositional decomposition hypothesis. |
| **TEPC** (arXiv:2503.14956, J. Royal Soc. Interface, March 2025) | Topology-Enabled Predictions from Chaos. Topological methods predict where traditional methods fail. Potential strong non-LLM baseline for anti-hash validation. |
| **OMEGA** (Sun et al., arXiv:2506.18880, Allen AI/UC Berkeley, 2025) | Confirms compositionality gap in scientific reasoning across three generalization axes. Fine-tuning helps exploratory but not compositional generalization. |
| **ABench-Physics** (arXiv:2507.04766, July 2025) | Dynamic problem variation engine. 22.5% performance drop static→dynamic. Validates anti-memorization design. |

### Related Work (Cite but Don't Build On)

- **ARC-AGI-2** (Chollet, arXiv:2505.11831): Efficiency metrics, compositional reasoning frontier-hard.
- **FrontierMath** (Glazer et al., arXiv:2411.04872): Guessproof answers. Dynamical systems naturally guessproof.
- **LiveBench** (White et al., ICLR 2025 Spotlight): Automatic refresh with objective ground truth. ChaosBench gets this for free via parameter spaces.
- **Stanford HAI Benchmark Saturation** (2025): Justifies why new benchmarks are needed.
- **"Why LLMs Aren't Scientists Yet"** (arXiv:2601.03315): Failure mode taxonomy (false confidence, memory degradation).
- **FEM-Bench** (arXiv:2512.20732): Scientific code generation benchmark.
- **Zhang et al. edge-of-chaos** (ICLR 2025): Intermediate-complexity data develops better reasoning. One sentence in discussion.
- **Neurosymbolic AI** (PNAS Nexus 2025): Compositional reasoning more efficient with symbolic methods.
- **Nature compositional neural subspaces** (2025, DOI:10.1038/s41586-025-09805-2): Brains compose via dynamical motifs.

---

## §14 Open Questions

### 14.1 Cheap Epiplexity Estimation

Prequential coding is too expensive for real-time use. Possible proxies:
- Permutation entropy × (1 − |ACF(1)|)
- Return map smoothness (R² from polynomial fit) — directly measures "can simple model extract structure?"
- For NeurIPS: precompute on frozen bank, speed doesn't matter. Only critical for live arena/web platform.

### 14.2 Does Epiplexity Predict Transfer?

If per-problem epiplexity correlates with per-problem contribution to superlinear Φ(n), that's a novel empirical result connecting Finzi (2026) and ChaosBench. Deferred — interesting but Δ(D) is the primary story.

### 14.3 Weighting Δ(D) by Epiplexity vs. h_KS

Unweighted Δ(D) is the clean baseline. Weighted variants are follow-up analysis. Same principle applies: could weight by epiplexity vs. by h_KS to separate "hard because noisy" from "hard because structurally complex."

### 14.4 When to Add PREDICT-LONG?

Purest test of epiplexity extraction. LLMs can learn invariant measures even when point prediction fails (Finzi Figure 11). Already in PRD §4.4. Strong candidate for post-NeurIPS. Medium effort, high theoretical value.

### 14.5 Reverse Ordering Experiment

Show question first, data second (vs. current data-first). Free experimental condition from the epiplexity paper's factorization-order result (Theorem 13). Include if time permits.

### 14.6 What Is T for LLMs?

The epiplexity framework parameterizes everything by compute budget T. For LLMs at inference, T is determined by architecture (fixed weights, context length). Treat as latent variable — ChaosBench measures the observable consequences via query efficiency. Different models have different effective T, which should produce different epiplexity extraction. This connects to why larger/better models should show more transfer. Discussion chapter material, no implementation needed.

### 14.7 Testing Static LLMs vs. Learning Systems

We're testing static LLMs at inference time. We can't measure learning. We CAN measure:
- **In-context reasoning efficiency:** Given N queries, how much generating structure can you recover? Smarter agents synchronize faster.
- **Transfer within a session:** If you've characterized system A and now get conjugate system B, do you recognize the relationship?
- **Compositional deduction:** Given you know the atoms, can you identify the composition?

The key insight: we're not testing whether they CAN learn to reason about chaos. We're testing whether their pre-trained representations already contain the right abstractions to decompose novel compositions on the fly. That's testing the quality of the prior, not the learning algorithm. In the limit, a future central AI would learn over time and get better — but given current paradigms, how well can we test them?

### 14.8 Closed vs. Open Vocabulary (Decided)

Closed vocabulary for NeurIPS: agent picks from ~20 known atoms. Coupling identification compensates for constrained search. Needs pilot data to validate that single-atom ID isn't trivial.

### 14.9 Computing Q* (Decided)

Use statistical estimation bounds for NeurIPS ("queries needed for MLE to distinguish most confusable atom pairs at coupling strength ε"). Precompute pairwise distinguishability for all atom pairs. For compositions, add coupling uncertainty. Q* = f(distinguishability, coupling_strength, noise). Defer ε-machine approach to journal.

### 14.10 Relationship to v2 Results

Gemini results (36% CLASSIFY, 55% IDENTIFY, 79% PREDICT on 1D maps) could serve as motivation for NeurIPS paper ("1D maps partially solvable, need harder problems"). Morgan decides whether v2 results appear in the paper.

### 14.11 dysts Reliability (Decided)

Write custom `ComposedSystem` on `scipy.solve_ivp`. Use dysts only for atom definitions, not integration. De-risks dysts dependency. dysts's SkewProduct class has a bug (returns None) — bypass entirely.

### 14.12 Cost Management (Decided)

Tool-calling format is expensive: 3 models × 100 problems × ~50 queries/problem. Could run $50-200+ per model per full run. Pilot with 5–6 atoms, 2–3 compositions, one model first. If Δ(D) shows up → commit to full build. Web platform solves scalability long-term.
