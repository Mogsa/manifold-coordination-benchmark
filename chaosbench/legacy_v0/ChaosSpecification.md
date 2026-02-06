# ChaosBench: A Rigorous Benchmark for Multi-Agent Reasoning via Chaotic Dynamical Systems

## Complete Specification Document

**Version**: 0.3 (Prototype)  
**Last Updated**: January 2026  
**Status**: Active Development — Dissertation Project

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Core Innovation: Why Chaotic Systems?](#3-core-innovation-why-chaotic-systems)
4. [The Φ(t) Evaluation Paradigm](#4-the-φt-evaluation-paradigm)
5. [Task Specification](#5-task-specification)
6. [Scoring and Correctness](#6-scoring-and-correctness)
7. [Observability, Cost, and Agency](#7-observability-cost-and-agency)
8. [Conditional vs Unconditional Settings](#8-conditional-vs-unconditional-settings)
9. [Multi-Agent Hypothesis](#9-multi-agent-hypothesis)
10. [Design Decisions and Rationale](#10-design-decisions-and-rationale)
11. [Known Limitations and Open Problems](#11-known-limitations-and-open-problems)
12. [Implementation Status](#12-implementation-status)
13. [Future Roadmap](#13-future-roadmap)
14. [Appendix: Mathematical Background](#appendix-mathematical-background)

---

## 1. Executive Summary

ChaosBench is a benchmark for evaluating reasoning systems (single agents, multi-agent collectives, or any computational solver) on prediction tasks over partially observable chaotic dynamical systems.

### What Makes This Benchmark Different

| Property | Existing Benchmarks (ARC, etc.) | ChaosBench |
|----------|--------------------------------|------------|
| **Complexity measure** | Heuristic or undefined | Rigorous: Kolmogorov-Sinai entropy (h_KS) |
| **Dimensionality** | High (~900D for 30×30 grids) | Low (1-3D), enabling theoretical analysis |
| **Evaluation metric** | Scalar accuracy | Curve Φ(t) revealing quality-throughput tradeoffs |
| **Task space** | Finite, memorizable | Infinite, parameterized |
| **Gaming prevention** | Implicit | Explicit: h_KS weighting + mandatory completion |
| **Observability** | Fixed, free | Parameterized, costly (quantifies "agency") |

### The Core Claim

A solver that **genuinely reasons** about dynamical systems—rather than pattern-matching or exploiting statistical regularities—should exhibit **superlinear Φ(t)**: initially slow (learning the structure), then accelerating as it exploits shared structure across the task distribution.

---

## 2. Motivation and Problem Statement

### 2.1 The Problem with ARC and Variants

The Abstraction and Reasoning Corpus (ARC) and its variants suffer from fundamental limitations for rigorous analysis:

1. **High dimensionality**: 30×30 grids = 900 dimensions. This makes theoretical analysis intractable and obscures what's actually being measured.

2. **No principled complexity measure**: Task "difficulty" is defined by human intuition or solve rates, not by any information-theoretic quantity. This makes it impossible to study scaling laws rigorously.

3. **No time-bounded analysis**: ARC measures accuracy, not efficiency. A solver that takes 10 seconds per task and one that takes 0.1 seconds get the same score if both are correct.

4. **Memorization risk**: Finite task sets can be memorized. Data contamination is a persistent concern.

5. **No connection to Kolmogorov complexity**: Despite claims about measuring "intelligence" or "abstraction," there's no formal connection to time-bounded Kolmogorov complexity (see the epiplexity paper for why this matters).

### 2.2 What We Want Instead

A benchmark that:

1. Has **provably rigorous complexity grounding** via information-theoretic measures
2. Operates in **low dimensions** where theoretical analysis is tractable
3. Measures **efficiency**, not just accuracy—enabling analysis of throughput vs quality tradeoffs
4. Has an **infinite parameterized task space** preventing memorization
5. **Cannot be gamed** by solving only easy tasks
6. Enables analysis of **multi-agent collaboration** and **decentralized reasoning**

### 2.3 The Research Questions

This benchmark is designed to answer:

1. **Quality-throughput tradeoffs**: How do different architectures (single powerful model vs. mixture of models vs. decentralized agents) trade off quality and speed?

2. **Learning dynamics**: Do solvers learn the structure of the task distribution? Evidence: superlinear Φ(t).

3. **Multi-agent advantage**: Can N collaborating agents achieve Ψ(t) > N × Ψ_single(t)? (Superlinear scaling with agents)

4. **Routing and specialization**: Can learned routing functions achieve Pareto-optimal quality-cost tradeoffs?

5. **Transfer within families**: Does learning easy instances of a system family help with hard instances?

---

## 3. Core Innovation: Why Chaotic Systems?

### 3.1 Kolmogorov-Sinai Entropy as Complexity Measure

For a chaotic dynamical system, the **Kolmogorov-Sinai entropy** h_KS quantifies the rate of information production:

$$h_{KS} = \sum_{\lambda_i > 0} \lambda_i$$

where λ_i are the Lyapunov exponents (by Pesin's theorem, for smooth systems with SRB measures).

**Why this is the right measure:**

- h_KS is **intrinsic** to the dynamics—it doesn't depend on how we choose to measure difficulty
- h_KS directly determines the **prediction horizon**: after time t, prediction error grows as exp(h_KS · t)
- h_KS is **computable** (via QR decomposition along trajectories) for the systems we use
- h_KS provides a **continuous difficulty spectrum** by varying system parameters

### 3.2 The Lyapunov Time

The **Lyapunov time** τ_λ = 1/λ_max is the characteristic timescale over which predictability is lost. After ~2-3 Lyapunov times, initial uncertainty has amplified to the scale of the attractor.

**Critical implication**: Prediction horizons must be calibrated relative to τ_λ, not in absolute steps. A horizon of 10 steps means very different things for different systems.

### 3.3 Why Low-Dimensional?

We restrict to 1-3 dimensional systems because:

1. **Theoretical tractability**: Lyapunov spectra, invariant measures, and attractor geometry are well-understood
2. **Computational efficiency**: No need for expensive high-dimensional integrators
3. **Interpretability**: We can visualize trajectories, attractors, and predictions
4. **Sufficient complexity**: Even 1D maps like the logistic map exhibit rich chaotic behavior

### 3.4 System Families

The benchmark includes parameterized families:

| Family | Dimension | Parameter | h_KS Range | Notes |
|--------|-----------|-----------|------------|-------|
| Logistic map | 1D | r ∈ [3.57, 4] | [0, ln(2)] | Exactly solvable h_KS |
| Tent map | 1D | μ ∈ [1, 2] | [0, ln(2)] | Piecewise linear |
| Hénon map | 2D | a ∈ [1.2, 1.4] | ~0.3-0.5 | Classic 2D chaos |
| Standard map | 2D | K ∈ [0.5, 5] | ~0-1.5 | Hamiltonian chaos |
| Lorenz (discretized) | 3D | ρ ∈ [20, 40] | ~0.7-1.1 | Continuous-time, Poincaré section |

Varying parameters within each family creates a continuous difficulty spectrum.

---

## 4. The Φ(t) Evaluation Paradigm

### 4.1 Definition

The primary evaluation object is **not a scalar** but a **curve**:

$$\Phi(t) = \sum_{\substack{\text{tasks } \tau \text{ solved} \\ \text{by wall-clock time } t}} w(h_{KS}(\tau)) \cdot \text{Score}(\tau)$$

where:
- t is wall-clock time
- w(h_KS) is the difficulty weighting function (a dataset hyperparameter)
- Score(τ) is the per-task score (see Section 6)

### 4.2 What Φ(t) Reveals

The shape of Φ(t) encodes rich information about the solver:

| Φ(t) Shape | Interpretation |
|------------|----------------|
| **Shallow initial slope** | Learning overhead—solver is adapting to task distribution |
| **Linear growth** | Solver treats each task independently, no learning |
| **Accelerating (superlinear)** | Solver is learning shared structure—easy tasks become trivial, freeing resources for hard tasks |
| **Plateau** | Solver has hit a capability ceiling |
| **Concave (decelerating)** | Solver is getting bogged down on hard tasks |

**The key hypothesis**: A solver that genuinely *reasons* about dynamical systems should show **superlinear Φ(t)**. Pure pattern-matching yields linear Φ(t).

### 4.3 Why Not a Scalar?

Scalar metrics (accuracy, F1, AUC) collapse the quality-throughput tradeoff into a single number, losing crucial information:

- A solver with 90% accuracy in 10 seconds and one with 90% accuracy in 100 seconds get the same score
- A solver that solves easy tasks fast and hard tasks slow looks identical to one with uniform speed

Φ(t) preserves this structure. Different solvers may have **different Φ(t) shapes** even with the same final Φ value.

### 4.4 Difficulty Weighting

The weighting function w(h_KS) is a **dataset hyperparameter** that controls what the benchmark rewards:

```python
# Options implemented:
DifficultyWeighting.constant    # w(h) = 1: pure throughput
DifficultyWeighting.linear      # w(h) = h: balanced (default)
DifficultyWeighting.quadratic   # w(h) = h²: reward hard tasks
DifficultyWeighting.inverse     # w(h) = 1/h: reward easy tasks
DifficultyWeighting.log         # w(h) = log(1+h): diminishing returns
```

**Design rationale**: Different research questions require different weightings:
- Studying throughput? Use `constant`
- Balanced analysis? Use `linear`
- Testing capability limits? Use `quadratic`

The benchmark can be configured to emphasize different regimes.

---

## 5. Task Specification

### 5.1 Task Definition

A task instance is:

```
Task = {
    system: ChaoticSystem,        # The hidden dynamical system
    observations: array[n_obs, d], # What the solver sees
    obs_times: array[n_obs],       # When observations were taken
    true_future: array[d],         # Ground truth (hidden)
    future_time: int,              # Prediction horizon
    h_ks: float,                   # Task complexity
    discretizer: DiscretizedSpace, # For probability computation
    true_bin: int,                 # Correct discrete bin
    conditional: bool,             # Is family revealed?
    family: str                    # System family (if conditional)
}
```

### 5.2 What the Solver Receives

The solver sees:
- Observations: A sequence of noisy state measurements
- Observation times: When each measurement was taken (may be sparse)
- Prediction horizon: How far ahead to predict
- (If conditional) System family: e.g., "logistic", "lorenz"

The solver does **not** see:
- True system parameters
- Noise-free trajectory
- h_KS value
- True future state

### 5.3 What the Solver Returns

The solver returns a **probability distribution** over the discretized state space:

```python
def predict(self, task: Task) -> np.ndarray:
    """
    Returns: array of shape (n_states,) summing to 1.0
    """
```

**Design rationale**: Requiring probability distributions (not point predictions) enables:
1. Proper scoring via negative log-likelihood
2. Uncertainty quantification
3. Distinguishing "confident and wrong" from "uncertain"

### 5.4 Discretization

The state space is discretized into bins:

- 1D: n_bins intervals in the appropriate range
- 2D+: n_bins^d grid (keep d ≤ 3 to avoid curse of dimensionality)

Default: n_bins = 20, giving 20 states for 1D, 400 for 2D, 8000 for 3D.

**Why discretize?** 
- Clean NLL computation without density estimation
- Unambiguous "correct/incorrect" for accuracy metrics
- Avoids issues with continuous probability densities

---

## 6. Scoring and Correctness

### 6.1 The Core Question: What Does "Correct" Mean?

This was the central unsolved problem in earlier versions. The user specified:

> "either regression or (better) would be likelihood of the functions given the partial observability, so p(x_t | x_1, x_t-1) and return NLL biased (multiplied?) by the difficulty function"

**Decision**: Use NLL on discretized space, weighted by difficulty.

### 6.2 Scoring Formula

For a single task:

```python
nll = -log(prob[true_bin])  # Negative log-likelihood
accuracy = exp(-nll)         # Probability-based accuracy
score = w(h_ks) * accuracy   # Weighted by difficulty
```

**Design rationale**:
- NLL captures how well the solver predicted the true outcome
- Difficulty is captured via w(h_KS) weighting, not in the accuracy term
- This avoids double-weighting that previously crushed scores for hard tasks

### 6.4 Binary Accuracy (Secondary Metric)

For interpretability, we also track:

```python
correct_bin = (argmax(probs) == true_bin)
```

This gives a simple "did the solver predict the right region?" metric.

---

## 7. Observability, Cost, and Agency

### 7.1 The Key Insight

> "The other key thing the benchmark MUST consider is that observability should have a cost, which allows us to quantify 'agency'. So more agents can observe more in parallel at lower cost like in the real-world."

Observations are not free. A solver that requests more observations pays a cost. This enables:
1. Distinguishing "good predictions from few observations" vs "good predictions from many observations"
2. Modeling the multi-agent advantage: N agents can collect N× observations in 1× time

### 7.2 Observation Parameters

Two parameters control partial observability:

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `n_obs` | Number of observations available | More = more information |
| `obs_density` | Temporal density (1.0 = contiguous, 0.25 = sparse) | Higher = more coherent signal |

**Design rationale**: These decouple "how much" from "how spread out":
- n_obs=50, density=1.0: 50 consecutive observations
- n_obs=50, density=0.25: 50 observations spread over 200 timesteps

### 7.3 Observation Cost (Future Extension)

The evaluator accepts an `obs_cost` parameter:

```python
evaluator = Evaluator(
    weighting=DifficultyWeighting.linear,
    obs_cost=0.01  # Cost per observation
)
```

When obs_cost > 0, the score becomes:

```python
task_score = w(h_ks) * accuracy - obs_cost * n_observations
```

This creates a **cost-adjusted metric** Ψ(t,n) where n is total observations used.

### 7.4 The 2D Evaluation Surface

With observation costs, the evaluation object becomes a **surface** Ψ(t, n):

```
Ψ(t, n) = cumulative score at time t using ≤ n observations
```

This reveals:
- Time-only tradeoffs (horizontal slices)
- Observation-only tradeoffs (vertical slices)
- Pareto frontier of optimal (t, n, Ψ) triples

---

## 8. Conditional vs Unconditional Settings

### 8.1 Two Evaluation Regimes

The user specified:

> "Yes reporting both conditional and unconditional would be valid"

**Conditional**: Solver knows the system family (e.g., "this is a Lorenz system") but not parameters.

**Unconditional**: Solver knows nothing—must infer both family and parameters from observations.

### 8.2 Why Both?

The gap between conditional and unconditional performance measures **the value of system identification**:

- If gap is small: Family knowledge doesn't help much (maybe all systems look similar in observation space)
- If gap is large: System identification is a crucial subtask

### 8.3 Current Implementation

```python
config = TaskConfig(conditional=True)   # Reveal family
config = TaskConfig(conditional=False)  # Hide family
```

When conditional=True, the task includes `family="logistic"` (or similar).

**Note**: The current ConditionalSolver doesn't properly exploit this information (see Section 11).

---

## 9. Multi-Agent Hypothesis

### 9.1 The Core Claim

> "The 'model' can be anything, it could be a decentralised multi-agent framework, or it could be a single centralised agent. It could be pre-trained models, it could be MLPs. This benchmark, however, will distinguish from others and allow analysis into what works best, as it cares about efficiency vs quality with respect to observability and agency."

The benchmark is **architecture-agnostic**. It evaluates any solver that implements the interface.

### 9.2 Multi-Agent Advantage Hypothesis

From Chat 1 (the most conceptually rich discussion):

> **Hypothesis**: Multi-agent should dominate because:
> - Agent A probes System 1 heavily, learns it
> - Agent B gets System 1 later, probes minimally using A's discovery
> - Net probe cost across the swarm decreases faster

Formally: For N agents with shared knowledge,

$$\Psi_N(t, n) > N \times \Psi_1(t, n)$$

at equal total observation budget n.

### 9.3 The Discovery Protocol (Future Extension)

Chat 1 proposed a **knowledge sharing mechanism**:

```python
Discovery = {
    task: Task,
    pattern: "Trajectory looks like Lorenz",
    answer: predicted_distribution,
    confidence: float
}

# Agents can:
# - Probe (costs λ per observation)
# - Query knowledge base (FREE)
# - Write discoveries (FREE)
```

**The asymmetry is key**: Observation is expensive, knowledge sharing is free. This models real-world collaboration where communication is cheap but data collection is costly.

### 9.4 What Multi-Agent Enables

1. **Parallel observation**: N agents collect N× data in 1× time
2. **Specialization**: Some agents probe, others predict
3. **Knowledge transfer**: Learning from others' discoveries
4. **Division of labor**: Route easy tasks to fast agents, hard tasks to capable agents

---

## 10. Design Decisions and Rationale

### 10.1 Why NLL Instead of MSE?

Earlier versions used MSE, which was **gameable by mean reversion**:

> "Mean reversion baselines achieved ~80% accuracy by exploiting the ergodic nature of chaotic attractors, where predicting the attractor center averages out over the prediction window."

NLL on discretized space avoids this because:
- Predicting uniform distribution gives NLL = log(n_states) ≈ 3.0
- Predicting the correct bin gives NLL ≈ 0
- Mean reversion gives intermediate NLL, not artificially good scores

### 10.2 Why Difficulty Weighting as Hyperparameter?

> "We could configure it to reward solving lots of easy tasks, or bias it to rewarding more the solution of lots of hard tasks throughput"

Different research questions need different weightings. Making it a hyperparameter:
- Enables studying "throughput focus" vs "capability focus" vs "balanced"
- Prevents overfitting benchmark design to one evaluation criterion
- Allows users to configure for their specific research question

### 10.3 Why Discretized Space Instead of Continuous?

Options considered:
1. **Gaussian output**: Simple but can't capture multimodality
2. **Mixture of Gaussians**: Expressive but complex interface
3. **Samples**: Flexible but NLL estimation is noisy
4. **Normalizing flows**: Powerful but requires deep learning

**Decision**: Discretized space with ~20 bins per dimension.

Rationale:
- Clean, unambiguous NLL computation
- Simple solver interface (just return probabilities)
- Sufficient resolution for the task
- Avoids density estimation issues

### 10.4 Why Stratified Sampling?

Tasks are stratified by h_KS quintile:

```python
tasks = generator.generate_batch(n_tasks, stratified=True)
```

This ensures:
- Balanced representation of easy, medium, and hard tasks
- Prevents accidentally drawing all-easy or all-hard batches
- Makes Φ(t) curves more comparable across runs

### 10.5 Why Fixed Batches with Mandatory Completion?

> "Must attempt ALL tasks—cannot game by solving only easy ones"

If solvers could skip tasks, they would:
- Solve only low-h_KS tasks
- Achieve high throughput with low quality
- Game the Φ(t) metric

Mandatory completion + h_KS weighting prevents this.

---

## 11. Known Limitations and Open Problems

### 11.1 h_KS Computation

~~**Previous state**: h_KS values were hand-tuned approximations.~~

**FIXED (v3.1)**: h_KS is now computed rigorously via Lyapunov exponents:
- 1D maps: Direct computation via `compute_lyapunov_1d()`
- 2D maps: QR method via `compute_lyapunov_spectrum()`
- Continuous flows (Lorenz): Integration method via `compute_lyapunov_continuous()`

Systems with h_KS < 0.1 are filtered out to exclude non-chaotic parameter regimes.

### 11.2 Prediction Horizon Scaling

~~**Previous state**: Fixed horizon = 10 steps.~~

**FIXED (v3.1)**: Horizon now scales with Lyapunov time:

```python
horizon = int(horizon_lyapunov_multiplier * system.lyapunov_time)  # default k=1.5
```

This makes prediction horizons ~1.5 Lyapunov times ahead—challenging but feasible.

### 11.3 ConditionalSolver Doesn't Do Conditional Prediction

**Current state**: The ConditionalSolver uses invariant measures (e.g., arcsine distribution for logistic map) which don't depend on parameters.

**Required fix**: Implement parameter estimation:

```python
# Estimate r from observations
r_estimates = [x_next / (x * (1-x)) for x, x_next in consecutive_pairs]
r_hat = median(r_estimates)

# Simulate forward
x = last_observation
for _ in range(horizon):
    x = r_hat * x * (1 - x)
return discretize(x)
```

**Impact**: Cannot test "does family knowledge help?" until this is fixed.

### 11.4 Scoring Formula

~~**Previous state**: Score = w(h_KS) × exp(-NLL / h_KS) — double-weighted difficulty.~~

**FIXED (v3.1)**: Score = w(h_KS) × exp(-NLL). Difficulty is captured once via the weighting function, not twice.

### 11.5 Multi-Agent Protocol Not Implemented

**Current state**: The multi-agent knowledge sharing described in Chat 1 is not implemented.

**Required work**:
1. Define knowledge base data structure
2. Implement probing cost mechanism
3. Create MultiAgentSolver wrapper
4. Track Ψ(t,n) surface

### 11.6 No Theoretical Bounds

**Mentioned but not implemented**:

> "Prove that no solver can achieve Φ(t) > C·t for some constant C depending on the observation model"

This would establish an information-theoretic ceiling. Likely provable via rate-distortion theory.

---

## 12. Implementation Status

### 12.1 What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| System families | ✅ Working | 5 families, parameterized |
| Task generation | ✅ Working | Stratified sampling |
| Discretization | ✅ Working | Clean NLL computation |
| Φ(t) tracking | ✅ Working | Cumulative curve |
| Baseline solvers | ✅ Working | 7 solvers implemented |
| Difficulty weighting | ✅ Working | 5 options |
| Conditional flag | ✅ Working | But solver doesn't exploit it |
| Visualization | ✅ Working | 7 diagnostic plots |

### 12.2 What Needs Work

| Component | Status | Priority |
|-----------|--------|----------|
| Correct h_KS values | ✅ Computed via Lyapunov | DONE |
| Horizon scaling | ✅ k × lyapunov_time | DONE |
| Filter non-chaotic systems | ✅ min_h_ks threshold | DONE |
| Scoring formula | ✅ No double-weighting | DONE |
| Proper conditional solver | ❌ Fake | HIGH |
| Observation cost | ⚠️ Implemented but untested | MEDIUM |
| Multi-agent protocol | ❌ Not started | MEDIUM |
| Theoretical bounds | ❌ Not started | LOW |

### 12.3 File Structure

```
chaosbench_v3.py          # Main implementation (~1100 lines)
CHAOSBENCH_SPECIFICATION.md  # This document

# Generated outputs:
phi_curves.png            # Φ(t) for all solvers
nll_vs_difficulty.png     # NLL vs h_KS scatter
accuracy_by_difficulty.png # Bar chart by difficulty band
weighting_comparison.png  # Effect of w(h_KS) choice
conditional_comparison.png # Conditional vs unconditional
observability_analysis.png # Effect of n_obs and density
complexity_spectrum.png   # h_KS distribution across systems
```

---

## 13. Future Roadmap

### Phase 1: Fix Critical Issues (Immediate)

1. **Compute correct h_KS values**
   - Implement QR method for Lyapunov spectrum
   - Or restrict to 1D maps with analytical h_KS
   - Validate against published values

2. **Scale horizon by Lyapunov time**
   - horizon = k × τ_λ where k ∈ [0.5, 2]
   - Verify prediction is feasible at chosen horizon

3. **Build proper parameter estimation solver**
   - Estimate parameters from observations
   - Use estimated parameters for prediction
   - This becomes the "smart baseline"

### Phase 2: Multi-Agent Extension (Near-term)

4. **Implement observation cost mechanism**
   - Track observations used per task
   - Compute Ψ(t,n) surface
   - Analyze Pareto frontier

5. **Implement knowledge sharing protocol**
   - Define discovery format
   - Build knowledge base
   - Create MultiAgentSolver wrapper

6. **Test multi-agent hypothesis**
   - Compare Ψ_N vs N × Ψ_1
   - Measure collaboration efficiency

### Phase 3: Theoretical Grounding (Medium-term)

7. **Prove information-theoretic bounds**
   - Lower bound on Φ(t) for any solver
   - Upper bound (oracle with known parameters)
   - Characterize achievable region

8. **Analyze transfer learning**
   - Within-family transfer
   - Cross-family transfer
   - Emergent capability detection

### Phase 4: Scale and Publish (Long-term)

9. **Evaluate LLM-based solvers**
   - Wrap API calls in Solver interface
   - Compare reasoning models vs fast models
   - Test routing strategies

10. **NeurIPS submission**
    - Full experimental evaluation
    - Theoretical results
    - Multi-agent analysis

---

## Appendix: Mathematical Background

### A.1 Lyapunov Exponents

For a dynamical system x_{n+1} = f(x_n), the Lyapunov exponent measures the rate of separation of nearby trajectories:

$$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \log |f'(x_i)|$$

For d-dimensional systems, there are d Lyapunov exponents λ_1 ≥ λ_2 ≥ ... ≥ λ_d.

### A.2 Kolmogorov-Sinai Entropy

The KS entropy is:

$$h_{KS} = \sum_{\lambda_i > 0} \lambda_i$$

By Pesin's theorem, this equals the rate of information production for systems with SRB (Sinai-Ruelle-Bowen) measures.

**Interpretation**: h_KS bits of information are produced per iteration. After t iterations, exp(h_KS · t) distinguishable trajectories are possible.

### A.3 Lyapunov Time

The Lyapunov time is:

$$\tau_\lambda = \frac{1}{\lambda_{\max}}$$

After ~τ_λ iterations, prediction error has grown by factor e ≈ 2.718.

After ~3τ_λ iterations, prediction error has grown by factor e³ ≈ 20.

After ~5τ_λ iterations, prediction error has grown by factor e⁵ ≈ 150.

**Practical limit**: Prediction beyond ~2-3 Lyapunov times is essentially impossible without exact knowledge of initial conditions.

### A.4 Invariant Measures

For ergodic chaotic systems, almost all trajectories spend time in region A proportional to μ(A), where μ is the invariant measure.

**Examples**:
- Logistic map (r=4): μ has density 1/(π√(x(1-x))) (arcsine distribution)
- Tent map (μ=2): μ is uniform on [0,1]
- Lorenz: μ is supported on the butterfly attractor

**Implication for benchmarking**: A solver that simply returns the invariant measure will achieve non-trivial scores without understanding dynamics. This is why NLL-based scoring is used instead of MSE.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | Jan 2026 | Initial formulation (Chat 0) |
| 0.2 | Jan 2026 | Added Φ(t) paradigm, multi-agent discussion |
| 0.3 | Jan 2026 | Full specification with NLL scoring, this document |

---

*This document captures the complete context for ChaosBench development. A new reader should be able to understand all design decisions, their rationale, known limitations, and the path forward.*