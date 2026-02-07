# ChaosBench: Comprehensive Technical Report

**Date:** 2026-01-31
**Status:** Phase 6 Complete (Hypothesis-Driven Redesign)
**Test Coverage:** 76 tests passing

---

## Executive Summary

ChaosBench is a rigorous benchmark for evaluating reasoning systems on prediction tasks over partially observable chaotic dynamical systems. Unlike traditional benchmarks (ARC, etc.), ChaosBench provides:

1. **Principled complexity measure** via Kolmogorov-Sinai entropy (h_KS)
2. **Efficiency-aware evaluation** via Φ(t) curves (quality × throughput)
3. **Infinite task space** preventing memorization
4. **Metacognitive agent protocol** for studying scientific reasoning

---

## 1. Mathematical Foundation

### 1.1 Kolmogorov-Sinai Entropy (h_KS)

The benchmark uses h_KS as the rigorous complexity measure:

$$h_{KS} = \sum_{\lambda_i > 0} \lambda_i$$

where λ_i are the Lyapunov exponents. By Pesin's theorem, this equals the rate of information production.

**Implementation:** `chaosbench/core/lyapunov.py`

| Method | Function | Use Case |
|--------|----------|----------|
| Direct formula | `compute_lyapunov_1d()` | 1D maps: λ = (1/n) Σ ln\|f'(x_i)\| |
| QR decomposition | `compute_lyapunov_spectrum()` | Multi-D maps |
| Variational equations | `compute_lyapunov_continuous()` | Continuous flows (Lorenz) |

**Validated against literature values:**
- Logistic map (r=4): λ = ln(2) ≈ 0.693 ✓
- Tent map (μ=2): λ = ln(2) ≈ 0.693 ✓
- Hénon map (a=1.4, b=0.3): λ₁ ≈ 0.42, λ₂ ≈ -1.62 ✓
- Lorenz (σ=10, ρ=28, β=8/3): λ₁ ≈ 0.906 ✓

### 1.2 Lyapunov Time

The prediction horizon is calibrated relative to the Lyapunov time:

$$\tau_\lambda = \frac{1}{\lambda_{\max}}$$

After ~τ_λ iterations, prediction error grows by factor e ≈ 2.718. The benchmark uses horizons of ~1.5 Lyapunov times.

---

## 2. Chaotic Systems Library

**Implementation:** `chaosbench/core/Chaosbench_v3.py`

### 2.1 System Families

| Family | Dimension | Parameters | h_KS Range | Equation |
|--------|-----------|------------|------------|----------|
| **Logistic** | 1D | r ∈ [3.57, 4.0] | [0, ln(2)] | x_{n+1} = r·x_n·(1-x_n) |
| **Tent** | 1D | μ ∈ [1.5, 2.0] | [0, ln(2)] | x_{n+1} = μ·min(x_n, 1-x_n) |
| **Hénon** | 2D | a ∈ [1.2, 1.4], b=0.3 | ~0.3-0.5 | x_{n+1} = 1 - a·x_n² + y_n |
| **Standard** | 2D | K ∈ [0.5, 5.0] | ~0-1.5 | p_{n+1} = p_n + K·sin(q_n) |
| **Lorenz** | 3D | ρ ∈ [20, 40] | ~0.7-1.1 | Discretized via RK4 |

### 2.2 System Architecture

```python
@dataclass
class ChaoticSystem:
    name: str
    family: str
    dim: int
    h_ks: float  # Computed via Lyapunov exponents
    lyapunov_time: float  # τ_λ = 1/λ_max

    def step(self, x: np.ndarray) -> np.ndarray: ...
    def trajectory(self, x0: np.ndarray, n_steps: int) -> np.ndarray: ...
```

### 2.3 Model Factory

**Implementation:** `chaosbench/core/models.py`

```python
from chaosbench.core.models import create_model, MODEL_PARAMS

model = create_model("logistic", {"r": 3.9})  # Returns LogisticMap instance
MODEL_PARAMS  # {"logistic": ["r"], "tent": ["mu"], "henon": ["a", "b"], ...}
```

---

## 3. Task Generation

### 3.1 Task Structure

```python
@dataclass
class Task:
    task_id: int
    system: ChaoticSystem
    observations: np.ndarray  # Shape: (n_obs, dim)
    obs_times: np.ndarray     # When observations were taken
    true_future: np.ndarray   # Ground truth (hidden from agent)
    future_time: int          # Prediction horizon
    h_ks: float               # Task complexity
    discretizer: DiscretizedSpace  # For probability computation
    true_bin: int             # Correct discrete bin
    conditional: bool         # Is system family revealed?
    family: str               # System family name
```

### 3.2 Task Configuration

```python
@dataclass
class TaskConfig:
    n_obs: int = 50                          # Observations available
    obs_density: float = 1.0                 # Temporal density
    horizon_lyapunov_multiplier: float = 1.5 # Horizon = k × τ_λ
    noise_std: float = 0.01                  # Observation noise
    n_bins: int = 20                         # Discretization resolution
    conditional: bool = True                 # Reveal system family?
    min_h_ks: float = 0.1                    # Filter non-chaotic systems
```

### 3.3 Stratified Sampling

Tasks are stratified by h_KS quintile to ensure balanced difficulty distribution:

```python
generator = TaskGenerator(config)
tasks = generator.generate_batch(n_tasks=100, stratified=True)
```

---

## 4. Baseline Solvers

**Implementation:** `chaosbench/core/Chaosbench_v3.py` (Solver classes)

| Solver | Strategy | Expected Performance |
|--------|----------|---------------------|
| **UniformSolver** | Return uniform distribution | Worst case baseline |
| **MeanSolver** | Predict mean of observations | Mean reversion baseline |
| **LastValueSolver** | Predict last observed value | Persistence baseline |
| **HistogramSolver** | Empirical distribution from observations | Frequentist baseline |
| **LinearExtrapolator** | Fit linear model, extrapolate | Simple dynamics baseline |
| **NearestNeighborSolver** | Find similar patterns in history | Pattern matching baseline |
| **ConditionalSolver** | Use family-specific heuristics | Informed baseline |

All solvers return probability distributions over discretized state space.

---

## 5. Scoring and Evaluation

### 5.1 Per-Task Scoring

```python
# Negative log-likelihood on discretized space
nll = -log(prob[true_bin])

# Probability-based accuracy
accuracy = exp(-nll)

# Weighted by difficulty
score = w(h_ks) × accuracy
```

**Scoring formula for predictions:**
```python
error = abs(prediction - actual)
score = exp(-error × 5)  # Exponential decay
```

### 5.2 Difficulty Weighting Functions

```python
DifficultyWeighting.constant   # w(h) = 1: pure throughput
DifficultyWeighting.linear     # w(h) = h: balanced (default)
DifficultyWeighting.quadratic  # w(h) = h²: reward hard tasks
DifficultyWeighting.inverse    # w(h) = 1/h: reward easy tasks
DifficultyWeighting.log        # w(h) = log(1+h): diminishing returns
```

### 5.3 Φ(t) Curve

The primary evaluation metric is not a scalar but a **curve**:

$$\Phi(t) = \sum_{\substack{\text{tasks solved by time } t}} w(h_{KS}) \cdot \text{Score}$$

**What Φ(t) shape reveals:**

| Shape | Interpretation |
|-------|----------------|
| Linear growth | No learning, treats tasks independently |
| Superlinear (accelerating) | Learning shared structure — key hypothesis |
| Concave (decelerating) | Bogged down on hard tasks |
| Plateau | Capability ceiling reached |

---

## 6. Metacognitive Agent Protocol

### 6.1 Agent Architecture

**Implementation:**
- Types: `chaosbench/agents/metacognitive_types.py`
- Session: `chaosbench/experiments/session.py`
- Agent: `chaosbench/agents/metacognitive_agent.py`

### 6.2 Agent Observation

```python
@dataclass
class AgentObservation:
    task_id: int
    observations: np.ndarray      # x_0, x_1, ..., x_49
    obs_times: np.ndarray
    prediction_horizon: int       # How far ahead to predict
    family: str | None            # System family (if conditional)
    learnings: str                # Persistent notepad
    last_feedback: Feedback | None
    last_backtest: BacktestFeedback | None
```

### 6.3 Agent Actions

| Action | JSON Format | Effect |
|--------|-------------|--------|
| **HYPOTHESIZE** | `{"action": "HYPOTHESIZE", "model": "logistic", "params": {"r": 3.85}}` | Test model against observations, get MAE |
| **FIT** | `{"action": "FIT", "model": "logistic"}` | Auto-fit parameters, get best MAE |
| **PREDICT** | `{"action": "PREDICT", "value": 0.42}` | Commit final prediction |
| **WRITE** | `{"action": "WRITE", "text": "..."}` | Record learnings |
| **DELETE** | `{"action": "DELETE", "section": "## Title"}` | Remove from learnings |
| **MOVE_ON** | `{"action": "MOVE_ON"}` | Accept score, next task |

### 6.4 Backtest & Fitting

**Implementation:**
- `chaosbench/core/backtest.py`
- `chaosbench/core/fitting.py`

```python
# Backtest: test specific parameters
result = backtest_model("logistic", {"r": 3.9}, observations)
# Returns: BacktestResult(mae=0.02, predicted_next=0.156)

# Fit: auto-estimate parameters
result = fit_model("logistic", observations)
# Returns: FitResult(params={"r": 3.87}, mae=0.01, predicted_next=0.394)
```

Fitting uses `scipy.optimize.minimize_scalar` with bounded search in chaotic regime.

### 6.5 Session Flow

```
┌─────────────────────────────────────────────────────────┐
│                    SESSION LOOP                         │
├─────────────────────────────────────────────────────────┤
│ for each task:                                          │
│   while turns < max_turns:                              │
│     1. Build observation (obs + learnings + backtest)   │
│     2. Agent returns (reasoning, action)                │
│     3. Handle action:                                   │
│        - HYPOTHESIZE → backtest, store feedback         │
│        - FIT → fit params, store feedback               │
│        - PREDICT → store prediction (no reveal)         │
│        - WRITE/DELETE → update learnings                │
│        - MOVE_ON → reveal score, reflection phase       │
│   4. Bank weighted score to Φ(t)                        │
└─────────────────────────────────────────────────────────┘
```

### 6.6 Learnings Mechanism

**Implementation:** `chaosbench/agents/learnings.py`

Persistent notepad across tasks. Agent can:
- WRITE markdown sections
- DELETE sections by title
- Read current learnings in each observation

**Research question:** Does accumulated knowledge transfer to future tasks? (Superlinear Φ(t) hypothesis)

---

## 7. System Prompts

### 7.1 Hypothesis-Driven Prompt

**File:** `chaosbench/prompts/hypothesis_system.txt`

Key instructions:
1. Form hypotheses about what system generated the data
2. Test hypotheses via HYPOTHESIZE action
3. Refine until MAE is low
4. Use model's prediction when confident
5. WRITE learnings after seeing result

### 7.2 Model Families in Prompt

Agent is given:
- `logistic (params: r)` — x_{n+1} = r × x_n × (1-x_n)
- `tent (params: mu)` — piecewise linear map
- `henon (params: a, b)` — 2D quadratic map
- `standard (params: K)` — Hamiltonian chaos
- `lorenz (params: sigma, rho, beta)` — 3D continuous flow

---

## 8. Visualization & Plots

### 8.1 Generated Plots

| Plot | File | Description |
|------|------|-------------|
| Φ(t) Curves | `phi_curves.png` | Performance over time for all solvers |
| NLL vs Difficulty | `nll_vs_difficulty.png` | Scatter + trend line per solver |
| Accuracy by Difficulty | `accuracy_by_difficulty.png` | Bar chart by h_KS band |
| Weighting Comparison | `weighting_comparison.png` | Effect of w(h_KS) choice |
| Complexity Spectrum | `complexity_spectrum.png` | h_KS distribution by family |
| Observability Analysis | `observability_analysis.png` | Effect of n_obs and density |

### 8.2 Plot Functions

```python
from chaosbench.core.Chaosbench_v3 import (
    plot_phi_curves,
    plot_nll_vs_difficulty,
    plot_accuracy_by_difficulty,
    plot_weighting_comparison,
    plot_conditional_vs_unconditional,
    plot_observability_analysis,
    plot_system_complexity_spectrum,
)
```

---

## 9. Testing Infrastructure

**76 tests across 15 test files:**

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_models.py` | 4 | Model factory |
| `test_backtest.py` | 4 | Backtest function |
| `test_fitting.py` | 5 | Parameter fitting |
| `test_metacognitive_types.py` | 15 | Types + parsing |
| `test_session.py` | 5 | Session runner |
| `test_trace.py` | 4 | Trace logging |
| `test_learnings.py` | - | Learnings manager |
| `test_hypothesis_integration.py` | 2 | Full flow integration |
| `test_chaosbench.py` | - | Core benchmark |
| `test_lyapunov.py` | - | Lyapunov computation |

**Run all tests:**
```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
source venv/bin/activate
pytest chaosbench/tests/ -v
```

---

## 10. CLI Runners

### 10.1 Metacognitive Agent Runner

**File:** `chaosbench/run_metacognitive.py`

```bash
python -m chaosbench.run_metacognitive \
    --model gemini/gemini-2.0-flash \
    --n-tasks 10 \
    --timeout 300 \
    --output session_output \
    --conditional
```

**Outputs:**
- `trace.md` — Turn-by-turn reasoning + actions
- `learnings.md` — Final learnings content
- `phi_curve.json` — Φ(t) curve data

### 10.2 Main Benchmark

```bash
python -m chaosbench.core.Chaosbench_v3
```

Runs full benchmark with 7 baseline solvers, generates all plots.

---

## 11. Development History

### Phase 1-3: Core Protocol (Complete)
- AgentObservation/AgentAction dataclasses
- Session runner with PREDICT/WRITE/DELETE/MOVE_ON
- LEARNINGS.md mechanics
- Trace logging

### Phase 4: Analysis (Pending)
- Trace viewer / pretty-printer
- Learnings evolution over session
- Transfer detection

### Phase 5: Design Fix (Complete)
- Identified feedback exploit (agent iterating toward revealed answer)
- Implemented blind prediction (no feedback until MOVE_ON)
- Added reflection phase

### Phase 6: Hypothesis-Driven Redesign (Complete)
- HYPOTHESIZE action (test model against observations)
- FIT action (auto-fit parameters)
- Backtest feedback (MAE + predicted_next)
- 76 tests passing

### Phase 7: Simple Benchmark (In Progress)
- Raw prediction (no scaffolding)
- Difficulty via horizon × noise
- Pasteable format for manual testing

---

## 12. Key Research Questions

1. **Quality-throughput tradeoffs:** How do different architectures trade off quality and speed?

2. **Learning dynamics:** Do agents learn structure? Evidence: superlinear Φ(t)

3. **Transfer within families:** Does learning easy instances help with hard ones?

4. **Scaffolding value:** Gap between raw prediction vs hypothesis-driven = value of tools

5. **Multi-agent advantage:** Can N agents achieve Ψ > N × Ψ_single? (Future work)

---

## 13. File Structure

```
chaosbench/
├── core/
│   ├── Chaosbench_v3.py      # Main benchmark (1309 lines)
│   ├── lyapunov.py           # Lyapunov computation (309 lines)
│   ├── models.py             # Model factory
│   ├── backtest.py           # Backtest function
│   ├── fitting.py            # Parameter fitting
│   └── ChaosSpecification.md # Full specification
├── agents/
│   ├── metacognitive_types.py # Data types
│   ├── metacognitive_agent.py # LLM agent
│   └── learnings.py           # Learnings manager
├── experiments/
│   ├── session.py             # Session runner (249 lines)
│   └── trace.py               # Trace logger
├── prompts/
│   ├── metacognitive_system.txt  # Original prompt
│   └── hypothesis_system.txt     # Hypothesis-driven prompt
├── tests/
│   └── (15 test files, 76 tests)
└── run_metacognitive.py       # CLI runner
```

---

## 14. Dependencies

```
numpy
scipy
matplotlib
litellm
python-dotenv
pytest
```

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| h_KS | Kolmogorov-Sinai entropy |
| λ_i | Lyapunov exponents |
| τ_λ | Lyapunov time = 1/λ_max |
| Φ(t) | Cumulative weighted score at time t |
| w(h) | Difficulty weighting function |
| NLL | Negative log-likelihood |

---

## Appendix B: Known Limitations

1. **ConditionalSolver doesn't properly exploit family information** — uses invariant measures, not parameter estimation

2. **Multi-agent protocol not implemented** — knowledge sharing mechanism designed but not built

3. **No theoretical bounds proven** — information-theoretic ceiling not established

4. **Simple scoring** — currently uses exponential decay of error, not full NLL on discretized space

---

*Report generated 2026-01-31. For the latest status, see `docs/plans/chaosbench-v4-task-plan.md`.*
