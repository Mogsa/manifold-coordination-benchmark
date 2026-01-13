# Manifold Coordination Benchmark: Implementation Plan

> **Purpose:** This document contains ALL context needed to implement the benchmark. Each section is self-contained with checkpoints that can be marked complete. A fresh Claude instance should be able to continue work from any checkpoint.

> **Status Update (2026-01-12):** Environment setup complete. Virtual environment created with all dependencies installed. Ready to begin Phase 1: Core Surface Engine implementation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mathematical Specification](#2-mathematical-specification)
3. [Observation Model](#3-observation-model)
4. [Turn Structure](#4-turn-structure)
5. [Communication Protocol](#5-communication-protocol)
6. [Scoring System](#6-scoring-system)
7. [Surface Design](#7-surface-design)
8. [Implementation Checkpoints](#8-implementation-checkpoints)
9. [File Structure](#9-file-structure)
10. [API Specifications](#10-api-specifications)
11. [Prompt Templates](#11-prompt-templates)
12. [Test Cases](#12-test-cases)
13. [Visualization Requirements](#13-visualization-requirements)
14. [Baselines](#14-baselines)
15. [Evaluation Protocol](#15-evaluation-protocol)

---

## 1. Project Overview

### 1.1 Research Question

Can LLM agents coordinate effectively when they have **asymmetric partial observations** of an environment and must communicate to solve a joint optimization problem?

### 1.2 Core Concept

Two LLM agents jointly control a single "player" navigating a 2D surface to find the global maximum:

- **Agent A** controls the x-coordinate
- **Agent B** controls the y-coordinate
- **The surface** f(x, y) has multiple peaks
- **Neither agent** can see the full surface — each sees only a 1D slice through the current position
- **Communication** is necessary to combine partial views and locate the maximum

### 1.3 Why This Tests Coordination

| Property | How It's Achieved |
|----------|-------------------|
| Information asymmetry | Agent A sees horizontal slice; Agent B sees vertical slice |
| Communication necessity | Neither slice alone reveals 2D peak structure |
| Joint action required | Position (x, y) requires both agents to move |
| Planning required | Limited turns forces strategic exploration |
| Abstract reasoning | Must combine 1D observations into 2D mental model |

### 1.4 Success Criteria for the Benchmark

- [ ] Agents with communication outperform agents without communication
- [ ] Performance scales with surface difficulty
- [ ] Random baseline scores significantly lower than LLM agents
- [ ] Results are reproducible across runs

---

## 2. Mathematical Specification

### 2.1 Domain

```
Surface domain:    [0, 10] × [0, 10]
Surface codomain:  ℝ (typically [0, 1] for normalized surfaces)
Surface function:  f: [0, 10] × [0, 10] → ℝ
```

### 2.2 State

```
Agent A position:  x_A ∈ [0, 10]
Agent B position:  y_B ∈ [0, 10]
View point:        p = (x_A, y_B)
Turn counter:      t ∈ {1, 2, ..., N}
```

### 2.3 Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Domain size | L | 10.0 | Surface spans [0, L] × [0, L] |
| Observation radius | R | 1.5 | Agents see slice of length 2R centered at position |
| Number of turns | N | 10 | Episode length |
| Slice samples | S | 11 | Number of discrete samples in each slice |
| Initial position | (x₀, y₀) | (5.0, 5.0) | Starting point (center of domain) |

### 2.4 Gradient Definition

```
∇f(x, y) = (∂f/∂x, ∂f/∂y)

Computed numerically:
  ∂f/∂x ≈ (f(x + ε, y) - f(x - ε, y)) / (2ε)
  ∂f/∂y ≈ (f(x, y + ε) - f(x, y - ε)) / (2ε)

Where ε = 0.001 (small perturbation)
```

---

## 3. Observation Model

### 3.1 Core Principle

Each agent sees a **1D slice** of the 2D surface through the current view point, perpendicular to the other agent's slice.

```
        y
        ▲
        │       ┃
        │       ┃  ← Agent B sees VERTICAL slice
        │       ┃
   y_B ─┼━━━━━━━●━━━━━━━━  ← Agent A sees HORIZONTAL slice
        │       ┃
        │       ┃
        │       ┃
        └───────┼──────────► x
              x_A

View point = intersection = (x_A, y_B)
```

### 3.2 Agent A Observation

After moving to position x_A (with Agent B at y_B):

```python
observation_A = {
    # Current position (both coordinates, for reference)
    "position": {
        "x": x_A,  # Agent A's position (what A controls)
        "y": y_B   # Agent B's position (for context)
    },

    # Value at the view point
    "value_at_position": f(x_A, y_B),

    # Partial derivative in x-direction
    "gradient_x": ∂f/∂x at (x_A, y_B),

    # Horizontal slice: f(x', y_B) for x' in [x_A - R, x_A + R]
    "slice": [
        {"x": x_A - R, "value": f(x_A - R, y_B)},
        {"x": x_A - R + step, "value": f(x_A - R + step, y_B)},
        ...
        {"x": x_A + R, "value": f(x_A + R, y_B)}
    ]
    # Total of S samples, evenly spaced
}
```

### 3.3 Agent B Observation

After moving to position y_B (with Agent A at x_A):

```python
observation_B = {
    # Current position (both coordinates, for reference)
    "position": {
        "x": x_A,  # Agent A's position (for context)
        "y": y_B   # Agent B's position (what B controls)
    },

    # Value at the view point
    "value_at_position": f(x_A, y_B),

    # Partial derivative in y-direction
    "gradient_y": ∂f/∂y at (x_A, y_B),

    # Vertical slice: f(x_A, y') for y' in [y_B - R, y_B + R]
    "slice": [
        {"y": y_B - R, "value": f(x_A, y_B - R)},
        {"y": y_B - R + step, "value": f(x_A, y_B - R + step)},
        ...
        {"y": y_B + R, "value": f(x_A, y_B + R)}
    ]
    # Total of S samples, evenly spaced
}
```

### 3.4 Boundary Handling

When observation radius extends beyond domain bounds:

```python
# Clamp slice to valid domain
x_min_slice = max(0, x_A - R)
x_max_slice = min(L, x_A + R)

# Samples are only taken within valid range
# This means edge positions have shorter slices
```

### 3.5 What Each Agent CANNOT See

| Agent | Cannot See |
|-------|------------|
| A | How f varies in y-direction; ∂f/∂y; vertical structure |
| B | How f varies in x-direction; ∂f/∂x; horizontal structure |

This asymmetry is **the core driver of communication necessity**.

---

## 4. Turn Structure

### 4.1 Episode Flow

```
INITIALIZATION:
  - Generate surface f(x, y) with known global maximum
  - Set initial position (x_A, y_B) = (5.0, 5.0)
  - Set turn counter t = 0

TURN LOOP (repeat N times):
  t ← t + 1

  Step 1: OBSERVE
    - Agent A receives observation_A (horizontal slice + ∂f/∂x)
    - Agent B receives observation_B (vertical slice + ∂f/∂y)

  Step 2: COMMUNICATE
    - Agent A sends message to Agent B
    - Agent B sends message to Agent A
    - (Messages are exchanged simultaneously or in sequence — see 4.2)

  Step 3: DECIDE
    - Agent A outputs new x-coordinate: x_A_new ∈ [0, 10]
    - Agent B outputs new y-coordinate: y_B_new ∈ [0, 10]

  Step 4: MOVE
    - Update position: (x_A, y_B) ← (x_A_new, y_B_new)

FINAL DECISION (after turn N):
  - Agents communicate one final time
  - Agent A outputs: x_final
  - Agent B outputs: y_final
  - Final answer: (x_final, y_final)

SCORING:
  - Compute score = f(x_final, y_final) / f(x_optimal, y_optimal)
```

### 4.2 Communication Timing Options

**Option A: Simultaneous exchange**
```
Both agents write messages without seeing the other's message.
Then both messages are revealed.
Simpler but may lead to miscoordination.
```

**Option B: Sequential exchange (RECOMMENDED)**
```
Agent A sends message first.
Agent B sees A's message, then sends response.
More natural dialogue flow.
Alternate who goes first each turn.

Detailed flow per turn:
1. Both agents receive observations simultaneously
2. Agent A generates message (sees only own observation)
3. Agent A's message is delivered to Agent B
4. Agent B generates message (sees own observation + A's message)
5. Agent B's message is delivered to Agent A
6. Both agents decide their new position (with full message context)
7. Position updates

Note: Alternate which agent speaks first each turn for fairness.
Turn 1,3,5,7,9: A speaks first
Turn 2,4,6,8,10: B speaks first
```

### 4.3 Turn Budget Interpretation

With N=10 turns:
- Turns 1-7: Exploration and model building
- Turns 8-9: Convergence toward suspected maximum
- Turn 10: Fine-tuning + final decision

Agents must balance **exploration** (sampling diverse locations) vs **exploitation** (refining around promising areas).

---

## 5. Communication Protocol

### 5.1 Format

**Free-form natural language.** No structured format enforced.

Agents should naturally convey:
- What their slice looks like (shape, peaks, valleys)
- Their gradient direction and magnitude
- Hypotheses about peak locations
- Proposed next position

### 5.2 Example Dialogue

```
=== TURN 1 ===

[Agent A → Agent B]
"I'm at x=5.0. My horizontal slice shows relatively flat terrain with a slight
increase toward x=7. The gradient ∂f/∂x = +0.12 confirms the surface rises to
my right. Value here is 0.34. I suggest we explore toward higher x — I'll
move to x=6.5."

[Agent B → Agent A]
"At y=5.0, my vertical slice has a clear peak around y=7.5 with value ~0.6.
Gradient ∂f/∂y = +0.25 points upward strongly. There might be a peak in the
upper region. I'll move to y=7.0 to investigate."

=== TURN 2 ===

[Agent A → Agent B]
"Now at x=6.5, y=7.0. Much better — value here is 0.58! My slice shows a peak
around x=7.2. Gradient is +0.18. We're getting closer to something. What do
you see in the y-direction?"

[Agent B → Agent A]
"Excellent! At this position, my slice peaks right around y=7.0-7.5 with values
~0.6-0.65. Gradient ∂f/∂y = +0.08, so we're near the top in y. I think there's
a peak near (7.2, 7.3). Let's converge there."

...

=== FINAL DECISION ===

[Agent A]: "Based on our exploration, the global maximum appears to be at
approximately x=7.2. Confirming this as my final answer."

[Agent B]: "Agreed. The peak is around y=7.3. My final answer is y=7.3."

Final answer: (7.2, 7.3)
```

### 5.3 Communication Constraints (Optional Extensions)

For harder variants:
- **Token limit:** Max 100 tokens per message
- **Numeric limit:** Can only report 3 numbers per message
- **Noisy channel:** Messages have 10% chance of word corruption

(Not implemented in base version)

---

## 6. Scoring System

### 6.1 Primary Metric: Normalized Score

```python
score = f(x_final, y_final) / f(x_optimal, y_optimal)
```

| Score | Interpretation |
|-------|----------------|
| 1.0 | Found exact global maximum |
| 0.9-0.99 | Very close to global maximum |
| 0.7-0.9 | Found a secondary peak or nearby |
| 0.5-0.7 | Moderate success |
| < 0.5 | Poor performance |

### 6.2 Secondary Metrics

```python
# Distance to true maximum
distance_error = sqrt((x_final - x_opt)^2 + (y_final - y_opt)^2)

# Did they find the right peak? (for multi-peak surfaces)
peak_identification = 1 if closest_peak(x_final, y_final) == global_max_peak else 0

# Exploration coverage (what fraction of domain was observed)
coverage = unique_area_observed / total_area

# Communication efficiency
tokens_used = total_tokens_in_all_messages
```

### 6.3 Aggregation Across Trials

For statistical validity:
- Run each (surface, agent_pair) combination K times (K ≥ 5)
- Report: mean score, std deviation, min, max
- Use different random seeds for LLM sampling

---

## 7. Surface Design

### 7.1 Surface Function Family

All surfaces are sums of Gaussian peaks:

```python
def gaussian_peak(x, y, cx, cy, height, sigma):
    """Single Gaussian peak centered at (cx, cy)."""
    return height * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def multi_peak_surface(x, y, peaks):
    """
    Surface with multiple Gaussian peaks.

    peaks: list of dicts with keys {cx, cy, height, sigma}
    """
    return sum(
        gaussian_peak(x, y, p['cx'], p['cy'], p['height'], p['sigma'])
        for p in peaks
    )
```

### 7.2 Difficulty Levels

#### Level 1: Single Peak
```python
peaks = [
    {"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.5}
]
# Trivial: just follow gradients to the peak
```

#### Level 2: Two Peaks, Clear Winner
```python
peaks = [
    {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},  # Decoy
    {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}   # Global max
]
# Must identify that (7.5, 7.5) is higher than (2.5, 2.5)
```

#### Level 3: Two Peaks, Similar Heights
```python
peaks = [
    {"cx": 2.5, "cy": 7.5, "height": 0.95, "sigma": 1.0},
    {"cx": 7.5, "cy": 2.5, "height": 1.00, "sigma": 1.0}
]
# Heights are close; requires careful comparison
# Peaks are in opposite corners — tests spatial reasoning
```

#### Level 4: Three+ Peaks
```python
peaks = [
    {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
    {"cx": 8.0, "cy": 2.0, "height": 0.7, "sigma": 1.0},
    {"cx": 5.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
]
# More candidates to evaluate and rank
```

#### Level 5: Peaks with Ridge
```python
peaks = [
    {"cx": 3.0, "cy": 3.0, "height": 0.8, "sigma": 1.0},
    {"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}
]
# Plus a ridge connecting them:
def ridge(x, y):
    # Elevated values along diagonal
    dist_to_diagonal = abs(x - y) / sqrt(2)
    return 0.3 * exp(-dist_to_diagonal**2 / 0.5)

# Ridge creates misleading gradients
```

### 7.3 Procedural Generation

```python
def generate_random_surface(difficulty: int, seed: int) -> Surface:
    """
    Generate a surface with controlled difficulty.

    Args:
        difficulty: 1-5, controls number of peaks and height similarity
        seed: random seed for reproducibility

    Returns:
        Surface object with known global maximum
    """
    rng = np.random.default_rng(seed)

    n_peaks = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[difficulty]

    peaks = []
    for i in range(n_peaks):
        peak = {
            "cx": rng.uniform(1.5, 8.5),
            "cy": rng.uniform(1.5, 8.5),
            "height": rng.uniform(0.5, 1.0),
            "sigma": rng.uniform(0.8, 1.5)
        }
        peaks.append(peak)

    # Ensure one peak is clearly the global max (for levels 1-2)
    if difficulty <= 2:
        max_idx = np.argmax([p["height"] for p in peaks])
        peaks[max_idx]["height"] = 1.0
        for i, p in enumerate(peaks):
            if i != max_idx:
                p["height"] = min(p["height"], 0.7)

    return Surface(peaks)
```

### 7.4 Pre-defined Test Surfaces

For reproducibility, define a fixed set of test surfaces:

```python
TEST_SURFACES = {
    "single_peak_center": {
        "peaks": [{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.5}],
        "optimal": (5.0, 5.0),
        "difficulty": 1
    },
    "single_peak_corner": {
        "peaks": [{"cx": 8.0, "cy": 8.0, "height": 1.0, "sigma": 1.2}],
        "optimal": (8.0, 8.0),
        "difficulty": 1
    },
    "two_peaks_clear": {
        "peaks": [
            {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},
            {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}
        ],
        "optimal": (7.5, 7.5),
        "difficulty": 2
    },
    "two_peaks_close": {
        "peaks": [
            {"cx": 3.0, "cy": 7.0, "height": 0.92, "sigma": 1.0},
            {"cx": 7.0, "cy": 3.0, "height": 1.00, "sigma": 1.0}
        ],
        "optimal": (7.0, 3.0),
        "difficulty": 3
    },
    "three_peaks": {
        "peaks": [
            {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
            {"cx": 8.0, "cy": 2.0, "height": 0.7, "sigma": 1.0},
            {"cx": 5.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
        ],
        "optimal": (5.0, 8.0),
        "difficulty": 4
    }
}
```

---

## 8. Implementation Checkpoints

### Phase 1: Core Surface Engine

#### Checkpoint 1.1: Surface Class
- [ ] **File:** `core/surface.py`
- [ ] **Class:** `Surface`
- [ ] **Requirements:**
  - [ ] Constructor takes list of peak definitions
  - [ ] `evaluate(x, y) -> float`: Returns f(x, y)
  - [ ] `gradient(x, y) -> Tuple[float, float]`: Returns (∂f/∂x, ∂f/∂y)
  - [ ] `get_optimal() -> Tuple[float, float, float]`: Returns (x_opt, y_opt, f_opt)
  - [ ] Handles boundary conditions (clamp to [0, L])
- [ ] **Tests:**
  - [ ] Single peak at (5, 5) evaluates correctly
  - [ ] Gradient points toward peak
  - [ ] Optimal location is correct

#### Checkpoint 1.2: Observation Generator
- [ ] **File:** `core/observation.py`
- [ ] **Class:** `ObservationGenerator`
- [ ] **Requirements:**
  - [ ] Constructor takes Surface and parameters (R, S)
  - [ ] `generate_observation_a(x_a, y_b) -> dict`: Horizontal slice + ∂f/∂x
  - [ ] `generate_observation_b(x_a, y_b) -> dict`: Vertical slice + ∂f/∂y
  - [ ] Slices have exactly S samples
  - [ ] Boundary handling for edge positions
- [ ] **Tests:**
  - [ ] Slice length is 2R (or less at boundaries)
  - [ ] Samples are evenly spaced
  - [ ] Gradient sign matches slice slope

#### Checkpoint 1.3: Episode State Manager
- [ ] **File:** `core/episode.py`
- [ ] **Class:** `Episode`
- [ ] **Requirements:**
  - [ ] Constructor takes Surface, initial position, N turns
  - [ ] `get_state() -> dict`: Current position, turn number
  - [ ] `step(x_new, y_new) -> Tuple[obs_a, obs_b]`: Move and observe
  - [ ] `is_done() -> bool`: Check if N turns completed
  - [ ] `get_score(x_final, y_final) -> float`: Compute normalized score
  - [ ] Tracks history of positions and observations
- [ ] **Tests:**
  - [ ] Episode ends after N turns
  - [ ] Score is 1.0 when final position equals optimal
  - [ ] History is recorded correctly

---

### Phase 2: Agent Framework

#### Checkpoint 2.1: Base Agent Interface
- [ ] **File:** `agents/base.py`
- [ ] **Class:** `BaseAgent` (abstract)
- [ ] **Requirements:**
  - [ ] `receive_observation(observation: dict) -> None`
  - [ ] `receive_message(message: str) -> None`
  - [ ] `generate_message() -> str`
  - [ ] `decide_position() -> float`: Returns new coordinate
  - [ ] `final_decision() -> float`: Returns final coordinate
  - [ ] Abstract methods for subclasses to implement

#### Checkpoint 2.2: Random Baseline Agent
- [ ] **File:** `agents/random_agent.py`
- [ ] **Class:** `RandomAgent(BaseAgent)`
- [ ] **Requirements:**
  - [ ] Ignores observations and messages
  - [ ] Returns random position in [0, 10]
  - [ ] Deterministic given seed
- [ ] **Tests:**
  - [ ] Same seed produces same sequence
  - [ ] Positions are within bounds

#### Checkpoint 2.3: Greedy Gradient Agent
- [ ] **File:** `agents/greedy_agent.py`
- [ ] **Class:** `GreedyAgent(BaseAgent)`
- [ ] **Requirements:**
  - [ ] Follows gradient direction (∂f/∂x for A, ∂f/∂y for B)
  - [ ] Step size parameter (default 1.0)
  - [ ] Clamps to domain bounds
  - [ ] No communication (messages are empty)
- [ ] **Tests:**
  - [ ] Moves in gradient direction
  - [ ] Respects domain bounds

#### Checkpoint 2.4: LLM Agent
- [ ] **File:** `agents/llm_agent.py`
- [ ] **Class:** `LLMAgent(BaseAgent)`
- [ ] **Requirements:**
  - [ ] Constructor takes: model name, agent role ('A' or 'B'), API key
  - [ ] Maintains conversation history
  - [ ] Formats observation into prompt
  - [ ] Parses coordinate from LLM response
  - [ ] Handles API errors gracefully
- [ ] **Configuration:**
  - [ ] System prompt loaded from file
  - [ ] Temperature parameter (default 0.7)
  - [ ] Max tokens per response
- [ ] **Tests:**
  - [ ] Prompt includes observation data
  - [ ] Coordinate extraction handles various formats
  - [ ] Conversation history accumulates

---

### Phase 3: Episode Runner

#### Checkpoint 3.1: Turn Executor
- [ ] **File:** `experiments/runner.py`
- [ ] **Class:** `EpisodeRunner`
- [ ] **Requirements:**
  - [ ] Constructor takes: Episode, Agent A, Agent B
  - [ ] `run_turn() -> dict`: Execute one turn (observe, communicate, decide, move)
  - [ ] `run_episode() -> dict`: Run all N turns + final decision
  - [ ] Returns full transcript (observations, messages, positions)
- [ ] **Tests:**
  - [ ] Turn count increments correctly
  - [ ] Both agents receive correct observations
  - [ ] Messages are exchanged between agents

#### Checkpoint 3.2: Result Logger
- [ ] **File:** `experiments/logger.py`
- [ ] **Class:** `ResultLogger`
- [ ] **Requirements:**
  - [ ] Save episode results to JSON
  - [ ] Include: surface config, agent configs, transcript, score
  - [ ] Timestamp and unique ID for each run
  - [ ] Load results for analysis
- [ ] **Output format:**
```json
{
  "id": "run_20240115_143022_abc123",
  "timestamp": "2024-01-15T14:30:22Z",
  "surface": {"name": "two_peaks_clear", "optimal": [7.5, 7.5]},
  "agents": {"A": "gpt-4", "B": "gpt-4"},
  "turns": [
    {
      "turn": 1,
      "position_before": [5.0, 5.0],
      "observation_a": {...},
      "observation_b": {...},
      "message_a": "...",
      "message_b": "...",
      "decision_a": 6.5,
      "decision_b": 7.0,
      "position_after": [6.5, 7.0]
    },
    ...
  ],
  "final_decision": [7.4, 7.6],
  "score": 0.97,
  "metrics": {...}
}
```

---

### Phase 4: Visualization

#### Checkpoint 4.1: 3D Surface Plot
- [ ] **File:** `visualization/plot3d.py`
- [ ] **Function:** `plot_surface(surface, show=True, save_path=None)`
- [ ] **Requirements:**
  - [ ] 3D wireframe or surface plot of f(x, y)
  - [ ] Colormap showing height
  - [ ] Axis labels
  - [ ] Mark optimal point with red dot
- [ ] **Library:** matplotlib with mplot3d or plotly

#### Checkpoint 4.2: Agent Position Overlay
- [ ] **File:** `visualization/plot3d.py`
- [ ] **Function:** `plot_episode(surface, episode_history, show=True, save_path=None)`
- [ ] **Requirements:**
  - [ ] Surface plot as base
  - [ ] Trajectory line showing agent path
  - [ ] Points for each turn position
  - [ ] Start point (green), end point (blue), optimal (red)
  - [ ] Turn numbers as labels

#### Checkpoint 4.3: Slice Visualization
- [ ] **File:** `visualization/slices.py`
- [ ] **Function:** `plot_slices(surface, x_a, y_b, R)`
- [ ] **Requirements:**
  - [ ] Two subplots: horizontal slice, vertical slice
  - [ ] Mark current position on each
  - [ ] Show gradient as arrow
  - [ ] Useful for debugging agent observations

#### Checkpoint 4.4: Animation (Optional)
- [ ] **File:** `visualization/animate.py`
- [ ] **Function:** `animate_episode(surface, episode_history, save_path)`
- [ ] **Requirements:**
  - [ ] Animated GIF or MP4
  - [ ] Shows agent moving turn by turn
  - [ ] Displays current observation slices
  - [ ] Shows messages as subtitles

---

### Phase 5: Evaluation Harness

#### Checkpoint 5.1: Batch Runner
- [ ] **File:** `experiments/eval.py`
- [ ] **Function:** `run_evaluation(surfaces, agent_configs, n_trials, output_dir)`
- [ ] **Requirements:**
  - [ ] Run all combinations of surfaces × agent configs
  - [ ] Multiple trials per combination
  - [ ] Save all results
  - [ ] Progress bar

#### Checkpoint 5.2: Statistical Analysis
- [ ] **File:** `experiments/analysis.py`
- [ ] **Functions:**
  - [ ] `compute_summary_stats(results) -> DataFrame`
  - [ ] `compare_agents(results, agent_a, agent_b) -> dict`
  - [ ] `plot_results(summary) -> Figure`
- [ ] **Requirements:**
  - [ ] Mean, std, min, max score per condition
  - [ ] Statistical significance tests
  - [ ] Bar charts with error bars

#### Checkpoint 5.3: Baseline Comparison
- [ ] **File:** `experiments/baselines.py`
- [ ] **Requirements:**
  - [ ] Run random baseline on all surfaces
  - [ ] Run greedy baseline on all surfaces
  - [ ] Compare to LLM agents
  - [ ] Generate comparison table

---

## 9. File Structure

```
manifold_benchmark/
│
├── core/
│   ├── __init__.py
│   ├── surface.py          # Surface class with peaks
│   ├── observation.py      # Observation generation
│   └── episode.py          # Episode state management
│
├── agents/
│   ├── __init__.py
│   ├── base.py             # Abstract base agent
│   ├── random_agent.py     # Random baseline
│   ├── greedy_agent.py     # Gradient-following baseline
│   └── llm_agent.py        # LLM-based agent
│
├── visualization/
│   ├── __init__.py
│   ├── plot3d.py           # 3D surface and trajectory plots
│   ├── slices.py           # 1D slice visualization
│   └── animate.py          # Episode animation
│
├── experiments/
│   ├── __init__.py
│   ├── runner.py           # Episode execution
│   ├── logger.py           # Result saving/loading
│   ├── eval.py             # Batch evaluation
│   ├── analysis.py         # Statistical analysis
│   └── baselines.py        # Baseline comparisons
│
├── prompts/
│   ├── agent_a_system.txt  # System prompt for Agent A
│   └── agent_b_system.txt  # System prompt for Agent B
│
├── tests/
│   ├── test_surface.py
│   ├── test_observation.py
│   ├── test_episode.py
│   └── test_agents.py
│
├── configs/
│   ├── surfaces.yaml       # Pre-defined test surfaces
│   └── experiments.yaml    # Experiment configurations
│
├── results/                # Output directory for runs
│   └── .gitkeep
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 10. API Specifications

### 10.1 Surface Class

```python
class Surface:
    """A 2D surface defined by Gaussian peaks."""

    def __init__(self, peaks: List[dict], domain_size: float = 10.0):
        """
        Args:
            peaks: List of peak definitions, each with keys:
                   - cx: float, x-coordinate of peak center
                   - cy: float, y-coordinate of peak center
                   - height: float, peak height (typically 0-1)
                   - sigma: float, peak width (standard deviation)
            domain_size: Size of domain [0, domain_size] × [0, domain_size]
        """
        pass

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate surface at point (x, y)."""
        pass

    def gradient(self, x: float, y: float, eps: float = 0.001) -> Tuple[float, float]:
        """Compute gradient (∂f/∂x, ∂f/∂y) at point (x, y)."""
        pass

    def get_optimal(self) -> Tuple[float, float, float]:
        """
        Return (x_opt, y_opt, f_opt) for global maximum.

        Implementation note: For sum of Gaussians, use grid search or scipy.optimize.
        Simple approach: return center of highest peak (approximate).
        Precise approach:
            from scipy.optimize import minimize
            result = minimize(lambda p: -self.evaluate(p[0], p[1]),
                             x0=[5, 5], bounds=[(0, L), (0, L)], method='L-BFGS-B')
            return (*result.x, -result.fun)
        """
        pass

    def to_dict(self) -> dict:
        """Serialize surface configuration."""
        pass

    @classmethod
    def from_dict(cls, config: dict) -> 'Surface':
        """Deserialize surface from configuration."""
        pass
```

### 10.2 ObservationGenerator Class

```python
class ObservationGenerator:
    """Generates agent observations from surface."""

    def __init__(self, surface: Surface, radius: float = 1.5, n_samples: int = 11):
        """
        Args:
            surface: The Surface to observe
            radius: Observation radius R
            n_samples: Number of samples in slice
        """
        pass

    def generate_observation_a(self, x_a: float, y_b: float) -> dict:
        """
        Generate observation for Agent A (horizontal slice).

        Returns:
            {
                "position": {"x": x_a, "y": y_b},
                "value_at_position": float,
                "gradient_x": float,
                "slice": [{"x": float, "value": float}, ...]
            }
        """
        pass

    def generate_observation_b(self, x_a: float, y_b: float) -> dict:
        """
        Generate observation for Agent B (vertical slice).

        Returns:
            {
                "position": {"x": x_a, "y": y_b},
                "value_at_position": float,
                "gradient_y": float,
                "slice": [{"y": float, "value": float}, ...]
            }
        """
        pass
```

### 10.3 Episode Class

```python
class Episode:
    """Manages episode state and progression."""

    def __init__(
        self,
        surface: Surface,
        initial_position: Tuple[float, float] = (5.0, 5.0),
        n_turns: int = 10,
        observation_generator: ObservationGenerator = None
    ):
        pass

    @property
    def current_position(self) -> Tuple[float, float]:
        """Current (x_a, y_b) position."""
        pass

    @property
    def turn_number(self) -> int:
        """Current turn (1-indexed)."""
        pass

    def step(self, x_new: float, y_new: float) -> Tuple[dict, dict]:
        """
        Move to new position and generate observations.

        Returns:
            (observation_a, observation_b)
        """
        pass

    def is_done(self) -> bool:
        """True if all turns completed."""
        pass

    def get_score(self, x_final: float, y_final: float) -> float:
        """Compute normalized score for final position."""
        pass

    def get_history(self) -> List[dict]:
        """Return full history of positions and observations."""
        pass
```

### 10.4 BaseAgent Class

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for agents."""

    def __init__(self, role: str):
        """
        Args:
            role: 'A' (controls x) or 'B' (controls y)
        """
        self.role = role
        self.observation_history = []
        self.message_history = []

    def receive_observation(self, observation: dict) -> None:
        """Store observation from environment."""
        self.observation_history.append(observation)

    def receive_message(self, message: str) -> None:
        """Store message from other agent."""
        self.message_history.append(message)

    @abstractmethod
    def generate_message(self) -> str:
        """Generate message to send to other agent."""
        pass

    @abstractmethod
    def decide_position(self) -> float:
        """Decide next position along controlled axis."""
        pass

    @abstractmethod
    def final_decision(self) -> float:
        """Make final position decision."""
        pass

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.observation_history = []
        self.message_history = []
```

### 10.5 LLMAgent Class

```python
class LLMAgent(BaseAgent):
    """
    LLM-based agent using API calls.

    API Key Configuration:
        Keys are read from environment variables:
        - OPENAI_API_KEY for GPT models (gpt-4, gpt-3.5-turbo)
        - ANTHROPIC_API_KEY for Claude models (claude-3-opus, claude-3-sonnet)

        Alternatively, pass api_key parameter to constructor.
    """

    def __init__(
        self,
        role: str,
        model: str = "gpt-4",
        api_key: str = None,
        system_prompt_path: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Args:
            role: 'A' or 'B'
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            api_key: API key (or uses environment variable)
            system_prompt_path: Path to system prompt file
            temperature: Sampling temperature
            max_tokens: Max tokens per response
        """
        pass

    def _format_observation(self, observation: dict) -> str:
        """Format observation as human-readable text."""
        pass

    def _build_prompt(self, include_decision: bool = False) -> List[dict]:
        """Build message list for API call."""
        pass

    def _parse_coordinate(self, response: str) -> float:
        """
        Extract coordinate from LLM response.

        Implementation:
            import re
            # Find all numbers in response
            matches = re.findall(r'[-+]?\d*\.?\d+', response)
            if matches:
                # Take the last number (usually the final answer)
                value = float(matches[-1])
                # Clamp to valid domain
                return max(0.0, min(self.domain_size, value))
            raise ValueError(f"Could not parse coordinate from: {response}")
        """
        pass

    def generate_message(self) -> str:
        """Call LLM to generate message."""
        pass

    def decide_position(self) -> float:
        """Call LLM to decide position."""
        pass

    def final_decision(self) -> float:
        """Call LLM for final decision."""
        pass

    # Error Handling Configuration
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30

    """
    Error Handling:
        On API error: Exponential backoff (1s, 2s, 4s) up to MAX_RETRIES
        On parsing failure: Retry with clarifying prompt:
            "Please respond with just a single number between 0 and 10."
        On repeated failure: Return current position (no movement)

        All errors should be logged for debugging.
    """
```

---

## 11. Prompt Templates

### 11.1 Agent A System Prompt

**File:** `prompts/agent_a_system.txt`

```
You are Agent A in a cooperative search task. You and Agent B are working together to find the highest point on a 2D surface.

SETUP:
- The surface is a function f(x, y) defined on [0, 10] × [0, 10]
- You control the x-coordinate
- Agent B controls the y-coordinate
- Together you determine the viewing position (x, y)

YOUR OBSERVATIONS:
- You see a HORIZONTAL slice of the surface at the current y position
- You see f(x', y) for x' values near your current x position
- You see ∂f/∂x (the gradient in the x-direction)
- You do NOT see how the surface varies in the y-direction

AGENT B's OBSERVATIONS:
- Agent B sees a VERTICAL slice at the current x position
- Agent B sees ∂f/∂y (the gradient in the y-direction)
- Agent B does NOT see the x-direction variation

YOUR GOAL:
- Communicate with Agent B to combine your partial views
- Together, locate the GLOBAL MAXIMUM of the surface
- You have limited turns, so explore strategically

EACH TURN:
1. You receive your observation (slice + gradient)
2. You exchange messages with Agent B
3. You decide where to move in x (Agent B decides y)

OUTPUT FORMAT:
When asked for a message, write your observations and reasoning.
When asked for a position, respond with a single number between 0 and 10.

Think step by step. Combine your horizontal view with Agent B's vertical view to build a 2D picture of the surface.
```

### 11.2 Agent B System Prompt

**File:** `prompts/agent_b_system.txt`

```
You are Agent B in a cooperative search task. You and Agent A are working together to find the highest point on a 2D surface.

SETUP:
- The surface is a function f(x, y) defined on [0, 10] × [0, 10]
- You control the y-coordinate
- Agent A controls the x-coordinate
- Together you determine the viewing position (x, y)

YOUR OBSERVATIONS:
- You see a VERTICAL slice of the surface at the current x position
- You see f(x, y') for y' values near your current y position
- You see ∂f/∂y (the gradient in the y-direction)
- You do NOT see how the surface varies in the x-direction

AGENT A's OBSERVATIONS:
- Agent A sees a HORIZONTAL slice at the current y position
- Agent A sees ∂f/∂x (the gradient in the x-direction)
- Agent A does NOT see the y-direction variation

YOUR GOAL:
- Communicate with Agent A to combine your partial views
- Together, locate the GLOBAL MAXIMUM of the surface
- You have limited turns, so explore strategically

EACH TURN:
1. You receive your observation (slice + gradient)
2. You exchange messages with Agent A
3. You decide where to move in y (Agent A decides x)

OUTPUT FORMAT:
When asked for a message, write your observations and reasoning.
When asked for a position, respond with a single number between 0 and 10.

Think step by step. Combine your vertical view with Agent A's horizontal view to build a 2D picture of the surface.
```

### 11.3 Turn Prompt Template

```
=== TURN {turn_number} of {total_turns} ===

CURRENT POSITION: ({x_a:.2f}, {y_b:.2f})

YOUR OBSERVATION:
{formatted_observation}

MESSAGE FROM {other_agent}:
"{other_agent_message}"

---

First, send a message to {other_agent} describing what you observe and your thoughts.
Then, decide your next {coordinate_name}-coordinate.

YOUR MESSAGE:
```

### 11.4 Final Decision Prompt Template

```
=== FINAL DECISION ===

You have completed {total_turns} turns of exploration.

SUMMARY OF YOUR OBSERVATIONS:
{observation_summary}

CONVERSATION HISTORY:
{message_history}

Based on all the information gathered, what is your final answer for the {coordinate_name}-coordinate of the global maximum?

Provide your reasoning, then state your final {coordinate_name} value (a single number between 0 and 10).

YOUR FINAL {coordinate_name}:
```

---

## 12. Test Cases

### 12.1 Surface Tests

```python
# test_surface.py

def test_single_peak_evaluation():
    """Surface with single peak evaluates correctly at peak."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    assert abs(surface.evaluate(5.0, 5.0) - 1.0) < 0.01
    assert surface.evaluate(0.0, 0.0) < 0.1

def test_gradient_points_toward_peak():
    """Gradient should point toward peak."""
    surface = Surface([{"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    gx, gy = surface.gradient(5.0, 5.0)
    assert gx > 0  # Peak is to the right
    assert gy > 0  # Peak is above

def test_gradient_zero_at_peak():
    """Gradient should be near zero at peak."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    gx, gy = surface.gradient(5.0, 5.0)
    assert abs(gx) < 0.01
    assert abs(gy) < 0.01

def test_get_optimal():
    """get_optimal returns correct peak location."""
    surface = Surface([
        {"cx": 3.0, "cy": 3.0, "height": 0.5, "sigma": 1.0},
        {"cx": 7.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
    ])
    x_opt, y_opt, f_opt = surface.get_optimal()
    assert abs(x_opt - 7.0) < 0.1
    assert abs(y_opt - 8.0) < 0.1
    assert abs(f_opt - 1.0) < 0.01

def test_multi_peak_surface():
    """Multiple peaks superimpose correctly."""
    surface = Surface([
        {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
        {"cx": 8.0, "cy": 8.0, "height": 0.5, "sigma": 1.0}
    ])
    # At each peak
    assert abs(surface.evaluate(2.0, 2.0) - 0.5) < 0.01
    assert abs(surface.evaluate(8.0, 8.0) - 0.5) < 0.01
    # At center (far from both peaks)
    assert surface.evaluate(5.0, 5.0) < 0.1
```

### 12.2 Observation Tests

```python
# test_observation.py

def test_horizontal_slice_samples():
    """Horizontal slice has correct number of samples."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    assert len(obs["slice"]) == 11

def test_slice_range():
    """Slice covers correct x range."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    x_values = [s["x"] for s in obs["slice"]]
    assert min(x_values) == 3.5  # 5.0 - 1.5
    assert max(x_values) == 6.5  # 5.0 + 1.5

def test_boundary_clipping():
    """Slice clips at domain boundary."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(0.5, 5.0)  # Near left edge
    x_values = [s["x"] for s in obs["slice"]]
    assert min(x_values) >= 0.0  # Clipped to boundary

def test_gradient_in_observation():
    """Observation includes correct gradient."""
    surface = Surface([{"cx": 7.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=1.5, n_samples=11)
    obs = obs_gen.generate_observation_a(5.0, 5.0)
    assert obs["gradient_x"] > 0  # Peak is to the right
```

### 12.3 Episode Tests

```python
# test_episode.py

def test_episode_initialization():
    """Episode starts at correct position."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=10)
    assert episode.current_position == (5.0, 5.0)
    assert episode.turn_number == 0

def test_episode_step():
    """Step updates position and returns observations."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=10)
    obs_a, obs_b = episode.step(6.0, 7.0)
    assert episode.current_position == (6.0, 7.0)
    assert episode.turn_number == 1
    assert "gradient_x" in obs_a
    assert "gradient_y" in obs_b

def test_episode_completion():
    """Episode is done after N turns."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)
    for i in range(3):
        assert not episode.is_done()
        episode.step(5.0, 5.0)
    assert episode.is_done()

def test_perfect_score():
    """Score is 1.0 when finding exact optimal."""
    surface = Surface([{"cx": 7.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)
    score = episode.get_score(7.0, 8.0)
    assert abs(score - 1.0) < 0.01
```

### 12.4 Agent Tests

```python
# test_agents.py

def test_random_agent_bounds():
    """Random agent stays within bounds."""
    agent = RandomAgent(role='A', seed=42)
    for _ in range(100):
        pos = agent.decide_position()
        assert 0 <= pos <= 10

def test_random_agent_deterministic():
    """Random agent is deterministic with same seed."""
    agent1 = RandomAgent(role='A', seed=42)
    agent2 = RandomAgent(role='A', seed=42)
    for _ in range(10):
        assert agent1.decide_position() == agent2.decide_position()

def test_greedy_agent_follows_gradient():
    """Greedy agent moves in gradient direction."""
    agent = GreedyAgent(role='A', step_size=1.0)
    # Positive gradient should increase position
    agent.receive_observation({"gradient_x": 0.5, "position": {"x": 5.0, "y": 5.0}})
    new_pos = agent.decide_position()
    assert new_pos > 5.0
```

---

## 13. Visualization Requirements

### 13.1 3D Surface Plot

**Function:** `plot_surface(surface, ax=None, show=True, save_path=None)`

**Requirements:**
- Grid resolution: 50×50 points minimum
- Use colormap: 'viridis' or 'plasma'
- Add colorbar showing z-values
- Mark global maximum with red sphere/point
- Label axes: "x", "y", "f(x,y)"
- Optional: wireframe overlay

**Example output:**
```
     z (height)
     ▲
     │    ╱╲
     │   ╱  ╲
     │  ╱    ╲    ╱╲
     │ ╱      ╲__╱  ╲
     │╱              ╲
     └──────────────────► x
    ╱
   ╱
  y
```

### 13.2 Episode Trajectory Plot

**Function:** `plot_episode(surface, history, ax=None, show=True, save_path=None)`

**Requirements:**
- Base: 3D surface (semi-transparent)
- Trajectory: line connecting all positions
- Turn markers: numbered spheres at each position
- Color coding:
  - Green: start position
  - Yellow: intermediate positions
  - Blue: final position
  - Red: optimal position
- Legend explaining markers

### 13.3 Slice Visualization

**Function:** `plot_slices(surface, x_a, y_b, radius, ax=None)`

**Requirements:**
- Two subplots side by side
- Left: horizontal slice f(x, y_b) for x in [x_a-R, x_a+R]
- Right: vertical slice f(x_a, y) for y in [y_b-R, y_b+R]
- Mark current position with vertical line
- Show gradient as arrow from current position
- Title showing position coordinates

### 13.4 Interactive Dashboard (Optional)

**File:** `visualization/dashboard.py`

**Requirements:**
- Plotly-based interactive 3D plot
- Slider to step through turns
- Display current observation slices
- Show message history panel
- Useful for debugging and demos

---

## 14. Baselines

### 14.1 Random Baseline

**Behavior:**
- Each turn, choose position uniformly at random from [0, 10]
- No communication (empty messages)
- Final decision: random position

**Expected performance:**
- Mean score: ~0.1-0.2 (depending on surface)
- High variance
- Establishes lower bound

### 14.2 Greedy Gradient Baseline

**Behavior:**
- Each turn, move step_size in gradient direction
- position_new = clamp(position_old + step_size * sign(gradient), 0, 10)
- No communication
- Final decision: current position

**Expected performance:**
- Mean score: ~0.5-0.8
- Should find A peak, but not necessarily global max
- May get stuck in local maxima

### 14.3 Oracle Baseline (Upper Bound)

**Behavior:**
- Agents receive FULL surface (no information asymmetry)
- Both can compute optimal directly
- No communication needed

**Expected performance:**
- Score: 1.0 (always finds global max)
- Establishes upper bound
- Validates scoring system

### 14.4 No-Communication LLM Baseline

**Behavior:**
- LLM agents receive observations
- Messages are blocked (empty strings)
- Must decide based only on own observations

**Expected performance:**
- Should perform between greedy and full-communication LLM
- Quantifies value of communication

---

## 15. Evaluation Protocol

### 15.1 Experiment Matrix

| Condition | Surfaces | Trials per surface | Total runs |
|-----------|----------|-------------------|------------|
| Random baseline | 5 test surfaces | 10 | 50 |
| Greedy baseline | 5 test surfaces | 10 | 50 |
| LLM (no comm) | 5 test surfaces | 5 | 25 |
| LLM (with comm) | 5 test surfaces | 5 | 25 |

### 15.2 Statistical Analysis

**Metrics to report:**
- Mean score ± standard deviation per condition
- Score distribution (histogram)
- Pairwise comparisons with t-tests or Mann-Whitney U

**Visualization:**
- Bar chart with error bars
- Box plots per condition
- Learning curves (if applicable)

### 15.3 Qualitative Analysis

For LLM runs, manually inspect:
- Communication quality (do agents share useful info?)
- Coordination (do they converge on same target?)
- Failure modes (where do they go wrong?)

Categorize failures as:
- Exploration failure (didn't find the peak region)
- Communication failure (shared wrong/incomplete info)
- Coordination failure (disagreed on target)
- Navigation failure (identified peak but moved wrong)

### 15.4 Reproducibility Requirements

- All random seeds must be logged
- LLM API temperature and parameters logged
- Surface configurations saved
- Full transcripts saved
- Code version tracked (git commit hash)

---

## Appendix A: Dependencies

```
# requirements.txt

numpy>=1.21.0
matplotlib>=3.5.0
plotly>=5.0.0  # For interactive viz
openai>=1.0.0  # For GPT API
anthropic>=0.5.0  # For Claude API
pyyaml>=6.0.0  # For config files
pytest>=7.0.0  # For testing
tqdm>=4.60.0  # For progress bars
pandas>=1.3.0  # For analysis
scipy>=1.7.0  # For statistics
```

---

## Appendix B: Quick Reference

### Key Parameters
```
Domain:     [0, 10] × [0, 10]
Radius:     R = 1.5
Samples:    S = 11
Turns:      N = 10
Start:      (5.0, 5.0)
```

### Score Interpretation
```
1.0      = Perfect (found global max)
0.9-0.99 = Excellent
0.7-0.9  = Good (secondary peak or close)
0.5-0.7  = Moderate
< 0.5    = Poor
```

### File Locations
```
Surface code:      core/surface.py
Observation code:  core/observation.py
Episode code:      core/episode.py
LLM agent:         agents/llm_agent.py
Prompts:           prompts/agent_a_system.txt, prompts/agent_b_system.txt
Results:           results/
```

---

## Appendix C: Checkpoint Summary

Copy this checklist to track progress:

```
PHASE 0: ENVIRONMENT SETUP
[X] 0.1 Create project directory structure
[X] 0.2 Create requirements.txt with all dependencies
[X] 0.3 Create virtual environment
[X] 0.4 Install dependencies (43 packages)
[X] 0.5 Configure .gitignore

PHASE 1: CORE ENGINE
[X] 1.1 Surface class
[X] 1.2 Observation generator
[ ] 1.3 Episode state manager

PHASE 2: AGENT FRAMEWORK
[ ] 2.1 Base agent interface
[ ] 2.2 Random baseline agent
[ ] 2.3 Greedy gradient agent
[ ] 2.4 LLM agent

PHASE 3: EPISODE RUNNER
[ ] 3.1 Turn executor
[ ] 3.2 Result logger

PHASE 4: VISUALIZATION
[ ] 4.1 3D surface plot
[ ] 4.2 Agent position overlay
[ ] 4.3 Slice visualization
[ ] 4.4 Animation (optional)

PHASE 5: EVALUATION
[ ] 5.1 Batch runner
[ ] 5.2 Statistical analysis
[ ] 5.3 Baseline comparison
```

---

## Appendix D: Model Assignment (Opus vs Sonnet)

### Summary Table

| Checkpoint | Model | Rationale |
|------------|-------|-----------|
| 1.1 Surface | Sonnet | Math is straightforward |
| 1.2 Observation | Sonnet | Data transformation |
| 1.3 Episode | Sonnet | State machine |
| 2.1 Base Agent | Sonnet | Boilerplate ABC |
| 2.2 Random Agent | Sonnet | Trivial |
| 2.3 Greedy Agent | Sonnet | Simple logic |
| **2.4 LLM Agent** | **OPUS** ⭐ | Complex API + parsing edge cases |
| **3.1 Turn Executor** | **OPUS** ⭐ | Orchestration correctness critical |
| 3.2 Logger | Sonnet | File I/O |
| 4.1-4.4 Visualization | Sonnet | Library-driven |
| 5.1 Batch Runner | Sonnet | Loop logic |
| **5.2 Statistics** | **OPUS** ⭐ | Research validity |
| 5.3 Baselines | Sonnet | Aggregation |
| **Prompts** | **OPUS** ⭐ | LLM behavior understanding |

### Opus-Required Components (Critical Thinking Needed)

**2.4 LLM Agent** — MOST COMPLEX
- API integration for OpenAI AND Anthropic (different SDKs)
- Response parsing with edge cases:
  - "My answer is 7.5" → extract 7.5
  - "approximately 7 to 8, let's say 7.5" → which number?
  - "Based on gradient 0.12, I should move to 6.8" → 0.12 or 6.8?
- Error handling: retries, backoff, timeouts
- A parsing bug silently corrupts ALL experiment data

**3.1 Turn Executor** — ORCHESTRATION CRITICAL
- Must correctly sequence: observe → A message → B message → decide → move
- Speaker alternation (odd turns: A first, even turns: B first)
- Wrong observation routing won't crash — produces garbage results

**5.2 Statistical Analysis** — RESEARCH VALIDITY
- Test selection: t-test vs Mann-Whitney (assumptions matter)
- Small sample handling (5 trials has low power)
- Multiple comparisons problem
- Wrong test → invalid dissertation conclusions

**Prompts** — LLM BEHAVIOR
- Subtle wording affects agent performance
- Must elicit parseable responses
- Failure mode anticipation

### Sonnet-Appropriate Components (Straightforward)

- **Phase 1**: Surface, Observation, Episode — math and state machines
- **Phase 2 (2.1-2.3)**: Base Agent, Random, Greedy — boilerplate and simple logic
- **Phase 3 (3.2)**: Logger — file I/O
- **Phase 4**: All visualization — library-driven, visual bugs are obvious
- **Phase 5 (5.1, 5.3)**: Batch Runner, Baselines — loops and aggregation

### Execution Order

**Sonnet Phase (build foundation first):**
```
1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 2.3 → 3.2 → 4.1 → 4.2 → 4.3 → 5.1 → 5.3
```

**Opus Phase (critical components):**
```
Prompts (write first, needed for 2.4)
   ↓
2.4 LLM Agent (depends on prompts)
   ↓
3.1 Turn Executor (depends on 2.4)
   ↓
5.2 Statistical Analysis (after data collected)
```

### Architecture Review Points (Opus)

1. After Phase 1: Review core classes before building agents
2. After Phase 2: Review agent integration points
3. After Phase 3: Review full episode pipeline
4. Before submission: Final correctness review

### Rationale

**Philosophy**: Use Opus where bugs would silently corrupt results rather than crash.

- A parsing error in LLM Agent produces wrong coordinates, not exceptions
- Swapped observations in Turn Executor runs fine but data is garbage
- Wrong statistical test gives valid-looking p-value but wrong conclusion

**The math**:
- Opus handles ~25% of components but ~75% of complexity
- Cost is higher per token, but debugging silent failures costs hours

---

*Document version: 1.1*
*Last updated: 2025-01-12*
*Author: Research planning session*
