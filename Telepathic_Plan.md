# Telepathic Benchmark: Implementation Plan

> **Purpose:** This document contains ALL context needed to implement the Telepathic benchmark. Each section is self-contained with checkpoints that can be marked complete. A fresh Claude instance should be able to continue work from any checkpoint.

> **Status:** DRAFT - Ready for implementation

> **Relationship to Other Benchmarks:** This benchmark shares the same repository as the Coordination (2-agent spatial) and Temporal (1-agent dynamics) benchmarks. Shared utilities are in `shared/`. See Section 9 for file structure.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mathematical Specification](#2-mathematical-specification)
3. [Function Domain](#3-function-domain)
4. [Observation Model](#4-observation-model)
5. [Episode Structure](#5-episode-structure)
6. [Communication Protocol](#6-communication-protocol)
7. [Scoring System](#7-scoring-system)
8. [Function Library & Complexity Tiers](#8-function-library--complexity-tiers)
9. [Implementation Checkpoints](#9-implementation-checkpoints)
10. [File Structure](#10-file-structure)
11. [API Specifications](#11-api-specifications)
12. [Prompt Templates](#12-prompt-templates)
13. [Test Cases](#13-test-cases)
14. [Visualization Requirements](#14-visualization-requirements)
15. [Baselines](#15-baselines)
16. [Evaluation Protocol](#16-evaluation-protocol)

---

## 1. Project Overview

### 1.1 Research Question

Can LLM agents **compress and transmit algorithmic rules** through a bandwidth-constrained channel, and does success require genuine compositional reasoning rather than memorization?

### 1.2 Core Concept: Agentic Fluidity

Two LLM agents play a **communication game** testing whether they can CREATE and LEARN a compositional language:

- **The Seer** invents a language mapping functions to abstract tokens, then teaches it to the Doer
- **The Doer** learns the language from demonstrations (no natural language), then applies it
- **The Constraint** forces genuine communication: NO natural language allowed, only tokens + numbers
- **Success** requires the Seer to CREATE a compositional grammar and the Doer to LEARN and GENERALIZE it

### 1.2.1 The "Agentic Fluidity" Hypothesis

**The Core Insight:** Current AI benchmarks test Fluid Reasoning in isolation (single-agent puzzles) and Communication in isolation (multi-agent English chat). **Agentic Fluidity** is the intersection: inducing a novel abstract concept AND inventing a protocol to transmit it.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC FLUIDITY                             │
│                                                                 │
│   True fluid intelligence occurs when agents face a phenomenon  │
│   for which NO WORD EXISTS in their training data.              │
│                                                                 │
│   The Seer must:                                                │
│   1. Recognize the function from samples                        │
│   2. INVENT a token encoding (α = sin, β = cos, ...)            │
│   3. TEACH this encoding to the Doer WITHOUT natural language   │
│                                                                 │
│   The Doer must:                                                │
│   1. LEARN token meanings from pure demonstration               │
│   2. GENERALIZE to compositions never explicitly shown          │
│                                                                 │
│   If they succeed, they haven't just solved a puzzle—           │
│   they have ALIGNED THEIR LATENT SPACES to create a shared,     │
│   ad-hoc language.                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2.2 Two-Level Generalization Test

```
LEVEL 1: Seer generalizes "how to make language"
├── Meta-trained with example language (x, y, z)
├── Must CREATE new language with (α, β, γ, δ, ε)
├── Must design TEACHABLE, COMPOSITIONAL encodings
└── Tests: Can LLMs do meta-learning for language creation?

LEVEL 2: Doer generalizes the invented language
├── Learns primitives α, β, γ from demonstrations
├── Tested on compositions (e.g., α-γ) never explicitly shown
└── Tests: Can LLMs learn compositional grammar from examples?
```

### 1.2.3 The Critical Constraint: NO Natural Language

```
┌─────────────────────────────────────────────────────────────────┐
│                 COMMUNICATION CONSTRAINTS                        │
│                                                                 │
│   ALLOWED:                                                      │
│   ├── Abstract tokens: α, β, γ, δ, ε                            │
│   ├── Numbers: 0, 1, 2, -0.84, 0.91, etc.                       │
│   ├── Structural symbols: → | - (for composition)               │
│   └── Line breaks / spacing                                     │
│                                                                 │
│   NOT ALLOWED:                                                  │
│   ├── Words: "means", "apply", "first", "example", "rule"       │
│   ├── Any natural language explanation                          │
│   └── Meta-commentary about the protocol                        │
│                                                                 │
│   This forces PURE DEMONSTRATION, not explanation.              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2.4 How Numbers Ground Abstract Tokens

The abstract tokens (α, β, γ) have **no inherent meaning**. Numbers provide the GROUNDING that makes them meaningful:

```
WITHOUT NUMBERS:
┌─────────────────────────┐
│ α                       │  → Doer: "α could mean anything.
│                         │          I have no idea."
└─────────────────────────┘

WITH NUMBERS:
┌─────────────────────────┐
│ α                       │  → Doer: "When input is 0, output is 0.
│ 0 → 0                   │          When input is 1, output is 0.84.
│ 1 → 0.84                │          This pattern looks like sin(x)!
│ 2 → 0.91                │          So α must mean sin."
└─────────────────────────┘

The numbers ARE the meaning. They show what the function DOES.
```

**Analogy:** Teaching "red" to an alien without shared language:
- Just saying "red" = meaningless sound
- Pointing at red apple, red fire truck, red blood = grounding through demonstration

**For composition:**
```
Seer teaches:
  α: 0→0, 1→0.84, 2→0.91        (Doer infers: α = sin)
  γ: 0→0, 1→1, 2→4              (Doer infers: γ = square)
  α-γ: 1→0.84                   (Doer sees ONE example)

Doer reasons:
  "α-γ at input 1 gives 0.84
   If I apply γ first: γ(1) = 1
   Then apply α: α(1) = 0.84 ✓
   So α-γ means: apply γ, then α!"

The numbers let Doer VERIFY their hypothesis about composition order.
```

### 1.3 The "Shannon Barrier" Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE SHANNON BARRIER                          │
│                                                                 │
│   If:   bandwidth (k tokens) << data (n sample points)          │
│   Then: memorization is MATHEMATICALLY IMPOSSIBLE               │
│                                                                 │
│   The ONLY way to succeed is to:                                │
│   1. Identify the underlying function (compression)             │
│   2. Encode it in abstract tokens (protocol invention)          │
│   3. Decode and apply to new input (generalization)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Why This Tests Compositional Reasoning

| Property | How It's Achieved |
|----------|-------------------|
| Forced compression | Bandwidth << data makes memorization impossible |
| Compositional structure | Functions are compositions of primitives (f ∘ g) |
| Measurable complexity | Token count vs K(f) estimate |
| Generalization test | Novel compositions of known primitives |
| Binary success | Output matches expected (within tolerance) or not |

### 1.5 Connection to Algorithmic Information Theory

This benchmark operationalizes Kolmogorov complexity:

- **K(f)** = length of shortest program computing f
- **Seer's task** = find minimal description of f
- **Message length** = upper bound on K(f) the agents can express
- **Φ metric** = measures if agents learned grammar vs lookup table

### 1.6 Comparison with Other Benchmarks

| Benchmark | Agents | Domain | Information Asymmetry | K-Complexity Test |
|-----------|--------|--------|----------------------|-------------------|
| Coordination | 2 | f(x,y) spatial | Perpendicular slices | Weak (implicit) |
| Temporal | 1 | f(x,t) evolving | Hidden dynamics | Medium (implicit) |
| **Telepathic** | 2 | f: ℝ→ℝ functions | **Bandwidth bottleneck** | **Strong (explicit)** |

### 1.7 Success Criteria

- [ ] Agents with shared vocabulary outperform random baseline significantly
- [ ] Φ ≈ 1 on novel compositions indicates compositional grammar
- [ ] Φ ≈ 0 on novel compositions indicates memorization
- [ ] Performance degrades gracefully with function complexity
- [ ] Agents fail on incompressible functions (sanity check)
- [ ] Results are reproducible across runs

---

## 2. Mathematical Specification

### 2.1 Domain

```
Function domain:     x ∈ [-5, 5] (input range for sampling)
Function codomain:   f(x) ∈ ℝ (real-valued outputs)
Sample points:       5 fixed points per function
Test points:         1-3 novel x values for evaluation
```

### 2.2 Agents

```
Seer (Sender):       Observes samples, produces message
Doer (Receiver):     Observes message + test input, produces output
Same LLM model:      Both agents use identical model (e.g., GPT-4)
```

### 2.3 Communication Channel

```
Vocabulary:          V = {α, β, γ, δ, ε, ζ, η, θ, ι, κ}  (10 tokens)
Vocabulary size:     |V| = 10
Max message length:  k = 5 tokens
Message space:       M ⊆ V* where |m| ≤ k for all m ∈ M
Channel capacity:    log₂(Σᵢ₌₁ᵏ |V|ⁱ) ≈ 16.6 bits
```

### 2.4 Function Space

```
Primitives:          P = {sin, cos, square, sqrt, abs, neg, exp, log, relu, sign}
Primitive count:     |P| = 10
Composition depth:   d ∈ {1, 2, 3} (for proof-of-concept)
Function:            f = pₐ ∘ pᵦ ∘ ... where pᵢ ∈ P
```

### 2.5 Success Criterion

```
For test input x_test and expected output y_expected = f(x_test):

success = |y_predicted - y_expected| / |y_expected| < ε

where ε = 0.01 (1% relative tolerance)

Special case: if |y_expected| < 0.01, use absolute tolerance:
success = |y_predicted - y_expected| < 0.01
```

---

## 3. Function Domain

### 3.1 Primitive Functions

All primitives are unparameterized functions f: ℝ → ℝ.

```python
PRIMITIVES = {
    # Trigonometric
    "sin": lambda x: math.sin(x),
    "cos": lambda x: math.cos(x),
    
    # Power functions
    "square": lambda x: x ** 2,
    "sqrt": lambda x: math.sqrt(abs(x)),  # Safe sqrt
    
    # Absolute/Sign
    "abs": lambda x: abs(x),
    "neg": lambda x: -x,
    "sign": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
    
    # Exponential/Log
    "exp": lambda x: math.exp(min(x, 10)),  # Capped to avoid overflow
    "log": lambda x: math.log(abs(x) + 0.01),  # Safe log
    
    # Neural network activation
    "relu": lambda x: max(0, x),
}
```

### 3.2 Function Safety

All primitives are made "safe" to handle edge cases:

```python
def safe_sqrt(x):
    """Square root of absolute value to handle negatives."""
    return math.sqrt(abs(x))

def safe_log(x):
    """Logarithm with offset to handle zero/negatives."""
    return math.log(abs(x) + 0.01)

def safe_exp(x):
    """Exponential with cap to prevent overflow."""
    return math.exp(min(x, 10))
```

### 3.3 Composition Operator

```python
def compose(*functions):
    """
    Compose functions right-to-left: (f ∘ g)(x) = f(g(x))
    
    compose(f, g, h)(x) = f(g(h(x)))
    """
    def composed(x):
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return composed

# Example: sin(x²) = compose(sin, square)
f = compose(PRIMITIVES["sin"], PRIMITIVES["square"])
f(2.0)  # Returns sin(4) ≈ -0.757
```

### 3.4 Sample Point Selection

Fixed sample points chosen to reveal function structure:

```python
SAMPLE_X_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]  # 5 points

def generate_samples(f):
    """Generate (x, f(x)) pairs for a function."""
    return [(x, f(x)) for x in SAMPLE_X_VALUES]

# Example output for f(x) = x²
# [(-2.0, 4.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]
```

### 3.5 Test Point Selection

Test points are different from sample points:

```python
TEST_X_VALUES = [-1.5, 0.5, 1.5]  # 3 test points, none in training

def generate_test(f):
    """Generate test cases for evaluation."""
    return [(x, f(x)) for x in TEST_X_VALUES]
```

### 3.6 Few-Shot Vocabulary Grounding

The benchmark uses few-shot examples to teach agents the token→function mappings. This section defines exactly what is shown vs. held out.

#### 3.6.1 Primitives Shown in Few-Shot (5 of 10)

```python
FEW_SHOT_PRIMITIVES = {
    "α": {
        "function": "sin",
        "samples": [(-2.0, -0.9093), (-1.0, -0.8415), (0.0, 0.0), (1.0, 0.8415), (2.0, 0.9093)],
        "test_examples": [(0.5, 0.4794), (1.5, 0.9975), (-1.5, -0.9975)]
    },
    "β": {
        "function": "cos",
        "samples": [(-2.0, -0.4161), (-1.0, 0.5403), (0.0, 1.0), (1.0, 0.5403), (2.0, -0.4161)],
        "test_examples": [(0.5, 0.8776), (1.5, 0.0707), (-1.5, 0.0707)]
    },
    "γ": {
        "function": "square",
        "samples": [(-2.0, 4.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 4.0)],
        "test_examples": [(0.5, 0.25), (1.5, 2.25), (-1.5, 2.25)]
    },
    "δ": {
        "function": "abs",
        "samples": [(-2.0, 2.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        "test_examples": [(0.5, 0.5), (1.5, 1.5), (-1.5, 1.5)]
    },
    "ε": {
        "function": "neg",
        "samples": [(-2.0, 2.0), (-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (2.0, -2.0)],
        "test_examples": [(0.5, -0.5), (1.5, -1.5), (-1.5, 1.5)]
    }
}
```

#### 3.6.2 Compositions Shown in Few-Shot (2 only)

```python
FEW_SHOT_COMPOSITIONS = {
    "α-γ": {
        "description": "sin(x²) - apply γ (square) first, then α (sin)",
        "primitives": ["α", "γ"],
        "samples": [(-2.0, -0.7568), (-1.0, 0.8415), (0.0, 0.0), (1.0, 0.8415), (2.0, -0.7568)],
        "test_examples": [(0.5, 0.2474), (1.5, 0.7781)]
    },
    "δ-α": {
        "description": "|sin(x)| - apply α (sin) first, then δ (abs)",
        "primitives": ["δ", "α"],
        "samples": [(-2.0, 0.9093), (-1.0, 0.8415), (0.0, 0.0), (1.0, 0.8415), (2.0, 0.9093)],
        "test_examples": [(0.5, 0.4794), (1.5, 0.9975)]
    }
}
```

#### 3.6.3 What Is Held Out (THE KEY TEST)

```
FEW-SHOT (shown):         α-γ (sin∘square), δ-α (abs∘sin)
HELD-OUT (tested):        All other pairs of {α, β, γ, δ, ε}

Novel compositions to test (18 total):
  α-β, α-δ, α-ε           (sin composed with cos, abs, neg)
  β-α, β-γ, β-δ, β-ε      (cos composed with others)
  γ-α, γ-β, γ-δ, γ-ε      (square composed with others)
  δ-β, δ-γ, δ-ε           (abs composed with others, excluding δ-α shown)
  ε-α, ε-β, ε-γ, ε-δ      (neg composed with others)

This ensures we test GENERALIZATION, not memorization of examples.
```

#### 3.6.4 Primitives NOT in Few-Shot (Future Extension)

```python
# These 5 primitives are NOT shown in few-shot examples
# Testing these would require agents to INFER new token mappings
# This is a harder extension for future work

HELD_OUT_PRIMITIVES = {
    "ζ": "sqrt",
    "η": "exp",
    "θ": "log",
    "ι": "relu",
    "κ": "sign"
}
```

---

## 4. Observation Model

### 4.1 Seer's Observation

The Seer receives input-output samples but NOT the function identity:

```python
seer_observation = {
    "samples": [
        {"x": -2.0, "y": 4.0},
        {"x": -1.0, "y": 1.0},
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 4.0}
    ],
    "task": "Identify the pattern and encode it using the token vocabulary."
}
```

### 4.2 Doer's Observation

The Doer receives ONLY the message and test input:

```python
doer_observation = {
    "message": "γ",  # Abstract token(s) from Seer
    "test_input": 1.5,
    "task": "Decode the message and compute the output for the test input."
}
```

### 4.3 What Each Agent CANNOT See

| Agent | Cannot See |
|-------|------------|
| Seer | Function name, test input values |
| Doer | Training samples, function name, correct output |

### 4.4 Information Flow

```
┌─────────────┐                              ┌─────────────┐
│    SEER     │                              │    DOER     │
├─────────────┤                              ├─────────────┤
│ Observes:   │      Message (≤5 tokens)     │ Observes:   │
│ - 5 samples │  ─────────────────────────►  │ - Message   │
│ - Vocabulary│                              │ - Test x    │
│             │                              │ - Vocabulary│
│ Outputs:    │                              │ Outputs:    │
│ - Message   │                              │ - f(x) pred │
└─────────────┘                              └─────────────┘
```

---

## 5. Episode Structure

### 5.1 Complete Episode Flow (4 Steps)

The episode structure reflects the "Agentic Fluidity" design where the Seer INVENTS a language and TEACHES it to the Doer through pure demonstration.

```
┌─────────────────────────────────────────────────────────────────┐
│              STEP 0: META-TRAINING (Seer only)                  │
│                                                                 │
│  The Seer learns HOW to create language from an example.        │
│  This happens via the system prompt — NOT during the episode.   │
│                                                                 │
│  System prompt shows:                                           │
│    - Example language using (x, y, z) tokens                    │
│    - How to demonstrate primitives with input→output pairs      │
│    - How to demonstrate compositions                            │
│                                                                 │
│  The Seer must GENERALIZE to create a NEW language              │
│  using (α, β, γ, δ, ε) tokens.                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 1: LANGUAGE CREATION (Seer decides)               │
│                                                                 │
│  The Seer decides what each token means:                        │
│    - Seer sees samples of each primitive function               │
│    - Seer assigns tokens: α=?, β=?, γ=?, δ=?, ε=?               │
│                                                                 │
│  CONSTRAINT: Seer must be CONSISTENT across the episode.        │
│  Once α is assigned to sin, it must always mean sin.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 2: TEACHING PHASE (Seer → Doer)                 │
│                                                                 │
│  The Seer teaches token meanings to the Doer.                   │
│                                                                 │
│  CRITICAL CONSTRAINT: NO NATURAL LANGUAGE                       │
│  Seer can ONLY send:                                            │
│    - Tokens: α, β, γ, δ, ε                                      │
│    - Numbers: 0, 1, 2, -0.84, 0.91, etc.                        │
│    - Structural symbols: → | -                                  │
│                                                                 │
│  Example teaching message (for α = sin):                        │
│  ┌─────────────────────────────────────────┐                    │
│  │ α                                       │                    │
│  │ 0 → 0                                   │                    │
│  │ 1 → 0.84                                │                    │
│  │ 2 → 0.91                                │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
│  Doer must INFER: "α means sin(x)"                              │
│                                                                 │
│  Seer also teaches composition rule with ONE example:           │
│  ┌─────────────────────────────────────────┐                    │
│  │ α-γ                                     │                    │
│  │ 1 → 0.84                                │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
│  Doer must VERIFY: "α-γ at 1: γ(1)=1, α(1)=0.84 ✓"              │
│  Doer learns: "X-Y means apply Y first, then X"                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 3: TESTING PHASE (novel compositions)           │
│                                                                 │
│  Now we test whether the Doer learned COMPOSITIONAL GRAMMAR.    │
│                                                                 │
│  For each test trial:                                           │
│    1. Seer observes samples of a composition (e.g., β-γ)        │
│    2. Seer sends message: "β-γ"                                 │
│    3. Doer receives message + test input x                      │
│    4. Doer computes: γ(x), then β(γ(x))                         │
│    5. Evaluation: compare to expected                           │
│                                                                 │
│  KEY: β-γ was NEVER explicitly taught!                          │
│  Doer must GENERALIZE the composition rule.                     │
│                                                                 │
│  Success here = learned compositional grammar                   │
│  Failure here = only memorized taught examples                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Single Trial Flow (Within Testing Phase)

```
┌─────────────────────────────────────────────────────────────────┐
│                      SINGLE TRIAL                               │
│                                                                 │
│  1. SETUP                                                       │
│     - Select function f from library                            │
│     - Generate samples: [(x₁,f(x₁)), ..., (x₅,f(x₅))]          │
│     - Select test point: x_test                                 │
│                                                                 │
│  2. SEER PHASE                                                  │
│     - Seer receives: samples                                    │
│     - Seer outputs: message m (≤5 tokens)                       │
│     - NO natural language, only: tokens + numbers + symbols     │
│                                                                 │
│  3. DOER PHASE                                                  │
│     - Doer receives: message m + test input x_test              │
│     - Doer decodes using learned token meanings                 │
│     - Doer outputs: predicted y_pred                            │
│                                                                 │
│  4. EVALUATION                                                  │
│     - Compute: y_expected = f(x_test)                           │
│     - Success: |y_pred - y_expected| / |y_expected| < 0.01      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Curriculum Phases (Agentic Fluidity Design)

The curriculum tests TWO levels of generalization:
1. **Level 1:** Seer generalizes "how to make language" (meta-learning)
2. **Level 2:** Doer generalizes the invented language (compositional grammar)

```
┌─────────────────────────────────────────────────────────────────┐
│     TEACHING PHASE: PRIMITIVE DEMONSTRATION (5 functions)       │
│                                                                 │
│  Goal: Seer teaches Doer the 5 primitive token meanings         │
│  Communication: ONLY tokens + numbers + structural symbols      │
│                                                                 │
│  For each primitive:                                            │
│    - Seer receives samples of the function                      │
│    - Seer sends: token + (input→output) demonstrations          │
│    - Doer receives and must INFER the function                  │
│                                                                 │
│  Example teaching sequence:                                     │
│    α | 0→0 | 1→0.84 | 2→0.91                                    │
│    β | 0→1 | 1→0.54 | 2→-0.42                                   │
│    γ | 0→0 | 1→1 | 2→4                                          │
│    δ | -1→1 | 0→0 | 1→1                                         │
│    ε | 1→-1 | 2→-2                                              │
│                                                                 │
│  Measure: Can Doer correctly predict unseen (x, f(x)) pairs?    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│   TEACHING PHASE: COMPOSITION RULE (1-2 examples only)          │
│                                                                 │
│  Goal: Teach the "X-Y" composition syntax                       │
│  Communication: ONLY tokens + numbers + structural symbols      │
│                                                                 │
│  Seer sends ONE composition example:                            │
│    α-γ | 1→0.84                                                 │
│                                                                 │
│  Doer must reason:                                              │
│    "α-γ at x=1 gives 0.84                                       │
│     If γ(1)=1 and α(1)=0.84...                                  │
│     Then α-γ means: apply γ first, then α!"                     │
│                                                                 │
│  This is the ONLY composition shown. All others are held out.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│     TEST PHASE: NOVEL COMPOSITIONS  ══════ THE KEY TEST ══════  │
│                                                                 │
│  Goal: Test compositional GENERALIZATION                        │
│  Functions: Compositions NEVER explicitly demonstrated          │
│  Trials: 18 novel compositions                                  │
│                                                                 │
│  Taught: α-γ (ONE composition example)                          │
│  Tested (18 novel combinations):                                │
│    - α-β, α-δ, α-ε                                              │
│    - β-α, β-γ, β-δ, β-ε                                         │
│    - γ-α, γ-β, γ-δ, γ-ε                                         │
│    - δ-α, δ-β, δ-γ, δ-ε                                         │
│    - ε-α, ε-β, ε-γ, ε-δ                                         │
│                                                                 │
│  Measure: Φ = accuracy_novel / accuracy_primitive               │
│                                                                 │
│  Interpretation:                                                │
│    Φ ≈ 1.0 → Learned compositional grammar (THE RULE)           │
│    Φ ≈ 0.0 → Only memorized the one taught example              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            TEST PHASE: INCOMPRESSIBLE CONTROLS                  │
│                                                                 │
│  Goal: Sanity check — agents SHOULD fail here                   │
│  Functions: Random mappings with no short description           │
│  Trials: 5                                                      │
│                                                                 │
│  Example:                                                       │
│    - f(x) = [random lookup: -2→3.7, -1→-1.2, 0→8.4, ...]        │
│    - No pattern exists, no token can encode this                │
│                                                                 │
│  Measure: Failure rate (should be ~100%)                        │
│  If agents PASS here → benchmark is broken (test leakage)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**MVP Trial Summary:**
| Phase | Trials | Purpose |
|-------|--------|---------|
| Teaching: Primitives | 5 | Seer teaches 5 tokens via demonstration |
| Teaching: Composition | 1 | Seer teaches composition rule with ONE example |
| Test: Novel Compositions | 18 | **THE KEY TEST** — measures Φ |
| Test: Incompressible | 5 | Sanity check |
| **Total** | **29** | |

### 5.4 Composition Hold-Out Strategy

With the teaching phase, the hold-out is defined by what's demonstrated:

```python
# MVP: Only 5 primitives (those shown in few-shot)
FEW_SHOT_PRIMITIVES = {
    "α": "sin",
    "β": "cos",
    "γ": "square",
    "δ": "abs",
    "ε": "neg"
}

# Compositions SHOWN in few-shot (agents have seen these)
FEW_SHOT_COMPOSITIONS = [
    ("α", "γ"),  # sin(x²) - SHOWN
    ("δ", "α"),  # |sin(x)| - SHOWN
]

# Compositions TESTED in Phase 2 (agents have NOT seen these)
# All pairs of {α,β,γ,δ,ε} minus the 2 shown above = 18 novel compositions
TEST_COMPOSITIONS = [
    ("α", "β"),  # sin(cos(x))
    ("α", "δ"),  # sin(|x|)
    ("α", "ε"),  # sin(-x)
    ("β", "α"),  # cos(sin(x))
    ("β", "γ"),  # cos(x²)
    ("β", "δ"),  # cos(|x|)
    ("β", "ε"),  # cos(-x)
    ("γ", "α"),  # (sin(x))²
    ("γ", "β"),  # (cos(x))²
    ("γ", "δ"),  # |x|²
    ("γ", "ε"),  # (-x)²
    ("δ", "β"),  # |cos(x)|
    ("δ", "γ"),  # |x²| = x²
    ("δ", "ε"),  # |-x| = |x|
    ("ε", "α"),  # -sin(x)
    ("ε", "β"),  # -cos(x)
    ("ε", "γ"),  # -x²
    ("ε", "δ"),  # -|x|
]

# Note: α-γ and δ-α are NOT in TEST_COMPOSITIONS (they were shown in few-shot)
```

---

## 6. Communication Protocol

### 6.1 The Critical Constraint: NO NATURAL LANGUAGE

```
┌─────────────────────────────────────────────────────────────────┐
│                 COMMUNICATION CONSTRAINTS                        │
│                                                                 │
│   ALLOWED in Seer→Doer messages:                                │
│   ├── Abstract tokens: α, β, γ, δ, ε                            │
│   ├── Numbers: 0, 1, 2, -0.84, 0.91, 3.14, etc.                 │
│   ├── Structural symbols: → | - (for composition)               │
│   └── Line breaks / spacing                                     │
│                                                                 │
│   NOT ALLOWED:                                                  │
│   ├── Words: "means", "apply", "first", "example", "rule"       │
│   ├── Labels: "sin", "cos", "square" (function names)           │
│   ├── Any natural language explanation                          │
│   └── Meta-commentary about the protocol                        │
│                                                                 │
│   WHY: This forces PURE DEMONSTRATION, not explanation.         │
│   The Doer must INFER meaning from patterns in numbers.         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Token Vocabulary

```python
# MVP vocabulary (5 tokens)
VOCABULARY = ["α", "β", "γ", "δ", "ε"]

# IMPORTANT: Token meanings are NOT pre-defined!
# The Seer INVENTS what each token means and TEACHES it to the Doer.
# The Seer could assign:
#   α = sin, β = cos, γ = square, δ = abs, ε = neg
# OR:
#   α = square, β = abs, γ = sin, δ = neg, ε = cos
# Any consistent assignment is valid!

# For evaluation purposes, we track what assignment the Seer chose.
```

### 6.3 Teaching Message Format

During the teaching phase, Seer sends messages like:

```
┌─────────────────────────────────────────────────────────────────┐
│  TEACHING A PRIMITIVE (e.g., sin)                               │
│                                                                 │
│  α                                                              │
│  0 → 0                                                          │
│  1 → 0.84                                                       │
│  2 → 0.91                                                       │
│                                                                 │
│  Doer receives this and must INFER: "α means sin(x)"            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TEACHING A COMPOSITION (e.g., sin ∘ square)                    │
│                                                                 │
│  α-γ                                                            │
│  1 → 0.84                                                       │
│                                                                 │
│  Doer receives this and must VERIFY:                            │
│    "α-γ at x=1 gives 0.84                                       │
│     If I try γ first: γ(1) = 1                                  │
│     Then α: α(1) = 0.84 ✓                                       │
│     So α-γ means apply γ first, then α!"                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Test Message Format

During the test phase, Seer sends compact messages:

```python
# Single primitive: one token
message = "α"

# Composition: hyphen-separated tokens
message = "α-β"      # Meaning: α ∘ β (apply β first, then α)
message = "α-β-γ"    # Meaning: α ∘ β ∘ γ

# Maximum length: 5 tokens (allowing depth-5 composition)
max_tokens = 5
```

### 6.5 Composition Semantics

**Right-to-left application** (standard mathematical convention):

```python
# Message "α-β" means α ∘ β
# So for input x: apply β first, then α
# (α ∘ β)(x) = α(β(x))

# Example: if Seer assigned α=sin, β=square
# Message "α-β" for input x=2:
# β(2) = 4
# α(4) = sin(4) ≈ -0.757
```

### 6.6 Bandwidth Constraint

```
Maximum message length: 5 tokens
Information per token:  log₂(5) ≈ 2.32 bits (MVP vocabulary)
Maximum information:    5 × 2.32 ≈ 11.6 bits

Compare to raw data:
  5 samples × 2 floats × 32 bits = 320 bits
  Compression ratio: 320 / 11.6 ≈ 28×

This forces genuine compression — cannot transmit raw samples.
```

### 6.7 Protocol Validation

```python
def validate_teaching_message(message: str) -> bool:
    """
    Validate a teaching message conforms to no-natural-language constraint.

    Allowed characters:
    - Greek letters: α, β, γ, δ, ε
    - Digits and decimal: 0-9, .
    - Signs: -, +
    - Arrow: →
    - Pipe: |
    - Whitespace: space, newline

    NOT allowed:
    - Letters a-z, A-Z
    - Words of any kind
    """
    import re

    # Pattern: only allowed characters
    allowed_pattern = r'^[αβγδε0-9\.\-\+→\|\s]+$'

    if not re.match(allowed_pattern, message):
        return False

    # Additional check: no sequences that look like words
    if re.search(r'[a-zA-Z]{2,}', message):
        return False

    return True
```

---

## 7. Scoring System

### 7.1 Trial-Level Scoring

```python
def score_trial(y_predicted, y_expected, epsilon=0.01):
    """
    Score a single trial.
    
    Returns:
        1 if prediction is within tolerance
        0 otherwise
    """
    if abs(y_expected) < epsilon:
        # Absolute tolerance for near-zero expected values
        return 1 if abs(y_predicted - y_expected) < epsilon else 0
    else:
        # Relative tolerance
        relative_error = abs(y_predicted - y_expected) / abs(y_expected)
        return 1 if relative_error < epsilon else 0
```

### 7.2 Phase-Level Metrics

```python
@dataclass
class PhaseMetrics:
    phase_name: str
    num_trials: int
    num_successes: int
    accuracy: float  # successes / trials
    
    # Token usage statistics
    avg_message_length: float
    message_consistency: float  # Same function → same message?

def compute_phase_metrics(trial_results: List[TrialResult]) -> PhaseMetrics:
    """Aggregate trial results into phase metrics."""
    pass
```

### 7.3 The Φ Metric (Primary Evaluation)

```python
def compute_phi(phase1_accuracy: float, phase3_accuracy: float) -> float:
    """
    Compute the Fluidity Ratio Φ.
    
    Φ = accuracy_on_novel_compositions / accuracy_on_primitives
    
    Interpretation:
        Φ ≈ 1.0: Agents learned compositional grammar
        Φ ≈ 0.0: Agents memorized lookup table
        Φ > 1.0: Suspicious (check for bugs)
    """
    if phase1_accuracy == 0:
        return 0.0  # Avoid division by zero
    return phase3_accuracy / phase1_accuracy
```

### 7.4 Secondary Metrics

```python
@dataclass
class BenchmarkResults:
    # Core metric
    phi: float                      # Fluidity ratio (main result)

    # Per-phase accuracy (3 phases with few-shot grounding)
    phase1_accuracy: float          # Primitives (should be high - few-shot taught)
    phase2_accuracy: float          # Novel compositions (THE KEY METRIC)
    phase3_failure_rate: float      # Incompressible (should be ~1.0)

    # Efficiency metrics
    avg_tokens_per_primitive: float
    avg_tokens_per_composition: float

    # Consistency metrics
    vocabulary_stability: float     # Same function → same token?
    composition_syntax_correct: float  # Proper use of hyphenation?

    # Error analysis
    common_failure_modes: Dict[str, int]
```

### 7.5 Score Interpretation Guide

| Φ Value | Interpretation | Conclusion |
|---------|----------------|------------|
| Φ ≥ 0.9 | Near-perfect generalization | Strong compositional grammar |
| 0.7 ≤ Φ < 0.9 | Good generalization | Partial compositionality |
| 0.5 ≤ Φ < 0.7 | Moderate generalization | Mixed strategy |
| 0.3 ≤ Φ < 0.5 | Weak generalization | Mostly memorization |
| Φ < 0.3 | Poor generalization | Lookup table / memorization |

---

## 8. Function Library & Complexity Tiers

### 8.1 Tier 1: Primitives (Depth 1)

**MVP uses only the 5 primitives shown in few-shot examples.**

```python
# ═══════════════════════════════════════════════════════════════════
# MVP PRIMITIVES (shown in few-shot, tokens α β γ δ ε)
# ═══════════════════════════════════════════════════════════════════

MVP_PRIMITIVES = {
    "sin": {                                    # Token: α
        "token": "α",
        "function": lambda x: math.sin(x),
        "description": "Sine function"
    },
    "cos": {                                    # Token: β
        "token": "β",
        "function": lambda x: math.cos(x),
        "description": "Cosine function"
    },
    "square": {                                 # Token: γ
        "token": "γ",
        "function": lambda x: x ** 2,
        "description": "Square function"
    },
    "abs": {                                    # Token: δ
        "token": "δ",
        "function": lambda x: abs(x),
        "description": "Absolute value"
    },
    "neg": {                                    # Token: ε
        "token": "ε",
        "function": lambda x: -x,
        "description": "Negation"
    },
}

# ═══════════════════════════════════════════════════════════════════
# HELD-OUT PRIMITIVES (NOT in few-shot, tokens ζ η θ ι κ)
# For future extension only
# ═══════════════════════════════════════════════════════════════════

HELD_OUT_PRIMITIVES = {
    "sqrt": {                                   # Token: ζ
        "token": "ζ",
        "function": lambda x: math.sqrt(abs(x)),
        "description": "Square root (safe)"
    },
    "exp": {                                    # Token: η
        "token": "η",
        "function": lambda x: math.exp(min(x, 10)),
        "description": "Exponential (capped)"
    },
    "log": {                                    # Token: θ
        "token": "θ",
        "function": lambda x: math.log(abs(x) + 0.01),
        "description": "Logarithm (safe)"
    },
    "relu": {                                   # Token: ι
        "token": "ι",
        "function": lambda x: max(0, x),
        "description": "ReLU activation"
    },
    "sign": {                                   # Token: κ
        "token": "κ",
        "function": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
        "description": "Sign function"
    }
}
```

### 8.2 Tier 2: Compositions (Depth 2)

```python
TIER_2_FUNCTIONS = {
    "sin_of_square": {
        "composition": ["sin", "square"],
        "function": lambda x: math.sin(x ** 2),
        "complexity": 2,
        "description": "sin(x²)"
    },
    "square_of_sin": {
        "composition": ["square", "sin"],
        "function": lambda x: math.sin(x) ** 2,
        "complexity": 2,
        "description": "(sin x)²"
    },
    "abs_of_sin": {
        "composition": ["abs", "sin"],
        "function": lambda x: abs(math.sin(x)),
        "complexity": 2,
        "description": "|sin x|"
    },
    # ... many more combinations
}
```

### 8.3 Tier 3: Deep Compositions (Depth 3) — Future Extension

```python
TIER_3_FUNCTIONS = {
    "sin_of_square_of_abs": {
        "composition": ["sin", "square", "abs"],
        "function": lambda x: math.sin(abs(x) ** 2),
        "complexity": 3,
        "description": "sin(|x|²)"
    },
    # ... for future implementation
}
```

### 8.4 Tier X: Incompressible Functions (Controls)

```python
def generate_incompressible_function(seed: int):
    """
    Generate a function with no short description.
    Used as negative control — agents SHOULD fail.
    """
    rng = np.random.default_rng(seed)
    
    # Random lookup table for sample points
    random_outputs = {
        -2.0: rng.uniform(-10, 10),
        -1.0: rng.uniform(-10, 10),
        0.0: rng.uniform(-10, 10),
        1.0: rng.uniform(-10, 10),
        2.0: rng.uniform(-10, 10),
    }
    
    def f(x):
        if x in random_outputs:
            return random_outputs[x]
        else:
            # For test points, also random
            return rng.uniform(-10, 10)
    
    return f, random_outputs

INCOMPRESSIBLE_FUNCTIONS = [
    generate_incompressible_function(seed=i) for i in range(10)
]
```

### 8.5 Training/Test Split for Compositions

```python
# All possible depth-2 compositions: 10 × 10 = 100
# We use a subset for tractability

TRAINING_COMPOSITIONS = [
    # Group A × Group A (same-group compositions)
    ("sin", "cos"),
    ("sin", "square"),
    ("cos", "square"),
    ("square", "abs"),
    ("sqrt", "abs"),
    
    # Group B × Group B
    ("neg", "exp"),
    ("exp", "log"),
    ("relu", "neg"),
    ("sign", "relu"),
    
    # Selected cross-group (A × B)
    ("sin", "neg"),
    ("sin", "relu"),
    ("cos", "exp"),
    ("square", "neg"),
    ("abs", "relu"),
]

TEST_COMPOSITIONS = [
    # Held-out cross-group combinations
    ("sin", "exp"),      # sin and exp never composed in training
    ("sin", "log"),
    ("cos", "neg"),
    ("cos", "relu"),
    ("square", "exp"),
    ("square", "log"),
    ("sqrt", "neg"),
    ("sqrt", "relu"),
    ("abs", "exp"),
    ("abs", "sign"),
]
```

---

## 9. Implementation Checkpoints

### Phase 0: Setup

#### Checkpoint 0.1: Directory Structure ✅
- [x] Create `telepathic/` directory structure
- [x] Verify shared utilities available
- [x] Create `__init__.py` files

### Phase 1: Core Engine ✅

#### Checkpoint 1.1: Function Library ✅
- [x] **File:** `telepathic/core/functions.py`
- [x] **Requirements:**
  - [x] Implement 5 MVP primitive functions (sin, cos, square, abs, neg)
  - [x] Safe handling of edge cases (sqrt of negative, log of zero)
  - [x] Composition operator `compose(*funcs)`
  - [x] Function registry with metadata
- [x] **Tests:**
  - [x] Each primitive evaluates correctly
  - [x] Compositions work in correct order
  - [x] Edge cases handled gracefully

#### Checkpoint 1.2: Sample Generator ✅
- [x] **File:** `telepathic/core/sampling.py`
- [x] **Requirements:**
  - [x] Generate fixed sample points for any function
  - [x] Generate test points (different from samples)
  - [x] Format samples for Seer observation
- [x] **Tests:**
  - [x] Correct number of samples
  - [x] Sample and test points don't overlap

#### Checkpoint 1.3: Protocol Manager ✅
- [x] **File:** `telepathic/core/protocol.py`
- [x] **Requirements:**
  - [x] Define vocabulary (10 tokens)
  - [x] Validate message format
  - [x] Parse message into token sequence
  - [x] Enforce bandwidth constraint (max 5 tokens)
- [x] **Tests:**
  - [x] Valid messages accepted
  - [x] Invalid messages rejected
  - [x] Token parsing correct

#### Checkpoint 1.4: Evaluation Engine ✅
- [x] **File:** `telepathic/core/evaluation.py`
- [x] **Requirements:**
  - [x] Score prediction against expected
  - [x] Relative tolerance (1%)
  - [x] Handle near-zero expected values
  - [x] Aggregate scores across trials
- [x] **Tests:**
  - [x] Correct predictions score 1
  - [x] Incorrect predictions score 0
  - [x] Edge cases (zero, very large) handled

#### Checkpoint 1.5: Few-Shot Generator ✅
- [x] **File:** `telepathic/core/few_shot.py`
- [x] **Requirements:**
  - [x] Generate primitive examples (5 primitives: sin, cos, square, abs, neg)
  - [x] Generate composition examples (2: α-γ, δ-α)
  - [x] Format examples for Seer system prompt
  - [x] Format examples for Doer system prompt
  - [x] Define held-out compositions for Phase 2 testing
- [x] **Tests:**
  - [x] All primitive examples have correct (x, y) pairs
  - [x] Composition examples have correct (x, y) pairs
  - [x] Formatting is consistent between Seer and Doer

---

### Phase 2: Agents ✅

#### Checkpoint 2.1: Base Agent ✅
- [x] **File:** `telepathic/agents/base.py`
- [x] **Requirements:**
  - [x] Abstract base class for agents
  - [x] Define interface for Seer and Doer
- [x] **Tests:**
  - [x] Interface is implementable

#### Checkpoint 2.2: Seer Agent ✅
- [x] **File:** `telepathic/agents/seer.py`
- [x] **Requirements:**
  - [x] Receive samples and vocabulary
  - [x] Generate message (≤5 tokens)
  - [x] Use shared LLM utilities
  - [x] Parse response for message
- [x] **Tests:**
  - [x] Generates valid messages
  - [x] Respects bandwidth constraint

#### Checkpoint 2.3: Doer Agent ✅
- [x] **File:** `telepathic/agents/doer.py`
- [x] **Requirements:**
  - [x] Receive message and test input
  - [x] Generate numerical prediction
  - [x] Use shared LLM utilities
  - [x] Parse response for number
- [x] **Tests:**
  - [x] Produces numerical output
  - [x] Handles malformed messages gracefully

#### Checkpoint 2.4: Random Baseline Agent ✅
- [x] **File:** `telepathic/agents/random_agent.py`
- [x] **Requirements:**
  - [x] Seer: Random tokens
  - [x] Doer: Random number in expected range
  - [x] Deterministic with seed
- [x] **Tests:**
  - [x] Same seed = same behavior

---

### Phase 3: Experiment Runner

#### Checkpoint 3.1: Trial Executor
- [ ] **File:** `telepathic/experiments/trial.py`
- [ ] **Requirements:**
  - [ ] Execute single Seer→Doer trial
  - [ ] Capture message, prediction, score
  - [ ] Handle errors gracefully
- [ ] **Tests:**
  - [ ] Complete trial returns result
  - [ ] Errors caught and logged

#### Checkpoint 3.2: Curriculum Manager
- [ ] **File:** `telepathic/experiments/curriculum.py`
- [ ] **Requirements:**
  - [ ] Generate trials for each phase
  - [ ] Manage training/test split
  - [ ] Track phase progression
- [ ] **Tests:**
  - [ ] Correct number of trials per phase
  - [ ] Compositions correctly partitioned

#### Checkpoint 3.3: Episode Runner
- [ ] **File:** `telepathic/experiments/runner.py`
- [ ] **Requirements:**
  - [ ] Run full curriculum
  - [ ] Aggregate results by phase
  - [ ] Compute Φ metric
  - [ ] Save results
- [ ] **Tests:**
  - [ ] Full run completes
  - [ ] Metrics computed correctly

#### Checkpoint 3.4: Result Logger
- [ ] **File:** `telepathic/experiments/logger.py`
- [ ] **Requirements:**
  - [ ] Save results to JSON
  - [ ] Timestamp and unique ID
  - [ ] Include all metadata
- [ ] **Tests:**
  - [ ] Results serializable
  - [ ] Loadable for analysis

---

### Phase 4: Visualization

#### Checkpoint 4.1: Function Plotter
- [ ] **File:** `telepathic/visualization/plot_function.py`
- [ ] **Requirements:**
  - [ ] Plot f(x) over domain
  - [ ] Mark sample points
  - [ ] Mark test points (different color)
- [ ] **Tests:**
  - [ ] Plots render correctly

#### Checkpoint 4.2: Trial Visualizer
- [ ] **File:** `telepathic/visualization/plot_trial.py`
- [ ] **Requirements:**
  - [ ] Show function, samples, prediction
  - [ ] Annotate with message sent
  - [ ] Color-code success/failure
- [ ] **Tests:**
  - [ ] Trial visualization complete

#### Checkpoint 4.3: Results Dashboard
- [ ] **File:** `telepathic/visualization/dashboard.py`
- [ ] **Requirements:**
  - [ ] Bar chart of accuracy by phase
  - [ ] Φ metric prominently displayed
  - [ ] Confusion matrix for token mappings
- [ ] **Tests:**
  - [ ] Dashboard renders

---

### Phase 5: Analysis

#### Checkpoint 5.1: Statistical Analysis
- [ ] **File:** `telepathic/experiments/analysis.py`
- [ ] **Requirements:**
  - [ ] Compute mean, std across runs
  - [ ] Confidence intervals
  - [ ] Statistical tests (Φ vs baselines)
- [ ] **Tests:**
  - [ ] Stats computed correctly

#### Checkpoint 5.2: Vocabulary Analysis
- [ ] **File:** `telepathic/experiments/vocabulary_analysis.py`
- [ ] **Requirements:**
  - [ ] Extract token→function mappings
  - [ ] Measure consistency across trials
  - [ ] Visualize emergent vocabulary
- [ ] **Tests:**
  - [ ] Mappings extracted correctly

---

## 10. File Structure

```
DISS/
│
├── shared/                           # Existing shared utilities
│   ├── __init__.py
│   ├── llm_utils.py                  # API calls, retry logic
│   ├── logging.py                    # Result logging
│   └── base_agent.py                 # Abstract agent interface
│
├── coordination/                     # Existing benchmark
│   └── ...
│
├── temporal/                         # Existing benchmark
│   └── ...
│
├── telepathic/                       # NEW: This benchmark
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── functions.py              # Primitive library + composition
│   │   ├── sampling.py               # Sample/test point generation
│   │   ├── protocol.py               # Vocabulary + message validation
│   │   ├── evaluation.py             # Scoring logic
│   │   └── few_shot.py               # Few-shot example generation
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract Seer/Doer interface
│   │   ├── seer.py                   # LLM Seer agent
│   │   ├── doer.py                   # LLM Doer agent
│   │   └── random_agent.py           # Random baseline
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── trial.py                  # Single trial execution
│   │   ├── curriculum.py             # Phase management
│   │   ├── runner.py                 # Full episode execution
│   │   ├── logger.py                 # Result saving
│   │   ├── analysis.py               # Statistical analysis
│   │   └── vocabulary_analysis.py    # Emergent vocabulary study
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_function.py          # Function visualization
│   │   ├── plot_trial.py             # Trial visualization
│   │   └── dashboard.py              # Results dashboard
│   │
│   ├── prompts/
│   │   ├── seer_system.txt           # Few-shot system prompt for Seer
│   │   ├── seer_trial.txt            # Few-shot trial prompt for Seer
│   │   ├── doer_system.txt           # Few-shot system prompt for Doer
│   │   ├── doer_trial.txt            # Few-shot trial prompt for Doer
│   │   ├── seer_system_zero.txt      # Zero-shot system prompt for Seer
│   │   ├── seer_trial_zero.txt       # Zero-shot trial prompt for Seer
│   │   ├── doer_system_zero.txt      # Zero-shot system prompt for Doer
│   │   └── doer_trial_zero.txt       # Zero-shot trial prompt for Doer
│   │
│   └── tests/
│       ├── test_functions.py
│       ├── test_sampling.py
│       ├── test_protocol.py
│       ├── test_evaluation.py
│       └── test_agents.py
│
├── results/
│   ├── coordination/
│   ├── temporal/
│   └── telepathic/                   # NEW
│
├── PLAN.md                           # Coordination benchmark plan
├── TEMPORAL_PLAN.md                  # Temporal benchmark plan
├── TELEPATHIC_PLAN.md                # THIS DOCUMENT
└── README.md
```

---

## 11. API Specifications

### 11.1 Function Class

```python
@dataclass
class PrimitiveFunction:
    """A single primitive function."""
    
    name: str                           # e.g., "sin"
    func: Callable[[float], float]      # The actual function
    description: str                    # Human-readable description
    complexity: int = 1                 # Always 1 for primitives
    
    def __call__(self, x: float) -> float:
        """Evaluate function at x."""
        return self.func(x)
    
    def to_dict(self) -> dict:
        """Serialize (without the callable)."""
        return {
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity
        }


@dataclass
class ComposedFunction:
    """A composition of primitive functions."""
    
    primitives: List[str]               # e.g., ["sin", "square"]
    complexity: int                     # len(primitives)
    
    def __call__(self, x: float) -> float:
        """Evaluate composition at x."""
        result = x
        for prim_name in reversed(self.primitives):
            result = PRIMITIVES[prim_name](result)
        return result
    
    @property
    def description(self) -> str:
        """Generate description like 'sin(square(x))'."""
        inner = "x"
        for prim in reversed(self.primitives):
            inner = f"{prim}({inner})"
        return inner
    
    def to_dict(self) -> dict:
        return {
            "primitives": self.primitives,
            "description": self.description,
            "complexity": self.complexity
        }
```

### 11.2 Sample Generator

```python
class SampleGenerator:
    """Generates input-output samples for functions."""
    
    SAMPLE_X = [-2.0, -1.0, 0.0, 1.0, 2.0]
    TEST_X = [-1.5, 0.5, 1.5]
    
    def __init__(self, function: Union[PrimitiveFunction, ComposedFunction]):
        self.function = function
    
    def generate_samples(self) -> List[Tuple[float, float]]:
        """Generate training samples."""
        return [(x, self.function(x)) for x in self.SAMPLE_X]
    
    def generate_tests(self) -> List[Tuple[float, float]]:
        """Generate test cases."""
        return [(x, self.function(x)) for x in self.TEST_X]
    
    def format_for_seer(self) -> str:
        """Format samples as text for Seer prompt."""
        samples = self.generate_samples()
        lines = ["Input-Output Samples:"]
        for x, y in samples:
            lines.append(f"  f({x:.1f}) = {y:.4f}")
        return "\n".join(lines)
```

### 11.3 Protocol Manager

```python
class Protocol:
    """Manages communication protocol."""
    
    VOCABULARY = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ"]
    MAX_TOKENS = 5
    SEPARATOR = "-"
    
    @classmethod
    def validate_message(cls, message: str) -> bool:
        """Check if message is valid."""
        tokens = message.split(cls.SEPARATOR)
        
        if len(tokens) > cls.MAX_TOKENS:
            return False
        
        for token in tokens:
            if token not in cls.VOCABULARY:
                return False
        
        return True
    
    @classmethod
    def parse_message(cls, message: str) -> List[str]:
        """Parse message into token list."""
        if not cls.validate_message(message):
            raise ValueError(f"Invalid message: {message}")
        return message.split(cls.SEPARATOR)
    
    @classmethod
    def format_vocabulary(cls) -> str:
        """Format vocabulary for prompts."""
        return ", ".join(cls.VOCABULARY)
```

### 11.4 Evaluation Engine

```python
class Evaluator:
    """Evaluates Doer predictions."""
    
    RELATIVE_TOLERANCE = 0.01  # 1%
    ABSOLUTE_TOLERANCE = 0.01  # For near-zero values
    
    @classmethod
    def score(cls, predicted: float, expected: float) -> int:
        """
        Score a single prediction.
        
        Returns 1 for success, 0 for failure.
        """
        if abs(expected) < cls.ABSOLUTE_TOLERANCE:
            # Use absolute tolerance for near-zero
            return 1 if abs(predicted - expected) < cls.ABSOLUTE_TOLERANCE else 0
        else:
            # Use relative tolerance
            relative_error = abs(predicted - expected) / abs(expected)
            return 1 if relative_error < cls.RELATIVE_TOLERANCE else 0
    
    @classmethod
    def compute_phi(cls, primitive_accuracy: float, novel_accuracy: float) -> float:
        """Compute the Fluidity Ratio Φ."""
        if primitive_accuracy == 0:
            return 0.0
        return novel_accuracy / primitive_accuracy
```

### 11.5 Base Agent Interface

```python
from abc import ABC, abstractmethod

class SeerAgent(ABC):
    """Abstract base class for Seer agents."""
    
    @abstractmethod
    def generate_message(
        self,
        samples: List[Tuple[float, float]],
        vocabulary: List[str],
        max_tokens: int
    ) -> str:
        """
        Generate message encoding the function.
        
        Args:
            samples: List of (x, f(x)) pairs
            vocabulary: Available tokens
            max_tokens: Maximum message length
            
        Returns:
            Message string (e.g., "α-β")
        """
        pass


class DoerAgent(ABC):
    """Abstract base class for Doer agents."""
    
    @abstractmethod
    def generate_prediction(
        self,
        message: str,
        test_input: float,
        vocabulary: List[str]
    ) -> float:
        """
        Generate prediction for test input.
        
        Args:
            message: Received message from Seer
            test_input: x value to evaluate
            vocabulary: Available tokens (for reference)
            
        Returns:
            Predicted f(test_input)
        """
        pass
```

### 11.6 LLM Agent Implementation

```python
class LLMSeerAgent(SeerAgent):
    """Seer agent using LLM."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        system_prompt_path: str = "prompts/seer_system.txt"
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = load_prompt(system_prompt_path)
    
    def generate_message(
        self,
        samples: List[Tuple[float, float]],
        vocabulary: List[str],
        max_tokens: int
    ) -> str:
        """Generate message using LLM."""
        # Format prompt
        user_prompt = self._format_trial_prompt(samples, vocabulary, max_tokens)
        
        # Call LLM
        response = call_llm(
            model=self.model,
            system=self.system_prompt,
            user=user_prompt,
            temperature=self.temperature
        )
        
        # Parse response for message
        message = self._parse_response(response)
        
        return message
    
    def _format_trial_prompt(self, samples, vocabulary, max_tokens) -> str:
        """Format the trial prompt."""
        pass  # Implementation details
    
    def _parse_response(self, response: str) -> str:
        """Extract message from LLM response."""
        pass  # Implementation details


class LLMDoerAgent(DoerAgent):
    """Doer agent using LLM."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        system_prompt_path: str = "prompts/doer_system.txt"
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = load_prompt(system_prompt_path)
    
    def generate_prediction(
        self,
        message: str,
        test_input: float,
        vocabulary: List[str]
    ) -> float:
        """Generate prediction using LLM."""
        # Format prompt
        user_prompt = self._format_trial_prompt(message, test_input, vocabulary)
        
        # Call LLM
        response = call_llm(
            model=self.model,
            system=self.system_prompt,
            user=user_prompt,
            temperature=self.temperature
        )
        
        # Parse response for number
        prediction = self._parse_response(response)
        
        return prediction
    
    def _parse_response(self, response: str) -> float:
        """Extract numerical prediction from LLM response."""
        pass  # Implementation details
```

### 11.7 Trial Result

```python
@dataclass
class TrialResult:
    """Result of a single Seer→Doer trial."""
    
    # Trial identification
    trial_id: str
    phase: str                          # "primitive", "trained", "novel", "incompressible"
    
    # Function info
    function_name: str
    function_complexity: int
    is_composition: bool
    composition_primitives: Optional[List[str]]
    
    # Execution
    samples: List[Tuple[float, float]]
    test_input: float
    expected_output: float
    
    # Agent outputs
    message: str
    message_tokens: List[str]
    message_length: int
    predicted_output: float
    
    # Scoring
    success: int                        # 1 or 0
    relative_error: Optional[float]     # If applicable
    
    # Timing
    seer_latency_ms: float
    doer_latency_ms: float
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        pass
```

### 11.8 Episode Result

```python
@dataclass
class EpisodeResult:
    """Result of full curriculum episode."""
    
    # Metadata
    episode_id: str
    timestamp: str
    model: str
    
    # Phase results
    phase1_trials: List[TrialResult]
    phase2_trials: List[TrialResult]
    phase3_trials: List[TrialResult]
    phase4_trials: List[TrialResult]
    
    # Aggregated metrics
    phase1_accuracy: float
    phase2_accuracy: float
    phase3_accuracy: float
    phase4_failure_rate: float
    
    # Primary metric
    phi: float
    
    # Token analysis
    vocabulary_mapping: Dict[str, str]  # token → most common function
    vocabulary_consistency: float
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        pass
    
    def save(self, path: str):
        """Save to JSON file."""
        pass
    
    @classmethod
    def load(cls, path: str) -> 'EpisodeResult':
        """Load from JSON file."""
        pass
```

---

## 12. Prompt Templates (Few-Shot Version)

**Key Design:** Both Seer and Doer receive identical few-shot examples teaching the token→function mappings. This ensures shared vocabulary without explicit rules.

### 12.1 Seer System Prompt

**File:** `telepathic/prompts/seer_system.txt`

```
You are the SEER in a function communication game.

YOUR TASK:
- You observe input-output samples of a mathematical function
- You must encode the function as a short message using abstract tokens
- Your message must be ≤5 tokens, separated by hyphens if multiple

VOCABULARY: α, β, γ, δ, ε

Learn the token meanings from these examples:

═══════════════════════════════════════════════════════════════════
EXAMPLE 1 (primitive):
Samples:
  f(-2.0) = -0.9093
  f(-1.0) = -0.8415
  f(0.0) = 0.0
  f(1.0) = 0.8415
  f(2.0) = 0.9093
MESSAGE: α
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 2 (primitive):
Samples:
  f(-2.0) = -0.4161
  f(-1.0) = 0.5403
  f(0.0) = 1.0
  f(1.0) = 0.5403
  f(2.0) = -0.4161
MESSAGE: β
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 3 (primitive):
Samples:
  f(-2.0) = 4.0
  f(-1.0) = 1.0
  f(0.0) = 0.0
  f(1.0) = 1.0
  f(2.0) = 4.0
MESSAGE: γ
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 4 (primitive):
Samples:
  f(-2.0) = 2.0
  f(-1.0) = 1.0
  f(0.0) = 0.0
  f(1.0) = 1.0
  f(2.0) = 2.0
MESSAGE: δ
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 5 (primitive):
Samples:
  f(-2.0) = 2.0
  f(-1.0) = 1.0
  f(0.0) = 0.0
  f(1.0) = -1.0
  f(2.0) = -2.0
MESSAGE: ε
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 6 (composition):
Samples:
  f(-2.0) = -0.7568
  f(-1.0) = 0.8415
  f(0.0) = 0.0
  f(1.0) = 0.8415
  f(2.0) = -0.7568
MESSAGE: α-γ
(This combines α and γ: first apply γ to x, then apply α to the result)
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 7 (composition):
Samples:
  f(-2.0) = 0.9093
  f(-1.0) = 0.8415
  f(0.0) = 0.0
  f(1.0) = 0.8415
  f(2.0) = 0.9093
MESSAGE: δ-α
(This combines δ and α: first apply α to x, then apply δ to the result)
═══════════════════════════════════════════════════════════════════

COMPOSITION RULE:
When you write "X-Y", it means: first apply Y, then apply X to that result.
So "α-γ" means: compute γ(x), then compute α(γ(x)).

OUTPUT FORMAT:
After analyzing the samples, output ONLY your message on a line starting with "MESSAGE:"
```

### 12.2 Seer Trial Prompt

**File:** `telepathic/prompts/seer_trial.txt`

```
═══════════════════════════════════════════════════════════════════
YOUR TURN - ENCODE THIS FUNCTION:

Samples:
  f(-2.0) = {y1}
  f(-1.0) = {y2}
  f(0.0) = {y3}
  f(1.0) = {y4}
  f(2.0) = {y5}

Analyze the pattern. Which primitive(s) from the examples produce this?
If it's a composition, remember: "X-Y" means apply Y first, then X.

MESSAGE:
═══════════════════════════════════════════════════════════════════
```

### 12.3 Doer System Prompt

**File:** `telepathic/prompts/doer_system.txt`

```
You are the DOER in a function communication game.

YOUR TASK:
- You receive a token message from the Seer
- You must decode it and compute the function output for a test input
- Output a single number

VOCABULARY: α, β, γ, δ, ε

Learn the token meanings from these examples:

═══════════════════════════════════════════════════════════════════
EXAMPLE 1 (primitive α):
Message: α
Test input: 0.5 → Output: 0.4794
Test input: 1.5 → Output: 0.9975
Test input: -1.5 → Output: -0.9975
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 2 (primitive β):
Message: β
Test input: 0.5 → Output: 0.8776
Test input: 1.5 → Output: 0.0707
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 3 (primitive γ):
Message: γ
Test input: 0.5 → Output: 0.25
Test input: 1.5 → Output: 2.25
Test input: -1.5 → Output: 2.25
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 4 (primitive δ):
Message: δ
Test input: 0.5 → Output: 0.5
Test input: -1.5 → Output: 1.5
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 5 (primitive ε):
Message: ε
Test input: 0.5 → Output: -0.5
Test input: -1.5 → Output: 1.5
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 6 (composition α-γ):
Message: α-γ
Test input: 0.5 → Output: 0.2474
Test input: 1.5 → Output: 0.7781
(α-γ means: first apply γ, then apply α to that result)
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
EXAMPLE 7 (composition δ-α):
Message: δ-α
Test input: 0.5 → Output: 0.4794
Test input: 1.5 → Output: 0.9975
(δ-α means: first apply α, then apply δ to that result)
═══════════════════════════════════════════════════════════════════

COMPOSITION RULE:
"X-Y" means: first compute Y(input), then compute X(Y(input)).
"X-Y-Z" means: first Z, then Y, then X.

OUTPUT FORMAT:
After your reasoning, output ONLY the number on a line starting with "PREDICTION:"
```

### 12.4 Doer Trial Prompt

**File:** `telepathic/prompts/doer_trial.txt`

```
═══════════════════════════════════════════════════════════════════
YOUR TURN - DECODE AND COMPUTE:

Message: {message}
Test input: {x_test}

Decode the token(s) and compute the output.
Remember: "X-Y" means apply Y first, then X.

PREDICTION:
═══════════════════════════════════════════════════════════════════
```

### 12.5 Zero-Shot Prompts (No Examples)

**File:** `telepathic/prompts/seer_system_zero.txt`

```
You are the SEER in a function communication game.

YOUR TASK:
- You observe input-output samples of a mathematical function
- You must encode the function using abstract tokens
- Your partner (the Doer) will only see your message and must compute outputs

VOCABULARY: α, β, γ, δ, ε

These tokens have NO pre-defined meaning. You must decide what each token represents.
Think about common mathematical functions: sin, cos, square, absolute value, negation, etc.

RULES:
- Use single tokens for single operations (e.g., "α")
- Use hyphens for composition: "X-Y" means apply Y first, then X
- Maximum 5 tokens per message

The Doer will try to understand your encoding. Be consistent!

OUTPUT FORMAT:
Output your message on a line starting with "MESSAGE:"
```

**File:** `telepathic/prompts/seer_trial_zero.txt`

```
═══════════════════════════════════════════════════════════════════
ENCODE THIS FUNCTION:

Samples:
  f(-2.0) = {y1}
  f(-1.0) = {y2}
  f(0.0) = {y3}
  f(1.0) = {y4}
  f(2.0) = {y5}

What mathematical function produces these outputs?
Encode it using your chosen token meanings.

MESSAGE:
═══════════════════════════════════════════════════════════════════
```

**File:** `telepathic/prompts/doer_system_zero.txt`

```
You are the DOER in a function communication game.

YOUR TASK:
- The Seer observed samples of a mathematical function
- The Seer encoded it as a token message
- You must decode the message and compute the output for a test input

VOCABULARY: α, β, γ, δ, ε

These tokens have NO pre-defined meaning. The Seer chose what each represents.
Think about common mathematical functions: sin, cos, square, absolute value, negation, etc.

RULES:
- Single tokens represent single operations
- "X-Y" means composition: apply Y first, then X to the result
- Try to infer what the Seer meant based on the function structure

OUTPUT FORMAT:
Output your numerical answer on a line starting with "PREDICTION:"
```

**File:** `telepathic/prompts/doer_trial_zero.txt`

```
═══════════════════════════════════════════════════════════════════
DECODE AND COMPUTE:

Message from Seer: {message}
Test input: {x_test}

What function did the Seer encode? Compute f({x_test}).

PREDICTION:
═══════════════════════════════════════════════════════════════════
```

---

## 13. Test Cases

### 13.1 Function Tests

```python
# telepathic/tests/test_functions.py

def test_primitive_sin():
    """Sin function evaluates correctly."""
    f = PRIMITIVES["sin"]
    assert abs(f(0) - 0.0) < 0.001
    assert abs(f(math.pi/2) - 1.0) < 0.001

def test_primitive_square():
    """Square function evaluates correctly."""
    f = PRIMITIVES["square"]
    assert f(2) == 4
    assert f(-3) == 9

def test_safe_sqrt():
    """Sqrt handles negative inputs."""
    f = PRIMITIVES["sqrt"]
    assert f(4) == 2.0
    assert f(-4) == 2.0  # sqrt(|-4|) = sqrt(4) = 2

def test_safe_log():
    """Log handles zero and negative inputs."""
    f = PRIMITIVES["log"]
    result = f(0)
    assert math.isfinite(result)  # Should not be -inf

def test_composition_order():
    """Composition applies functions right-to-left."""
    # sin(x²) at x=0 should be sin(0) = 0
    f = compose(PRIMITIVES["sin"], PRIMITIVES["square"])
    assert abs(f(0) - 0.0) < 0.001
    
    # (sin x)² at x=π/2 should be 1² = 1
    g = compose(PRIMITIVES["square"], PRIMITIVES["sin"])
    assert abs(g(math.pi/2) - 1.0) < 0.001

def test_deep_composition():
    """Three-level composition works."""
    # |sin(x²)| at x=2
    f = compose(PRIMITIVES["abs"], PRIMITIVES["sin"], PRIMITIVES["square"])
    expected = abs(math.sin(4))
    assert abs(f(2) - expected) < 0.001
```

### 13.2 Protocol Tests

```python
# telepathic/tests/test_protocol.py

def test_valid_single_token():
    """Single token message is valid."""
    assert Protocol.validate_message("α") == True

def test_valid_composition():
    """Composition message is valid."""
    assert Protocol.validate_message("α-β") == True
    assert Protocol.validate_message("α-β-γ") == True

def test_invalid_token():
    """Unknown token rejected."""
    assert Protocol.validate_message("x") == False
    assert Protocol.validate_message("α-x") == False

def test_too_long():
    """Message exceeding max length rejected."""
    assert Protocol.validate_message("α-β-γ-δ-ε-ζ") == False  # 6 tokens

def test_parse_message():
    """Message parsing works."""
    assert Protocol.parse_message("α") == ["α"]
    assert Protocol.parse_message("α-β-γ") == ["α", "β", "γ"]
```

### 13.3 Evaluation Tests

```python
# telepathic/tests/test_evaluation.py

def test_exact_match():
    """Exact prediction scores 1."""
    assert Evaluator.score(predicted=1.0, expected=1.0) == 1

def test_within_tolerance():
    """Prediction within 1% scores 1."""
    assert Evaluator.score(predicted=1.005, expected=1.0) == 1

def test_outside_tolerance():
    """Prediction outside 1% scores 0."""
    assert Evaluator.score(predicted=1.02, expected=1.0) == 0

def test_near_zero():
    """Near-zero uses absolute tolerance."""
    assert Evaluator.score(predicted=0.005, expected=0.0) == 1
    assert Evaluator.score(predicted=0.02, expected=0.0) == 0

def test_phi_calculation():
    """Phi metric computed correctly."""
    assert Evaluator.compute_phi(1.0, 1.0) == 1.0
    assert Evaluator.compute_phi(1.0, 0.5) == 0.5
    assert Evaluator.compute_phi(0.8, 0.4) == 0.5
```

### 13.4 Sampling Tests

```python
# telepathic/tests/test_sampling.py

def test_sample_count():
    """Generates correct number of samples."""
    f = PRIMITIVES["sin"]
    gen = SampleGenerator(f)
    samples = gen.generate_samples()
    assert len(samples) == 5

def test_test_count():
    """Generates correct number of test points."""
    f = PRIMITIVES["sin"]
    gen = SampleGenerator(f)
    tests = gen.generate_tests()
    assert len(tests) == 3

def test_no_overlap():
    """Sample and test points don't overlap."""
    sample_x = set(SampleGenerator.SAMPLE_X)
    test_x = set(SampleGenerator.TEST_X)
    assert sample_x.isdisjoint(test_x)
```

---

## 14. Visualization Requirements

### 14.1 Function Plot

**Function:** `plot_function(f, samples, tests, title, save_path=None)`

**Requirements:**
- Plot f(x) over domain [-5, 5]
- Mark sample points with blue circles
- Mark test points with red triangles
- Add grid and axis labels
- Title shows function description

**Example:**
```
      f(x)
        │       ╱╲
   1.0 ─┤      ╱  ╲       ● sample
        │     ╱    ╲      ▲ test
   0.0 ─┼────╱──────╲─────────
        │   ╱        ╲
  -1.0 ─┤  ╱          ╲
        └──┴──┴──┴──┴──┴──► x
         -2  -1  0  1  2
```

### 14.2 Trial Visualization

**Function:** `plot_trial(trial_result, save_path=None)`

**Requirements:**
- Show function curve
- Mark sample points (what Seer saw)
- Mark test point with prediction vs expected
- Display message in annotation
- Color-code success (green) / failure (red)

### 14.3 Results Dashboard

**Function:** `plot_dashboard(episode_result, save_path=None)`

**Requirements:**
- Bar chart: accuracy by phase
- Large Φ metric display
- Token→function mapping table
- Error distribution histogram

### 14.4 Vocabulary Visualization

**Function:** `plot_vocabulary(episode_results, save_path=None)`

**Requirements:**
- Heatmap: token × function usage frequency
- Cluster analysis of token usage
- Consistency score per token

---

## 15. Baselines

### 15.1 Random Baseline

**Behavior:**
- Seer: Random token(s) from vocabulary
- Doer: Random number in range [-10, 10]

**Expected performance:**
- Phase 1 accuracy: ~10% (chance matching)
- Φ: ~1.0 (both equally bad)
- Establishes lower bound

### 15.2 Memorization Baseline

**Behavior:**
- Seer: Hash function inputs → deterministic token
- Doer: Memorize (function_hash, output) pairs from training

**Expected performance:**
- Phase 1 accuracy: High (memorized)
- Phase 2 accuracy: High (if compositions were trained)
- Phase 3 accuracy: ~0% (novel compositions fail)
- Φ: ~0 (the "pattern recognizer" signature)

### 15.3 Oracle Baseline (Upper Bound)

**Behavior:**
- Seer: Knows correct token mapping
- Doer: Knows correct token→function mapping

**Expected performance:**
- All phases: ~100% accuracy
- Φ: 1.0
- Validates evaluation pipeline

---

## 16. Evaluation Protocol

### 16.1 Experiment Configuration (Two Conditions)

```python
EXPERIMENT_CONFIG = {
    "model": "gemini/gemini-2.5-flash",  # Using Gemini 2.5 Flash
    "temperature": 0.7,
    "num_runs": 1,  # Per condition for MVP

    # 3-phase curriculum
    "phase1_trials": 15,   # 3 per primitive × 5 primitives
    "phase2_trials": 18,   # Novel compositions (THE KEY TEST)
    "phase3_trials": 5,    # Incompressible controls

    # Total: 38 trials per condition × 2 conditions = 76 trials
    "conditions": ["few-shot", "zero-shot"]
}
```

### 16.2 Run Protocol (Two Conditions)

```
For each condition (few-shot, zero-shot):
    1. Load appropriate prompts
    2. Execute Phase 1 (primitives)
    3. Execute Phase 2 (novel compositions)
    4. Execute Phase 3 (incompressible)
    5. Compute Φ for this condition

Compare conditions:
    - Φ_few_shot vs Φ_zero_shot
    - Δ = Φ_few_shot - Φ_zero_shot (effect of grounding)

Expected outcomes:
    - Few-shot: High Φ (can generalize from examples)
    - Zero-shot: Low Φ (can't invent consistent protocol)
    - Δ > 0.5 would be a strong finding
```

### 16.3 Statistical Analysis

```python
def analyze_results(results: List[EpisodeResult]):
    """Compute aggregate statistics."""
    
    phi_values = [r.phi for r in results]
    
    return {
        "phi_mean": np.mean(phi_values),
        "phi_std": np.std(phi_values),
        "phi_ci_95": confidence_interval(phi_values, 0.95),
        
        "phase1_mean": np.mean([r.phase1_accuracy for r in results]),
        "phase3_mean": np.mean([r.phase3_accuracy for r in results]),
        
        # Statistical test: is Φ > 0.5 (better than chance)?
        "phi_ttest": ttest_1samp(phi_values, 0.5),
    }
```

### 16.4 Reporting

Results should include:

1. **Primary metric:** Φ (mean ± std)
2. **Per-phase accuracy:** Bar chart with error bars
3. **Vocabulary analysis:** Emergent token mappings
4. **Failure analysis:** Common error patterns
5. **Comparison to baselines:** Table of all methods

---

## Appendix A: Quick Reference

### Key Parameters (MVP)
```
Domain:           x ∈ [-5, 5]
Sample points:    5 fixed: [-2.0, -1.0, 0.0, 1.0, 2.0]
Test points:      3 points: [-1.5, 0.5, 1.5]
Vocabulary:       5 tokens (α, β, γ, δ, ε) for MVP
Max message:      5 tokens
Tolerance:        1% relative
Primitives:       5 (sin, cos, square, abs, neg)
Model:            gemini/gemini-2.5-flash
Conditions:       2 (few-shot, zero-shot)
Total trials:     76 (38 per condition)
```

### Few-Shot Coverage
```
FEW-SHOT PRIMITIVES:     α=sin, β=cos, γ=square, δ=abs, ε=neg
FEW-SHOT COMPOSITIONS:   α-γ (sin∘square), δ-α (abs∘sin)
HELD-OUT COMPOSITIONS:   All other pairs (18 combinations)
HELD-OUT PRIMITIVES:     ζ=sqrt, η=exp, θ=log, ι=relu, κ=sign (future)
```

### Φ Interpretation
```
Φ ≥ 0.9   = Strong compositional grammar (learned the RULE)
0.7 ≤ Φ   = Good compositionality
0.5 ≤ Φ   = Partial compositionality
Φ < 0.5   = Memorization dominant (only remembers few-shot examples)
```

### File Locations
```
Functions:        telepathic/core/functions.py
Few-shot:         telepathic/core/few_shot.py
Protocol:         telepathic/core/protocol.py
Seer agent:       telepathic/agents/seer.py
Doer agent:       telepathic/agents/doer.py
Runner:           telepathic/experiments/runner.py
Prompts:          telepathic/prompts/
Results:          results/telepathic/
```

---

## Appendix B: Checkpoint Summary

```
PHASE 0: SETUP
[x] 0.1 Directory structure

PHASE 1: CORE ENGINE
[x] 1.1 Function library (5 MVP primitives)
[x] 1.2 Sample generator
[x] 1.3 Protocol manager
[x] 1.4 Evaluation engine
[x] 1.5 Few-shot generator

PHASE 2: AGENTS
[x] 2.1 Base agent interface
[x] 2.2 Seer agent (with few-shot prompt)
[x] 2.3 Doer agent (with few-shot prompt)
[x] 2.4 Random baseline

PHASE 3: EXPERIMENTS
[ ] 3.1 Trial executor
[ ] 3.2 Curriculum manager (3 phases: primitives, novel, incompressible)
[ ] 3.3 Episode runner
[ ] 3.4 Result logger

PHASE 4: VISUALIZATION (optional for MVP)
[ ] 4.1 Function plotter
[ ] 4.2 Trial visualizer
[ ] 4.3 Results dashboard

PHASE 5: ANALYSIS (optional for MVP)
[ ] 5.1 Statistical analysis
[ ] 5.2 Vocabulary analysis
```

---

## Appendix C: Future Extensions

### Extension 1: Parameterized Functions
Add primitives with parameters (e.g., `add_c`, `mul_c`):
- Requires parameter transmission protocol
- Tests if agents can communicate numerical values
- Significantly increases complexity

### Extension 2: Depth-3+ Compositions
Test deeper function nesting:
- How does Φ degrade with depth?
- Is there a "compositionality horizon"?

### Extension 3: Neural Network Agents
Replace LLMs with trained neural networks:
- Compare MDL-trained vs cross-entropy trained
- Directly test "pattern maker vs recognizer" hypothesis

### Extension 4: Multi-Round Learning
Allow agents to learn over multiple episodes:
- Does vocabulary stabilize?
- Does Φ improve with experience?

---

*Document version: 3.0*
*Created: 2026-01-14*
*Updated: 2026-01-17 (Agentic Fluidity redesign: Seer invents & teaches language, no natural language constraint)*
*Status: SPECIFICATION UPDATED - Implementation requires agent redesign*