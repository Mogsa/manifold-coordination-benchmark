# Telepathic Benchmark v2: Binary Lambda Calculus Compression

> **Purpose**: Test whether AI agents can compress mathematical functions into short programs, demonstrating genuine understanding through Kolmogorov complexity approximation.

> **Status**: SPECIFICATION - Ready for Implementation

> **Relationship to v1**: This replaces the "Agentic Fluidity" two-agent design. The two-agent teaching variant is scaffolded for future work.

---

## Table of Contents

1. [Core Thesis](#1-core-thesis)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [The Agent's Freedom: Encoding Choice](#3-the-agents-freedom-encoding-choice)
4. [Benchmark Protocol](#4-benchmark-protocol)
5. [Scoring System](#5-scoring-system)
6. [Lambda Calculus Specification](#6-lambda-calculus-specification)
7. [Numerical Evaluation Pipeline](#7-numerical-evaluation-pipeline)
8. [Function Test Suite](#8-function-test-suite)
9. [Implementation Architecture](#9-implementation-architecture)
10. [Implementation Checkpoints](#10-implementation-checkpoints)
11. [Baselines](#11-baselines)
12. [Verification Criteria](#12-verification-criteria)
13. [Future Extensions](#13-future-extensions)

---

## 1. Core Thesis

> **An agent that truly understands a function can compress it into a short program. The ability to compress—not just predict—demonstrates structural understanding.**

### 1.1 What This Tests

The benchmark tests **Automated Scientific Discovery**: Can an AI agent observe noisy data and re-derive mathematics from first principles?

To succeed, the agent must:

1. **Act as a Computer Engineer**: Design a number encoding and arithmetic from pure lambda calculus
2. **Act as a Mathematician**: Derive approximations (Taylor series, polynomials) for functions
3. **Act as a Data Scientist**: Balance precision vs code size (MDL tradeoff)

**The agent has FULL FREEDOM over encoding choice.** They may use Church numerals (simple but slow), binary lists (complex but efficient), or invent something entirely new.

### 1.2 Key Innovation

| Innovation | Description |
|------------|-------------|
| Noise forces compression | Noisy samples prevent memorization |
| Bits as score | Program length directly measures understanding |
| Active probing | Agent decides when it has gathered enough evidence |
| Universal measurement | All outputs compiled to BLC for objective scoring |

---

## 2. Theoretical Foundations

This section establishes the theoretical basis for using program compression as a measure of understanding. We present the key theorems, explain their relevance, and explicitly acknowledge where we use heuristic approximations.

---

### 2.1 Kolmogorov Complexity

#### 2.1.1 Definition

**Kolmogorov Complexity** (also called algorithmic complexity or descriptive complexity) measures the intrinsic information content of an object.

```
DEFINITION: The Kolmogorov complexity of string x with respect to
universal Turing machine U is:

    K_U(x) = min { |p| : U(p) = x }

That is: the length of the shortest program p (in bits) such that
U running p outputs x.
```

For functions, we extend this to:

```
K_U(f) = min { |p| : U(p, x) = f(x) for all x in domain }
```

**Plain English:** K(f) is the length of the shortest program that computes f. A function with low K(f) has exploitable structure; a function with high K(f) is essentially random.

#### 2.1.2 The Uncomputability Theorem

```
THEOREM (Kolmogorov, 1965; Chaitin, 1966):

    K(x) is not computable. There exists no algorithm that takes
    arbitrary input x and outputs K(x).

PROOF SKETCH:
    Suppose K were computable. Consider the program:
    "Find the first string x with K(x) > n and output it."
    This program has length O(log n) but outputs a string with K(x) > n.
    Contradiction.
```

**Why this matters for our benchmark:**

| Implication | Consequence |
|-------------|-------------|
| We cannot compute K(f) | We can only measure UPPER BOUNDS |
| Agent's program \|P\| ≥ K(f) | Better agents → tighter upper bounds |
| We measure compression ability | Not the true Kolmogorov complexity |

**Explicit acknowledgment:** When we say "approximate K(f)", we mean: measure how well the agent compresses, knowing the true K(f) is unknowable.

#### 2.1.3 The Invariance Theorem

```
THEOREM (Solomonoff, 1964; Kolmogorov, 1965):

    For any two universal Turing machines U₁ and U₂, there exists
    a constant c (depending only on U₁ and U₂, not on x) such that:

    |K_{U₁}(x) - K_{U₂}(x)| ≤ c

PROOF SKETCH:
    U₁ can simulate U₂ with a fixed-size interpreter program.
    So any program for U₂ can be converted to one for U₁ by
    prepending this interpreter. The overhead is constant.
```

**Why this matters for our benchmark:**

We use Binary Lambda Calculus (BLC) as our reference machine. The invariance theorem tells us:

- The CHOICE of reference machine doesn't fundamentally matter
- Different machines give K values differing by at most a constant
- BLC is a valid choice (and has nice properties: simple, well-studied)

**Explicit acknowledgment:** Our bit counts are BLC-specific. Comparisons are valid within our benchmark but absolute values depend on the BLC constant.

#### 2.1.4 The Incompressibility Theorem

```
THEOREM (Kolmogorov, 1965):

    For any length n, at least 2^n - 2^{n-c} + 1 strings of length n
    satisfy K(x) ≥ n - c.

    In other words: MOST strings are incompressible.

COROLLARY:
    A random string x of length n satisfies K(x) ≥ n - O(1)
    with high probability.
```

**Why this matters for our benchmark:**

This theorem justifies our **Tier 4: Incompressible Functions** (random lookups).

| If agent... | Then... |
|-------------|---------|
| Succeeds on Tier 4 | Benchmark is broken (impossible by theorem) |
| Fails on Tier 4 | Expected behavior — validates the benchmark |

Random functions have K(f) ≈ size of lookup table. No compression is possible. An agent that "succeeds" must be cheating (e.g., memorizing test set).

---

### 2.2 Minimum Description Length (MDL)

#### 2.2.1 The Principle

**Minimum Description Length** is a formalization of Occam's Razor for model selection. It originates from Rissanen (1978).

```
PRINCIPLE (Two-Part MDL):

    The best hypothesis H for data D minimizes:

    L(H, D) = L(H) + L(D|H)

    Where:
        L(H)   = description length of the hypothesis (in bits)
        L(D|H) = description length of data given hypothesis (in bits)
```

**Plain English:**
- L(H) = how complex is your model?
- L(D|H) = how well does it fit? (bits to encode residual errors)
- Best model balances complexity and fit

#### 2.2.2 Connection to Our Scoring

Our scoring formula directly implements MDL:

```
SCORE = |P|_BLC        + ERROR_PENALTY
        ↑                ↑
        L(H)             L(D|H)
        model complexity  residual encoding
```

| MDL Component | Our Implementation |
|---------------|-------------------|
| L(H) | BLC bit length of agent's program |
| L(D\|H) | Σ log₂(1 + \|errorᵢ\| × k) |

#### 2.2.3 Justification for Error Penalty Form

**The theoretical ideal:** L(D|H) should be the bits needed to optimally encode the prediction errors. For Gaussian errors with variance σ², this would be:

```
L(D|H) ≈ (n/2) × log₂(2πeσ²)  for n data points
```

**Our approximation:** We use `log₂(1 + |error| × k)` per test point.

```
WHY THIS FORM:

1. Logarithmic scaling matches information theory
   - Small errors need few bits to specify
   - Large errors need more bits

2. The "+1" prevents log(0) and ensures minimum 0 bits for perfect fit

3. The "× k" (k=100) scales errors to meaningful bit counts
   - Error of 0.01 → log₂(1 + 1) = 1 bit
   - Error of 0.1  → log₂(1 + 10) ≈ 3.5 bits
   - Error of 1.0  → log₂(1 + 100) ≈ 6.7 bits
```

**Explicit acknowledgment:** This is a HEURISTIC, not a derivation from first principles. A fully rigorous MDL approach would use:
- Prequential (predictive) coding, or
- Normalized Maximum Likelihood (NML), or
- Bayesian model evidence

We accept this approximation because:
1. It has the correct qualitative behavior
2. It's computationally simple
3. It enables fair comparison between agents

---

### 2.3 Rate-Distortion Theory

#### 2.3.1 The Framework

**Rate-Distortion Theory** (Shannon, 1959) addresses lossy compression: what's the minimum bit rate needed to represent a source with bounded distortion?

```
DEFINITION: For source X and reconstruction X̂ with distortion
measure d(x, x̂), the rate-distortion function is:

    R(D) = min_{p(x̂|x): E[d(X,X̂)] ≤ D} I(X; X̂)

Where:
    D = maximum allowed expected distortion
    I(X; X̂) = mutual information (bits)
    R(D) = minimum rate (bits) to achieve distortion ≤ D
```

#### 2.3.2 Connection to Our Benchmark

Our benchmark IS a rate-distortion problem:

```
MAPPING:

    Rate R        ↔  Program length |P|_BLC
    Distortion D  ↔  Prediction error on test set
    Source X      ↔  The unknown function f
    Encoder       ↔  The agent's compression strategy
```

The agent must navigate the rate-distortion tradeoff:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   High rate (long program)  →  Low distortion (accurate)    │
│   Low rate (short program)  →  High distortion (errors)     │
│                                                             │
│   OPTIMAL: Find the "knee" of the R(D) curve                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.3.3 Why This Matters

Rate-distortion theory tells us:

1. **There exists a fundamental limit** — you cannot compress below R(D) and achieve distortion ≤ D
2. **The tradeoff is unavoidable** — shorter programs necessarily sacrifice accuracy (for complex functions)
3. **Our scoring rewards efficiency** — agents near the R(D) curve score better than those far from it

**Explicit acknowledgment:** We don't compute R(D) analytically (intractable for arbitrary functions). The benchmark empirically explores this tradeoff and compares agents' efficiency.

---

### 2.4 Generalization: Why Compression Implies Understanding

#### 2.4.1 The Core Insight

Why should a short program that fits noisy data generalize to new data? This isn't obvious — couldn't it be a coincidence?

The answer comes from **learning theory**: short hypotheses that fit noisy data MUST capture real structure. They cannot be fitting noise.

#### 2.4.2 Blumer's Occam Theorem (1987)

```
THEOREM (Blumer, Ehrenfeucht, Haussler, Warmuth, 1987):

    Let H be a hypothesis class with |H| hypotheses.
    Let S be a sample of m examples drawn i.i.d. from distribution D.
    Let h ∈ H be consistent with S (zero training error).

    Then with probability at least 1 - δ:

        error_D(h) ≤ (1/m)(ln|H| + ln(1/δ))

    Where error_D(h) is the true error under distribution D.
```

#### 2.4.3 The Description Length Version

When hypotheses are described by bit strings:

```
COROLLARY:

    If hypothesis h can be described in b bits, then h is
    effectively one of 2^b possible hypotheses.

    Substituting |H| = 2^b into Blumer's theorem:

        error_D(h) ≤ (1/m)(b × ln(2) + ln(1/δ))
                   ≈ (b + log(1/δ)) / m

    Shorter description (smaller b) → lower error bound.
```

#### 2.4.4 Application to Our Benchmark

```
WHY COMPRESSION → GENERALIZATION:

1. Agent produces program P with |P|_BLC = b bits
2. P is consistent with noisy probe samples (fits the data)
3. By Blumer's theorem: P's true error is bounded by O(b/m)
4. Short program + fits noisy data → MUST generalize

THE NOISE IS CRITICAL:
- Without noise: agent could memorize samples (high b, but fits perfectly)
- With noise (σ=0.1): fitting noise requires high b
- Short program that fits noisy data → found real structure, not noise
```

**Why Gaussian noise with σ=0.1:**

| Property | Rationale |
|----------|-----------|
| Gaussian | Maximum entropy for given variance; no structure to exploit |
| σ=0.1 | Large enough to prevent exact memorization; small enough that signal dominates |

**Explicit acknowledgment:** The choice σ=0.1 is empirical, not derived from theory. It could be tuned based on function ranges.

---

### 2.5 Solomonoff Induction

#### 2.5.1 The Universal Prior

**Solomonoff Induction** (1964) defines the theoretically optimal predictor using a universal prior over programs.

```
DEFINITION (Solomonoff Prior):

    The prior probability of program p is:

        P(p) = 2^{-|p|}

    That is: probability decreases exponentially with program length.

INTERPRETATION:
    - 10-bit program: probability 2^{-10} ≈ 0.001
    - 100-bit program: probability 2^{-100} ≈ 10^{-30}
    - Shorter programs are EXPONENTIALLY more likely a priori
```

#### 2.5.2 Solomonoff's Theorem

```
THEOREM (Solomonoff, 1964):

    Let M(x) = Σ_p 2^{-|p|} where sum is over all programs p
    that output string beginning with x.

    M(x) is a universal predictor: it dominates any computable
    predictor up to a multiplicative constant.

    Specifically, for any computable probability distribution Q:

        M(x) ≥ 2^{-K(Q)} × Q(x)
```

#### 2.5.3 Connection to Our Benchmark

Solomonoff induction embodies the principle:

```
SHORTER PROGRAMS ARE MORE LIKELY TO BE CORRECT

Our benchmark tests: Can agents approximate Solomonoff-style reasoning?

- Given noisy samples, the agent should PREFER short explanations
- This is exactly what MDL scoring rewards
- An agent that produces short, accurate programs is approximating
  the Solomonoff ideal
```

**Explicit acknowledgment:** Solomonoff induction is UNCOMPUTABLE (it requires summing over all programs). Our benchmark tests whether agents can achieve practical approximations through explicit program synthesis.

---

### 2.6 Acknowledged Approximations and Heuristics

We explicitly acknowledge where our benchmark uses approximations rather than theoretically optimal methods:

| Component | Theoretical Ideal | Our Implementation | Justification |
|-----------|-------------------|-------------------|---------------|
| **K(f)** | True Kolmogorov complexity | Agent's \|P\|_BLC | K is uncomputable; we measure upper bounds. Invariance theorem ensures BLC is valid. |
| **L(D\|H)** | Optimal error coding (NML or prequential) | log₂(1 + \|e\| × k) | Heuristic with correct qualitative behavior. Simpler to compute. |
| **Probe cost** | Information-theoretic value of query | 5 bits per probe | Rough estimate: ~log₂(20) for choosing one of ~20 meaningful probe points in (0,1]. |
| **Test set size** | Infinite domain evaluation | 20 uniform samples | Practical limitation. Theory suggests O(1/ε²) samples for ε-accuracy. |
| **Noise level** | Derived from signal properties | σ = 0.1 fixed | Empirical choice balancing signal preservation and memorization prevention. |
| **Reference machine** | Any UTM | Binary Lambda Calculus | Invariance theorem: choice affects scores by at most a constant. BLC is simple and well-studied. |
| **Timeout** | Unbounded computation | 10M β-reductions | Practical necessity. Implicitly penalizes slow encodings (related to Kt complexity). |

#### 2.6.1 What These Approximations Mean

**For benchmark validity:**
- Comparisons BETWEEN agents are meaningful (same approximations apply to all)
- Absolute scores are BLC-specific and heuristic-dependent
- Tier 4 (incompressible) validates that the benchmark isn't trivially gameable

**For dissertation claims:**
- Claim: "Agent A compresses better than Agent B" ✓ (valid comparison)
- Claim: "Agent A achieves K(f)" ✗ (K is unknowable)
- Claim: "Agent A approximates K(f) with upper bound b bits" ✓ (valid)

---

## 3. The Agent's Freedom: Encoding Choice

### 3.1 The Core Principle

**The agent chooses their own encoding.** We do not mandate how numbers are represented. The agent must:

1. Design an encoding for numbers in pure lambda calculus
2. Implement arithmetic operations using that encoding
3. Build their function approximation on top

**This IS the test.** The encoding choice is part of what we measure.

### 3.2 The Tradeoff

Different encodings have different tradeoffs:

| Encoding | Program Complexity | Execution Speed | When to Use |
|----------|-------------------|-----------------|-------------|
| **Church numerals** | Very simple | O(n²) in value — SLOW | Tiny numbers only |
| **Binary lists** | Complex (needs ADD, MUL) | O(n²) in bits — FAST | Large numbers |
| **Fixed approximation** | Depends | Depends | If agent finds clever shortcut |

**Example: Church numerals**
```python
# Simple to write...
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
MUL = lambda m: lambda n: lambda f: m(n(f))

# ...but 100 × 100 requires 10,000 function applications
# Large numbers will TIMEOUT
```

**Example: Binary lists**
```python
# Complex to write (needs full adder, carry logic)...
BIT0 = lambda x: lambda y: x
BIT1 = lambda x: lambda y: y
CONS = lambda h: lambda t: lambda s: s(h)(t)
# ... plus 50+ lines of arithmetic

# ...but 100 × 100 only requires ~50 operations
# Scales to any size
```

### 3.3 What This Tests

By giving the agent full control, we test:

| Capability | How It's Tested |
|------------|-----------------|
| **Engineering judgment** | Does agent choose an encoding that will actually work? |
| **MDL tradeoff** | Simple encoding + big program vs complex encoding + small program |
| **Computational reasoning** | Does agent understand execution costs? |
| **Creativity** | Can agent invent novel encodings we didn't anticipate? |

**The scoring handles this automatically**: if the agent chooses a bad encoding, either:
- Their program times out (∞ score), or
- Their program is huge (high bit count), or
- Their approximation is poor (high error penalty)

A good encoding choice leads to a good score. We don't need to mandate it.

---

## 4. Benchmark Protocol

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK PROTOCOL                            │
│                                                                 │
│  INPUT:  Unknown function f, accessible only through probing    │
│  OUTPUT: Pure λ-calculus program P that computes f              │
│  SCORE:  |P|_BLC + error_penalty + probe_penalty                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Active Probing

```
PROTOCOL:

  probe_count = 0
  samples = []

  REPEAT:
    Agent sends: PROBE(x) where x ∈ (0, 1]
    Environment returns: y = f(x) + ε where ε ~ N(0, σ²)

    samples.append((x, y))
    probe_count += 1

  UNTIL Agent sends: DONE

AGENT'S GOAL:
  - Identify the underlying pattern
  - Distinguish signal from noise
  - Gather sufficient evidence
  - Minimize probe count (probes have cost)
```

### 4.3 Phase 2: Program Synthesis

```
PROTOCOL:

  Agent outputs a STRUCTURED RESPONSE with two parts:

  1. REASONING (for research analysis):
     - What pattern did the agent identify from the samples?
     - What encoding strategy did they choose and why?
     - What mathematical approach (Taylor series, polynomial fit, etc.)?
     - Any tradeoffs considered (accuracy vs program size)?

  2. PROGRAM: A pure λ-calculus program P

  REQUIREMENTS FOR PROGRAM:
    - P must be valid λ-calculus syntax (Python lambda only)
    - P must compile to BLC without error
    - P must terminate within timeout on all test inputs
    - P must include encode/decode functions for I/O

  AGENT'S CHOICE:
    - How to encode numbers (Church, binary, custom...)
    - How to implement arithmetic
    - How to approximate the function

  OUTPUT FORMAT (JSON):
    {
      "reasoning": "I observed that y ≈ x² based on the probe samples...
                    Choosing Church numerals for simplicity...
                    Using MUL(x)(x) for squaring...",
      "program": "MUL = lambda m: lambda n: lambda f: m(n(f))\nSQUARE = lambda x: MUL(x)(x)"
    }

  CLAIM:
    "P computes (approximately) the function f"
```

**Why capture reasoning?**

| Research Value | What It Reveals |
|----------------|-----------------|
| Pattern recognition | How does the LLM identify structure from noisy data? |
| Engineering choices | Does it understand encoding tradeoffs (Church vs binary)? |
| Mathematical insight | Can it derive approximations (Taylor series) from samples? |
| MDL intuition | Does it reason about compression vs accuracy tradeoffs? |

This is NOT scored — it's for dissertation analysis of the agent's thought process.

### 4.4 Phase 3: Evaluation

```
PROTOCOL:

  STEP 1: Compile to BLC
    blc_program = compile_to_blc(P)
    compression_bits = len(blc_program)

  STEP 2: Generate held-out test points
    test_inputs = uniform_sample((0, 1], n=20)  -- Fixed per function, not probed

  STEP 3: Execute and measure error (see Section 7 for details)
    FOR each x_test in test_inputs:

      -- Try pattern matching first (fast path)
      y_pred = pattern_match_evaluate(P, x_test)

      -- Fall back to beta reduction if pattern matching fails
      IF y_pred == UNRECOGNIZED:
        y_pred = beta_reduce_evaluate(P, x_test, SCALE=1000, timeout=10M)

      IF y_pred == TIMEOUT_ERROR:
        RETURN Score = ∞

      y_true = f(x_test)  -- Ground truth (no noise)
      errors.append(|y_pred - y_true|)

  STEP 4: Compute final score
    SCORE = compression_bits + error_penalty + probe_penalty
```

---

## 5. Scoring System

### 5.1 Score Formula

```
SCORE = |P|_BLC + ERROR_PENALTY + PROBE_PENALTY

Where:

|P|_BLC = length of BLC-compiled program in bits

ERROR_PENALTY = Σᵢ log₂(1 + |predicted_i - actual_i| × k)
  Where k = 100 (error scaling factor)

PROBE_PENALTY = n_probes × C_probe
  Where C_probe = 5 bits per probe

LOWER IS BETTER
```

### 5.2 Justification for Error Penalty

The logarithmic penalty follows MDL principles:

```
log₂(1 + error × k) ≈ bits needed to "correct" the error

Small errors:  log₂(1 + 0.01 × 100) = log₂(2) = 1 bit
Medium errors: log₂(1 + 0.1 × 100) = log₂(11) ≈ 3.5 bits
Large errors:  log₂(1 + 1.0 × 100) = log₂(101) ≈ 6.7 bits

Smooth pressure toward accuracy without overwhelming
compression score for small approximation errors.
```

### 5.3 Justification for Probe Penalty

```
Each probe provides information, but probing is "easy."
The real test is: Can you COMPRESS the information?

C_probe = 5 bits ≈ log₂(20) — the "cost" of choosing
one of ~20 meaningful probe points in (0,1].
```

### 5.4 Hyperparameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Noise std dev | σ | 0.1 | Gaussian noise on probe samples |
| Domain | (a, b] | (0, 1] | Input range (excludes 0 for log safety) |
| **Scale factor** | **SCALE** | **1000** | **Fixed-point scaling (resolution 0.001)** |
| Error scale | k | 100 | Multiplier in error penalty |
| Probe cost | C_probe | 5 bits | Cost per probe |
| Timeout | T | 10,000,000 | Max beta reductions |
| Test set size | N_test | 20 | Held-out test points (uniform, fixed per function) |

---

## 6. Lambda Calculus Specification

### 6.0 The Key Constraint: Python Lambda Calculus Only

**CRITICAL RULES**:

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM WRITES:  Pure lambda calculus using Python's lambda syntax │
│                                                                 │
│  ALLOWED:     lambda, function application, variable names      │
│                                                                 │
│  NOT ALLOWED: +, -, *, /, if, for, while, lists, dicts,        │
│               built-in functions, imports, ANYTHING ELSE        │
│                                                                 │
│  LLM CAN:     Test/verify programs in Python interpreter        │
│                                                                 │
│  WE COMPILE:  Python lambdas → BLC for scoring                  │
└─────────────────────────────────────────────────────────────────┘
```

The LLM uses Python syntax (which it knows well) but is restricted to PURE lambda calculus. No Python features except `lambda` and application.

### 6.1 What the LLM Can Write

**ALLOWED** (pure lambda calculus in Python syntax):
```python
# Abstraction
lambda x: x                      # Identity
lambda x: lambda y: x            # TRUE / First
lambda x: lambda y: y            # FALSE / Second

# Application
f(x)                             # Apply f to x
(lambda x: x)(5)                 # Apply identity to 5

# Variable binding (using Python assignment for readability)
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y
AND = lambda a: lambda b: a(b)(FALSE)

# Nested lambdas
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
```

**NOT ALLOWED** (these are NOT lambda calculus):
```python
# NO arithmetic operators
lambda x: x + 1                  # FORBIDDEN
lambda x: x * 2                  # FORBIDDEN

# NO conditionals
lambda x: x if x > 0 else -x     # FORBIDDEN

# NO built-in functions
lambda x: abs(x)                 # FORBIDDEN
lambda x: math.sin(x)            # FORBIDDEN

# NO data structures
lambda x: [x, x]                 # FORBIDDEN
lambda x: {'a': x}               # FORBIDDEN
```

### 6.2 Interactive Testing Environment

The LLM has access to a Python interpreter to TEST its lambda calculus programs:

```python
# LLM can define and test interactively:

>>> TRUE = lambda x: lambda y: x
>>> FALSE = lambda x: lambda y: y
>>> AND = lambda a: lambda b: a(b)(FALSE)

>>> AND(TRUE)(TRUE) == TRUE
True

>>> AND(TRUE)(FALSE) == FALSE
True

# LLM can test their own encoding (whatever they choose):
>>> # If using Church numerals:
>>> ZERO = lambda f: lambda x: x
>>> SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
>>> church_to_int = lambda n: n(lambda x: x + 1)(0)  # Helper for testing
>>> church_to_int(SUCC(SUCC(ZERO)))
2

>>> # If using binary lists:
>>> BIT0 = lambda x: lambda y: x
>>> CONS = lambda h: lambda t: lambda s: s(h)(t)
>>> # ... test their implementation
```

This allows the LLM to **iteratively develop and debug** its solution before final submission.

### 6.3 Syntax Rules

```bnf
<term> ::= <variable>                    -- Variable names
         | "lambda" <variable> ":" <term>  -- Abstraction
         | <term> "(" <term> ")"         -- Application
         | "(" <term> ")"                -- Grouping

<variable> ::= [a-zA-Z_][a-zA-Z0-9_]*    -- Python identifiers
```

**That's it.** No other Python constructs allowed.

### 6.4 Variable Assignments (For Readability)

The LLM MAY use Python variable assignment to name lambda terms:

```python
# This is allowed (just naming):
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
TWO = SUCC(ONE)

# The FINAL PROGRAM must be a single lambda expression
# that we compile to BLC
```

**Compilation**: We inline all variable references to get a single lambda term, then compile to BLC.

### 6.5 Example Encodings (For Reference Only)

**The agent is NOT required to use any of these.** These are examples of possible approaches:

#### Option A: Church Numerals (Simple but Slow)

```python
# Very simple to define
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))
MUL = lambda m: lambda n: lambda f: m(n(f))

# Warning: O(n) in VALUE for operations
# 100 × 100 = 10,000 function applications → may timeout
```

#### Option B: Binary Lists (Complex but Fast)

```python
# Bits
BIT0 = lambda x: lambda y: x    # Represents 0 (same as TRUE)
BIT1 = lambda x: lambda y: y    # Represents 1 (same as FALSE)

# Lists (Scott encoding)
NIL  = lambda s: lambda z: z
CONS = lambda h: lambda t: lambda s: s(h)(t)

# Number 5 = 101 in binary (LSB first)
FIVE = CONS(BIT1)(CONS(BIT0)(CONS(BIT1)(NIL)))

# Requires implementing ADD, MUL as ~50 lines of lambda calculus
# But operations are O(log n) — scales to large numbers
```

#### Option C: Custom Encoding

The agent may invent something we haven't thought of. As long as:
- It's pure lambda calculus
- It compiles to BLC
- It produces correct outputs

**The benchmark doesn't care HOW — only that it works.**

### 6.6 BLC Compilation

```
BLC Encoding Rules:

  Abstraction:  λM     → 00 ++ blc(M)
  Application:  M N    → 01 ++ blc(M) ++ blc(N)
  Variable:     i      → 1^(i+1) ++ 0

Where i = De Bruijn index (0 = innermost binding)

Examples:
  Identity (λx.x):     00 10           = 4 bits
  True (λx.λy.x):      00 00 110       = 7 bits
  False (λx.λy.y):     00 00 10        = 6 bits
```

### 6.7 Execution

```
Reduction: Normal order (leftmost-outermost)
Timeout:   10,000,000 beta reductions
Result:    Final lambda term (agent's encoding) or TIMEOUT_ERROR
```

---

## 7. Numerical Evaluation Pipeline

This section explains how we bridge lambda calculus programs to numerical testing.

### 7.1 The Core Problem

The LLM outputs pure lambda calculus. We need to test it on actual numbers to measure error.

```
Lambda World                    Numerical World
─────────────────              ─────────────────
λf.λx.f(f(x))      ────────►   f(3.5) = ???
```

### 7.2 Fixed-Point Scaling

All numerical values are converted to integers via fixed-point scaling:

```
SCALE = 1000

Input:  x = 0.35  →  x_int = 350
Output: y = 0.1225 → y_int = 123 (rounded)

Resolution: 0.001 (three decimal places)
Max value:  (0, 1] → (0, 1000] as integers
```

**Why SCALE = 1000?**

| Scale | Resolution | x² at x=1 | Church MUL cost | Timeout safe? |
|-------|------------|-----------|-----------------|---------------|
| 100   | 0.01       | 10,000    | ~10K reductions | ✓ Yes |
| 1000  | 0.001      | 1,000,000 | ~1M reductions  | ✓ Yes |

With domain (0, 1], SCALE=1000 is safe (well under 10M timeout) and provides 0.001 resolution — 100× finer than the noise floor (σ = 0.1).

### 7.3 Two-Stage Evaluation

We evaluate lambda programs using two approaches:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                          │
│                                                                 │
│  STAGE 1: Pattern Matching (Fast Path)                         │
│    - Parse lambda calculus into AST                            │
│    - Recognize standard combinators (MUL, ADD, SUCC, etc.)     │
│    - Convert to equivalent Python expression                    │
│    - Evaluate numerically                                       │
│                                                                 │
│  STAGE 2: Beta Reduction (Fallback)                            │
│    - If pattern matching fails (novel structure)               │
│    - Scale input to integer, encode as Church numeral          │
│    - Execute via beta reduction (up to 10M steps)              │
│    - Decode result back to number                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 Stage 1: Pattern Matching

We maintain a dictionary of known lambda combinators and their numerical meanings:

| Lambda Pattern | Numerical Equivalent |
|----------------|---------------------|
| `MUL(a)(b)` | `a * b` |
| `ADD(a)(b)` | `a + b` |
| `SUB(a)(b)` | `a - b` |
| `SUCC(n)` | `n + 1` |
| `PRED(n)` | `n - 1` |
| `POW(a)(n)` | `a ** n` |
| `CHURCH_N` | `N` (the number) |

**Example:**
```python
# LLM writes:
SQUARE = lambda x: MUL(x)(x)

# We parse and recognize:
#   SQUARE(x) = MUL(x)(x) → x * x

# We generate Python:
def square(x: float) -> float:
    return x * x

# We test on actual numbers
```

### 7.5 Stage 2: Beta Reduction (Fallback)

When the LLM invents novel structures we don't recognize, we fall back to direct execution:

```python
def evaluate_via_beta_reduction(program: LambdaTerm, x: float) -> float:
    SCALE = 1000

    # 1. Scale float to integer
    x_int = round(x * SCALE)

    # 2. Encode as Church numeral
    x_church = int_to_church(x_int)

    # 3. Apply program and reduce
    result_term = beta_reduce(
        App(program, x_church),
        max_steps=10_000_000
    )

    # 4. Decode Church numeral to integer
    y_int = church_to_int(result_term)

    # 5. Unscale to float
    return y_int / SCALE
```

**When does this happen?**

| Situation | Example | Fallback needed? |
|-----------|---------|------------------|
| Standard multiplication | `MUL(x)(x)` | No - pattern match |
| Novel number encoding | Custom representation | Yes |
| Clever mathematical trick | Unrecognized identity | Yes |
| Malformed program | Doesn't compute anything | Yes (will timeout or error) |

### 7.6 Why Allow Novel Structures?

The benchmark tests **creativity**, not just knowledge of standard combinators. An LLM might:

1. Invent a more efficient encoding we didn't anticipate
2. Exploit a mathematical identity we don't recognize
3. Find a compression strategy that's genuinely novel

Beta reduction as fallback ensures we can evaluate ANY valid lambda calculus, even structures we've never seen.

### 7.7 Scaling for Different Operations

The LLM must account for how scaling affects operations:

| Operation | Input Scale | Output Scale | LLM Must Handle |
|-----------|-------------|--------------|-----------------|
| x | 1000 | 1000 | Nothing |
| x + y | 1000 | 1000 | Nothing |
| x × y | 1000 | 1,000,000 | Divide by SCALE |
| x² | 1000 | 1,000,000 | Divide by SCALE |
| x³ | 1000 | 1,000,000,000 | Divide by SCALE² |

**Example for x²:**
```
True: 0.5² = 0.25
Scaled input: 500
Naive: 500 × 500 = 250,000 (wrong scale!)
Correct: 250,000 / 1000 = 250 → 0.25 ✓
```

The LLM must implement division by SCALE in their program, OR we normalize at the boundary (design choice - see Section 7.8).

### 7.8 Design Decision: Who Handles Scale Normalization?

**Option A: LLM handles internally**
- LLM knows SCALE = 1000
- LLM divides by SCALE after multiplication
- More work for LLM, but "purer" test

**Option B: We normalize at boundary**
- We tell LLM the operation type (e.g., "quadratic")
- We apply appropriate unscaling based on function degree
- Easier for LLM, but requires function metadata

**Current choice: Option A** — The LLM handles all scaling. This is part of the engineering challenge.

---

## 8. Function Test Suite

### 8.1 Tier 1: Polynomials (Low Complexity)

| ID | Function | Formula | Expected Score |
|----|----------|---------|----------------|
| P1 | Identity | f(x) = x | TBD |
| P2 | Square | f(x) = x² | TBD |
| P3 | Cube | f(x) = x³ | TBD |
| P4 | Linear | f(x) = 2x + 1 | TBD |
| P5 | Quadratic | f(x) = x² + 2x + 1 = (x+1)² | TBD |

### 8.2 Tier 2: Transcendentals (Medium Complexity)

| ID | Function | Hint (Taylor series converges in (0,1]) | Expected Score |
|----|----------|----------------------------------------|----------------|
| T1 | Sine | x - x³/6 + x⁵/120 | TBD |
| T2 | Cosine | 1 - x²/2 + x⁴/24 | TBD |
| T3 | Exponential | 1 + x + x²/2 + x³/6 | TBD |
| T4 | Logarithm | ln(1+x) ≈ x - x²/2 + x³/3 | TBD |

### 8.3 Tier 3: Compositions (High Complexity)

| ID | Function | Formula | Expected Score |
|----|----------|---------|----------------|
| C1 | Sin of square | sin(x²) | TBD |
| C2 | Gaussian | e^(-x²) | TBD |
| C3 | Identity (tricky) | sin²(x) + cos²(x) = 1 | TBD |

### 8.4 Tier 4: Incompressible (Control)

| ID | Description | Expected Behavior |
|----|-------------|-------------------|
| R1 | Random lookup | Agent SHOULD FAIL |
| R2 | Pseudo-random | Very high score |

**Purpose**: If agent succeeds on incompressible functions, the benchmark is broken.

---

## 9. Implementation Architecture

### 9.1 Directory Structure

```
telepathic/
├── core/
│   ├── __init__.py
│   ├── lambda_parser.py      # Parse λ-calculus syntax (Python lambda)
│   ├── blc_compiler.py       # Compile to Binary Lambda Calculus
│   ├── blc_interpreter.py    # Execute BLC with timeout
│   ├── functions.py          # Test function library
│   ├── sampling.py           # Noisy sample generation
│   └── evaluation.py         # Scoring (bits + error + probes)
│
├── agents/
│   ├── __init__.py
│   ├── base.py               # Abstract agent interface
│   ├── llm_agent.py          # LLM-based agent
│   └── baselines.py          # Random, memorization, oracle
│
├── experiments/
│   ├── __init__.py
│   ├── environment.py        # Active probing environment
│   ├── runner.py             # Experiment orchestrator
│   └── analysis.py           # Results analysis
│
├── vocabulary/               # SCAFFOLD FOR VARIANT B
│   ├── __init__.py
│   ├── base.py               # Vocabulary interface (future)
│   └── compiler.py           # Vocabulary compilation (future)
│
├── prompts/
│   ├── agent_system.txt      # System prompt for LLM agent
│   └── agent_probe.txt       # Probing prompt template
│
└── tests/
    ├── test_lambda_parser.py
    ├── test_blc_compiler.py
    ├── test_blc_interpreter.py
    └── test_scoring.py
```

### 9.2 Key Interfaces

```python
from dataclasses import dataclass

# Synthesis Result (includes reasoning for research analysis)
@dataclass
class SynthesisResult:
    reasoning: str   # Agent's thought process (for dissertation analysis)
    program: str     # Pure λ-calculus program

# Agent Interface
class Agent(ABC):
    @abstractmethod
    def probe(self, environment: Environment) -> None:
        """Interactive probing phase. Agent decides when to stop."""
        pass

    @abstractmethod
    def synthesize(self) -> SynthesisResult:
        """Return reasoning + λ-calculus program."""
        pass

# Environment Interface
class Environment:
    def __init__(self, function: Callable, noise_std: float = 0.1):
        self.f = function
        self.noise_std = noise_std
        self.probes = []  # List of (x, y) tuples

    def probe(self, x: float) -> float:
        """Return noisy sample at x."""
        y = self.f(x) + np.random.normal(0, self.noise_std)
        self.probes.append((x, y))
        return y

# Score Result
@dataclass
class Score:
    compression_bits: int    # |P|_BLC
    error_penalty: float     # Σ log₂(1 + |e| × 100)
    probe_penalty: int       # n_probes × 5
    total: float             # Sum of above (lower is better)

    # For analysis (not part of score)
    errors: list[float]      # Individual test point errors
    reasoning: str           # Agent's reasoning (preserved for research)

# Scorer Interface
def score(result: SynthesisResult, environment: Environment) -> Score:
    """Compute total score for a synthesis result."""
    blc = compile_to_blc(parse(result.program))
    errors = evaluate_on_test_set(blc, environment.f)

    compression_bits = len(blc)
    error_penalty = sum(log2(1 + e * 100) for e in errors)
    probe_penalty = len(environment.probes) * 5

    return Score(
        compression_bits=compression_bits,
        error_penalty=error_penalty,
        probe_penalty=probe_penalty,
        total=compression_bits + error_penalty + probe_penalty,
        errors=errors,
        reasoning=result.reasoning,
    )
```

---

## 10. Implementation Checkpoints

### Phase 1: Lambda Calculus Infrastructure

- [ ] **1.1** Lambda parser (Python lambda syntax → AST)
- [ ] **1.2** De Bruijn index conversion
- [ ] **1.3** BLC compiler (AST → binary string)
- [ ] **1.4** BLC interpreter with timeout
- [ ] **1.5** Unit tests for all above

### Phase 2: Benchmark Environment

- [ ] **2.1** Function library (all 4 tiers)
- [ ] **2.2** Noisy sample generation
- [ ] **2.3** Active probing interface
- [ ] **2.4** Held-out test point generation
- [ ] **2.5** Environment class

### Phase 3: Scoring & Evaluation

- [ ] **3.1** BLC bit counting
- [ ] **3.2** Error penalty computation
- [ ] **3.3** Probe penalty computation
- [ ] **3.4** Total score aggregation
- [ ] **3.5** Score dataclass (includes reasoning for research analysis)
- [ ] **3.6** Result reporting and export

### Phase 4: Agents

- [ ] **4.1** SynthesisResult dataclass (reasoning + program)
- [ ] **4.2** Abstract agent interface (probe + synthesize → SynthesisResult)
- [ ] **4.3** Random baseline agent
- [ ] **4.4** Memorization baseline agent
- [ ] **4.5** Oracle baseline agent
- [ ] **4.6** LLM agent implementation (with JSON output parsing)
- [ ] **4.7** Agent prompts (must request reasoning + program in JSON format)

### Phase 5: Experiments & Analysis

- [ ] **5.1** Experiment runner
- [ ] **5.2** Results logging
- [ ] **5.3** Score correlation analysis
- [ ] **5.4** Visualization

---

## 11. Baselines

### 11.1 Random Baseline

```python
class RandomAgent(Agent):
    def probe(self, env):
        for _ in range(10):
            env.probe(random.uniform(0.01, 1.0))

    def synthesize(self) -> SynthesisResult:
        return SynthesisResult(
            reasoning="No analysis performed. Returning identity function.",
            program="lambda x: x"
        )
```

**Expected**: Very high score (poor accuracy)

### 11.2 Memorization Baseline

```python
class MemorizationAgent(Agent):
    def synthesize(self) -> SynthesisResult:
        return SynthesisResult(
            reasoning="Memorizing all probe samples as lookup table.",
            program=self.build_lookup_table()  # Huge λ-calculus IF chain
        )
```

**Expected**: ∞ (fails on held-out test points)

### 11.3 Oracle Baseline

```python
class OracleAgent(Agent):
    def __init__(self, true_function_name):
        self.name = true_function_name

    def synthesize(self) -> SynthesisResult:
        return SynthesisResult(
            reasoning=f"Oracle knows true function is {self.name}.",
            program=OPTIMAL_PROGRAMS[self.name]
        )
```

**Expected**: Best possible score (lower bound)

---

## 12. Verification Criteria

### 12.1 Sanity Checks

| Check | Expected Result |
|-------|-----------------|
| Polynomials | Short programs (relative to other tiers) |
| Transcendentals | Medium programs |
| Incompressible | Agent FAILS (∞ or huge score) |
| Random baseline | Very high scores |
| Oracle baseline | Lowest scores |

### 12.2 Correlation Test

Score should correlate with true K(f):
- Simpler functions → lower scores
- Complex functions → higher scores
- Plot: Score vs estimated complexity

### 12.3 Noise Robustness

- Same function with different noise seeds → similar scores
- Higher noise → slightly higher scores (harder to identify)

---

## 13. Future Extensions

### 13.1 Variant B: Vocabulary Invention

**Deferred for later implementation**

```
INPUT:  Multiple related functions f₁, f₂, ..., fₙ
OUTPUT: Vocabulary V + Messages M₁, M₂, ..., Mₙ
SCORE:  |V| + Σ|Mᵢ| + errors + probes
```

Tests whether agent can **amortize** common structure (encoding, arithmetic, primitives) across functions.

### 13.2 Two-Agent Teaching

**Deferred for later implementation**

One agent invents vocabulary, teaches another agent through demonstration. Tests theory of mind and emergent communication.

### 13.3 Deeper Compositions

Test depth-3+ compositions: sin(cos(x²)), etc.

### 13.4 Multi-Agent Competition

Multiple agents compete on same functions. Compare compression strategies.

---

## Appendix A: Example Walkthrough

### Function: f(x) = x²

**Phase 1: Probing**
```
Agent: PROBE(0.1) → (0.1, 0.012)
Agent: PROBE(0.5) → (0.5, 0.247)
Agent: PROBE(0.7) → (0.7, 0.493)
Agent: PROBE(1.0) → (1.0, 0.98)
Agent: "Pattern: values ≈ x²! DONE"

Total probes: 4
```

**Phase 2: Synthesis**
```python
# Agent chooses their encoding and implements arithmetic
# (Could be Church numerals, binary lists, or something custom)

# Example using Church numerals (simple case):
MUL = lambda m: lambda n: lambda f: m(n(f))

# Final program: f(x) = x²
SQUARE = lambda x: MUL(x)(x)
```

**Phase 3: Evaluation**
```
BLC compilation: TBD bits
Test points: {0.15, 0.35, 0.6, 0.85, ...}
All predictions correct within tolerance.
Error penalty: TBD bits
Probe penalty: 4 × 5 = 20 bits

TOTAL SCORE: TBD bits
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| BLC | Binary Lambda Calculus — compact encoding for λ-terms |
| K(f) | Kolmogorov complexity — length of shortest program computing f |
| MDL | Minimum Description Length — principle that best model minimizes total description |
| De Bruijn index | Variable representation using binding depth instead of names |
| Church numeral | Encoding of n as `lambda f: lambda x: f(f(...f(x)))` (n times) — simple but slow |
| Binary list | Encoding of n as list of bits — complex but efficient |
| Agent's choice | The encoding is NOT mandated — agent decides their own representation |

---

*Document version: 2.4*
*Created: 2026-01-17*
*Updated: 2026-01-18 — SCALE=1000 (0.001 resolution), domain (0, 1], expected scores TBD, Section 7 (Numerical Evaluation Pipeline)*
*Status: SPECIFICATION - Ready for Implementation*
