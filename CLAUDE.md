# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⚠️ IMPORTANT: How to Work With Morgan

**Morgan is a researcher, not a coder.** They excel at:
- Planning and architecture
- Debugging and problem-solving
- Understanding what benchmarks MEAN for LLM research
- Interpreting results and drawing conclusions

**Morgan is NOT interested in:**
- Writing code themselves
- Learning syntax details
- Watching you code silently for long stretches

### Collaboration Model

| Morgan does | Claude does |
|-------------|-------------|
| Makes research decisions | Writes the code |
| Predicts agent behavior | Runs experiments |
| Interprets what results mean | Generates visualizations |
| Explains design choices (for dissertation) | Builds infrastructure |
| Asks "what does this MEAN?" | Explains mechanics |

### BEFORE Starting Any Major Work, ASK:

1. **Research questions:** "What hypothesis are we testing with this component?"
2. **Expected behavior:** "What do you predict will happen when we run this?"
3. **Design tradeoffs:** "I could do X or Y - which aligns better with your research goals?"
4. **Success criteria:** "How will we know if this is working correctly?"

### AFTER Building Something, DISCUSS:

1. "Here's what I built - does this match your mental model?"
2. "What does this result tell us about LLM reasoning?"
3. "Is this benchmark actually testing what you intended?"
4. "How would you explain this finding to your supervisor?"

### Don't Just Execute - Engage

- If given a task like "do phase 1", STOP and ask research questions first
- Surface assumptions that need Morgan's input
- Explain WHY you're making architectural choices
- Connect code decisions back to research implications

### The Architect Protocol (For Code Design)

**Before implementing any component, ASK Morgan:**
1. "What are the inputs and outputs of this component?"
2. "What are the steps/transformations between them?"
3. "What could go wrong or needs special handling?"

**After implementing, VERIFY with Morgan:**
1. "Before I run this - predict what you expect to happen"
2. "Now let's change one thing deliberately - what do you expect to break?"

**This gives Morgan ownership of the LOGIC while Claude handles the SYNTAX.**

Morgan understands: classes, loops, variables, data flow, architecture
Claude handles: syntax, libraries, edge cases, boilerplate

---

## Project Overview

**Manifold Benchmark Suite** — Benchmarks for testing LLM reasoning capabilities.

### Coordination Benchmark (Complete)
Two LLM agents jointly control a single "player" navigating a 2D surface f(x,y) to find the global maximum:
- Agent A controls x-coordinate, sees horizontal slices
- Agent B controls y-coordinate, sees vertical slices
- Neither can see the full surface — communication is necessary

### Telepathic Benchmark v2 (Active Priority)
**Single-agent compression benchmark testing Kolmogorov complexity approximation:**
- Agent observes noisy function samples via active probing
- Must synthesize a pure λ-calculus program computing the function
- Agent has FULL FREEDOM over encoding (Church numerals, binary lists, custom)
- Program compiled to Binary Lambda Calculus (BLC) for objective scoring
- **Score = program bits + error penalty + probe penalty** (lower is better)
- Tests: Can LLMs compress → demonstrating structural understanding?

See `Telepathic_Plan_v2.md` for full specification.

### Telepathic Benchmark v1 (Paused)
Two LLM agents play a communication game testing compositional reasoning:
- **Seer** observes input-output samples of a function, encodes it as abstract tokens
- **Doer** receives only the token message, must compute output for new inputs
- **Bandwidth constraint** (≤5 tokens) forces genuine compression, not memorization
- **Key metric: Φ** = accuracy on novel compositions / accuracy on primitives
- Tests whether LLMs learn compositional grammar vs. lookup tables

*Status: Phases 1-2 complete (core + agents). May be finished later.*

### Temporal Benchmark (Paused)
A single LLM agent navigates a 1D surface f(x,t) that evolves over time:
- Agent controls x-position, sees local slices + gradients
- Surface changes according to hidden dynamics (peaks move, grow, shrink)
- Agent must learn patterns and predict future states

*Status: Phase 0 complete (repository restructure). May be finished later.*

## Current Status

| Benchmark | Status | Plan File |
|-----------|--------|-----------|
| Coordination | ✅ Complete (all 5 phases) | `PLAN.md` |
| **Telepathic v2** | 🎯 **Active Priority** — Implementation starting | `Telepathic_Plan_v2.md` |
| Telepathic v1 | ⏸️ Paused (Phases 1-2 done) | `Telepathic_Plan.md` |
| Temporal | ⏸️ Paused (Phase 0 done) | `Temporal_PLAN.md` |

**Next Steps:** Implement Telepathic v2 starting with Phase 1 (Lambda Calculus Infrastructure). See `Telepathic_Plan_v2.md` Section 10 for checkpoints.

## Architecture Summary

```
DISS/
├── shared/             # Shared utilities for all benchmarks
│   ├── gaussians.py    # Gaussian peak math (1D and 2D)
│   ├── llm_utils.py    # LiteLLM wrapper, retry logic, parsing
│   ├── logging.py      # Base result logging
│   └── base_agent.py   # Abstract agent interfaces
│
├── coordination/       # 2-agent coordination benchmark (Complete)
│   ├── core/           # Surface, observation, episode logic
│   ├── agents/         # Agent implementations (random, greedy, LLM)
│   ├── visualization/  # 3D plots, trajectory visualization
│   ├── experiments/    # Episode runner, evaluation harness
│   ├── prompts/        # LLM system prompts
│   └── tests/          # Unit tests
│
├── temporal/           # Temporal tracking benchmark (Paused)
│   ├── core/           # (To be implemented)
│   ├── agents/         # (To be implemented)
│   └── ...
│
├── telepathic/         # Telepathic benchmarks (v1 paused, v2 active)
│   ├── core/           # v1: Functions, sampling, protocol, evaluation, few-shot
│   │                   # v2: Will add lambda_parser, blc_compiler, blc_interpreter
│   ├── agents/         # v1: LLM Seer/Doer + random baselines (Complete)
│   │                   # v2: Will add compression agent
│   ├── experiments/    # Trial runner (To be implemented for v2)
│   ├── prompts/        # Agent prompts
│   └── tests/          # Unit tests
│
├── PLAN.md             # Coordination benchmark specification
├── Temporal_PLAN.md    # Temporal benchmark specification
├── Telepathic_Plan.md  # Telepathic v1 specification (Seer/Doer two-agent)
└── Telepathic_Plan_v2.md  # Telepathic v2 specification (BLC compression) ← ACTIVE
```

## Development Commands

```bash
# Activate virtual environment (REQUIRED)
source venv/bin/activate

# Run coordination tests
pytest coordination/tests/ -v

# Run telepathic tests
pytest telepathic/tests/ -v

# Run specific test file
pytest coordination/tests/test_surface.py -v
pytest telepathic/tests/test_core.py -v
pytest telepathic/tests/test_agents.py -v

# Run coordination episode
python -m coordination.experiments.runner --surface two_peaks_clear

# Run coordination evaluation
python -m coordination.experiments.eval --config configs/experiments.yaml

# Telepathic v1 debug scripts (sanity checks with Gemini)
python -m telepathic.debug_conversation  # Test primitive (square)
python -m telepathic.debug_novel         # Test novel composition (cos∘square)

# Telepathic v2 (once implemented)
# pytest telepathic/tests/test_lambda_parser.py -v
# pytest telepathic/tests/test_blc_compiler.py -v
# pytest telepathic/tests/test_blc_interpreter.py -v
```

## Environment Setup

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set API keys (for LLM agents)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# 3. Verify setup
python --version  # Should show Python 3.13.7
pytest --version  # Should show pytest 9.0.2
```

## Key Design Decisions

### Coordination Benchmark
| Decision | Choice |
|----------|--------|
| Information asymmetry | Perpendicular 1D slices |
| Movement | Continuous coordinates in [0, 10] |
| Observation | Slice radius R=1.5, 11 samples + gradient |
| Turns | N=10 exploration + final decision |
| Scoring | f(final) / f(optimal) |

### Temporal Benchmark
| Decision | Choice |
|----------|--------|
| Observation | Local slice radius R=0.5, 11 samples + ∂f/∂x + ∂f/∂t |
| Timesteps | T=20 per run, max 10 exploration runs |
| Mode | Function mode (batch 20 positions per run) |
| Scoring | Cumulative reward / optimal reward |

### Telepathic Benchmark v2 (Active)
| Decision | Choice |
|----------|--------|
| Agent freedom | Agent chooses own encoding (Church numerals, binary lists, custom) |
| Input syntax | Pure λ-calculus via Python lambda (no +, *, if, etc.) |
| Compilation target | Binary Lambda Calculus (BLC) |
| Domain | (0, 1] — excludes 0 for log safety |
| Scale factor | SCALE=1000 (resolution 0.001) |
| Noise | Gaussian σ=0.1 on probe samples |
| Scoring | `bits + Σlog₂(1+|error|×100) + probes×5` |
| Timeout | 10M β-reductions |
| Test set | 20 held-out points (uniform, no noise) |

### Telepathic Benchmark v1 (Paused)
| Decision | Choice |
|----------|--------|
| Primitives | 5 MVP: sin(α), cos(β), square(γ), abs(δ), neg(ε) |
| Sample points | 5 fixed: [-2, -1, 0, 1, 2] |
| Test points | 3 fixed: [-1.5, 0.5, 1.5] |
| Max message | 5 tokens, hyphen-separated |
| Tolerance | 1% relative (absolute for near-zero) |
| Key metric | Φ = novel_accuracy / primitive_accuracy |

## Workflow

**Current Priority: Telepathic v2** — See `Telepathic_Plan_v2.md` Section 10 for checkpoints.

**Paused benchmarks (may resume later):**
- Telepathic v1: See `Telepathic_Plan.md`
- Temporal: See `Temporal_PLAN.md`
- Coordination: Complete — See `PLAN.md`

### Implementation Steps

1. Read the checkpoint requirements in the relevant PLAN.md
2. Read the API specification for the component
3. Read the test cases for expected behavior
4. Implement the component
5. Run tests: `pytest <module>/tests/test_<file>.py -v`
6. Mark checkpoint complete in PLAN.md Appendix
7. Commit with checkpoint reference

### Telepathic v2 Phases

| Phase | Focus | Key Components |
|-------|-------|----------------|
| 1 | Lambda Calculus Infrastructure | Parser, De Bruijn, BLC compiler/interpreter |
| 2 | Benchmark Environment | Function library, noisy sampling, probing interface |
| 3 | Scoring & Evaluation | Bit counting, error penalty, total score |
| 4 | Agents | Base interface, baselines (random, memorization, oracle), LLM agent |
| 5 | Experiments & Analysis | Runner, logging, visualization |
