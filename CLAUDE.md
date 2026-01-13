# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Manifold Coordination Benchmark** — A benchmark for testing LLM multi-agent coordination on a joint optimization task.

Two LLM agents jointly control a single "player" navigating a 2D surface f(x,y) to find the global maximum:
- Agent A controls x-coordinate, sees horizontal slices
- Agent B controls y-coordinate, sees vertical slices
- Neither can see the full surface — communication is necessary

## Current Status & Next Steps

**Environment:** Setup complete with virtual environment and all dependencies installed
**Current Phase:** Phase 1 - Core Surface Engine
**Next Checkpoint:** 1.1 Surface Class (manifold_benchmark/core/surface.py)

Check PLAN.md "Appendix C: Checkpoint Summary" for detailed progress tracking and "Section 8: Implementation Checkpoints" for requirements.

## Essential Reading

**PLAN.md** — Complete implementation specification with all requirements, API specs, and test cases. READ THE RELEVANT CHECKPOINT SECTION before starting work on any component.

## Architecture Summary

```
manifold_benchmark/
├── core/           # Surface, observation, episode logic
├── agents/         # Agent implementations (random, greedy, LLM)
├── visualization/  # 3D plots, trajectory visualization
├── experiments/    # Episode runner, evaluation harness
├── prompts/        # LLM system prompts
└── tests/          # Unit tests
```

## Key Design Decisions

| Decision | Choice |
|----------|--------|
| Information asymmetry | Perpendicular 1D slices (Agent A: horizontal, Agent B: vertical) |
| Movement | Continuous coordinates in [0, 10] |
| Observation | Slice of radius R=1.5 with 11 samples + partial derivative |
| Turns | N=10 exploration turns + final decision |
| Scoring | f(final) / f(optimal), normalized to [0, 1] |

## Implementation Status

Check PLAN.md "Appendix C: Checkpoint Summary" for current progress.

## Model Assignment (Opus vs Sonnet)

See PLAN.md "Appendix D" for detailed reasoning. Quick reference:

**Use OPUS for:**
- 2.4 LLM Agent (parsing edge cases, API integration)
- 3.1 Turn Executor (orchestration correctness)
- 5.2 Statistical Analysis (research validity)
- Prompts (LLM behavior understanding)
- Architecture reviews at phase boundaries

**Use SONNET for:**
- All Phase 1 (Surface, Observation, Episode)
- 2.1-2.3 (Base Agent, Random, Greedy)
- 3.2 Logger, 4.x Visualization, 5.1/5.3 (straightforward)

## Development Commands

```bash
# Activate virtual environment (REQUIRED before any other commands)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_surface.py

# Run tests with verbose output
pytest tests/ -v

# Run single episode (after Phase 3 implementation)
python -m manifold_benchmark.experiments.runner --surface two_peaks_clear

# Run full evaluation (after Phase 5 implementation)
python -m manifold_benchmark.experiments.eval --config configs/experiments.yaml
```

## Environment Setup

**Virtual environment is already configured.** To work in this project:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set API keys (only needed for Phase 2.4: LLM Agent)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# 3. Verify setup
python --version  # Should show Python 3.13.7
pytest --version  # Should show pytest 9.0.2
```

## Workflow

**For each checkpoint:**
1. Read the checkpoint requirements in PLAN.md Section 8
2. Read the API specification in PLAN.md Section 10 for the component
3. Read the test cases in PLAN.md Section 12 for expected behavior
4. Implement the component
5. Run tests: `pytest tests/test_<module>.py -v`
6. Mark checkpoint complete in PLAN.md Appendix C
7. Commit with checkpoint reference: `git commit -m "Complete checkpoint 1.1: Surface class"`
8. Update "Current Status" in this file when completing major phases

## Critical Implementation Details

**Information Asymmetry (Core Design):**
- Agent A sees horizontal slice: f(x', y_b) for x' in [x_a - R, x_a + R]
- Agent B sees vertical slice: f(x_a, y') for y' in [y_b - R, y_b + R]
- Agent A receives ∂f/∂x (NOT ∂f/∂y), Agent B receives ∂f/∂y (NOT ∂f/∂x)
- Slices are perpendicular and centered at view point (x_a, y_b)

**Key Parameters:**
```python
DOMAIN = [0, 10] × [0, 10]
OBSERVATION_RADIUS = 1.5
N_SAMPLES = 11
N_TURNS = 10
INITIAL_POSITION = (5.0, 5.0)
```

**Surface Functions:**
All surfaces are sums of Gaussian peaks defined by (cx, cy, height, sigma). See PLAN.md Section 7 for pre-defined test surfaces.

**Scoring:**
Score = f(x_final, y_final) / f(x_optimal, y_optimal), normalized to [0, 1]
