# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Manifold Coordination Benchmark** — A benchmark for testing LLM multi-agent coordination on a joint optimization task.

Two LLM agents jointly control a single "player" navigating a 2D surface f(x,y) to find the global maximum:
- Agent A controls x-coordinate, sees horizontal slices
- Agent B controls y-coordinate, sees vertical slices
- Neither can see the full surface — communication is necessary

## Key Documents

- **PLAN.md** — Complete implementation specification with checkpoints. READ THIS FIRST when starting work.

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

## Commands

```bash
# Run tests
pytest tests/

# Run single episode (once implemented)
python -m experiments.runner --surface two_peaks_clear

# Run full evaluation (once implemented)
python -m experiments.eval --config configs/experiments.yaml
```

## Working with This Codebase

1. Read PLAN.md Section 8 (Implementation Checkpoints) to find current phase
2. Each checkpoint has specific requirements and tests to pass
3. Mark checkpoints complete in PLAN.md as you finish them
4. Run tests after each checkpoint: `pytest tests/test_<module>.py`
