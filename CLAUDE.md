# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Manifold Benchmark Suite** — Benchmarks for testing LLM reasoning capabilities.

### Coordination Benchmark (Complete)
Two LLM agents jointly control a single "player" navigating a 2D surface f(x,y) to find the global maximum:
- Agent A controls x-coordinate, sees horizontal slices
- Agent B controls y-coordinate, sees vertical slices
- Neither can see the full surface — communication is necessary

### Temporal Benchmark (In Progress)
A single LLM agent navigates a 1D surface f(x,t) that evolves over time:
- Agent controls x-position, sees local slices + gradients
- Surface changes according to hidden dynamics (peaks move, grow, shrink)
- Agent must learn patterns and predict future states

## Current Status

| Benchmark | Status |
|-----------|--------|
| Coordination | ✅ Complete (all 5 phases) |
| Temporal | 🔄 Phase 0 Complete, Phase 1 next |

**Next Steps:** Begin Phase 1 of Temporal Benchmark (Core Engine). See Temporal_PLAN.md.

## Architecture Summary

```
DISS/
├── shared/             # Shared utilities for both benchmarks
│   ├── gaussians.py    # Gaussian peak math (1D and 2D)
│   ├── llm_utils.py    # LiteLLM wrapper, retry logic, parsing
│   ├── logging.py      # Base result logging
│   └── base_agent.py   # Abstract agent interfaces
│
├── coordination/       # 2-agent coordination benchmark
│   ├── core/           # Surface, observation, episode logic
│   ├── agents/         # Agent implementations (random, greedy, LLM)
│   ├── visualization/  # 3D plots, trajectory visualization
│   ├── experiments/    # Episode runner, evaluation harness
│   ├── prompts/        # LLM system prompts
│   └── tests/          # Unit tests
│
├── temporal/           # Temporal tracking benchmark (in progress)
│   ├── core/           # (To be implemented)
│   ├── agents/         # (To be implemented)
│   └── ...
│
├── PLAN.md             # Coordination benchmark specification
└── Temporal_PLAN.md    # Temporal benchmark specification
```

## Development Commands

```bash
# Activate virtual environment (REQUIRED)
source venv/bin/activate

# Run coordination tests
pytest coordination/tests/ -v

# Run specific test file
pytest coordination/tests/test_surface.py -v

# Run coordination episode
python -m coordination.experiments.runner --surface two_peaks_clear

# Run coordination evaluation
python -m coordination.experiments.eval --config configs/experiments.yaml
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

## Workflow

**For Coordination:** See PLAN.md for checkpoint details.

**For Temporal:** See Temporal_PLAN.md for checkpoint details.

1. Read the checkpoint requirements in the relevant PLAN.md
2. Read the API specification for the component
3. Read the test cases for expected behavior
4. Implement the component
5. Run tests: `pytest <module>/tests/test_<file>.py -v`
6. Mark checkpoint complete in PLAN.md Appendix
7. Commit with checkpoint reference
