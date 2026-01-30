# Metacognitive Agent Implementation Status

**Date:** 2026-01-30
**Status:** In Progress - Debugging Scoring Issue

---

## What We Built

All 7 tasks from the implementation plan are complete:

| Component | File | Status |
|-----------|------|--------|
| Data Types | `chaosbench/agents/metacognitive_types.py` | ✅ Done |
| Learnings Manager | `chaosbench/agents/learnings.py` | ✅ Done |
| Trace Logger | `chaosbench/experiments/trace.py` | ✅ Done |
| Session Runner | `chaosbench/experiments/session.py` | ✅ Done (modified) |
| LLM Agent | `chaosbench/agents/metacognitive_agent.py` | ✅ Done |
| CLI Runner | `chaosbench/run_metacognitive.py` | ✅ Done |
| Integration Test | `chaosbench/tests/test_integration.py` | ✅ Done |

---

## Design Changes Made

### Problem Discovered
The original design gave feedback after each PREDICT, allowing the agent to iterate toward a revealed answer (playing "hot/cold"). This made the task trivially easy - agent got perfect scores by just adjusting toward the shown answer.

### Fix Applied
Changed to two-phase design:

**Phase 1: Prediction (blind)**
- Agent sees observations
- Agent can PREDICT multiple times (refining thinking)
- NO feedback until commit
- Agent says MOVE_ON to commit

**Phase 2: Reflection (after MOVE_ON)**
- Agent sees result (prediction, actual, score)
- Agent can WRITE learnings
- Then next task begins

### Files Modified
- `chaosbench/experiments/session.py` - No feedback during PREDICT, only after MOVE_ON
- `chaosbench/prompts/metacognitive_system.txt` - Updated to explain two-phase flow

---

## Current Issue

**Agent never says MOVE_ON.** It keeps predicting 20 times (hitting the turn limit) without committing. Still getting perfect scores somehow, which needs investigation.

Possible causes:
1. Agent not understanding it needs to commit
2. Scoring bug
3. The last prediction happens to be correct by luck

### To Debug Next
1. Check if feedback is truly hidden (trace shows no feedback on PREDICT turns - looks correct)
2. Verify scoring function with known values
3. Update prompt to make MOVE_ON more explicit
4. Consider forcing single-shot prediction (only 1 PREDICT allowed)

---

## How to Run

```bash
cd "/Users/morgan/Desktop/Year 3/Diss/DISS"
source venv/bin/activate

# Run 5 tasks with Gemini 2.0 Flash
python -m chaosbench.run_metacognitive --model "gemini/gemini-2.0-flash" --n-tasks 5 --conditional --output session_output_test

# Check results
cat session_output_test/learnings.md
cat session_output_test/trace.md
```

---

## Key Files

- **Implementation plan:** `docs/plans/2026-01-30-chaosbench-v4-implementation.md`
- **Design spec:** `docs/plans/2026-01-30-chaosbench-metacognitive-agent-design.md`
- **System prompt:** `chaosbench/prompts/metacognitive_system.txt`
- **Session runner (main logic):** `chaosbench/experiments/session.py`
- **Learning journal:** `FORMORGAN.md`

---

## Next Steps

1. Debug why agent never says MOVE_ON
2. Debug why scores are perfect (1.00) with blind prediction
3. Consider simplifying to single-shot prediction
4. Once working, run longer experiments to test if agent uses WRITE
5. Analyze results for dissertation

---

## Model Configuration

Currently using:
- Model: `gemini/gemini-2.0-flash` (Gemini 3 Flash has issues with JSON output format)
- API key in `.env` as `GEMINI_API_KEY`
- LiteLLM for API calls
- dotenv loading added to `shared/llm_utils.py`
