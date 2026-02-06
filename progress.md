# Progress Log: ChaosBench v2

## Session: 2026-02-06

### Phase 1-2 (Complete, previous sessions)
- 4 atoms implemented (logistic, tent, damped_linear, rotation)
- 27 validated problems in frozen bank (mini_bank.json)
- 7 quality gates + 5 baselines
- 3 verifiers (classify, identify, predict)
- 107 tests passing

### Phase 3 (Complete, this session)
- Created 5 source files: agents/protocol.py, prompts.py, parsing.py, llm_agent.py, experiment/runner.py
- Created 2 test files: test_parsing.py, test_runner.py
- Fixed "quasiperiodic" substring bug (sort labels longest-first)
- Fixed confidence regex not matching negatives
- **44 new tests, 92 total, all passing**
- Ran Gemini on all 27 problems: CLASSIFY=36%, IDENTIFY=55%, PREDICT=79%
- Diagnosed CLASSIFY failures: Gemini can't distinguish chaotic/quasiperiodic without analytical tools
- Diagnosed IDENTIFY failures: "logistic" default bias from training data

### Phase 4 (Planning, this session)
- Designed adversarial arena with Morgan
- Four-phase round: PROPOSE -> BLIND SOLVE -> PEER REVIEW -> CONSENSUS
- Key decisions: Gemini-only, both ground truths tracked, chaosbench/arena/ module
- NeurIPS-style Likert scales defined (1-6 question quality, 1-6 answer correctness, 1-5 confidence)
- Three-axis reputation: proposing, solving, reviewing
- Plan documented in task_plan.md, findings.md, progress.md
- **Status: Ready to implement**

### Files Modified This Session
- Created: `chaosbench/agents/protocol.py`
- Created: `chaosbench/agents/prompts.py`
- Created: `chaosbench/agents/parsing.py`
- Created: `chaosbench/agents/llm_agent.py`
- Created: `chaosbench/experiment/runner.py`
- Created: `chaosbench/tests/test_parsing.py`
- Created: `chaosbench/tests/test_runner.py`
- Updated: `FORMORGAN.md` (Phase 3 section)

### Next Steps
1. Implement arena/protocol.py (dataclasses)
2. Implement arena/prompts.py (3 role prompts)
3. Implement arena/parsing.py (proposal + review parsing)
4. Implement arena/consensus.py (consensus + reputation)
5. Implement arena/runner.py (main loop)
6. Implement tests/test_arena.py (mocked LLM)
7. Run live arena with Gemini (~$0.30-0.50)

### Test Results
```
pytest chaosbench/tests/ -v
92 passed in ~3s
```
