# Task for Claude Opus: Complete Phase 2 Implementation

## Context

You are continuing work on the **Manifold Coordination Benchmark** dissertation project. Phase 1 (Core Surface Engine) is complete and tested. Phase 2 (Agent Framework) has been partially implemented.

## Current Status

### ✅ Already Completed (by Sonnet):

**Checkpoint 2.1 - BaseAgent Interface:**
- File: `manifold_benchmark/agents/base.py`
- Abstract base class with 3 concrete methods (receive_observation, receive_message, reset)
- 3 abstract methods (generate_message, decide_position, final_decision)
- Verified: Cannot be instantiated directly (raises TypeError as expected)

**Checkpoint 2.2 - RandomAgent:**
- File: `manifold_benchmark/agents/random_agent.py`
- Returns random positions in [0, 10]
- Deterministic with seed (uses numpy.random.default_rng)
- No communication (empty messages)

**Checkpoint 2.3 - GreedyAgent:**
- File: `manifold_benchmark/agents/greedy_agent.py`
- Follows gradient direction: `new_pos = current + step_size * sign(gradient)`
- Uses gradient_x for Agent A, gradient_y for Agent B
- Clamps to [0, domain_size] bounds
- No communication (empty messages)

**System Prompts:**
- File: `manifold_benchmark/prompts/agent_a_system.txt`
- File: `manifold_benchmark/prompts/agent_b_system.txt`
- Both prompts created and ready for LLM agents

### ❌ Still Need to Complete:

1. **Checkpoint 2.4 - LLMAgent** (CRITICAL - Your main task)
2. **Test suite** for all agents (`test_agents.py`)
3. **Run tests** and verify all pass
4. **Update PLAN.md** Appendix C to mark checkpoints complete
5. **Git commit** with checkpoint references

## Your Primary Task: Implement Checkpoint 2.4 (LLMAgent)

### Why Opus is Needed for This

Per PLAN.md Appendix D, this checkpoint requires Opus because:
- **Complex API integration**: Must support both OpenAI AND Anthropic APIs with different SDK methods
- **Critical parsing logic**: Coordinate extraction from LLM responses with many edge cases
- **Silent failure risk**: Parsing bugs won't crash but will corrupt all experiment data
- **Error handling**: Must gracefully handle API failures, retries, timeouts

### Implementation Requirements

**File:** `manifold_benchmark/agents/llm_agent.py`

**Key Features Needed:**

1. **API Client Initialization** (`_init_api_client`)
   - Detect model type (OpenAI vs Anthropic) from model string
   - Initialize appropriate client (openai.OpenAI or anthropic.Anthropic)
   - Read API keys from environment variables or constructor parameter
   - Store api_type ('openai' or 'anthropic') for later use

2. **System Prompt Loading** (`_load_system_prompt`)
   - Default path: `prompts/agent_{role}_system.txt`
   - Allow custom path via constructor
   - Read and store prompt text

3. **Observation Formatting** (`_format_observation`)
   - Convert observation dict to human-readable text
   - Include: position, value_at_position, gradient (x or y based on role), slice values
   - Format clearly for LLM understanding

4. **Prompt Building** (`_build_prompt`)
   - Start with system prompt
   - Add observation history with formatted observations
   - Interleave messages from other agent
   - Optionally add decision request at end

5. **Coordinate Parsing** (`_parse_coordinate`) - MOST CRITICAL
   - Use regex to find all numbers: `re.findall(r'[-+]?\d*\.?\d+', response)`
   - Take LAST number as the answer (handles "gradient is 0.5, I'll move to 7.2")
   - Clamp to [0, domain_size]
   - Raise ValueError if no numbers found

6. **API Calling with Retry** (`_call_api`)
   - Handle OpenAI: `client.chat.completions.create()`
   - Handle Anthropic: `client.messages.create()` (NOTE: system prompt goes in separate parameter)
   - Exponential backoff: 1s, 2s, 4s on failure
   - MAX_RETRIES = 3
   - TIMEOUT_SECONDS = 30

7. **Public Methods:**
   - `generate_message()`: Call API to generate message for other agent
   - `decide_position()`: Call API to get coordinate, parse it, handle errors
   - `final_decision()`: Call API with final decision prompt

**Critical Edge Cases to Handle:**

Coordinate parsing must work with:
- "7.5" → 7.5
- "My answer is 7.5" → 7.5
- "gradient 0.12, move to 6.8" → 6.8 (NOT 0.12!)
- "approximately 7 to 8, say 7.5" → 7.5
- "x=6.5" → 6.5
- Values outside [0, 10] → clamp to bounds
- No numbers in response → raise ValueError, trigger retry with clarifying prompt

**Error Handling Strategy:**
```
On API error:
  - Retry with exponential backoff (up to 3 times)
  - If all fail: return current_position (no movement)

On parse error:
  - Retry with clarifying prompt: "Please respond with just a single number between 0 and 10."
  - If still fails: return current_position

All errors should print messages for debugging
```

## Reference Materials

**Implementation Plan:**
- Location: `/Users/morgan/.claude/plans/tranquil-crafting-rain.md`
- Contains complete pseudocode for LLMAgent (lines 260-570)

**Specification:**
- Location: `PLAN.md` in project root
- Section 10.5: LLMAgent API specification
- Section 8: Checkpoint 2.4 requirements

**Existing Phase 1 Code:**
- `manifold_benchmark/core/surface.py` - Surface class
- `manifold_benchmark/core/observation.py` - ObservationGenerator (creates observations)
- `manifold_benchmark/core/episode.py` - Episode manager

## Test Suite Requirements

**File:** `manifold_benchmark/tests/test_agents.py`

Must include tests for:

**BaseAgent:**
- Cannot be instantiated (abstract class)

**RandomAgent:**
- `test_random_agent_bounds()` - positions stay in [0, 10]
- `test_random_agent_deterministic()` - same seed produces same sequence
- `test_random_agent_ignores_observations()` - doesn't use observations

**GreedyAgent:**
- `test_greedy_agent_follows_gradient()` - positive gradient increases position
- `test_greedy_agent_respects_bounds()` - clamping works
- `test_greedy_agent_negative_gradient()` - negative gradient decreases position

**LLMAgent:**
- `test_llm_agent_coordinate_parsing_simple()` - "7.5" → 7.5
- `test_llm_agent_coordinate_parsing_sentence()` - "My answer is 7.5" → 7.5
- `test_llm_agent_coordinate_parsing_multiple_numbers()` - "gradient 0.12, move to 6.8" → 6.8
- `test_llm_agent_coordinate_clamping()` - values clamped to [0, 10]
- `test_llm_agent_observation_formatting()` - observation dict converted to text

Note: LLM tests should NOT make actual API calls. Use mocking or test parsing logic directly.

## Verification Steps

After implementation, run:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest manifold_benchmark/tests/test_agents.py -v

# Expected: All tests pass (10-15 tests total)

# Test imports work
python -c "from manifold_benchmark.agents.base import BaseAgent; from manifold_benchmark.agents.random_agent import RandomAgent; from manifold_benchmark.agents.greedy_agent import GreedyAgent; from manifold_benchmark.agents.llm_agent import LLMAgent; print('All imports successful')"
```

## Deliverables

When you're done, you should have:

1. ✅ `manifold_benchmark/agents/llm_agent.py` - Fully implemented
2. ✅ `manifold_benchmark/tests/test_agents.py` - Complete test suite
3. ✅ All tests passing (pytest shows 0 failures)
4. ✅ Updated PLAN.md Appendix C with Phase 2 checkpoints marked complete
5. ✅ Git commits:
   - "Complete checkpoint 2.4: LLM Agent"
   - "Add comprehensive test suite for Phase 2 agents"
   - "Complete Phase 2: Agent Framework"

## Important Notes

- Do NOT implement checkpoint 2.4 with Sonnet - use Opus (that's you!)
- Follow the implementation plan closely - it has detailed pseudocode
- Pay special attention to coordinate parsing - this is where bugs hide
- Test thoroughly - parsing errors won't crash but corrupt experiment data
- The LLMAgent will be used in Phase 3 (Episode Runner) so it must be robust

## Questions to Consider

If anything is unclear:
1. Check the implementation plan at `/Users/morgan/.claude/plans/tranquil-crafting-rain.md`
2. Check PLAN.md sections 8 and 10
3. Look at existing Phase 1 code for patterns (type hints, docstrings, structure)
4. Ask the user for clarification

## Success Criteria

Phase 2 is complete when:
- ✅ All 4 agent classes implemented (base + 3 concrete)
- ✅ All tests written and passing
- ✅ LLMAgent can parse coordinates from various response formats
- ✅ Error handling prevents crashes and provides useful fallbacks
- ✅ Code matches quality of Phase 1 (docstrings, type hints, clean structure)

Good luck! This is the most complex checkpoint in Phase 2, but the plan has all the details you need.
