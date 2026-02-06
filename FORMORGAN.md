# FORMORGAN: What We Built and Why

This is your learning journal. When you look back at this code in 6 months, this file will remind you what we were thinking.

---

## The Metacognitive Agent: What's the Big Idea?

Imagine you're a scientist studying a mystery box. You poke it, observe what happens, write notes, and gradually build intuition. Some boxes follow simple rules; others are chaotic and unpredictable. The smart scientist learns *when to give up* on a hard box and move to an easier one.

That's exactly what our metacognitive agent does with chaotic dynamical systems.

**The research question:** Can an LLM accumulate useful knowledge about chaotic systems over time? Does it learn meta-strategies like "logistic maps oscillate between attractors" or "high Lyapunov exponent means don't bother predicting far ahead"?

---

## Architecture: The Restaurant Analogy

Think of a restaurant:

```
┌─────────────────────────────────────────────────────────────┐
│  SessionRunner (the restaurant manager)                     │
│  - Seats customers (generates tasks)                        │
│  - Times the meal (timeout handling)                        │
│  - Tracks the bill (Φ scoring)                              │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ LearningsManager│  │   TraceLogger   │                   │
│  │ (the menu notes)│  │ (the receipts)  │                   │
│  │ - Persists      │  │ - Every turn    │                   │
│  │   across tasks  │  │ - Every task    │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MetacognitiveAgent (the chef)                       │   │
│  │  - Sees the order (AgentObservation)                 │   │
│  │  - Cooks something (calls Gemini)                    │   │
│  │  - Plates it (AgentAction)                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Data flows like this:**
1. SessionRunner generates a task (chaotic time series)
2. It packages the task into an `AgentObservation` (what the agent sees)
3. Agent calls Gemini, gets back reasoning + JSON action
4. SessionRunner interprets the action:
   - `PREDICT`: Score it, give feedback
   - `WRITE`: Add to learnings notepad
   - `DELETE`: Remove from learnings
   - `MOVE_ON`: Bank the score, next task
5. Repeat until timeout

---

## The Components: What Each File Does

### Data Types (`metacognitive_types.py`)

The "vocabulary" of the system. Three dataclasses:

```python
Feedback      # What you see after predicting: "You said 0.5, actual was 0.3, score 0.7"
AgentObservation  # Everything the agent sees: task_id, time series, learnings, feedback
AgentAction   # What agent can do: PREDICT(0.42), WRITE("insight"), DELETE("## Old"), MOVE_ON
```

Plus `parse_action()` which extracts JSON from LLM responses. This is trickier than it sounds—LLMs write reasoning *before* the JSON, so we regex out the action.

**Design lesson:** Define your protocol clearly with types. It prevents "wait, what fields does this have?" confusion later.

### Learnings Manager (`learnings.py`)

A simple text buffer that persists across tasks. The agent writes markdown notes:

```markdown
# My Learnings

## Logistic Maps
They oscillate between two values when r > 3.5

## Tent Maps
Linear—just extrapolate the slope
```

The `delete()` method uses regex to remove sections by header. This lets the agent prune outdated insights.

**Design lesson:** Keep state managers dead simple. This is 40 lines. It could be 200 lines with "features" that would never get used.

### Trace Logger (`trace.py`)

Records everything for your dissertation analysis:
- Every turn (reasoning, action, feedback)
- Every task (family, h_KS, final score, duration)
- Exports to markdown for easy reading

**Design lesson:** Log more than you think you need. You can always ignore logs; you can't recover data you didn't capture.

### Session Runner (`session.py`)

The orchestrator. This is where the main loop lives:

```python
for task in tasks:
    while not done:
        obs = build_observation(task, feedback)
        reasoning, action = agent(obs)

        if action == PREDICT:
            feedback = score_prediction(action.value, task)
        elif action == WRITE:
            learnings.write(action.text)
        elif action == MOVE_ON:
            break

    phi += weight(task.h_ks) * score
```

**Key design decision:** The agent can predict multiple times per task. Each prediction gets feedback. Only the *last* prediction before MOVE_ON counts for scoring. This lets the agent iterate and improve.

**Design lesson:** The orchestrator should be *boring*. It just shuffles data between components. All the interesting logic lives in the agent or the scorer.

### LLM Agent (`metacognitive_agent.py`)

Wraps Gemini API calls. The interesting part is `_format_observation()`:

```python
## Task 1
**System family:** logistic
**Observations:** [0.234, 0.512, 0.891, ...]
**Predict:** x_50

**Last attempt:**
- Your prediction: 0.500
- Actual value: 0.312
- Score: 0.45

---
**Your Learnings:**
# My Learnings
## Logistic Maps
They oscillate...
```

The agent sees its own notes every turn. This is how "learning" persists.

**Design lesson:** The agent is *stateless*. It doesn't track conversation history internally. The SessionRunner manages state by including learnings in every observation. This makes the agent simple and testable.

---

## Bugs We Hit and How We Fixed Them

### Bug 1: 2D Arrays from ChaosBench

**Symptom:** Integration test crashed with "too many indices for array"

**Cause:** ChaosBench returns observations as 2D arrays `(n_obs, dim)` but our formatter assumed 1D.

**Fix:** Added `.flatten()` before formatting:
```python
obs_flat = obs.observations.flatten()
obs_str = ", ".join(f"{x:.3f}" for x in obs_flat[:10])
```

**Lesson:** Integration tests exist to catch exactly these mismatches between "what tests mock" and "what production provides."

### Bug 2: Mock Path for call_llm

**Symptom:** Tests weren't mocking the LLM—they were making real API calls.

**Cause:** Patched `shared.llm_utils.call_llm` but needed to patch where it's *imported*: `chaosbench.agents.metacognitive_agent.call_llm`

**Fix:**
```python
@patch('chaosbench.agents.metacognitive_agent.call_llm')  # Not shared.llm_utils!
```

**Lesson:** When mocking, patch at the *import site*, not the definition site. This is Python mocking 101 but trips everyone up at least once.

---

## Technologies and Why

| Tech | Why |
|------|-----|
| **Dataclasses** | Python's built-in way to define "bags of data" without boilerplate. Cleaner than dicts, lighter than full classes. |
| **LiteLLM** | Unified API for OpenAI, Anthropic, Google, etc. Switch models by changing a string. |
| **pytest** | Industry standard. Fixtures, markers, good error messages. |
| **Markdown for traces** | Human-readable. You'll be copy-pasting traces into your dissertation. |
| **JSON for Φ(t) curves** | Easy to load into pandas/matplotlib for plotting. |

---

## How Good Engineers Think

### 1. Start with the interface, not the implementation

We defined `AgentObservation` and `AgentAction` *before* writing any logic. This forced clarity: "What does the agent see? What can it do?" The implementation followed naturally.

### 2. Make the happy path obvious

Look at `SessionRunner.run()`. The main loop is 30 lines of straightforward code. Error handling, logging, edge cases—they're all there, but they don't obscure the core flow.

### 3. Test at the right level

- **Unit tests** for data types and parsing (fast, isolated)
- **Integration test** for the full stack (catches real bugs)
- **No tests** for the CLI script (it's just glue)

We didn't write tests for things that are obvious or tested by dependencies.

### 4. Resist the urge to generalize

The `LearningsManager` could have:
- Version history
- Search functionality
- Token limit enforcement
- Export to different formats

We built none of that. YAGNI (You Ain't Gonna Need It). If we need it later, we'll add it.

### 5. Log everything, but structure it

The `TraceLogger` captures every turn. But it's structured—`Turn` objects, not raw strings. This means we can analyze traces programmatically later.

---

## What's Next?

You now have:
```bash
python -m chaosbench.run_metacognitive --n-tasks 50 --conditional
```

This runs 50 tasks, saves traces, learnings, and Φ(t) curves.

**For your dissertation:**
1. Run sessions with different conditions (conditional vs. blind)
2. Analyze traces—does the agent develop useful heuristics?
3. Plot Φ(t) curves—does it improve over time?
4. Compare agent learnings to ground truth about each system family

**The big question:** Does the agent's "learning" transfer? If it learns about logistic maps in tasks 1-10, does it perform better on logistic maps in tasks 40-50?

---

## Glossary

| Term | Meaning |
|------|---------|
| **h_KS** | Kolmogorov-Sinai entropy. Measures "how chaotic" a system is. Higher = harder to predict. |
| **Φ(t)** | Cumulative weighted score over time. The "skill curve." |
| **Conditional** | Agent sees the system family (logistic, tent, Lorenz). Blind = doesn't see it. |
| **MOVE_ON** | Agent's way of saying "I've tried enough, let's go to the next task." |

---

## Session Status (2026-01-30)

### What Changed This Session

**Problem Found:** Original design let agent see the answer after each PREDICT, so it just iterated toward revealed answers ("hot/cold" game). Not a real prediction task.

**Fix Applied:** Two-phase design:
1. **Phase 1 (Prediction):** Agent predicts BLIND - no feedback until MOVE_ON
2. **Phase 2 (Reflection):** After MOVE_ON, agent sees result and can WRITE learnings

### Current Issue to Debug

Agent keeps predicting (20 times, hitting turn limit) but **never says MOVE_ON**. Still getting perfect scores somehow - needs investigation.

### Files Modified
- `session.py` - Changed to hide feedback until MOVE_ON
- `metacognitive_system.txt` - Updated prompt explaining two-phase flow
- Added `dotenv` loading to `shared/llm_utils.py`
- Default model changed to `gemini/gemini-2.0-flash` (Gemini 3 has JSON output issues)

### Next Steps
1. Debug why agent never commits (MOVE_ON)
2. Verify scoring is actually blind
3. Consider single-shot prediction (force commitment)
4. Run experiments once working

### Status Doc
See `docs/plans/2026-01-30-metacognitive-agent-status.md` for full details.

---

*Last updated: 2026-01-30 (mid-session, debugging blind prediction)*

---

## ChaosBench v2: The Rebuild (2026-02-06)

### Why v2? The v1 Autopsy

v1 had a metacognitive agent that was supposed to "learn" about chaotic systems over time. Three problems killed it:

1. **Prompt anchoring** — Agent would copy concrete numeric examples from the system prompt and predict those. Garbage in, garbage out.
2. **Scaffolding obscured signal** — The HYPOTHESIZE/FIT/PREDICT tool loop was so complex that it was impossible to tell if the agent was *reasoning* or just *following instructions*.
3. **NLL scoring was gameable** — Agent learned that predicting the mean always gives a decent NLL score. Mean reversion isn't intelligence.

**The v2 philosophy:** Strip everything back. Three simple question types (CLASSIFY, IDENTIFY, PREDICT). No metacognitive scaffolding. Raw reasoning ability, directly measured.

### Architecture: The Grammar of Chaos

Think of it like linguistics. Atoms are words, connectives (post-MVP) are grammar rules:

```
Atom (logistic, tent, damped_linear, rotation)
  ↓ iterate(), derivative(), lyapunov(), regime()
DynamicalSystem (wraps atom + observation model)
  ↓ observe(), future_trajectory()
Problem (system + question type + ground truth)
  ↓ verified by gates + baselines
Bank (collection of validated problems)
```

**Why this separation?** Because each layer has a different rate of change:
- Atoms are mathematical facts (never change)
- Systems add experimental noise (tune for difficulty)
- Problems add questions (swap out for different experiments)
- Banks add validation (ensure quality)

### The Parameter Selection Story

This one's worth remembering. Our first attempt used r=3.831 for the "chaotic" logistic problem. Turns out r=3.831 falls *exactly* in the period-3 window — it's periodic, not chaotic (λ=-0.37). This is the Sharkovskii theorem in action: period-3 implies chaos, but the *window* of period-3 stability is a gap in the chaotic band.

**Lesson:** Never trust round numbers for chaotic systems. We verified r=3.891 numerically (λ≈0.49) and hard-coded it.

Similarly, tent map mu=2.0 causes degeneration (every orbit eventually maps to zero). That was a v1 bug that cost hours of debugging. v2 hard-caps at mu≤1.95.

### The ACF Gate Surprise

We set the autocorrelation gate threshold at 0.05 (expecting chaotic systems to have near-zero ACF at lag 1). The logistic map at r=3.891 has ACF(1)≈-0.53. This is *correct* — chaotic logistic maps below r=4 have strong negative lag-1 autocorrelation. It's only at r=4 (the fully chaotic case) that ACF drops to near-zero.

**Fix:** Raised threshold to 0.95. The gate's real purpose is to catch periodic/fixed-point trajectories being mislabelled as chaotic, not to be a chaos detector. The Lyapunov + PE gates handle the actual chaos verification.

**Lesson for the dissertation:** "Chaotic" doesn't mean "random-looking." Chaos has structure — it's deterministic complexity, not noise. The negative ACF in the logistic map reflects the stretching-and-folding that defines chaos.

### What We Built (Phase 1-2 Summary)

| Module | Files | What It Does |
|--------|-------|-------------|
| `grammar/` | atoms, registry, system, connectives | 4 atoms with analytical properties, factory, observation model |
| `problems/` | verification, factory, bank | 3 verifiers, problem generator, 36-problem mini-bank |
| `scoring/` | difficulty | Composite difficulty: (1+h_ks)(1+depth)(1+10σ) |
| `validation/` | gates, baselines | 7 quality gates + 5 baselines for sanity checking |
| `tests/` | 8 new test files | 107 new tests, all passing |

**Result:** 36 problems generated, 24 validated (67% pass rate). All 4 families × 3 question types represented. Bank frozen to JSON.

### Key Numbers

- 146 total tests pass (107 new + 39 existing)
- 24/36 problems validated
- Difficulty range: 1.10 (damped linear, classify) to 1.80 (tent mu=1.574, classify)
- Zero regressions on existing lyapunov tests

### What's Next

Phase 4-5: Baselines, experiment analysis. The foundation and Gemini agent are ready — now we run experiments and interpret results.

*Updated: 2026-02-06 (Phase 1-2 complete, all tests green)*

---

## Phase 3: Gemini Solver + Quick Runner (2026-02-06)

### What We Built

Five source files and two test files to send problems to Gemini and get scored results:

| File | Purpose |
|------|---------|
| `agents/protocol.py` | `Solution`, `TaskResult` dataclasses + `Agent` protocol |
| `agents/prompts.py` | System prompt (no anchoring!) + per-problem user prompt |
| `agents/parsing.py` | Extract CLASSIFY/IDENTIFY/PREDICT answers from LLM responses |
| `agents/llm_agent.py` | Concrete `LLMAgent` wrapping `shared.llm_utils.call_llm` |
| `experiment/runner.py` | Load bank, run agent, score, print summary |

### The Parsing Problem

LLMs don't give you structured output — they give you essays with an answer buried somewhere. Our parser uses a three-tier strategy:

1. **ANSWER: line** — If the agent follows instructions, great. Extract from there.
2. **Pattern matching** — Look for known labels / bracketed lists / comma-separated numbers.
3. **Fallback** — Grab whatever floats exist in the response.

The key insight: **never crash on bad output**. Return empty string / empty list, score it as 0, log the raw response for debugging. A single API failure shouldn't kill a 27-problem run.

### The "quasiperiodic" Bug

First test run had a subtle bug: "periodic" is a substring of "quasiperiodic". When scanning the ANSWER line for known labels, "periodic" matched before "quasiperiodic" was checked.

**Fix:** Sort labels longest-first before scanning. This is a general lesson: when matching against a set of patterns where one is a substring of another, check the longer one first.

### Design Decisions Worth Remembering

1. **Reused `shared.llm_utils.call_llm` directly** — No new LLM wrapper. It has retry logic, .env loading, supports gemini/ prefix via LiteLLM. Don't reinvent.

2. **Task history is empty in MVP** — The protocol supports passing previous task results to the agent (sequential condition), but MVP uses independent mode (empty list). Sequential experiments are Phase 5.

3. **No concrete numeric examples in system prompt** — v1's fatal flaw was putting example numbers in prompts; the agent would anchor on them and predict those numbers. v2 system prompt says "think step by step" and nothing more specific.

4. **Loading from frozen JSON skips system_metadata/observation_detail** — The runner only needs observations, ground truth, and metadata. We set the heavy objects to `None` and move on. Full Problem reconstruction from JSON would require re-running the atoms, which is wasteful for evaluation.

### Running It

```bash
# Unit tests (no API calls)
pytest chaosbench/tests/test_parsing.py chaosbench/tests/test_runner.py -v

# Live run (costs ~$0.05)
python -m chaosbench.experiment.runner
```

### Numbers

- 44 new tests, all passing
- 92 total tests (44 new + 48 existing), zero regressions
- 27 problems in the bank ready for Gemini

*Updated: 2026-02-06 (Phase 3 complete, ready for live Gemini run)*
