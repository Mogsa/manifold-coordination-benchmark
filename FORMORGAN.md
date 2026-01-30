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
