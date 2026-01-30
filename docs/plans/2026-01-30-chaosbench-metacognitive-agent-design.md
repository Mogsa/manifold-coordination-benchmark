# ChaosBench v4: Metacognitive Agent Protocol

**Date**: 2026-01-30
**Status**: DESIGN IN PROGRESS
**Goal**: Design an LLM agent protocol that tests scientific reasoning via explicit metacognition on chaotic systems

---

## Research Hypothesis

> Reasoning systems that build transferable internal models of dynamical structure will exhibit **superlinear Φ(t)** growth across related tasks, while systems relying on task-specific heuristics will show **linear or sublinear** growth.

**Core insight**: The reasoning trace itself becomes dissertation data, not just final accuracy.

---

## Design Decisions (Agreed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Retry policy** | Agent decides, time cost | Tests metacognition; self-regulates via Φ(t) |
| **Feedback** | Score + true value | Enough signal to learn; agent must infer "why" |
| **Task transition** | Agent calls MOVE_ON | Transition decision is part of agency |
| **Probing** | Passive for v1.0 (50 obs given); Active probing in v1.1 | Test base case first, add probing once core loop works |
| **Learnings format** | Freeform markdown, agent curates | See natural organization; curation is agency |
| **Single vs multi** | Single for v1 | But design learnings interface for multi-agent later |

---

## Design Decisions (Finalized)

### Score counting: LAST
The last prediction before MOVE_ON is banked. Matches scientific process — your final understanding matters.

### Action format: JSON
Reliable parsing, unambiguous. Agent reasoning captured separately.

### Feedback: MINIMAL
Agent sees only: prediction, actual value, score. Must infer *why* it was wrong — that inference is part of what we measure.

---

## Protocol Specification (Draft)

### Session Structure

```
Session starts with:
  - Empty LEARNINGS.md
  - Task queue (stratified by h_KS)
  - Wall-clock timer starts

For each task:
  Agent sees:
    - Observations (x_0, ..., x_n)
    - Prediction horizon
    - Current LEARNINGS.md
    - (if conditional) System family hint

  Agent actions:
    - PREDICT(value) → gets feedback
    - WRITE(text) → append to learnings
    - EDIT(old, new) → modify learnings
    - DELETE(section) → remove from learnings
    - MOVE_ON → bank score, next task

  Loop until MOVE_ON or global timeout

Session ends when:
  - All tasks completed, OR
  - Global time limit reached
```

### Scoring

- Each PREDICT costs wall-clock time (API call + reasoning)
- When MOVE_ON called, **last** prediction score is banked
- Φ(t) = Σ w(h_KS) × accuracy over wall time
- Frivolous retries hurt Φ(t) (time passes, score doesn't improve)

### Agent Interface (v1.0)

```python
@dataclass
class AgentObservation:
    """What the agent sees each turn."""
    task_id: int
    observations: np.ndarray      # Shape: (50, dim) — all given upfront
    obs_times: np.ndarray         # When each observation taken
    prediction_horizon: int       # How far ahead to predict
    family: str | None            # System family (if conditional)
    learnings: str                # Current LEARNINGS.md content
    last_feedback: Feedback | None  # Feedback from last PREDICT (if any)

@dataclass
class Feedback:
    """What agent sees after PREDICT."""
    prediction: float
    actual: float
    score: float

@dataclass
class AgentAction:
    """What agent can do (v1.0)."""
    action: Literal["PREDICT", "WRITE", "DELETE", "MOVE_ON"]
    value: float | None = None     # For PREDICT
    text: str | None = None        # For WRITE
    section: str | None = None     # For DELETE

# v1.1 adds: PROBE action
# {"action": "PROBE", "n": 10}  — reveal n more observations
```

---

## System Prompt (v1.0)

```
You are a scientist studying unknown dynamical systems. Your goal is to
predict future states from past observations.

## Each Task

You observe a time series and must predict the next value:
- Observations: x_0, x_1, ..., x_49 (50 values)
- Your job: predict x_50

After each PREDICT, you see:
- Actual value
- Your score (0 = far off, 1 = perfect)

You may predict multiple times per task before moving on.

## Actions

First write your reasoning, then output ONE action as JSON.

PREDICT — Make or revise prediction
{"action": "PREDICT", "value": 0.42}

WRITE — Add to your learnings (persists across tasks)
{"action": "WRITE", "text": "Your note here"}

DELETE — Remove from learnings by section header
{"action": "DELETE", "section": "## Section Header"}

MOVE_ON — Accept current score, proceed to next task
{"action": "MOVE_ON"}

## Your Learnings

You have a persistent notepad. It appears below each task. Use it to
record patterns, mistakes, and insights that may help future tasks.

## Constraints

- Session has a time limit
- Some systems may be fundamentally hard to predict
- Knowing when to move on is valuable

Begin.
```

---

## Trace Format (For Dissertation Analysis)

```markdown
## Task 7: logistic (h_KS=0.67)

### Turn 1
**Observation**: 50 samples, horizon=2

**Agent reasoning**: (from LLM output)
> Looking at the density distribution, this appears bimodal
> with clusters around 0.15 and 0.85. Classic logistic map
> signature for r in [3.5, 3.9] range...

**Action**: PREDICT(0.82)

**Feedback**:
- Prediction: 0.82
- Actual: 0.15
- Score: 0.23

### Turn 2
**Agent reasoning**:
> I was wrong — predicted upper band continuation but it
> jumped. This suggests r > 3.85 where the map is fully
> chaotic. Should predict based on invariant density, not
> trajectory extrapolation.

**Action**: WRITE("For logistic r>3.85: use invariant density, not trajectory continuation")

### Turn 3
**Action**: PREDICT(0.41)

**Feedback**:
- Prediction: 0.41
- Actual: 0.38
- Score: 0.71

### Turn 4
**Action**: MOVE_ON

**Task result**: Score 0.71, 4 turns, 12.3 seconds
```

---

## Φ(t) Calculation

**Score banked on MOVE_ON** — only the last PREDICT score counts.

```
Φ(t) = Σ w(h_KS) × accuracy
```

where accuracy = exp(-NLL) and w(h_KS) is difficulty weighting (default: linear).

**Time is wall-clock** — every API call costs real time. Retries hurt Φ(t) if score doesn't improve.

**What the curve reveals:**

| Shape | Meaning |
|-------|---------|
| Steep early, then steady | Fast on easy, slower on hard |
| Shallow early, then steep | Learning overhead, then acceleration (**transfer!**) |
| Linear | No learning, constant per-task effort |
| Plateaus | Getting stuck on hard tasks |

---

## Implementation Phases

### Phase 1: Core Protocol
- [ ] AgentObservation / AgentAction dataclasses
- [ ] Session runner (task loop with PREDICT/WRITE/DELETE/MOVE_ON)
- [ ] LEARNINGS.md read/write/delete mechanics
- [ ] JSON action parsing from LLM output
- [ ] Trace logging (every turn, every task)

### Phase 2: LLM Integration
- [ ] Gemini API wrapper (already have key set up)
- [ ] System prompt injection
- [ ] Message formatting (observations + learnings + feedback)
- [ ] Response parsing (reasoning + JSON action)

### Phase 3: Evaluation
- [ ] Φ(t) curve tracking
- [ ] Per-task metrics (attempts, time, final score)
- [ ] Session summary (total Φ, tasks completed, learnings size)

### Phase 4: Analysis (Post-Run)
- [ ] Trace viewer / pretty-printer
- [ ] Learnings evolution over session
- [ ] Transfer detection (did learnings help?)

---

## Version Scope

### v1.0 (Current Target)
- Single agent
- Passive observations (50 given upfront)
- Freeform learnings (WRITE/EDIT/DELETE)
- JSON action format
- Gemini API (already set up)

**Actions:** PREDICT, WRITE, EDIT, DELETE, MOVE_ON

### v1.1 (After base case validated)
- Active probing: 10 initial, PROBE(n) for more up to 50
- Tests: "Does agent know how much data it needs?"

**Actions:** adds PROBE

### v2+ (Future)
- Multi-agent with shared learnings
- Observation cost budget
- Structured learnings format

---

## Next Steps

1. Resolve open questions (score counting, action format, feedback detail)
2. Write full protocol specification
3. Implement agent interface
4. Create system prompt for LLM
5. Run initial tests with Gemini
