# Findings: ChaosBench v4 Design

## Design Decisions Made

### Passive First (v1.0)
- Start with 50 observations given upfront
- Probing deferred to v1.1
- Rationale: Test base case (can LLM reason about chaos?) before adding complexity

### Simplified Actions
- PREDICT, WRITE, DELETE, MOVE_ON (no EDIT)
- JSON format for reliable parsing
- Agent writes reasoning before JSON action

### Minimal Feedback
- Agent sees only: prediction, actual value, score
- Must infer *why* it was wrong — that inference is part of what we measure

### Score Counting: Last
- Last prediction before MOVE_ON is banked
- Matches scientific process — final understanding matters

---

## Open Questions

### Transfer Detection
- How do we measure if learnings actually helped?
- Possible: ablation (run with vs without learnings)
- Possible: correlate learnings content with score improvement

### Task Ordering
- Currently stratified by h_KS
- Should we group by family to make transfer easier to observe?
- Or random to test generalization?

---

## References

- Design doc: `docs/plans/2026-01-30-chaosbench-metacognitive-agent-design.md`
- ChaosBench spec: `chaosbench/core/ChaosSpecification.md`
- Existing implementation: `chaosbench/core/Chaosbench_v3.py`
