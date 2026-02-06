# CLAUDE.md

## How to Work With Morgan

**Morgan is a researcher, not a coder.** Morgan makes research decisions, predicts agent behavior, interprets results, explains findings for dissertation. Claude writes code, runs experiments, builds infrastructure.

**Rules:**
- Don't just execute — engage. Ask research questions before major work.
- Surface assumptions. Explain WHY you're making architectural choices.
- Before implementing: "What are the inputs/outputs? What could go wrong?"
- After implementing: "Predict what happens. Now what if we change X?"
- Morgan owns the LOGIC, Claude handles the SYNTAX.

---

## Project: ChaosBench v2

Benchmark testing whether LLMs can reason about chaotic dynamical systems. Generates problems from a grammar of dynamical systems, validated through quality gates, difficulty-scored. Three types: CLASSIFY, IDENTIFY, PREDICT.

**Current state:** Phase 3 done (Gemini: 36% CLASSIFY, 55% IDENTIFY, 79% PREDICT). Phase 4 in progress (adversarial arena). Master spec: `docs/Chaos_IMO`.

**Archived benchmarks in `archive/` — do not touch unless Morgan asks.**

## Directory

```
chaosbench/           # ACTIVE — core/, grammar/, problems/, validation/,
                      #   scoring/, agents/, experiment/, data/, tests/, visualization/
shared/               # LiteLLM wrapper, utilities
docs/                 # Chaos_IMO (master spec), plans/
results/              # Phase 3 analysis plots
archive/              # coordination (done), telepathic (paused), temporal (stub)
task_plan.md          # THE to-do list (current phase)
findings.md           # Research results
progress.md           # Session log
```

## Commands

```bash
source venv/bin/activate
pytest chaosbench/tests/ -v          # 213 tests
python --version                      # 3.13.7
```

## Design Decisions

| Decision | Choice |
|----------|--------|
| Atoms | logistic, tent, damped_linear, rotation |
| Types | CLASSIFY, IDENTIFY, PREDICT |
| Validation | Quality gates + baseline checks |
| Scoring | Weighted accuracy by difficulty |
| Agent | LLM via LiteLLM (Gemini) |

## Workflow

**Hooks enforce planning:** PreToolUse shows task_plan.md. PostToolUse reminds to update. Stop checks completion.

**For new features/phases:**
1. **Brainstorm** (`superpowers:brainstorming`) — Socratic design with Morgan → save to `docs/plans/`
2. **Plan** — Overwrite `task_plan.md` with checklist (`- [ ]` / `- [x]`)
3. **Implement** (`superpowers:subagent-driven-development` for multi-task) — check boxes as done, log to findings.md + progress.md, run tests after each task
4. **Verify** (`superpowers:verification-before-completion`) — all tests pass, all boxes checked

**Rules:** 2-Action Rule (save findings every 2 operations). 3-Strike Rule (escalate after 3 failures).

**Session recovery:** Read task_plan.md → findings.md → progress.md → resume from first unchecked task.
