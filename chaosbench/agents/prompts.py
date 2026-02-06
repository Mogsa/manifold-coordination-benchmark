"""Prompt formatting for ChaosBench v2 agents.

Follows PRD §11 structure: system prompt + per-problem user prompt.
MVP adaptation: only lists families/regimes present in the mini-bank.
"""

from __future__ import annotations

from chaosbench.agents.protocol import TaskResult
from chaosbench.problems.factory import Problem

# MVP families and regimes (only those in the mini-bank)
MVP_FAMILIES = ["logistic", "tent", "damped_linear", "rotation"]
MVP_REGIMES = ["chaotic", "periodic", "quasiperiodic", "fixed_point"]


def format_system_prompt() -> str:
    """Static system prompt — no concrete numeric examples to avoid anchoring."""
    return (
        "You are a scientist analysing time-series data from unknown dynamical systems.\n"
        "Each task gives you a sequence of noisy observations and asks a specific question.\n"
        "Think step by step. Consider the data's structure, variance, and long-term behaviour.\n"
        "After your reasoning, provide your answer in the exact format requested.\n"
        "Also state your confidence as a number between 0 and 1, like: confidence: 0.8"
    )


def format_problem_prompt(
    problem: Problem, task_history: list[TaskResult] | None = None
) -> str:
    """Build the user message for a single problem."""
    parts: list[str] = []

    # Data
    obs = problem.observations
    obs_str = ", ".join(f"{v:.6f}" for v in obs)
    parts.append(f"DATA ({len(obs)} observations):\n{obs_str}")

    # Metadata shown to agent
    meta = problem.metadata
    domain = meta.get("domain", (0, 1))
    obs_mode = meta.get("observation_mode")
    noise_std = meta.get("noise_std", 0)
    if obs_mode == "noisy" or (obs_mode is None and noise_std and noise_std > 0):
        obs_mode_text = "Observations include additive Gaussian noise."
    elif obs_mode == "clean" or (obs_mode is None):
        obs_mode_text = "Observations are noise-free."
    else:
        obs_mode_text = "Observation noise mode is unspecified."
    parts.append(
        f"\nDomain: [{domain[0]}, {domain[1]}]"
        f"\n{obs_mode_text}"
        f"\nNoise level (std): {meta.get('noise_std', 'unknown')}"
        f"\nNumber of points: {meta.get('n_points', len(obs))}"
    )

    # Question
    q_instruction = format_question_instruction(
        problem.question_type.value, problem.question_params
    )
    parts.append(f"\nQUESTION:\n{q_instruction}")

    # Task history (capped at 3000 chars)
    if task_history:
        history_str = _format_history(task_history)
        if history_str:
            parts.append(f"\nPREVIOUS TASKS (for context):\n{history_str}")

    return "\n".join(parts)


def format_question_instruction(question_type: str, question_params: dict) -> str:
    """Per-type instruction block."""
    if question_type == "classify":
        labels = ", ".join(MVP_REGIMES)
        return (
            "Classify the dynamical regime of this system.\n"
            f"Valid answers: {labels}\n"
            "Reply with exactly one label on a line starting with ANSWER:"
        )

    if question_type == "identify":
        families = ", ".join(MVP_FAMILIES)
        return (
            "Identify which family of dynamical system generated this data.\n"
            f"Valid answers: {families}\n"
            "Reply with exactly one family name on a line starting with ANSWER:"
        )

    if question_type == "predict":
        K = question_params.get("horizon", 10)
        return (
            f"Predict the next {K} values of this time series.\n"
            f"Reply with exactly {K} comma-separated numbers on a line starting with ANSWER:"
        )

    return f"Unknown question type: {question_type}"


def _format_history(task_history: list[TaskResult], max_chars: int = 3000) -> str:
    """Format task history, capped at max_chars."""
    lines: list[str] = []
    total = 0
    for tr in task_history:
        preview = ", ".join(f"{v:.4f}" for v in tr.observations_preview[:20])
        line = (
            f"- [{tr.question_type}] score={tr.raw_score:.2f} "
            f"answer={tr.agent_answer} data=[{preview}...]"
        )
        if total + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)
