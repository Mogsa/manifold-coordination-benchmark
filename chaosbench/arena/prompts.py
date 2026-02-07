"""Prompt formatting for the 3 arena roles: proposer, solver, reviewer.

Each role gets a tailored system prompt. Solver reuses existing prompts
from agents/prompts.py.
"""

from __future__ import annotations

from chaosbench.agents.prompts import format_problem_prompt, format_system_prompt
from chaosbench.arena.protocol import Proposal, RoundResult, SolveResult
from chaosbench.grammar.registry import ATOM_REGISTRY
from chaosbench.problems.factory import Problem


def _grammar_info() -> str:
    """Summarize available families, params, and regimes for the proposer."""
    lines = ["Available dynamical system families:\n"]
    for name, spec in ATOM_REGISTRY.items():
        ranges = ", ".join(
            f"{p}: [{lo}, {hi}]" for p, (lo, hi) in spec.param_ranges.items()
        )
        lines.append(f"- {name}: parameters {ranges}")
    lines.append("\nQuestion types: classify, identify, predict")
    lines.append("Regimes: chaotic, periodic, quasiperiodic, fixed_point")
    return "\n".join(lines)


def _round_summary(result: RoundResult) -> str:
    """One-line summary of a past round for the proposer."""
    p = result.proposal
    status = "PASSED" if result.validation_passed else "FAILED"
    parts = [
        f"Round {result.round_number}: {p.family}({p.params}) {p.question_type}"
        f" → {status}"
    ]
    if result.validation_passed and result.solves:
        n_correct = sum(
            1 for s in result.solves
            if s.answer == result.math_ground_truth
        )
        parts.append(f", {n_correct}/{len(result.solves)} solved correctly")
    return "".join(parts)


def format_proposer_prompt(
    round_history: list[RoundResult],
) -> tuple[str, str]:
    """Build system + user prompts for the proposer role.

    Returns (system_prompt, user_prompt).
    """
    system = (
        "You are a creative mathematician designing benchmark problems for "
        "dynamical systems.\n"
        "Your goal: propose problems that are VALID (pass quality gates) and "
        "DISCRIMINATING (separate strong solvers from weak ones).\n"
        "Choose parameters that produce interesting dynamics — avoid trivial "
        "or degenerate cases.\n"
        "Propose 3 different problems. Vary the families and question types — "
        "don't repeat the same family or type in all three.\n"
        "Respond with a JSON array of 3 objects (no markdown fences)."
    )

    user_parts = [_grammar_info()]

    user_parts.append(
        "\nOutput format (JSON array of 3 objects):\n"
        '[{"family": "logistic", "params": {"r": 3.85}, '
        '"question_type": "classify", '
        '"reasoning": "why this is hard", '
        '"mock_answer": "chaotic", "mock_confidence": 0.9}, '
        '{"family": "tent", "params": {"mu": 1.7}, '
        '"question_type": "identify", '
        '"reasoning": "tent is tricky to identify", '
        '"mock_answer": "tent", "mock_confidence": 0.8}, '
        '{"family": "rotation", "params": {"theta": 0.41}, '
        '"question_type": "predict", '
        '"reasoning": "irrational rotation is hard to predict", '
        '"mock_answer": "0.3,0.7,0.1", "mock_confidence": 0.6}]'
    )

    if round_history:
        user_parts.append("\nPast rounds:")
        for r in round_history[-5:]:  # Last 5 rounds
            user_parts.append(f"  {_round_summary(r)}")
        user_parts.append(
            "\nLearn from past rounds: avoid families/params that failed "
            "validation, and try to create more discriminating problems."
        )

    return system, "\n".join(user_parts)


def format_solver_prompt(problem: Problem) -> tuple[str, str]:
    """Build system + user prompts for the solver role.

    Reuses existing agent prompts, adds explanation instruction.
    """
    system = format_system_prompt()
    user = format_problem_prompt(problem)
    user += (
        "\n\nAfter your reasoning, provide:\n"
        "ANSWER: <your answer>\n"
        "EXPLANATION: <brief explanation, under 500 characters>\n"
        "confidence: <0-1>"
    )
    return system, user


def format_reviewer_prompt(
    proposal: Proposal,
    problem: Problem,
    solves: list[SolveResult],
) -> tuple[str, str]:
    """Build system + user prompts for the reviewer role.

    Reviewer sees metadata + solver answers, NOT the raw time series.
    """
    system = (
        "You are a calibrated reviewer for a scientific benchmark on "
        "dynamical systems.\n"
        "Rate the question quality and each solver's answer using Likert "
        "scales.\n"
        "Be honest and calibrated — avoid defaulting to middle ratings.\n"
        "Respond with a single JSON object (no markdown fences)."
    )

    user_parts = [
        "Problem metadata:",
        f"  Family (claimed): {proposal.family}",
        f"  Parameters: {proposal.params}",
        f"  Question type: {proposal.question_type}",
        f"  Proposer reasoning: {proposal.reasoning}",
        "",
        "Solver answers:",
    ]

    for s in solves:
        user_parts.append(
            f"  {s.solver_id}: answer={s.answer}, "
            f"explanation={s.explanation[:200]}, confidence={s.confidence}"
        )

    user_parts.extend([
        "",
        "Rating scales:",
        "  question_quality (1-6): 1=trivial/invalid, 3=average, "
        "6=excellent discriminator",
        "  answer_ratings (1-6 per solver): 1=clearly wrong, 3=plausible, "
        "6=certainly correct",
        "  confidence (1-5): your confidence in your own ratings",
        "",
        "Based on the problem metadata and your analysis, also provide your "
        "own answer to the question (correct_answer). For classify: a regime "
        "label. For identify: a family name. For predict: a list of floats.",
        "",
        "Output format (JSON):",
        '{"question_quality": 4, "question_reasoning": "why", '
        '"answer_ratings": {"solver_0": 5, "solver_1": 2}, '
        '"confidence": 3, "correct_answer": "chaotic"}',
    ])

    return system, "\n".join(user_parts)
