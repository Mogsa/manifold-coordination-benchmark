"""Quick evaluation runner for ChaosBench v2.

Loads the frozen mini-bank, sends each problem to an agent, scores results.

Usage:
    python -m chaosbench.experiment.runner
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from chaosbench.agents.protocol import Agent, Solution, TaskResult
from chaosbench.problems.factory import GroundTruth, Problem, QuestionType
from chaosbench.problems.verification import verify_classify, verify_identify, verify_predict
from chaosbench.scoring.difficulty import weighted_score


BANK_PATH = Path(__file__).parent.parent / "data" / "mini_bank.json"


def load_problems(path: str | Path = BANK_PATH) -> list[Problem]:
    """Load problems from a frozen JSON bank into Problem objects."""
    with open(path) as f:
        bank = json.load(f)

    problems = []
    for entry in bank["problems"]:
        gt = GroundTruth(
            regime=entry["ground_truth"].get("regime"),
            family=entry["ground_truth"].get("family"),
            future_values=(
                np.array(entry["ground_truth"]["future_values"])
                if entry["ground_truth"].get("future_values") is not None
                else None
            ),
            params=entry["ground_truth"].get("params"),
        )

        # Build a lightweight Problem (no system_metadata/observation_detail needed)
        p = Problem(
            problem_id=entry["problem_id"],
            question_type=QuestionType(entry["question_type"]),
            observations=np.array(entry["observations"]),
            question_params=entry.get("question_params", {}),
            metadata=entry["metadata"],
            ground_truth=gt,
            system_metadata=None,  # type: ignore[arg-type]
            observation_detail=None,  # type: ignore[arg-type]
            difficulty=entry["difficulty"],
            stage1_passed=entry.get("stage1_passed"),
            stage2_results=entry.get("stage2_results"),
        )
        problems.append(p)
    return problems


def score_solution(solution: Solution, problem: Problem) -> dict:
    """Score a solution using the appropriate verifier."""
    qt = problem.question_type.value
    gt = problem.ground_truth

    if qt == "classify":
        raw = verify_classify(solution.answer, gt.regime)
        correct = gt.regime
    elif qt == "identify":
        raw = verify_identify(solution.answer, gt.family)
        correct = gt.family
    elif qt == "predict":
        preds = solution.answer if isinstance(solution.answer, list) else []
        result = verify_predict(
            np.array(preds) if preds else np.array([]),
            gt.future_values,
            problem.observations,
        )
        raw = result["raw_score"]
        correct = f"[{len(gt.future_values)} values]"
    else:
        raw = 0.0
        correct = None

    return {
        "raw_score": raw,
        "weighted_score": weighted_score(raw, problem.difficulty),
        "correct_answer": correct,
    }


def run_bank(
    agent: Agent,
    problems: list[Problem],
    verbose: bool = True,
) -> list[dict]:
    """Run agent on all problems, score each, return results."""
    results = []
    task_history: list[TaskResult] = []  # Empty for independent condition (MVP)

    for i, problem in enumerate(problems):
        qt = problem.question_type.value
        pid = problem.problem_id[:8]

        if verbose:
            print(f"[{i+1}/{len(problems)}] {pid} ({qt})...", end=" ", flush=True)

        try:
            solution = agent.solve(problem, task_history)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results.append({
                "problem_id": problem.problem_id,
                "question_type": qt,
                "agent_answer": None,
                "correct_answer": None,
                "raw_score": 0.0,
                "weighted_score": 0.0,
                "confidence": 0.0,
                "reasoning": str(e),
                "error": True,
            })
            continue

        scores = score_solution(solution, problem)

        result = {
            "problem_id": problem.problem_id,
            "question_type": qt,
            "agent_answer": solution.answer,
            "correct_answer": scores["correct_answer"],
            "raw_score": scores["raw_score"],
            "weighted_score": scores["weighted_score"],
            "confidence": solution.confidence,
            "reasoning": solution.reasoning_trace[:500],
            "error": False,
        }
        results.append(result)

        if verbose:
            mark = "✓" if scores["raw_score"] > 0 else "✗"
            ans_str = str(solution.answer)
            if len(ans_str) > 40:
                ans_str = ans_str[:40] + "..."
            print(f"{mark} score={scores['raw_score']:.2f} answer={ans_str}")

    if verbose:
        print("\n" + "=" * 60)
        print_summary(results)

    return results


def print_summary(results: list[dict]) -> None:
    """Print aggregate scores by question type and family."""
    by_type: dict[str, list[float]] = defaultdict(list)
    errors = sum(1 for r in results if r.get("error"))

    for r in results:
        if not r.get("error"):
            by_type[r["question_type"]].append(r["raw_score"])

    print("SUMMARY")
    print("-" * 40)
    for qt in ["classify", "identify", "predict"]:
        scores = by_type.get(qt, [])
        if scores:
            mean = sum(scores) / len(scores)
            correct = sum(1 for s in scores if s > 0)
            print(f"  {qt:12s}: {mean:.3f} avg ({correct}/{len(scores)} correct)")
        else:
            print(f"  {qt:12s}: no problems")

    all_scores = [r["raw_score"] for r in results if not r.get("error")]
    if all_scores:
        print(f"  {'overall':12s}: {sum(all_scores)/len(all_scores):.3f} avg")
    if errors:
        print(f"  errors: {errors}/{len(results)}")


if __name__ == "__main__":
    from chaosbench.agents.llm_agent import LLMAgent

    print("ChaosBench v2 — Quick Evaluation Runner")
    print("=" * 60)

    problems = load_problems()
    print(f"Loaded {len(problems)} problems from {BANK_PATH.name}")

    agent = LLMAgent()
    print(f"Agent: {agent.agent_id}")
    print("=" * 60 + "\n")

    results = run_bank(agent, problems, verbose=True)
