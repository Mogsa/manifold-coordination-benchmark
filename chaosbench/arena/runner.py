"""Arena runner — orchestrates the 4-phase adversarial round loop.

Usage:
    python -m chaosbench.arena.runner --rounds 10
"""

from __future__ import annotations

import argparse
from typing import Optional

from shared.llm_utils import call_llm

from chaosbench.arena.consensus import (
    compare_consensus_to_math,
    compute_discrimination,
    update_reputation,
)
from chaosbench.arena.parsing import parse_proposal, parse_proposals, parse_review, parse_solve_result
from chaosbench.arena.prompts import (
    format_proposer_prompt,
    format_reviewer_prompt,
    format_solver_prompt,
)
from chaosbench.arena.protocol import Reputation, RoundResult
from chaosbench.problems.bank import validate_problem
from chaosbench.problems.factory import QuestionType, create_problem


def _call(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int = 4000,
) -> str:
    """Wrapper around call_llm with system/user message format."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return call_llm(
        model, messages, temperature=temperature, max_tokens=max_tokens
    )


def run_round(
    round_num: int,
    reputations: dict[str, Reputation],
    round_history: list[RoundResult],
    model: str = "gemini/gemini-2.0-flash",
    n_solvers: int = 3,
    n_reviewers: int = 2,
    verbose: bool = True,
) -> RoundResult:
    """Execute one arena round: propose → solve → review → consensus."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")

    # --- Phase 1: PROPOSE (multi-proposal curation) ---
    if verbose:
        print("\n[Phase 1] Proposing problems...")

    system, user = format_proposer_prompt(round_history)
    proposer_response = _call(model, system, user, temperature=0.9)
    proposals = parse_proposals(proposer_response, proposer_id="proposer_0")

    if verbose:
        print(f"  Got {len(proposals)} proposals")

    # Validate each proposal, select the best valid one
    qt_map = {
        "classify": QuestionType.CLASSIFY,
        "identify": QuestionType.IDENTIFY,
        "predict": QuestionType.PREDICT,
    }

    best_proposal = None
    best_problem = None
    best_details = None
    best_difficulty = -1.0

    for i, prop in enumerate(proposals):
        question_type = qt_map.get(prop.question_type, QuestionType.CLASSIFY)
        try:
            prob = create_problem(
                family=prop.family,
                params=prop.params,
                question_type=question_type,
            )
        except Exception as e:
            if verbose:
                print(f"  Proposal {i} ({prop.family}/{prop.question_type}): "
                      f"creation failed — {e}")
            continue

        passed, details = validate_problem(prob)
        if passed:
            if verbose:
                print(f"  Proposal {i} ({prop.family}/{prop.question_type}): "
                      f"VALID (difficulty={prob.difficulty:.2f})")
            if prob.difficulty > best_difficulty:
                best_proposal = prop
                best_problem = prob
                best_details = details
                best_difficulty = prob.difficulty
        else:
            if verbose:
                print(f"  Proposal {i} ({prop.family}/{prop.question_type}): "
                      f"FAILED — {_summarize_gates(details)}")

    # If no valid proposal, use the first one and let it fail gracefully
    if best_proposal is None:
        proposal = proposals[0]
        question_type = qt_map.get(proposal.question_type, QuestionType.CLASSIFY)
        try:
            problem = create_problem(
                family=proposal.family,
                params=proposal.params,
                question_type=question_type,
            )
        except Exception as e:
            if verbose:
                print(f"  No valid proposals. Creation failed: {e}")
            result = RoundResult(
                round_number=round_num,
                proposal=proposal,
                validation_passed=False,
                validation_details={"error": str(e)},
            )
            result.reputation_updates = update_reputation(reputations, result)
            return result

        passed, details = validate_problem(problem)
        if verbose:
            print(f"  No valid proposals. Using first (FAILED: "
                  f"{_summarize_gates(details)})")
        result = RoundResult(
            round_number=round_num,
            proposal=proposal,
            problem=problem,
            validation_passed=False,
            validation_details=details,
        )
        result.reputation_updates = update_reputation(reputations, result)
        return result

    proposal = best_proposal
    problem = best_problem
    details = best_details

    if verbose:
        print(f"  Selected: {proposal.family}({proposal.params}) "
              f"{proposal.question_type} (difficulty={problem.difficulty:.2f})")

    # --- Phase 2: BLIND SOLVE ---
    if verbose:
        print(f"\n[Phase 2] Solving ({n_solvers} solvers)...")

    system, user = format_solver_prompt(problem)
    solves = []
    for i in range(n_solvers):
        solver_id = f"solver_{i}"
        try:
            response = _call(model, system, user, temperature=0.5)
            solve = parse_solve_result(
                response, solver_id,
                proposal.question_type, problem.question_params,
            )
        except Exception as e:
            if verbose:
                print(f"  {solver_id} failed: {e}")
            from chaosbench.arena.protocol import SolveResult
            solve = SolveResult(
                solver_id=solver_id, answer="", explanation=str(e),
                confidence=0.0,
            )
        solves.append(solve)
        if verbose:
            ans_str = str(solve.answer)
            if len(ans_str) > 40:
                ans_str = ans_str[:40] + "..."
            print(f"  {solver_id}: {ans_str} (conf={solve.confidence:.2f})")

    # --- Phase 3: PEER REVIEW ---
    if verbose:
        print(f"\n[Phase 3] Reviewing ({n_reviewers} reviewers)...")

    solver_ids = [s.solver_id for s in solves]
    reviews = []
    for i in range(n_reviewers):
        reviewer_id = f"reviewer_{i}"
        try:
            system_r, user_r = format_reviewer_prompt(proposal, problem, solves)
            response = _call(model, system_r, user_r, temperature=0.3)
            review = parse_review(response, reviewer_id, solver_ids)
        except Exception as e:
            if verbose:
                print(f"  {reviewer_id} failed: {e}")
            from chaosbench.arena.protocol import Review
            review = Review(
                reviewer_id=reviewer_id, question_quality=3,
                question_reasoning=str(e),
                answer_ratings={sid: 3 for sid in solver_ids},
                confidence=2,
            )
        reviews.append(review)
        if verbose:
            print(f"  {reviewer_id}: quality={review.question_quality}, "
                  f"ratings={review.answer_ratings}")

    # --- Phase 4: CONSENSUS ---
    if verbose:
        print("\n[Phase 4] Computing consensus...")

    from chaosbench.arena.consensus import compute_consensus_answer
    consensus = compute_consensus_answer(
        solves, reviews, proposal.question_type, reputations
    )
    gt = _get_math_ground_truth(problem)
    matches = compare_consensus_to_math(consensus, problem)
    disc = compute_discrimination(solves, problem)

    if verbose:
        print(f"  Consensus: {consensus}")
        print(f"  Ground truth: {_format_gt(gt)}")
        print(f"  Match: {matches}")
        print(f"  Discrimination: {disc:.4f}")

    result = RoundResult(
        round_number=round_num,
        proposal=proposal,
        problem=problem,
        validation_passed=True,
        validation_details=details,
        solves=solves,
        reviews=reviews,
        consensus_answer=consensus,
        math_ground_truth=gt,
        consensus_matches_math=matches,
    )
    result.reputation_updates = update_reputation(reputations, result)
    return result


def run_arena(
    n_rounds: int = 10,
    model: str = "gemini/gemini-2.0-flash",
    n_solvers: int = 3,
    n_reviewers: int = 2,
    verbose: bool = True,
) -> list[RoundResult]:
    """Run the full adversarial arena."""
    if verbose:
        print("ChaosBench v2 — Adversarial Arena")
        print(f"Model: {model}")
        print(f"Rounds: {n_rounds}, Solvers: {n_solvers}, Reviewers: {n_reviewers}")

    reputations: dict[str, Reputation] = {}
    history: list[RoundResult] = []

    for i in range(1, n_rounds + 1):
        result = run_round(
            round_num=i,
            reputations=reputations,
            round_history=history,
            model=model,
            n_solvers=n_solvers,
            n_reviewers=n_reviewers,
            verbose=verbose,
        )
        # Update reputations from this round
        reputations = result.reputation_updates
        history.append(result)

        if verbose:
            print_round_summary(result)

    if verbose:
        print_arena_summary(history)

    return history


def print_round_summary(result: RoundResult) -> None:
    """Print a brief round summary line."""
    p = result.proposal
    status = "PASS" if result.validation_passed else "FAIL"
    line = (
        f"\n  Round {result.round_number} [{status}] "
        f"{p.family}({p.question_type})"
    )
    if result.validation_passed and result.consensus_matches_math is not None:
        match = "correct" if result.consensus_matches_math else "wrong"
        line += f" → consensus {match}"
    print(line)


def print_arena_summary(results: list[RoundResult]) -> None:
    """Print aggregate analysis after all rounds."""
    print(f"\n{'='*60}")
    print("ARENA SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for r in results if r.validation_passed)
    print(f"Validation: {passed}/{total} proposals passed ({100*passed/total:.0f}%)")

    # Question type distribution
    type_counts: dict[str, int] = {}
    for r in results:
        qt = r.proposal.question_type
        type_counts[qt] = type_counts.get(qt, 0) + 1
    print(f"Types proposed: {type_counts}")

    # Consensus accuracy on validated problems
    validated = [r for r in results if r.validation_passed]
    if validated:
        correct = sum(1 for r in validated if r.consensus_matches_math)
        print(f"Consensus accuracy: {correct}/{len(validated)} "
              f"({100*correct/len(validated):.0f}%)")

    # Reputation summary
    if results and results[-1].reputation_updates:
        print("\nFinal reputations:")
        for aid, rep in sorted(results[-1].reputation_updates.items()):
            print(f"  {aid}: propose={rep.propose_score:.2f} "
                  f"solve={rep.solve_score:.2f} "
                  f"review={rep.review_score:.2f} "
                  f"(rounds={rep.rounds_participated})")


def _summarize_gates(details: dict) -> str:
    """Brief summary of which gates failed."""
    gates = details.get("stage1_gates", [])
    failed = [name for name, passed, _ in gates if not passed]
    return ", ".join(failed) if failed else "unknown"


def _get_math_ground_truth(problem) -> object:
    """Extract displayable ground truth."""
    gt = problem.ground_truth
    qt = problem.question_type.value
    if qt == "classify":
        return gt.regime
    if qt == "identify":
        return gt.family
    if qt == "predict":
        return gt.future_values
    return None


def _format_gt(gt: object) -> str:
    """Format ground truth for display."""
    if gt is None:
        return "None"
    s = str(gt)
    return s[:60] + "..." if len(s) > 60 else s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChaosBench Adversarial Arena")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--model", default="gemini/gemini-2.0-flash")
    parser.add_argument("--solvers", type=int, default=3)
    parser.add_argument("--reviewers", type=int, default=2)
    args = parser.parse_args()
    run_arena(
        n_rounds=args.rounds,
        model=args.model,
        n_solvers=args.solvers,
        n_reviewers=args.reviewers,
    )
