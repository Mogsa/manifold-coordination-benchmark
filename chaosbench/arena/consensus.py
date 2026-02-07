"""Consensus aggregation and reputation scoring for the arena.

Aggregates reviewer ratings into consensus answers, computes problem
discrimination, and updates agent reputations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from chaosbench.arena.protocol import (
    Proposal,
    Reputation,
    Review,
    RoundResult,
    SolveResult,
)
from chaosbench.problems.factory import Problem
from chaosbench.problems.verification import (
    verify_classify,
    verify_identify,
    verify_predict,
)


def _get_ground_truth(problem: Problem) -> Any:
    """Extract the ground truth answer from a problem."""
    qt = problem.question_type.value
    gt = problem.ground_truth
    if qt == "classify":
        return gt.regime
    if qt == "identify":
        return gt.family
    if qt == "predict":
        return gt.future_values
    return None


def _score_answer(answer: Any, problem: Problem) -> float:
    """Score a single answer against ground truth."""
    qt = problem.question_type.value
    gt = problem.ground_truth
    if qt == "classify":
        return verify_classify(answer, gt.regime)
    if qt == "identify":
        return verify_identify(answer, gt.family)
    if qt == "predict":
        preds = answer if isinstance(answer, list) else []
        result = verify_predict(
            np.array(preds) if preds else np.array([]),
            gt.future_values,
            problem.observations,
        )
        return result["raw_score"]
    return 0.0


# Asymmetric recommendation scale (1-6 Likert → weight)
# Negative ratings penalize more than positive ratings reward
REC_SCALE = {1: -1.0, 2: -0.5, 3: -0.15, 4: 0.3, 5: 0.7, 6: 1.0}

# Confidence multiplier (1-5 → multiplier)
CONF_SCALE = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.2, 5: 1.4}


def compute_weighted_rating(
    rating: int, confidence: int, reputation: float = 1.0
) -> float:
    """Compute asymmetric weighted score for a single rating."""
    rec = REC_SCALE.get(rating, 0.0)
    conf = CONF_SCALE.get(confidence, 1.0)
    return rec * conf * reputation


def compute_consensus_answer(
    solves: list[SolveResult],
    reviews: list[Review],
    question_type: str,
    reputations: Optional[dict[str, "Reputation"]] = None,
) -> Any:
    """Aggregate reviews into a consensus answer.

    Uses SPIRE-style asymmetric Likert weights and reputation-weighted reviews.
    Reviewer-provided correct_answer values are fused as additional votes.
    Ties broken by solver confidence.
    """
    if not solves or not reviews:
        return None

    if reputations is None:
        reputations = {}

    # Compute weighted score per solver using asymmetric Likert + reputation
    solver_scores: dict[str, float] = {}
    for review in reviews:
        rep = reputations.get(review.reviewer_id)
        rep_weight = rep.review_score if rep and rep.review_score > 0 else 1.0
        for sid, rating in review.answer_ratings.items():
            weighted = compute_weighted_rating(rating, review.confidence, rep_weight)
            solver_scores[sid] = solver_scores.get(sid, 0.0) + weighted

    # Fuse reviewer correct_answer values as additional votes
    # Each reviewer's answer is matched to the solver with the same answer,
    # adding a bonus weighted by reviewer confidence × reputation
    if question_type in ("classify", "identify"):
        for review in reviews:
            if review.correct_answer is None:
                continue
            rep = reputations.get(review.reviewer_id)
            rep_weight = rep.review_score if rep and rep.review_score > 0 else 1.0
            conf = CONF_SCALE.get(review.confidence, 1.0)
            bonus = conf * rep_weight
            reviewer_ans = str(review.correct_answer).strip().lower()
            for solve in solves:
                if str(solve.answer).strip().lower() == reviewer_ans:
                    solver_scores[solve.solver_id] = (
                        solver_scores.get(solve.solver_id, 0.0) + bonus
                    )

    # Find the solve with the best weighted score (ties: highest confidence)
    best_solve = max(
        solves,
        key=lambda s: (solver_scores.get(s.solver_id, 0.0), s.confidence),
    )
    return best_solve.answer


def compute_discrimination(
    solves: list[SolveResult], problem: Problem
) -> float:
    """Compute how well a problem discriminates between solvers.

    Uses variance of solver scores directly. Uniformly easy/hard rounds
    naturally yield variance 0.
    """
    if len(solves) < 2:
        return 0.0

    scores = [_score_answer(s.answer, problem) for s in solves]
    return float(np.var(scores))


def compare_consensus_to_math(
    consensus_answer: Any, problem: Problem
) -> bool:
    """Check if consensus answer matches mathematical ground truth."""
    if consensus_answer is None:
        return False
    return _score_answer(consensus_answer, problem) > 0


def _running_average(old: float, new: float, n: int) -> float:
    """Incremental running average: avg_{n} = (avg_{n-1} * (n-1) + new) / n."""
    if n <= 1:
        return new
    return (old * (n - 1) + new) / n


def update_reputation(
    current: dict[str, Reputation],
    round_result: RoundResult,
) -> dict[str, Reputation]:
    """Update reputations based on a completed round.

    - Propose score: validation pass (0/1) averaged over rounds.
    - Solve score: raw accuracy on this round's problem.
    - Review score: correlation between reviewer ratings and math ground truth.
    """
    updated = {k: Reputation(**v.__dict__) for k, v in current.items()}

    proposal = round_result.proposal
    problem = round_result.problem

    # --- Proposer reputation ---
    pid = proposal.proposer_id
    if pid not in updated:
        updated[pid] = Reputation(agent_id=pid)
    rep = updated[pid]
    rep.rounds_participated += 1
    propose_val = 1.0 if round_result.validation_passed else 0.0
    if round_result.validation_passed and problem:
        disc = compute_discrimination(round_result.solves, problem)
        propose_val = 0.5 + 0.5 * min(disc * 10, 1.0)  # Bonus for discrimination
    rep.propose_score = _running_average(
        rep.propose_score, propose_val, rep.rounds_participated
    )

    if not round_result.validation_passed or problem is None:
        return updated

    # --- Solver reputations ---
    for solve in round_result.solves:
        sid = solve.solver_id
        if sid not in updated:
            updated[sid] = Reputation(agent_id=sid)
        rep = updated[sid]
        rep.rounds_participated += 1
        score = _score_answer(solve.answer, problem)
        rep.solve_score = _running_average(
            rep.solve_score, score, rep.rounds_participated
        )

    # --- Reviewer reputations ---
    # Score reviewers by how well their ratings correlate with math truth
    solver_truth = {}
    for solve in round_result.solves:
        solver_truth[solve.solver_id] = _score_answer(solve.answer, problem)

    for review in round_result.reviews:
        rid = review.reviewer_id
        if rid not in updated:
            updated[rid] = Reputation(agent_id=rid)
        rep = updated[rid]
        rep.rounds_participated += 1

        # Compute agreement: do high ratings go to correct solvers?
        if review.answer_ratings and solver_truth:
            agreements = []
            for sid, rating in review.answer_ratings.items():
                truth = solver_truth.get(sid, 0.0)
                # Rating > 3 should predict correct (truth > 0), and vice versa
                reviewer_positive = rating > 3
                actually_correct = truth > 0
                agreements.append(1.0 if reviewer_positive == actually_correct else 0.0)
            review_score = sum(agreements) / len(agreements) if agreements else 0.5
        else:
            review_score = 0.5

        rep.review_score = _running_average(
            rep.review_score, review_score, rep.rounds_participated
        )

    return updated
