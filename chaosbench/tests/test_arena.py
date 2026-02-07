"""Tests for ChaosBench v2 Phase 4 — Adversarial Arena.

Tests all arena components with mocked LLM calls. Follows existing
class-based pattern, no shared fixtures beyond conftest.py.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from chaosbench.arena.protocol import (
    Proposal,
    Reputation,
    Review,
    RoundResult,
    SolveResult,
)


# ───────────────────────────────────────────────────────────
# Task 1: Protocol Dataclasses
# ───────────────────────────────────────────────────────────

class TestArenaProtocol:
    """Dataclass construction and field access."""

    def test_proposal_construction(self):
        p = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.8},
            question_type="classify", reasoning="test", mock_answer="chaotic",
            mock_confidence=0.9,
        )
        assert p.family == "logistic"
        assert p.params == {"r": 3.8}
        assert p.mock_confidence == 0.9

    def test_solve_result_construction(self):
        s = SolveResult(
            solver_id="s0", answer="chaotic", explanation="looks chaotic",
            confidence=0.8,
        )
        assert s.solver_id == "s0"
        assert s.answer == "chaotic"

    def test_review_construction(self):
        r = Review(
            reviewer_id="r0", question_quality=5,
            question_reasoning="good question",
            answer_ratings={"s0": 4, "s1": 2}, confidence=3,
        )
        assert r.question_quality == 5
        assert r.answer_ratings["s0"] == 4

    def test_reputation_defaults(self):
        rep = Reputation(agent_id="a0")
        assert rep.propose_score == 0.0
        assert rep.solve_score == 0.0
        assert rep.review_score == 0.0
        assert rep.rounds_participated == 0

    def test_round_result_defaults(self):
        prop = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.5},
            question_type="classify", reasoning="", mock_answer="",
            mock_confidence=0.5,
        )
        rr = RoundResult(round_number=1, proposal=prop)
        assert rr.problem is None
        assert rr.validation_passed is False
        assert rr.solves == []
        assert rr.reviews == []

    def test_solve_result_with_list_answer(self):
        s = SolveResult(
            solver_id="s0", answer=[0.1, 0.2, 0.3], explanation="predicted",
            confidence=0.7,
        )
        assert isinstance(s.answer, list)
        assert len(s.answer) == 3


# ───────────────────────────────────────────────────────────
# Task 2: Prompts
# ───────────────────────────────────────────────────────────

class TestArenaPrompts:
    """Prompt formatters produce correct structure."""

    def test_proposer_prompt_contains_grammar(self):
        from chaosbench.arena.prompts import format_proposer_prompt
        system, user = format_proposer_prompt([])
        assert "mathematician" in system.lower() or "benchmark" in system.lower()
        assert "logistic" in user
        assert "tent" in user
        assert "classify" in user.lower()

    def test_proposer_prompt_with_history(self):
        from chaosbench.arena.prompts import format_proposer_prompt
        prop = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.5},
            question_type="classify", reasoning="test", mock_answer="chaotic",
            mock_confidence=0.9,
        )
        rr = RoundResult(round_number=1, proposal=prop, validation_passed=True)
        system, user = format_proposer_prompt([rr])
        assert "Round 1" in user
        assert "PASSED" in user

    def test_solver_prompt_reuses_agents(self):
        from chaosbench.arena.prompts import format_solver_prompt
        from chaosbench.problems.factory import QuestionType, create_problem
        problem = create_problem("logistic", {"r": 3.891}, QuestionType.CLASSIFY)
        system, user = format_solver_prompt(problem)
        assert "scientist" in system.lower()
        assert "DATA" in user
        assert "ANSWER:" in user
        assert "EXPLANATION:" in user

    def test_reviewer_prompt_contains_likert(self):
        from chaosbench.arena.prompts import format_reviewer_prompt
        from chaosbench.problems.factory import QuestionType, create_problem
        prop = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.891},
            question_type="classify", reasoning="test", mock_answer="chaotic",
            mock_confidence=0.9,
        )
        problem = create_problem("logistic", {"r": 3.891}, QuestionType.CLASSIFY)
        solves = [
            SolveResult(solver_id="s0", answer="chaotic", explanation="yes",
                        confidence=0.8),
        ]
        system, user = format_reviewer_prompt(prop, problem, solves)
        assert "reviewer" in system.lower()
        assert "1-6" in user
        assert "s0" in user


# ───────────────────────────────────────────────────────────
# Task 3: Parsing
# ───────────────────────────────────────────────────────────

class TestArenaParsing:
    """Fail-safe parsing for proposals, solve results, and reviews."""

    def test_parse_valid_proposal(self):
        from chaosbench.arena.parsing import parse_proposal
        response = json.dumps({
            "family": "tent", "params": {"mu": 1.5},
            "question_type": "identify",
            "reasoning": "tent map is tricky",
            "mock_answer": "tent", "mock_confidence": 0.8,
        })
        p = parse_proposal(response)
        assert p.family == "tent"
        assert p.params["mu"] == 1.5
        assert p.question_type == "identify"

    def test_parse_proposal_with_markdown_fence(self):
        from chaosbench.arena.parsing import parse_proposal
        response = '```json\n{"family": "logistic", "params": {"r": 3.7}, "question_type": "classify", "reasoning": "near chaos edge", "mock_answer": "periodic", "mock_confidence": 0.6}\n```'
        p = parse_proposal(response)
        assert p.family == "logistic"
        assert p.question_type == "classify"

    def test_parse_proposal_invalid_family_falls_back(self):
        from chaosbench.arena.parsing import parse_proposal
        response = json.dumps({
            "family": "unknown_map", "params": {"x": 1.0},
            "question_type": "classify", "reasoning": "", "mock_answer": "",
            "mock_confidence": 0.5,
        })
        p = parse_proposal(response)
        assert p.family == "logistic"  # Fallback

    def test_parse_proposal_garbage_returns_defaults(self):
        from chaosbench.arena.parsing import parse_proposal
        p = parse_proposal("this is not json at all!!!")
        assert p.family == "logistic"
        assert p.params == {"r": 3.5}
        assert p.question_type == "classify"

    def test_parse_proposal_clamps_params(self):
        from chaosbench.arena.parsing import parse_proposal
        response = json.dumps({
            "family": "logistic", "params": {"r": 99.0},
            "question_type": "classify", "reasoning": "",
            "mock_answer": "", "mock_confidence": 0.5,
        })
        p = parse_proposal(response)
        assert p.params["r"] == 4.0  # Clamped to max

    def test_parse_solve_result_classify(self):
        from chaosbench.arena.parsing import parse_solve_result
        response = "I think this is chaotic.\nANSWER: chaotic\nEXPLANATION: high variance\nconfidence: 0.85"
        s = parse_solve_result(response, "s0", "classify", {})
        assert s.answer == "chaotic"
        assert s.confidence == 0.85
        assert "high variance" in s.explanation

    def test_parse_solve_result_predict(self):
        from chaosbench.arena.parsing import parse_solve_result
        response = "ANSWER: 0.5, 0.6, 0.7\nEXPLANATION: trending up\nconfidence: 0.6"
        s = parse_solve_result(response, "s0", "predict", {"horizon": 3})
        assert isinstance(s.answer, list)
        assert len(s.answer) == 3

    def test_parse_valid_review(self):
        from chaosbench.arena.parsing import parse_review
        response = json.dumps({
            "question_quality": 5,
            "question_reasoning": "good discriminator",
            "answer_ratings": {"s0": 4, "s1": 2},
            "confidence": 4,
        })
        r = parse_review(response, "r0", ["s0", "s1"])
        assert r.question_quality == 5
        assert r.answer_ratings["s0"] == 4
        assert r.answer_ratings["s1"] == 2
        assert r.confidence == 4

    def test_parse_review_clamps_values(self):
        from chaosbench.arena.parsing import parse_review
        response = json.dumps({
            "question_quality": 10,  # Out of range
            "question_reasoning": "",
            "answer_ratings": {"s0": -1},  # Out of range
            "confidence": 99,  # Out of range
        })
        r = parse_review(response, "r0", ["s0"])
        assert r.question_quality == 6  # Clamped
        assert r.answer_ratings["s0"] == 1  # Clamped
        assert r.confidence == 5  # Clamped

    def test_parse_review_garbage_returns_defaults(self):
        from chaosbench.arena.parsing import parse_review
        r = parse_review("not json at all", "r0", ["s0", "s1"])
        assert r.question_quality == 3
        assert r.answer_ratings["s0"] == 3
        assert r.confidence == 2

    def test_parse_review_regex_fallback(self):
        from chaosbench.arena.parsing import parse_review
        response = "The question is good. QUALITY: 5\nCONFIDENCE: 4"
        r = parse_review(response, "r0", ["s0"])
        assert r.question_quality == 5
        assert r.confidence == 4

    def test_parse_proposal_missing_params_uses_midpoint(self):
        from chaosbench.arena.parsing import parse_proposal
        response = json.dumps({
            "family": "logistic", "params": {},
            "question_type": "classify", "reasoning": "",
            "mock_answer": "", "mock_confidence": 0.5,
        })
        p = parse_proposal(response)
        assert p.params["r"] == 3.25  # midpoint of (2.5, 4.0)


# ───────────────────────────────────────────────────────────
# Task 4: Consensus + Reputation
# ───────────────────────────────────────────────────────────

class TestConsensus:
    """Consensus aggregation, discrimination, and reputation."""

    def _make_problem(self, qt="classify"):
        from chaosbench.problems.factory import QuestionType, create_problem
        qt_map = {
            "classify": QuestionType.CLASSIFY,
            "identify": QuestionType.IDENTIFY,
            "predict": QuestionType.PREDICT,
        }
        return create_problem("logistic", {"r": 3.891}, qt_map[qt])

    def test_consensus_picks_highest_rated_solver(self):
        from chaosbench.arena.consensus import compute_consensus_answer
        solves = [
            SolveResult(solver_id="s0", answer="periodic", explanation="",
                        confidence=0.8),
            SolveResult(solver_id="s1", answer="chaotic", explanation="",
                        confidence=0.7),
        ]
        reviews = [
            Review(reviewer_id="r0", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 2, "s1": 5}, confidence=3),
            Review(reviewer_id="r1", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 3, "s1": 6}, confidence=3),
        ]
        result = compute_consensus_answer(solves, reviews, "classify")
        assert result == "chaotic"  # s1 has higher avg rating

    def test_consensus_empty_returns_none(self):
        from chaosbench.arena.consensus import compute_consensus_answer
        assert compute_consensus_answer([], [], "classify") is None

    def test_discrimination_varied_scores(self):
        from chaosbench.arena.consensus import compute_discrimination
        problem = self._make_problem("classify")
        # One correct, one wrong
        solves = [
            SolveResult(solver_id="s0", answer="chaotic", explanation="",
                        confidence=0.8),
            SolveResult(solver_id="s1", answer="periodic", explanation="",
                        confidence=0.7),
        ]
        disc = compute_discrimination(solves, problem)
        assert disc > 0  # Variance in scores

    def test_discrimination_uniform_correct(self):
        from chaosbench.arena.consensus import compute_discrimination
        problem = self._make_problem("classify")
        # All correct → solve_rate=1 → discrimination=0
        solves = [
            SolveResult(solver_id="s0", answer="chaotic", explanation="",
                        confidence=0.9),
            SolveResult(solver_id="s1", answer="chaotic", explanation="",
                        confidence=0.8),
        ]
        disc = compute_discrimination(solves, problem)
        assert disc == 0.0

    def test_discrimination_uniform_wrong(self):
        from chaosbench.arena.consensus import compute_discrimination
        problem = self._make_problem("classify")
        # All wrong → solve_rate=0 → discrimination=0
        solves = [
            SolveResult(solver_id="s0", answer="periodic", explanation="",
                        confidence=0.5),
            SolveResult(solver_id="s1", answer="periodic", explanation="",
                        confidence=0.5),
        ]
        disc = compute_discrimination(solves, problem)
        assert disc == 0.0

    def test_discrimination_empty(self):
        from chaosbench.arena.consensus import compute_discrimination
        problem = self._make_problem("classify")
        assert compute_discrimination([], problem) == 0.0

    def test_discrimination_predict_partial_scores(self):
        from chaosbench.arena.consensus import compute_discrimination

        problem = self._make_problem("predict")
        truth = problem.ground_truth.future_values
        exact = truth.tolist()
        partial = truth.copy()
        partial[1] = partial[1] + 10.0  # Keep first point correct, fail at step 2

        solves = [
            SolveResult(solver_id="s0", answer=exact, explanation="", confidence=0.9),
            SolveResult(
                solver_id="s1",
                answer=partial.tolist(),
                explanation="",
                confidence=0.7,
            ),
        ]

        disc = compute_discrimination(solves, problem)
        assert disc > 0.0

    def test_compare_consensus_correct(self):
        from chaosbench.arena.consensus import compare_consensus_to_math
        problem = self._make_problem("classify")
        assert compare_consensus_to_math("chaotic", problem) is True

    def test_compare_consensus_wrong(self):
        from chaosbench.arena.consensus import compare_consensus_to_math
        problem = self._make_problem("classify")
        assert compare_consensus_to_math("periodic", problem) is False

    def test_compare_consensus_none(self):
        from chaosbench.arena.consensus import compare_consensus_to_math
        problem = self._make_problem("classify")
        assert compare_consensus_to_math(None, problem) is False

    def test_reputation_update_proposer(self):
        from chaosbench.arena.consensus import update_reputation
        prop = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.891},
            question_type="classify", reasoning="", mock_answer="",
            mock_confidence=0.5,
        )
        rr = RoundResult(
            round_number=1, proposal=prop, validation_passed=False,
        )
        reps = update_reputation({}, rr)
        assert "p0" in reps
        assert reps["p0"].propose_score == 0.0  # Failed validation
        assert reps["p0"].rounds_participated == 1

    def test_reputation_update_full_round(self):
        from chaosbench.arena.consensus import update_reputation
        problem = self._make_problem("classify")
        prop = Proposal(
            proposer_id="p0", family="logistic", params={"r": 3.891},
            question_type="classify", reasoning="", mock_answer="chaotic",
            mock_confidence=0.9,
        )
        solves = [
            SolveResult(solver_id="s0", answer="chaotic", explanation="",
                        confidence=0.9),
            SolveResult(solver_id="s1", answer="periodic", explanation="",
                        confidence=0.5),
        ]
        reviews = [
            Review(reviewer_id="r0", question_quality=5,
                   question_reasoning="",
                   answer_ratings={"s0": 5, "s1": 2}, confidence=4),
        ]
        rr = RoundResult(
            round_number=1, proposal=prop, problem=problem,
            validation_passed=True, solves=solves, reviews=reviews,
        )
        reps = update_reputation({}, rr)
        assert reps["s0"].solve_score == 1.0  # Correct
        assert reps["s1"].solve_score == 0.0  # Wrong
        assert reps["r0"].review_score > 0  # Rated correctly

    def test_reputation_running_average(self):
        from chaosbench.arena.consensus import _running_average
        assert _running_average(0.0, 1.0, 1) == 1.0
        assert _running_average(1.0, 0.0, 2) == 0.5
        assert abs(_running_average(0.5, 1.0, 3) - 2/3) < 1e-10

    def test_consensus_predict_type(self):
        from chaosbench.arena.consensus import compute_consensus_answer
        solves = [
            SolveResult(solver_id="s0", answer=[0.1, 0.2], explanation="",
                        confidence=0.8),
            SolveResult(solver_id="s1", answer=[0.3, 0.4], explanation="",
                        confidence=0.6),
        ]
        reviews = [
            Review(reviewer_id="r0", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 5, "s1": 2}, confidence=3),
        ]
        result = compute_consensus_answer(solves, reviews, "predict")
        assert result == [0.1, 0.2]  # s0 rated highest


# ───────────────────────────────────────────────────────────
# Task 5: Runner (with mocked LLM)
# ───────────────────────────────────────────────────────────

class TestArenaRunner:
    """Full round execution with mocked LLM calls."""

    def _mock_responses(self):
        """Return canned LLM responses for proposer, solvers, reviewers."""
        proposal_json = json.dumps({
            "family": "logistic", "params": {"r": 3.891},
            "question_type": "classify",
            "reasoning": "Near chaos boundary",
            "mock_answer": "chaotic", "mock_confidence": 0.9,
        })
        solve_response = (
            "The data shows high variance and no periodicity.\n"
            "ANSWER: chaotic\n"
            "EXPLANATION: Sensitive dependence on initial conditions\n"
            "confidence: 0.85"
        )
        review_json = json.dumps({
            "question_quality": 5,
            "question_reasoning": "good discriminator",
            "answer_ratings": {"solver_0": 5, "solver_1": 4, "solver_2": 3},
            "confidence": 4,
        })

        responses = [proposal_json]  # 1 proposer call
        responses.extend([solve_response] * 3)  # 3 solver calls
        responses.extend([review_json] * 2)  # 2 reviewer calls
        return responses

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_passes_validation(self, mock_llm):
        from chaosbench.arena.runner import run_round

        mock_llm.side_effect = self._mock_responses()
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", verbose=False,
        )
        assert result.validation_passed is True
        assert result.problem is not None
        assert len(result.solves) == 3
        assert len(result.reviews) == 2
        assert result.consensus_answer is not None
        assert result.consensus_matches_math is not None

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_failed_proposal(self, mock_llm):
        from chaosbench.arena.runner import run_round

        # Propose garbage that creates an invalid problem
        bad_proposal = json.dumps({
            "family": "logistic", "params": {"r": 2.5},
            "question_type": "predict",
            "reasoning": "test", "mock_answer": "", "mock_confidence": 0.5,
        })
        mock_llm.return_value = bad_proposal
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", verbose=False,
        )
        # r=2.5 is fixed_point → may fail chaotic gates for predict or pass
        # Either way, round should complete without crash
        assert result.round_number == 1
        assert result.proposal.family == "logistic"

    @patch("chaosbench.arena.runner.create_problem", side_effect=RuntimeError("boom"))
    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_problem_creation_failure_preserves_reputation(
        self, mock_llm, _mock_create_problem
    ):
        from chaosbench.arena.runner import run_round

        proposal_json = json.dumps({
            "family": "logistic", "params": {"r": 3.891},
            "question_type": "classify",
            "reasoning": "test",
            "mock_answer": "chaotic",
            "mock_confidence": 0.9,
        })
        mock_llm.return_value = proposal_json

        existing = {
            "solver_0": Reputation(
                agent_id="solver_0",
                solve_score=0.6,
                rounds_participated=2,
            ),
        }
        result = run_round(
            round_num=3,
            reputations=existing,
            round_history=[],
            model="test-model",
            verbose=False,
        )

        assert result.validation_passed is False
        assert "proposer_0" in result.reputation_updates
        assert "solver_0" in result.reputation_updates
        assert result.reputation_updates["solver_0"].solve_score == 0.6
        assert result.reputation_updates["solver_0"].rounds_participated == 2

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_has_reputation_updates(self, mock_llm):
        from chaosbench.arena.runner import run_round

        mock_llm.side_effect = self._mock_responses()
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", verbose=False,
        )
        assert len(result.reputation_updates) > 0
        assert "proposer_0" in result.reputation_updates

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_arena_multiple_rounds(self, mock_llm):
        from chaosbench.arena.runner import run_arena

        # Need enough responses for 2 rounds (6 calls each)
        responses = self._mock_responses() * 3
        mock_llm.side_effect = responses
        results = run_arena(n_rounds=2, model="test-model", verbose=False)
        assert len(results) == 2

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_solver_count(self, mock_llm):
        from chaosbench.arena.runner import run_round

        # 1 proposer + 2 solvers + 2 reviewers = 5 calls
        proposal_json = json.dumps({
            "family": "logistic", "params": {"r": 3.891},
            "question_type": "classify", "reasoning": "test",
            "mock_answer": "chaotic", "mock_confidence": 0.9,
        })
        solve_resp = "ANSWER: chaotic\nconfidence: 0.8"
        review_json = json.dumps({
            "question_quality": 4, "question_reasoning": "",
            "answer_ratings": {"solver_0": 4, "solver_1": 3},
            "confidence": 3,
        })
        mock_llm.side_effect = [proposal_json, solve_resp, solve_resp,
                                 review_json, review_json]
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", n_solvers=2, n_reviewers=2, verbose=False,
        )
        assert len(result.solves) == 2
        assert len(result.reviews) == 2

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_identify_type(self, mock_llm):
        from chaosbench.arena.runner import run_round

        proposal_json = json.dumps({
            "family": "tent", "params": {"mu": 1.5},
            "question_type": "identify", "reasoning": "tent is hard",
            "mock_answer": "tent", "mock_confidence": 0.8,
        })
        solve_resp = "ANSWER: tent\nEXPLANATION: triangular map\nconfidence: 0.7"
        review_json = json.dumps({
            "question_quality": 4, "question_reasoning": "decent",
            "answer_ratings": {"solver_0": 5, "solver_1": 3},
            "confidence": 3,
        })
        mock_llm.side_effect = [proposal_json, solve_resp, solve_resp,
                                 review_json, review_json]
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", n_solvers=2, n_reviewers=2, verbose=False,
        )
        assert result.proposal.question_type == "identify"
        assert result.proposal.family == "tent"


# ───────────────────────────────────────────────────────────
# Task 6: SPIRE-Inspired Improvements
# ───────────────────────────────────────────────────────────

class TestReviewCorrectAnswer:
    """Review dataclass accepts and stores correct_answer."""

    def test_review_correct_answer_field(self):
        r = Review(
            reviewer_id="r0", question_quality=5,
            question_reasoning="good",
            answer_ratings={"s0": 4}, confidence=3,
            correct_answer="chaotic",
        )
        assert r.correct_answer == "chaotic"

    def test_review_correct_answer_default_none(self):
        r = Review(
            reviewer_id="r0", question_quality=4,
            question_reasoning="ok",
            answer_ratings={"s0": 3}, confidence=3,
        )
        assert r.correct_answer is None


class TestParseProposals:
    """Multi-proposal parsing."""

    def test_parse_proposals_json_array(self):
        from chaosbench.arena.parsing import parse_proposals
        data = json.dumps([
            {"family": "logistic", "params": {"r": 3.8},
             "question_type": "classify", "reasoning": "a",
             "mock_answer": "chaotic", "mock_confidence": 0.9},
            {"family": "tent", "params": {"mu": 1.5},
             "question_type": "identify", "reasoning": "b",
             "mock_answer": "tent", "mock_confidence": 0.8},
            {"family": "rotation", "params": {"theta": 0.41},
             "question_type": "predict", "reasoning": "c",
             "mock_answer": "0.3", "mock_confidence": 0.6},
        ])
        proposals = parse_proposals(data)
        assert len(proposals) == 3
        assert proposals[0].family == "logistic"
        assert proposals[1].family == "tent"
        assert proposals[2].family == "rotation"
        assert proposals[0].question_type == "classify"
        assert proposals[1].question_type == "identify"
        assert proposals[2].question_type == "predict"

    def test_parse_proposals_fallback_single(self):
        from chaosbench.arena.parsing import parse_proposals
        data = json.dumps({
            "family": "logistic", "params": {"r": 3.891},
            "question_type": "classify", "reasoning": "test",
            "mock_answer": "chaotic", "mock_confidence": 0.9,
        })
        proposals = parse_proposals(data)
        assert len(proposals) == 1
        assert proposals[0].family == "logistic"

    def test_parse_proposals_garbage_returns_default(self):
        from chaosbench.arena.parsing import parse_proposals
        proposals = parse_proposals("totally not json")
        assert len(proposals) == 1
        assert proposals[0].family == "logistic"  # default


class TestParseReviewCorrectAnswer:
    """Parsing extracts correct_answer from review JSON."""

    def test_parse_review_with_correct_answer(self):
        from chaosbench.arena.parsing import parse_review
        response = json.dumps({
            "question_quality": 5,
            "question_reasoning": "good",
            "answer_ratings": {"s0": 4},
            "confidence": 3,
            "correct_answer": "chaotic",
        })
        r = parse_review(response, "r0", ["s0"])
        assert r.correct_answer == "chaotic"

    def test_parse_review_without_correct_answer(self):
        from chaosbench.arena.parsing import parse_review
        response = json.dumps({
            "question_quality": 4,
            "question_reasoning": "ok",
            "answer_ratings": {"s0": 3},
            "confidence": 3,
        })
        r = parse_review(response, "r0", ["s0"])
        assert r.correct_answer is None


class TestAsymmetricLikert:
    """Asymmetric Likert weights and weighted ratings."""

    def test_weighted_rating_asymmetric(self):
        from chaosbench.arena.consensus import compute_weighted_rating
        # Rating 1 should be more negative than rating 6 is positive (asymmetric)
        score_1 = compute_weighted_rating(1, 3)  # -1.0 * 1.0 = -1.0
        score_6 = compute_weighted_rating(6, 3)  # 1.0 * 1.0 = 1.0
        assert score_1 < 0
        assert score_6 > 0
        # Rating 2 is -0.5, rating 5 is 0.7 — penalize more than reward
        score_2 = compute_weighted_rating(2, 3)
        score_5 = compute_weighted_rating(5, 3)
        assert abs(score_2) < abs(score_5)  # -0.5 vs 0.7
        # Rating 3 is slightly negative, rating 4 slightly positive
        score_3 = compute_weighted_rating(3, 3)
        score_4 = compute_weighted_rating(4, 3)
        assert score_3 < 0
        assert score_4 > 0

    def test_weighted_rating_with_confidence(self):
        from chaosbench.arena.consensus import compute_weighted_rating
        # Higher confidence amplifies signal
        low_conf = compute_weighted_rating(5, 1)   # 0.7 * 0.4 = 0.28
        high_conf = compute_weighted_rating(5, 5)  # 0.7 * 1.4 = 0.98
        assert high_conf > low_conf
        assert abs(high_conf) > abs(low_conf)

    def test_weighted_rating_with_reputation(self):
        from chaosbench.arena.consensus import compute_weighted_rating
        # Higher reputation scales contribution
        low_rep = compute_weighted_rating(5, 3, reputation=0.5)
        high_rep = compute_weighted_rating(5, 3, reputation=1.0)
        assert high_rep > low_rep

    def test_consensus_with_reputation_weighting(self):
        from chaosbench.arena.consensus import compute_consensus_answer
        solves = [
            SolveResult(solver_id="s0", answer="periodic", explanation="",
                        confidence=0.8),
            SolveResult(solver_id="s1", answer="chaotic", explanation="",
                        confidence=0.7),
        ]
        # Both reviewers give opposing ratings, but one has high reputation
        reviews = [
            Review(reviewer_id="r_high", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 2, "s1": 6}, confidence=4),
            Review(reviewer_id="r_low", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 6, "s1": 2}, confidence=4),
        ]
        reps = {
            "r_high": Reputation(agent_id="r_high", review_score=0.9,
                                 rounds_participated=5),
            "r_low": Reputation(agent_id="r_low", review_score=0.1,
                                rounds_participated=5),
        }
        result = compute_consensus_answer(
            solves, reviews, "classify", reputations=reps
        )
        # r_high prefers s1 (chaotic), r_low prefers s0 (periodic)
        # r_high has 9x reputation, so s1 should win
        assert result == "chaotic"

    def test_consensus_fuses_reviewer_answers(self):
        from chaosbench.arena.consensus import compute_consensus_answer
        solves = [
            SolveResult(solver_id="s0", answer="periodic", explanation="",
                        confidence=0.5),
            SolveResult(solver_id="s1", answer="chaotic", explanation="",
                        confidence=0.5),
        ]
        # Neutral ratings (both get 3 → -0.15 each), but reviewer says
        # correct_answer is "chaotic" — this should tip the balance
        reviews = [
            Review(reviewer_id="r0", question_quality=4,
                   question_reasoning="",
                   answer_ratings={"s0": 3, "s1": 3}, confidence=4,
                   correct_answer="chaotic"),
        ]
        result = compute_consensus_answer(solves, reviews, "classify")
        assert result == "chaotic"  # Fused reviewer answer tips to s1


class TestRunnerMultiProposal:
    """Runner selects best proposal from multiple candidates."""

    @patch("chaosbench.arena.runner.call_llm")
    def test_run_round_multi_proposal(self, mock_llm):
        from chaosbench.arena.runner import run_round

        # Proposer returns 3 proposals as JSON array
        proposals_json = json.dumps([
            {"family": "logistic", "params": {"r": 3.891},
             "question_type": "classify", "reasoning": "chaos edge",
             "mock_answer": "chaotic", "mock_confidence": 0.9},
            {"family": "tent", "params": {"mu": 1.5},
             "question_type": "identify", "reasoning": "tent hard",
             "mock_answer": "tent", "mock_confidence": 0.8},
            {"family": "logistic", "params": {"r": 3.7},
             "question_type": "predict", "reasoning": "near chaos",
             "mock_answer": "0.5", "mock_confidence": 0.7},
        ])
        solve_resp = "ANSWER: chaotic\nconfidence: 0.8"
        review_json = json.dumps({
            "question_quality": 4, "question_reasoning": "",
            "answer_ratings": {"solver_0": 4, "solver_1": 3, "solver_2": 3},
            "confidence": 3,
        })
        mock_llm.side_effect = [
            proposals_json,
            solve_resp, solve_resp, solve_resp,
            review_json, review_json,
        ]
        result = run_round(
            round_num=1, reputations={}, round_history=[],
            model="test-model", verbose=False,
        )
        # Should pick a valid proposal (at least one of the three should pass)
        assert result.round_number == 1
        assert result.proposal is not None
