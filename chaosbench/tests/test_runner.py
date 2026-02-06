"""Tests for ChaosBench v2 experiment runner (mocked LLM)."""

from unittest.mock import patch

import numpy as np
import pytest

from chaosbench.agents.llm_agent import LLMAgent
from chaosbench.agents.protocol import Solution
from chaosbench.experiment.runner import load_problems, run_bank, score_solution
from chaosbench.problems.factory import GroundTruth, Problem, QuestionType


def _make_problem(qt: str, **kwargs) -> Problem:
    """Create a minimal Problem for testing."""
    defaults = {
        "problem_id": "test_001",
        "question_type": QuestionType(qt),
        "observations": np.linspace(0.1, 0.9, 200),
        "question_params": {},
        "metadata": {"domain": [0, 1], "n_points": 200, "noise_std": 0.01, "stride": 1},
        "ground_truth": GroundTruth(),
        "system_metadata": None,
        "observation_detail": None,
        "difficulty": 1.1,
    }
    defaults.update(kwargs)
    return Problem(**defaults)


class TestScoreSolution:
    def test_classify_correct(self):
        p = _make_problem("classify", ground_truth=GroundTruth(regime="chaotic"))
        sol = Solution(answer="chaotic", confidence=0.9, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["raw_score"] == 1.0

    def test_classify_wrong(self):
        p = _make_problem("classify", ground_truth=GroundTruth(regime="chaotic"))
        sol = Solution(answer="periodic", confidence=0.9, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["raw_score"] == 0.0

    def test_identify_correct(self):
        p = _make_problem("identify", ground_truth=GroundTruth(family="logistic"))
        sol = Solution(answer="logistic", confidence=0.8, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["raw_score"] == 1.0

    def test_predict_perfect(self):
        future = np.array([0.5, 0.6, 0.7])
        p = _make_problem(
            "predict",
            ground_truth=GroundTruth(future_values=future),
            question_params={"horizon": 3},
        )
        sol = Solution(answer=[0.5, 0.6, 0.7], confidence=0.8, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["raw_score"] == 1.0

    def test_predict_empty(self):
        future = np.array([0.5, 0.6, 0.7])
        p = _make_problem(
            "predict",
            ground_truth=GroundTruth(future_values=future),
            question_params={"horizon": 3},
        )
        sol = Solution(answer=[], confidence=0.5, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["raw_score"] == 0.0

    def test_weighted_score(self):
        p = _make_problem(
            "classify",
            ground_truth=GroundTruth(regime="chaotic"),
            difficulty=2.0,
        )
        sol = Solution(answer="chaotic", confidence=0.9, reasoning_trace="")
        result = score_solution(sol, p)
        assert result["weighted_score"] == 2.0


class TestRunBank:
    @patch("chaosbench.agents.llm_agent.call_llm")
    def test_run_bank_mocked(self, mock_llm):
        """Full run with mocked LLM returning canned responses."""
        # Canned responses per question type
        responses = {
            "classify": "This is chaotic.\nANSWER: chaotic\nconfidence: 0.8",
            "identify": "Looks like logistic.\nANSWER: logistic\nconfidence: 0.7",
            "predict": "ANSWER: 0.5, 0.5, 0.5\nconfidence: 0.6",
        }

        problems = [
            _make_problem("classify", ground_truth=GroundTruth(regime="chaotic")),
            _make_problem(
                "identify",
                problem_id="test_002",
                ground_truth=GroundTruth(family="logistic"),
            ),
            _make_problem(
                "predict",
                problem_id="test_003",
                ground_truth=GroundTruth(future_values=np.array([0.5, 0.5, 0.5])),
                question_params={"horizon": 3},
            ),
        ]

        # Return the appropriate canned response for each call
        call_count = [0]
        def side_effect(*args, **kwargs):
            qt = problems[call_count[0]].question_type.value
            call_count[0] += 1
            return responses[qt]

        mock_llm.side_effect = side_effect

        agent = LLMAgent()
        results = run_bank(agent, problems, verbose=False)

        assert len(results) == 3
        assert all(not r["error"] for r in results)
        assert results[0]["raw_score"] == 1.0  # classify correct
        assert results[1]["raw_score"] == 1.0  # identify correct

    @patch("chaosbench.agents.llm_agent.call_llm")
    def test_run_bank_handles_error(self, mock_llm):
        """Agent error doesn't crash the run."""
        mock_llm.side_effect = Exception("API timeout")

        problems = [
            _make_problem("classify", ground_truth=GroundTruth(regime="chaotic")),
        ]

        agent = LLMAgent()
        results = run_bank(agent, problems, verbose=False)

        assert len(results) == 1
        assert results[0]["error"] is True
        assert results[0]["raw_score"] == 0.0


class TestLoadProblems:
    def test_load_bank(self):
        """Load the real frozen bank and check structure."""
        from chaosbench.experiment.runner import BANK_PATH
        if not BANK_PATH.exists():
            pytest.skip("mini_bank.json not found")

        problems = load_problems()
        assert len(problems) > 0

        for p in problems:
            assert isinstance(p, Problem)
            assert isinstance(p.question_type, QuestionType)
            assert len(p.observations) > 0
            assert p.difficulty > 0
