"""Arena protocol dataclasses for ChaosBench v2 Phase 4.

Defines the data model for the adversarial arena: proposals, solve results,
reviews, reputation tracking, and round results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from chaosbench.problems.factory import Problem


QuestionType = Literal["classify", "identify", "predict"]
Params = dict[str, float]
AnswerRatings = dict[str, int]
ReputationMap = dict[str, "Reputation"]


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    """Clamp to [lo, hi], returning default on parse failure."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, numeric))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    """Clamp to [lo, hi], returning default on parse failure."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, numeric))


def _coerce_numeric_params(params: Any) -> Params:
    """Convert a params mapping to float values, dropping invalid entries."""
    if not isinstance(params, dict):
        return {}
    cleaned: Params = {}
    for key, value in params.items():
        try:
            cleaned[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return cleaned


@dataclass
class Proposal:
    """An LLM-generated problem proposal."""

    proposer_id: str
    family: str              # From ATOM_REGISTRY keys
    params: Params           # e.g. {"r": 3.891}
    question_type: QuestionType | str
    reasoning: str           # Why this problem is hard
    mock_answer: str         # Proposer's own answer
    mock_confidence: float  # 0-1

    def __post_init__(self) -> None:
        self.proposer_id = str(self.proposer_id).strip()
        self.family = str(self.family).strip().lower()
        self.params = _coerce_numeric_params(self.params)
        qt = str(self.question_type).strip().lower()
        self.question_type = qt if qt in ("classify", "identify", "predict") else "classify"
        self.reasoning = str(self.reasoning)[:500]
        self.mock_answer = str(self.mock_answer)
        self.mock_confidence = _clamp_float(self.mock_confidence, 0.0, 1.0, 0.5)


@dataclass
class SolveResult:
    """A solver's answer to a proposed problem."""

    solver_id: str
    answer: Any             # str or list[float]
    explanation: str        # <=500 chars
    confidence: float       # 0-1

    def __post_init__(self) -> None:
        self.solver_id = str(self.solver_id).strip()
        self.explanation = str(self.explanation)[:500]
        self.confidence = _clamp_float(self.confidence, 0.0, 1.0, 0.5)


@dataclass
class Review:
    """A reviewer's assessment of question quality and solver answers."""

    reviewer_id: str
    question_quality: int  # 1-6 Likert
    question_reasoning: str
    answer_ratings: AnswerRatings  # {solver_id: int (1-6)}
    confidence: int  # 1-5
    correct_answer: Any = None  # Reviewer's own answer to the problem

    def __post_init__(self) -> None:
        self.reviewer_id = str(self.reviewer_id).strip()
        self.question_quality = _clamp_int(self.question_quality, 1, 6, 3)
        self.question_reasoning = str(self.question_reasoning)[:500]
        self.confidence = _clamp_int(self.confidence, 1, 5, 2)
        if not isinstance(self.answer_ratings, dict):
            self.answer_ratings = {}
            return
        self.answer_ratings = {
            str(solver_id): _clamp_int(score, 1, 6, 3)
            for solver_id, score in self.answer_ratings.items()
        }


@dataclass
class Reputation:
    """Running performance tracker for an arena agent."""

    agent_id: str
    propose_score: float = 0.0       # Running average
    solve_score: float = 0.0
    review_score: float = 0.0
    rounds_participated: int = 0

    def __post_init__(self) -> None:
        self.agent_id = str(self.agent_id).strip()
        self.propose_score = _clamp_float(self.propose_score, 0.0, 1.0, 0.0)
        self.solve_score = _clamp_float(self.solve_score, 0.0, 1.0, 0.0)
        self.review_score = _clamp_float(self.review_score, 0.0, 1.0, 0.0)
        try:
            rounds = int(self.rounds_participated)
        except (TypeError, ValueError):
            rounds = 0
        self.rounds_participated = max(0, rounds)


@dataclass
class RoundResult:
    """Complete record of one arena round."""

    round_number: int
    proposal: Proposal
    problem: Optional[Problem] = None          # None if validation failed
    validation_passed: bool = False
    validation_details: dict = field(default_factory=dict)
    solves: list[SolveResult] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    consensus_answer: Optional[Any] = None
    math_ground_truth: Optional[Any] = None
    consensus_matches_math: Optional[bool] = None
    reputation_updates: ReputationMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.round_number = int(self.round_number)
        except (TypeError, ValueError):
            self.round_number = 0
        self.round_number = max(0, self.round_number)
        if self.validation_details is None:
            self.validation_details = {}
        if self.solves is None:
            self.solves = []
        if self.reviews is None:
            self.reviews = []
        if self.reputation_updates is None:
            self.reputation_updates = {}
