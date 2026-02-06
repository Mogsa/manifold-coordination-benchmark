"""
Problem factory — generates benchmark problems from atoms.

Each problem combines:
- A dynamical system (atom + parameters)
- A question type (CLASSIFY, IDENTIFY, PREDICT)
- Noisy observations
- Hidden ground truth
- Difficulty metadata
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from chaosbench.grammar.registry import create_atom
from chaosbench.grammar.system import DynamicalSystem, SystemMetadata, Observation
from chaosbench.scoring.difficulty import composite_difficulty


class QuestionType(Enum):
    CLASSIFY = "classify"
    IDENTIFY = "identify"
    PREDICT = "predict"


@dataclass
class GroundTruth:
    """Hidden answers — never shown to the agent."""

    regime: Optional[str] = None           # For CLASSIFY
    family: Optional[str] = None           # For IDENTIFY
    future_values: Optional[np.ndarray] = None  # For PREDICT
    params: Optional[dict] = None          # Full parameter dict


@dataclass
class Problem:
    """A single benchmark problem."""

    problem_id: str
    question_type: QuestionType
    observations: np.ndarray               # Noisy data (shown to agent)
    question_params: dict                  # e.g. {'horizon': 10} for PREDICT
    metadata: dict                         # Shown to agent (domain, n_points, etc.)
    ground_truth: GroundTruth              # Hidden from agent
    system_metadata: SystemMetadata        # Full system info (hidden)
    observation_detail: Observation        # Full observation object (hidden)
    difficulty: float
    stage1_passed: Optional[bool] = None
    stage2_results: Optional[dict] = None


def _make_problem_id(
    family: str,
    params: dict,
    question_type: QuestionType,
    ic_seed: int,
    noise_seed: int,
    n_points: int,
    noise_std: float,
    stride: int,
    burn_in: int,
) -> str:
    """Deterministic problem ID from components."""
    payload = {
        "family": family,
        "params": params,
        "question_type": question_type.value,
        "ic_seed": ic_seed,
        "noise_seed": noise_seed,
        "n_points": n_points,
        "noise_std": noise_std,
        "stride": stride,
        "burn_in": burn_in,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def create_problem(
    family: str,
    params: dict,
    question_type: QuestionType,
    ic_seed: int = 42,
    noise_seed: int = 123,
    n_points: int = 200,
    noise_std: float = 0.01,
    stride: int = 1,
    burn_in: int = 500,
) -> Problem:
    """Create a single benchmark problem.

    Args:
        family: Atom family name.
        params: Parameter dict for the atom.
        question_type: What the agent must answer.
        ic_seed: Seed for initial condition.
        noise_seed: Seed for observation noise.
        n_points: Number of observation points.
        noise_std: Gaussian noise standard deviation.
        stride: Keep every stride-th iterate when recording observations.
        burn_in: Number of pre-recording iterations to reach attractor.

    Returns:
        A Problem ready for validation and presentation.
    """
    atom = create_atom(family, params)
    system = DynamicalSystem(atom)
    obs = system.observe(
        n_points=n_points,
        noise_std=noise_std,
        stride=stride,
        ic_seed=ic_seed,
        noise_seed=noise_seed,
        burn_in=burn_in,
    )
    meta = system.metadata()
    diff = composite_difficulty(meta.h_ks, grammar_depth=0, noise_std=noise_std)
    problem_id = _make_problem_id(
        family=family,
        params=params,
        question_type=question_type,
        ic_seed=ic_seed,
        noise_seed=noise_seed,
        n_points=n_points,
        noise_std=noise_std,
        stride=stride,
        burn_in=burn_in,
    )

    # Build ground truth based on question type
    gt = GroundTruth(params=params)
    question_params: dict = {}

    if question_type == QuestionType.CLASSIFY:
        gt.regime = meta.regime

    elif question_type == QuestionType.IDENTIFY:
        gt.family = family

    elif question_type == QuestionType.PREDICT:
        horizon = system.prediction_horizon()
        gt.future_values = system.future_trajectory(obs, horizon)
        question_params["horizon"] = horizon

    # Metadata shown to agent
    shown_metadata = {
        "domain": meta.domain,
        "n_points": n_points,
        "noise_std": noise_std,
        "observation_mode": "clean" if noise_std == 0.0 else "noisy",
        "stride": stride,
    }

    return Problem(
        problem_id=problem_id,
        question_type=question_type,
        observations=obs.data,
        question_params=question_params,
        metadata=shown_metadata,
        ground_truth=gt,
        system_metadata=meta,
        observation_detail=obs,
        difficulty=diff,
    )
