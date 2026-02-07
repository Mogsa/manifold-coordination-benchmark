"""
Mini-bank generation — creates, validates, and serializes the problem bank.

The mini-bank is 4 families × 3 param sets × 3 question types = 36 problems.
After validation (Stage 1 gates + Stage 2 baselines), expect 12-18+ valid problems.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from chaosbench.grammar.connectives import AffineConjugacy
from chaosbench.grammar.registry import ATOM_REGISTRY, CONJ_BANK_PARAMS, MINI_BANK_PARAMS, create_atom
from chaosbench.problems.factory import (
    Problem,
    QuestionType,
    create_problem,
)
from chaosbench.problems.verification import verify_predict
from chaosbench.validation.baselines import (
    baseline_ar5,
    baseline_classify_moments,
    baseline_identify_return_map,
    baseline_mean_reversion,
    baseline_persistence,
    validate_stage2_predict,
)
from chaosbench.validation.gates import validate_stage1


PERIODICITY_MAX_PERIOD = 1000
PERIODICITY_MIN_REPEATS = 3
DEFAULT_REASONING_QUESTION_TYPES = (
    QuestionType.CLASSIFY,
    QuestionType.PREDICT,
)


def _normalize_question_types(
    question_types: Optional[Iterable[QuestionType]],
) -> List[QuestionType]:
    """Normalize question type selection, preserving order and uniqueness."""
    if question_types is None:
        return list(QuestionType)

    normalized: List[QuestionType] = []
    seen = set()
    for qt in question_types:
        if not isinstance(qt, QuestionType):
            raise TypeError(
                "question_types must contain QuestionType values, "
                f"got {type(qt).__name__}"
            )
        if qt in seen:
            continue
        seen.add(qt)
        normalized.append(qt)

    if not normalized:
        raise ValueError("question_types cannot be empty")
    return normalized


def _format_noise_sigma(noise_std: float) -> str:
    """Format noise std for stable track names and filenames."""
    return f"{noise_std:.3f}".replace(".", "p")


def _stage1_trajectory(problem: Problem) -> np.ndarray:
    """Build a trajectory long enough for strict periodicity checks."""
    obs = problem.observation_detail
    base = obs.clean_data

    # Strict periodicity gate needs enough points to test all periods <= 1000.
    if problem.system_metadata.regime != "chaotic":
        return base

    required_n = 2 * PERIODICITY_MAX_PERIOD * PERIODICITY_MIN_REPEATS
    if len(base) >= required_n:
        return base

    atom = create_atom(problem.system_metadata.family, problem.system_metadata.params)
    # For conjugated problems, wrap in AffineConjugacy so trajectory matches domain
    if problem.system_metadata.grammar_depth > 0:
        atom = AffineConjugacy(atom, a=CONJ_BANK_PARAMS["a"], b=CONJ_BANK_PARAMS["b"])
    start = float(base[0]) if len(base) > 0 else float(obs.x0)
    return atom.trajectory(start, required_n)


def generate_mini_bank(
    ic_seed: int = 42,
    noise_seed: int = 123,
    n_points: int = 200,
    noise_std: float = 0.01,
    question_types: Optional[Iterable[QuestionType]] = None,
    include_conjugated: bool = True,
) -> List[Problem]:
    """Generate a mini-bank over selected question types.

    Generates depth-0 problems from MINI_BANK_PARAMS, then optionally
    depth-1 conjugated problems from each family's hardest param set.
    """
    selected_types = _normalize_question_types(question_types)
    problems = []

    # Depth-0: bare atoms
    for family, param_list in MINI_BANK_PARAMS.items():
        for params in param_list:
            for qt in selected_types:
                p = create_problem(
                    family=family,
                    params=params,
                    question_type=qt,
                    ic_seed=ic_seed,
                    noise_seed=noise_seed,
                    n_points=n_points,
                    noise_std=noise_std,
                )
                problems.append(p)

    # Depth-1: conjugated (last param set per family)
    if include_conjugated:
        for family, param_list in MINI_BANK_PARAMS.items():
            hardest_params = param_list[-1]
            for qt in selected_types:
                p = create_problem(
                    family=family,
                    params=hardest_params,
                    question_type=qt,
                    ic_seed=ic_seed,
                    noise_seed=noise_seed,
                    n_points=n_points,
                    noise_std=noise_std,
                    conjugacy=CONJ_BANK_PARAMS,
                )
                problems.append(p)

    return problems


def generate_track_banks(
    ic_seed: int = 42,
    noise_seed: int = 123,
    n_points: int = 200,
    question_types: Optional[Iterable[QuestionType]] = None,
    noisy_noise_stds: Optional[Iterable[float]] = None,
) -> Dict[str, List[Problem]]:
    """Generate separate clean/noisy banks for fair split reporting."""
    if question_types is None:
        question_types = DEFAULT_REASONING_QUESTION_TYPES

    tracks: Dict[str, List[Problem]] = {}
    tracks["clean"] = generate_mini_bank(
        ic_seed=ic_seed,
        noise_seed=noise_seed,
        n_points=n_points,
        noise_std=0.0,
        question_types=question_types,
    )

    if noisy_noise_stds is None:
        noisy_noise_stds = (0.01,)

    for sigma in noisy_noise_stds:
        if sigma <= 0.0:
            raise ValueError(f"noisy_noise_stds must be > 0, got {sigma}")
        track_name = f"noisy_sigma_{_format_noise_sigma(sigma)}"
        tracks[track_name] = generate_mini_bank(
            ic_seed=ic_seed,
            noise_seed=noise_seed,
            n_points=n_points,
            noise_std=sigma,
            question_types=question_types,
        )

    return tracks


def validate_problem(problem: Problem) -> Tuple[bool, dict]:
    """Run Stage 1 gates and Stage 2 baselines on a problem.

    Returns (passed, details_dict).
    """
    meta = problem.system_metadata
    obs = problem.observation_detail
    spec = ATOM_REGISTRY[meta.family]
    gate_traj = _stage1_trajectory(problem)

    # Stage 1: hard gates on the clean trajectory
    s1_passed, s1_results = validate_stage1(
        trajectory=gate_traj,
        family=meta.family,
        params=meta.params,
        valid_ranges=spec.param_ranges,
        domain=meta.domain,
        regime=meta.regime,
        lambda_max=meta.lambda_max,
        periodicity_max_period=PERIODICITY_MAX_PERIOD,
    )
    problem.stage1_passed = s1_passed

    details: dict = {
        "stage1_passed": s1_passed,
        "stage1_gates": [(name, passed, reason) for name, passed, reason in s1_results],
    }

    if not s1_passed:
        return False, details

    # Stage 2: baseline validation
    s2: dict = {}

    if problem.question_type == QuestionType.PREDICT:
        gt_future = problem.ground_truth.future_values
        K = len(gt_future)
        data = obs.data

        # Run PREDICT baselines
        for name, fn in [
            ("persistence", baseline_persistence),
            ("mean_reversion", baseline_mean_reversion),
            ("ar5", baseline_ar5),
        ]:
            preds = fn(data, K)
            result = verify_predict(preds, gt_future, data)
            s2[name] = result

        # Validate against tau_lambda
        best_k = max(r["k_eff"] for r in s2.values())
        s2_check = validate_stage2_predict(
            {"k_eff": best_k}, meta.tau_lambda
        )
        s2["validation"] = s2_check
        passed = s1_passed and s2_check["passed"]

    elif problem.question_type == QuestionType.CLASSIFY:
        pred = baseline_classify_moments(obs.data)
        s2["moments_pred"] = pred
        s2["correct"] = pred == meta.regime
        passed = s1_passed  # CLASSIFY baselines are informational at problem level

    elif problem.question_type == QuestionType.IDENTIFY:
        pred = baseline_identify_return_map(obs.data)
        s2["return_map_pred"] = pred
        s2["correct"] = pred == meta.family
        passed = s1_passed  # IDENTIFY baselines are informational at problem level

    else:
        passed = s1_passed

    details["stage2"] = s2
    problem.stage2_results = s2
    return passed, details


def validate_and_filter_bank(problems: List[Problem]) -> List[Problem]:
    """Validate all problems and return only those that pass."""
    valid = []
    for p in problems:
        passed, _ = validate_problem(p)
        if passed:
            valid.append(p)
    return valid


def freeze_bank(problems: List[Problem], path: str, track_name: Optional[str] = None) -> None:
    """Serialize a problem bank to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    bank = []
    for p in problems:
        entry = {
            "problem_id": p.problem_id,
            "question_type": p.question_type.value,
            "observations": p.observations.tolist(),
            "question_params": p.question_params,
            "metadata": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in p.metadata.items()
            },
            "ground_truth": {
                "regime": p.ground_truth.regime,
                "family": p.ground_truth.family,
                "future_values": (
                    p.ground_truth.future_values.tolist()
                    if p.ground_truth.future_values is not None
                    else None
                ),
                "params": p.ground_truth.params,
            },
            "system_metadata": {
                "family": p.system_metadata.family,
                "params": p.system_metadata.params,
                "regime": p.system_metadata.regime,
                "h_ks": p.system_metadata.h_ks,
                "lambda_max": p.system_metadata.lambda_max,
                "tau_lambda": p.system_metadata.tau_lambda,
                "dim": p.system_metadata.dim,
                "grammar_depth": p.system_metadata.grammar_depth,
                "domain": list(p.system_metadata.domain),
            },
            "difficulty": p.difficulty,
            "stage1_passed": p.stage1_passed,
            "stage2_results": _serialize_stage2(p.stage2_results),
        }
        bank.append(entry)

    payload = {"version": "2.0", "n_problems": len(bank), "problems": bank}
    if track_name is not None:
        payload["track"] = track_name

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_and_freeze_track_banks(
    output_dir: str,
    ic_seed: int = 42,
    noise_seed: int = 123,
    n_points: int = 200,
    question_types: Optional[Iterable[QuestionType]] = None,
    noisy_noise_stds: Optional[Iterable[float]] = None,
) -> Dict[str, dict]:
    """Generate, validate, and freeze clean/noisy track banks."""
    os.makedirs(output_dir, exist_ok=True)
    tracks = generate_track_banks(
        ic_seed=ic_seed,
        noise_seed=noise_seed,
        n_points=n_points,
        question_types=question_types,
        noisy_noise_stds=noisy_noise_stds,
    )

    summary: Dict[str, dict] = {}
    for track_name, problems in tracks.items():
        valid = validate_and_filter_bank(problems)
        out_path = os.path.join(output_dir, f"mini_bank_{track_name}.json")
        freeze_bank(valid, out_path, track_name=track_name)
        summary[track_name] = {
            "path": out_path,
            "generated": len(problems),
            "validated": len(valid),
        }
    return summary


def _serialize_stage2(s2: Optional[dict]) -> Optional[dict]:
    """Make stage2 results JSON-serializable."""
    if s2 is None:
        return None
    out = {}
    for k, v in s2.items():
        if isinstance(v, dict):
            out[k] = {
                kk: _serialize_value(vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = _serialize_value(v)
    return out


def _serialize_value(v):
    """Convert a single value to JSON-serializable form."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def load_bank(path: str) -> dict:
    """Load a frozen bank from JSON."""
    with open(path) as f:
        return json.load(f)
