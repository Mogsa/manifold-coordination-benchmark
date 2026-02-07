"""Service layer for tool-first benchmark sessions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
import uuid
from pathlib import Path
from random import Random
from typing import Any, Iterable, Optional

import numpy as np

from chaosbench.agents.prompts import format_problem_prompt, format_system_prompt
from chaosbench.arena.runner import run_arena
from chaosbench.api.storage import SessionStore
from chaosbench.grammar.registry import MINI_BANK_PARAMS, create_atom
from chaosbench.problems.bank import validate_problem
from chaosbench.problems.factory import GroundTruth, Problem, QuestionType, create_problem
from chaosbench.problems.verification import verify_classify, verify_identify, verify_predict
from chaosbench.scoring.difficulty import weighted_score


VALID_QUESTION_TYPES = {"classify", "identify", "predict"}
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_sessions.db"


class SessionNotFoundError(ValueError):
    """Raised when a session id does not exist."""


class InvalidSessionStateError(ValueError):
    """Raised when an operation is not allowed for the current session state."""


class ArenaRunNotFoundError(ValueError):
    """Raised when an arena run id does not exist."""


class BenchmarkService:
    """Create, update, and score benchmark sessions."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        rng_seed: int | None = 0,
    ):
        self._store = SessionStore(db_path)
        self._rng = Random(rng_seed)

    def create_session(
        self,
        initial_points: int = 40,
        question_types: Optional[Iterable[str]] = None,
        session_label: str | None = None,
        include_system_prompt: bool = True,
        n_points: int = 200,
        noise_std: float = 0.01,
    ) -> dict[str, Any]:
        """Create a new benchmark session.

        A fresh problem is generated from the mini-bank family/parameter template.
        """
        selected_types = self._normalize_question_types(question_types)
        problem = self._sample_valid_problem(
            selected_types=selected_types,
            n_points=n_points,
            noise_std=noise_std,
        )

        max_points = int(len(problem.observations))
        revealed_points = min(max_points, max(1, int(initial_points)))
        session_id = uuid.uuid4().hex

        record = {
            "session_id": session_id,
            "status": "active",
            "session_label": (session_label or "").strip() or None,
            "problem_id": problem.problem_id,
            "question_type": problem.question_type.value,
            "family": problem.system_metadata.family,
            "params": dict(problem.system_metadata.params),
            "difficulty": float(problem.difficulty),
            "metadata": dict(problem.metadata),
            "question_params": dict(problem.question_params),
            "observations": problem.observations.astype(float).tolist(),
            "clean_data": problem.observation_detail.clean_data.astype(float).tolist(),
            "ground_truth": {
                "regime": problem.ground_truth.regime,
                "family": problem.ground_truth.family,
                "params": dict(problem.ground_truth.params or {}),
            },
            "revealed_points": revealed_points,
            "max_points": max_points,
            "data_requests": 0,
        }

        session = self._store.create_session(record)
        self._store.append_event(
            session_id,
            "session_created",
            {
                "question_type": session["question_type"],
                "problem_id": session["problem_id"],
                "revealed_points": revealed_points,
                "max_points": max_points,
            },
        )
        return self._public_session(session, include_system_prompt=include_system_prompt)

    def get_session(
        self,
        session_id: str,
        include_system_prompt: bool = True,
    ) -> dict[str, Any]:
        """Return public state for a session."""
        session = self._must_get_session(session_id)
        return self._public_session(session, include_system_prompt=include_system_prompt)

    def request_more_data(
        self,
        session_id: str,
        n_points: int = 20,
        include_system_prompt: bool = False,
    ) -> dict[str, Any]:
        """Reveal the next chunk of datapoints in an active session."""
        session = self._must_get_session(session_id)
        self._ensure_active(session)

        chunk = max(1, int(n_points))
        start = int(session["revealed_points"])
        end = min(int(session["max_points"]), start + chunk)
        observations = session["observations"]
        new_points = observations[start:end]

        updated = self._store.patch_session(
            session_id,
            revealed_points=end,
            data_requests=int(session["data_requests"]) + 1,
        )
        assert updated is not None
        self._store.append_event(
            session_id,
            "data_requested",
            {
                "requested": chunk,
                "granted": len(new_points),
                "new_revealed_points": end,
            },
        )

        response = self._public_session(
            updated,
            include_system_prompt=include_system_prompt,
        )
        response["new_points"] = new_points
        response["granted_points"] = len(new_points)
        response["lm_update_text"] = self._format_data_update_text(response, new_points)
        return response

    def submit_answer(self, session_id: str, answer: Any) -> dict[str, Any]:
        """Score and close a session."""
        session = self._must_get_session(session_id)
        self._ensure_active(session)

        score = self._score_answer(session, answer)
        updated = self._store.patch_session(
            session_id,
            status="completed",
            submitted_answer=answer,
            score=score,
        )
        assert updated is not None

        self._store.append_event(
            session_id,
            "answer_submitted",
            {
                "raw_score": score["raw_score"],
                "weighted_score": score["weighted_score"],
            },
        )

        result = self._public_session(updated, include_system_prompt=False)
        result["submitted_answer"] = answer
        result["score"] = score
        result["result_text"] = self._format_result_text(result, score)
        return result

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent session summaries for admin/debug use."""
        rows = self._store.list_sessions(limit=limit)
        summaries: list[dict[str, Any]] = []
        for s in rows:
            summaries.append(
                {
                    "session_id": s["session_id"],
                    "created_at": s["created_at"],
                    "status": s["status"],
                    "question_type": s["question_type"],
                    "family": s["family"],
                    "revealed_points": s["revealed_points"],
                    "max_points": s["max_points"],
                    "data_requests": s["data_requests"],
                    "score": s["score"],
                }
            )
        return summaries

    def run_arena_experiment(
        self,
        model: str = "gemini/gemini-2.0-flash",
        n_rounds: int = 5,
        n_solvers: int = 3,
        n_reviewers: int = 2,
        include_rounds: bool = True,
    ) -> dict[str, Any]:
        """Run the arena loop and persist round-level results."""
        run_id = uuid.uuid4().hex
        self._store.create_arena_run(
            {
                "run_id": run_id,
                "status": "running",
                "model": model,
                "n_rounds": n_rounds,
                "n_solvers": n_solvers,
                "n_reviewers": n_reviewers,
                "payload": {"note": "arena run started"},
            }
        )

        try:
            results = self._execute_arena(
                n_rounds=n_rounds,
                model=model,
                n_solvers=n_solvers,
                n_reviewers=n_reviewers,
            )
        except Exception as exc:
            failed = self._store.patch_arena_run(
                run_id,
                status="failed",
                payload={"error": str(exc)},
            )
            if failed is None:
                raise
            raise RuntimeError(f"Arena run failed: {exc}") from exc

        validated_rounds = sum(1 for r in results if r.validation_passed)
        consensus_correct = sum(1 for r in results if r.consensus_matches_math is True)

        serialized_rounds = [self._serialize_round_result(r) for r in results]
        for r, payload in zip(results, serialized_rounds):
            self._store.add_arena_round(
                run_id=run_id,
                round_number=r.round_number,
                validation_passed=r.validation_passed,
                consensus_matches_math=r.consensus_matches_math,
                payload=payload,
            )

        final_reputation = {}
        if results and results[-1].reputation_updates:
            for aid, rep in results[-1].reputation_updates.items():
                final_reputation[aid] = self._to_jsonable(asdict(rep))

        payload = {
            "validated_fraction": (
                float(validated_rounds) / float(len(results)) if results else 0.0
            ),
            "consensus_accuracy_on_validated": (
                float(consensus_correct) / float(validated_rounds)
                if validated_rounds
                else 0.0
            ),
            "final_reputation": final_reputation,
        }

        self._store.patch_arena_run(
            run_id,
            status="completed",
            total_rounds=len(results),
            validated_rounds=validated_rounds,
            consensus_correct=consensus_correct,
            payload=payload,
        )
        return self.get_arena_run(run_id, include_rounds=include_rounds)

    def get_arena_run(self, run_id: str, include_rounds: bool = True) -> dict[str, Any]:
        """Fetch a persisted arena run."""
        run = self._store.get_arena_run(run_id)
        if run is None:
            raise ArenaRunNotFoundError(f"Unknown run_id '{run_id}'")

        rounds = self._store.list_arena_rounds(run_id) if include_rounds else []
        return self._public_arena_run(run, rounds if include_rounds else None)

    def list_arena_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        """List recent arena runs."""
        rows = self._store.list_arena_runs(limit=limit)
        out: list[dict[str, Any]] = []
        for run in rows:
            out.append(self._public_arena_run(run, rounds=None))
        return out

    @staticmethod
    def _execute_arena(
        n_rounds: int,
        model: str,
        n_solvers: int,
        n_reviewers: int,
    ):
        return run_arena(
            n_rounds=n_rounds,
            model=model,
            n_solvers=n_solvers,
            n_reviewers=n_reviewers,
            verbose=False,
        )

    def _public_arena_run(
        self,
        run: dict[str, Any],
        rounds: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = run.get("payload") or {}
        summary = {
            "run_id": run["run_id"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "status": run["status"],
            "model": run["model"],
            "n_rounds": run["n_rounds"],
            "n_solvers": run["n_solvers"],
            "n_reviewers": run["n_reviewers"],
            "total_rounds": run["total_rounds"],
            "validated_rounds": run["validated_rounds"],
            "consensus_correct": run["consensus_correct"],
            "payload": payload,
        }
        if rounds is not None:
            summary["rounds"] = [r["payload"] for r in rounds]
        return summary

    def _serialize_round_result(self, result) -> dict[str, Any]:
        proposal = result.proposal
        payload: dict[str, Any] = {
            "round_number": result.round_number,
            "proposal": {
                "proposer_id": proposal.proposer_id,
                "family": proposal.family,
                "params": self._to_jsonable(proposal.params),
                "question_type": proposal.question_type,
                "reasoning": proposal.reasoning,
                "mock_answer": self._to_jsonable(proposal.mock_answer),
                "mock_confidence": proposal.mock_confidence,
            },
            "validation_passed": result.validation_passed,
            "validation_details": self._to_jsonable(result.validation_details),
            "consensus_answer": self._to_jsonable(result.consensus_answer),
            "math_ground_truth": self._to_jsonable(result.math_ground_truth),
            "consensus_matches_math": result.consensus_matches_math,
            "solves": [self._to_jsonable(asdict(s)) for s in result.solves],
            "reviews": [self._to_jsonable(asdict(r)) for r in result.reviews],
            "reputation_updates": {
                aid: self._to_jsonable(asdict(rep))
                for aid, rep in result.reputation_updates.items()
            },
        }

        if result.problem is not None:
            payload["problem"] = {
                "problem_id": result.problem.problem_id,
                "question_type": result.problem.question_type.value,
                "difficulty": float(result.problem.difficulty),
                "metadata": self._to_jsonable(result.problem.metadata),
                "question_params": self._to_jsonable(result.problem.question_params),
            }
        else:
            payload["problem"] = None
        return payload

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if is_dataclass(value):
            return cls._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(v) for v in value]
        return value

    def _sample_valid_problem(
        self,
        selected_types: set[str],
        n_points: int,
        noise_std: float,
        max_attempts: int = 40,
    ) -> Problem:
        """Sample a valid benchmark problem from mini-bank families."""
        families = list(MINI_BANK_PARAMS.keys())
        type_map = {
            "classify": QuestionType.CLASSIFY,
            "identify": QuestionType.IDENTIFY,
            "predict": QuestionType.PREDICT,
        }

        last_error: Exception | None = None
        for _ in range(max_attempts):
            family = self._rng.choice(families)
            params = dict(self._rng.choice(MINI_BANK_PARAMS[family]))
            qt_value = self._rng.choice(sorted(selected_types))
            qt = type_map[qt_value]

            ic_seed = self._rng.randint(1, 10_000_000)
            noise_seed = self._rng.randint(1, 10_000_000)

            try:
                problem = create_problem(
                    family=family,
                    params=params,
                    question_type=qt,
                    ic_seed=ic_seed,
                    noise_seed=noise_seed,
                    n_points=n_points,
                    noise_std=noise_std,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                continue

            passed, _ = validate_problem(problem)
            if passed:
                return problem

        if last_error is not None:
            raise RuntimeError(f"Failed to build session problem: {last_error}")
        raise RuntimeError("Failed to build a valid session problem")

    @staticmethod
    def _normalize_question_types(question_types: Optional[Iterable[str]]) -> set[str]:
        if question_types is None:
            return set(VALID_QUESTION_TYPES)

        selected: set[str] = set()
        for qt in question_types:
            value = str(qt).strip().lower()
            if value not in VALID_QUESTION_TYPES:
                raise ValueError(
                    f"Invalid question type '{qt}'. Valid: {sorted(VALID_QUESTION_TYPES)}"
                )
            selected.add(value)

        if not selected:
            raise ValueError("question_types cannot be empty")
        return selected

    def _must_get_session(self, session_id: str) -> dict[str, Any]:
        session = self._store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Unknown session_id '{session_id}'")
        return session

    @staticmethod
    def _ensure_active(session: dict[str, Any]) -> None:
        if session["status"] != "active":
            raise InvalidSessionStateError(
                f"Session is '{session['status']}', expected 'active'"
            )

    def _public_session(
        self,
        session: dict[str, Any],
        include_system_prompt: bool,
    ) -> dict[str, Any]:
        revealed = int(session["revealed_points"])
        max_points = int(session["max_points"])

        prompt = self._build_prompt_text(session, include_system_prompt=include_system_prompt)

        return {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "status": session["status"],
            "session_label": session["session_label"],
            "problem_id": session["problem_id"],
            "question_type": session["question_type"],
            "family_hint": "hidden",
            "difficulty": session["difficulty"],
            "revealed_points": revealed,
            "max_points": max_points,
            "remaining_points": max(0, max_points - revealed),
            "data_requests": int(session["data_requests"]),
            "tool_actions": [
                "request_more_data(n_points)",
                "submit_answer(answer)",
            ],
            "prompt": prompt,
        }

    def _build_prompt_text(
        self,
        session: dict[str, Any],
        include_system_prompt: bool,
    ) -> dict[str, str]:
        problem = self._build_visible_problem(session)
        system_prompt = format_system_prompt()
        user_prompt = format_problem_prompt(problem, task_history=None)
        protocol = (
            "TOOL PROTOCOL:\n"
            "1) If you need more observations, call request_more_data with n_points.\n"
            "2) When you are ready, call submit_answer with your final answer.\n"
            "3) Keep reasoning concise; final answer must match required format."
        )

        combined = (
            (f"SYSTEM PROMPT:\n{system_prompt}\n\n" if include_system_prompt else "")
            + f"TASK PROMPT:\n{user_prompt}\n\n{protocol}"
        )

        return {
            "system_prompt": system_prompt if include_system_prompt else "",
            "task_prompt": user_prompt,
            "tool_protocol": protocol,
            "combined_text": combined,
        }

    @staticmethod
    def _build_visible_problem(session: dict[str, Any]) -> Problem:
        revealed = int(session["revealed_points"])
        observations = np.asarray(session["observations"][:revealed], dtype=float)
        meta = dict(session["metadata"])
        meta["n_points"] = revealed

        return Problem(
            problem_id=session["problem_id"],
            question_type=QuestionType(session["question_type"]),
            observations=observations,
            question_params=dict(session["question_params"]),
            metadata=meta,
            ground_truth=GroundTruth(),
            system_metadata=None,  # type: ignore[arg-type]
            observation_detail=None,  # type: ignore[arg-type]
            difficulty=float(session["difficulty"]),
        )

    def _score_answer(self, session: dict[str, Any], answer: Any) -> dict[str, Any]:
        qt = session["question_type"]
        gt = session["ground_truth"]

        if qt == "classify":
            raw = verify_classify(answer, gt.get("regime"))
            details = {"correct_answer": gt.get("regime")}
        elif qt == "identify":
            raw = verify_identify(answer, gt.get("family"))
            details = {"correct_answer": gt.get("family")}
        else:
            predictions = self._coerce_prediction_answer(answer)
            true_future = self._compute_predict_truth(session)
            training_data = np.asarray(
                session["observations"][: int(session["revealed_points"])],
                dtype=float,
            )
            result = verify_predict(
                np.asarray(predictions, dtype=float),
                true_future,
                training_data,
            )
            raw = result["raw_score"]
            details = {
                "correct_answer": true_future.tolist(),
                "k_eff": result["k_eff"],
                "K": result["K"],
                "epsilon": result["epsilon"],
            }

        weighted = weighted_score(float(raw), float(session["difficulty"]))
        return {
            "raw_score": float(raw),
            "weighted_score": float(weighted),
            **details,
        }

    @staticmethod
    def _coerce_prediction_answer(answer: Any) -> list[float]:
        if isinstance(answer, list):
            out: list[float] = []
            for value in answer:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    continue
            return out

        if isinstance(answer, tuple):
            return BenchmarkService._coerce_prediction_answer(list(answer))

        if isinstance(answer, str):
            matches = re.findall(r"-?\d+(?:\.\d+)?", answer)
            out = []
            for match in matches:
                try:
                    out.append(float(match))
                except ValueError:
                    continue
            return out

        return []

    @staticmethod
    def _compute_predict_truth(session: dict[str, Any]) -> np.ndarray:
        horizon = int(session["question_params"].get("horizon", 10))
        clean = np.asarray(session["clean_data"], dtype=float)
        revealed = int(session["revealed_points"])
        if revealed <= 0:
            raise ValueError("Cannot score predict answer with zero revealed points")

        atom = create_atom(session["family"], dict(session["params"]))
        x = float(clean[revealed - 1])
        truth = np.empty(horizon, dtype=float)
        for i in range(horizon):
            x = atom.iterate(x)
            truth[i] = x
        return truth

    @staticmethod
    def _format_data_update_text(response: dict[str, Any], new_points: list[float]) -> str:
        if not new_points:
            return (
                "No additional datapoints are available. "
                "You can submit your final answer now."
            )

        shown = ", ".join(f"{v:.6f}" for v in new_points)
        return (
            "NEW DATA POINTS:\n"
            f"{shown}\n\n"
            f"Revealed: {response['revealed_points']}/{response['max_points']} points. "
            "You may request more data or submit your answer."
        )

    @staticmethod
    def _format_result_text(response: dict[str, Any], score: dict[str, Any]) -> str:
        lines = [
            "SESSION COMPLETE",
            f"Raw score: {score['raw_score']:.4f}",
            f"Weighted score: {score['weighted_score']:.4f}",
        ]

        if response["question_type"] == "predict":
            lines.append(f"k_eff: {score['k_eff']} / {score['K']}")
            lines.append(f"epsilon: {score['epsilon']:.6f}")

        lines.append(f"Correct answer: {score['correct_answer']}")
        return "\n".join(lines)
