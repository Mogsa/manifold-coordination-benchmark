"""SQLite persistence for benchmark API sessions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_JSON_FIELDS = {
    "metadata",
    "question_params",
    "observations",
    "clean_data",
    "ground_truth",
    "params",
    "submitted_answer",
    "score",
}


class SessionStore:
    """Persistent storage for API sessions and audit events."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_label TEXT,
                    problem_id TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    family TEXT NOT NULL,
                    params TEXT NOT NULL,
                    difficulty REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    question_params TEXT NOT NULL,
                    observations TEXT NOT NULL,
                    clean_data TEXT NOT NULL,
                    ground_truth TEXT NOT NULL,
                    revealed_points INTEGER NOT NULL,
                    max_points INTEGER NOT NULL,
                    data_requests INTEGER NOT NULL DEFAULT 0,
                    submitted_answer TEXT,
                    score TEXT
                );

                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS arena_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    n_rounds INTEGER NOT NULL,
                    n_solvers INTEGER NOT NULL,
                    n_reviewers INTEGER NOT NULL,
                    total_rounds INTEGER NOT NULL DEFAULT 0,
                    validated_rounds INTEGER NOT NULL DEFAULT 0,
                    consensus_correct INTEGER NOT NULL DEFAULT 0,
                    payload TEXT
                );

                CREATE TABLE IF NOT EXISTS arena_rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    validation_passed INTEGER NOT NULL,
                    consensus_matches_math INTEGER,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES arena_runs(run_id)
                );
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _from_json(value: Any) -> Any:
        if value is None:
            return None
        return json.loads(value)

    def create_session(self, record: dict[str, Any]) -> dict[str, Any]:
        now = self._now_iso()
        payload = {
            "session_id": record["session_id"],
            "created_at": now,
            "updated_at": now,
            "status": record["status"],
            "session_label": record.get("session_label"),
            "problem_id": record["problem_id"],
            "question_type": record["question_type"],
            "family": record["family"],
            "params": self._to_json(record["params"]),
            "difficulty": float(record["difficulty"]),
            "metadata": self._to_json(record["metadata"]),
            "question_params": self._to_json(record["question_params"]),
            "observations": self._to_json(record["observations"]),
            "clean_data": self._to_json(record["clean_data"]),
            "ground_truth": self._to_json(record["ground_truth"]),
            "revealed_points": int(record["revealed_points"]),
            "max_points": int(record["max_points"]),
            "data_requests": int(record.get("data_requests", 0)),
            "submitted_answer": None,
            "score": None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, updated_at, status, session_label,
                    problem_id, question_type, family, params,
                    difficulty, metadata, question_params, observations,
                    clean_data, ground_truth, revealed_points, max_points,
                    data_requests, submitted_answer, score
                ) VALUES (
                    :session_id, :created_at, :updated_at, :status, :session_label,
                    :problem_id, :question_type, :family, :params,
                    :difficulty, :metadata, :question_params, :observations,
                    :clean_data, :ground_truth, :revealed_points, :max_points,
                    :data_requests, :submitted_answer, :score
                )
                """,
                payload,
            )
        return self.get_session(payload["session_id"])

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_events (session_id, ts, event_type, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    self._now_iso(),
                    event_type,
                    self._to_json(payload),
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None

        out = dict(row)
        for field in _JSON_FIELDS:
            out[field] = self._from_json(out[field])
        return out

    def patch_session(self, session_id: str, **updates: Any) -> dict[str, Any] | None:
        if not updates:
            return self.get_session(session_id)

        payload = {}
        for key, value in updates.items():
            if key in _JSON_FIELDS and value is not None:
                payload[key] = self._to_json(value)
            else:
                payload[key] = value

        payload["updated_at"] = self._now_iso()
        assignments = ", ".join(f"{k} = :{k}" for k in payload)
        payload["session_id"] = session_id

        with self._connect() as conn:
            conn.execute(
                f"UPDATE sessions SET {assignments} WHERE session_id = :session_id",
                payload,
            )
        return self.get_session(session_id)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()

        sessions: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in _JSON_FIELDS:
                item[field] = self._from_json(item[field])
            sessions.append(item)
        return sessions

    def create_arena_run(self, record: dict[str, Any]) -> dict[str, Any]:
        now = self._now_iso()
        payload = {
            "run_id": record["run_id"],
            "created_at": now,
            "updated_at": now,
            "status": record.get("status", "running"),
            "model": record["model"],
            "n_rounds": int(record["n_rounds"]),
            "n_solvers": int(record["n_solvers"]),
            "n_reviewers": int(record["n_reviewers"]),
            "total_rounds": int(record.get("total_rounds", 0)),
            "validated_rounds": int(record.get("validated_rounds", 0)),
            "consensus_correct": int(record.get("consensus_correct", 0)),
            "payload": self._to_json(record.get("payload", {})),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO arena_runs (
                    run_id, created_at, updated_at, status, model, n_rounds,
                    n_solvers, n_reviewers, total_rounds, validated_rounds,
                    consensus_correct, payload
                ) VALUES (
                    :run_id, :created_at, :updated_at, :status, :model, :n_rounds,
                    :n_solvers, :n_reviewers, :total_rounds, :validated_rounds,
                    :consensus_correct, :payload
                )
                """,
                payload,
            )
        return self.get_arena_run(payload["run_id"])

    def patch_arena_run(self, run_id: str, **updates: Any) -> dict[str, Any] | None:
        if not updates:
            return self.get_arena_run(run_id)

        payload: dict[str, Any] = {}
        for key, value in updates.items():
            if key == "payload" and value is not None:
                payload[key] = self._to_json(value)
            else:
                payload[key] = value

        payload["updated_at"] = self._now_iso()
        assignments = ", ".join(f"{k} = :{k}" for k in payload)
        payload["run_id"] = run_id

        with self._connect() as conn:
            conn.execute(
                f"UPDATE arena_runs SET {assignments} WHERE run_id = :run_id",
                payload,
            )
        return self.get_arena_run(run_id)

    def add_arena_round(
        self,
        run_id: str,
        round_number: int,
        validation_passed: bool,
        consensus_matches_math: bool | None,
        payload: dict[str, Any],
    ) -> None:
        consensus_val: int | None
        if consensus_matches_math is None:
            consensus_val = None
        else:
            consensus_val = 1 if consensus_matches_math else 0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO arena_rounds (
                    run_id, round_number, validation_passed,
                    consensus_matches_math, payload
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(round_number),
                    1 if validation_passed else 0,
                    consensus_val,
                    self._to_json(payload),
                ),
            )

    def get_arena_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM arena_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None

        out = dict(row)
        out["payload"] = self._from_json(out["payload"])
        return out

    def list_arena_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM arena_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()

        runs: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = self._from_json(item["payload"])
            runs.append(item)
        return runs

    def list_arena_rounds(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM arena_rounds
                WHERE run_id = ?
                ORDER BY round_number ASC
                """,
                (run_id,),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = self._from_json(item["payload"])
            item["validation_passed"] = bool(item["validation_passed"])
            if item["consensus_matches_math"] is not None:
                item["consensus_matches_math"] = bool(item["consensus_matches_math"])
            out.append(item)
        return out
