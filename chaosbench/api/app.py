"""FastAPI app exposing tool-callable benchmark endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chaosbench.api.service import (
    ArenaRunNotFoundError,
    BenchmarkService,
    DEFAULT_DB_PATH,
    InvalidSessionStateError,
    SessionNotFoundError,
)


class CreateSessionRequest(BaseModel):
    initial_points: int = Field(default=40, ge=1, le=500)
    question_types: list[str] | None = Field(default=None)
    session_label: str | None = Field(default=None, max_length=120)
    include_system_prompt: bool = True
    n_points: int = Field(default=200, ge=20, le=1000)
    noise_std: float = Field(default=0.01, ge=0.0, le=1.0)


class RequestDataRequest(BaseModel):
    n_points: int = Field(default=20, ge=1, le=500)
    include_system_prompt: bool = False


class SubmitAnswerRequest(BaseModel):
    answer: Any


class ArenaRunRequest(BaseModel):
    model: str = "gemini/gemini-2.0-flash"
    n_rounds: int = Field(default=5, ge=1, le=100)
    n_solvers: int = Field(default=3, ge=1, le=20)
    n_reviewers: int = Field(default=2, ge=1, le=20)
    include_rounds: bool = True


def create_app(
    db_path: str | Path | None = None,
    seed: int | None = 0,
) -> FastAPI:
    """Create a configured FastAPI app instance."""
    service = BenchmarkService(db_path=db_path or DEFAULT_DB_PATH, rng_seed=seed)

    app = FastAPI(
        title="ChaosBench Tool API",
        version="0.1.0",
        summary="Tool-first benchmark API for connector/action integration",
    )
    app.state.service = service

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "chaosbench-tool-api",
            "docs": "/docs",
            "health": "/healthz",
        }

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/sessions")
    def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        try:
            return app.state.service.create_session(
                initial_points=req.initial_points,
                question_types=req.question_types,
                session_label=req.session_label,
                include_system_prompt=req.include_system_prompt,
                n_points=req.n_points,
                noise_std=req.noise_std,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}")
    def get_session(
        session_id: str,
        include_system_prompt: bool = True,
    ) -> dict[str, Any]:
        try:
            return app.state.service.get_session(
                session_id,
                include_system_prompt=include_system_prompt,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/request-data")
    def request_data(session_id: str, req: RequestDataRequest) -> dict[str, Any]:
        try:
            return app.state.service.request_more_data(
                session_id,
                n_points=req.n_points,
                include_system_prompt=req.include_system_prompt,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidSessionStateError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/submit-answer")
    def submit_answer(session_id: str, req: SubmitAnswerRequest) -> dict[str, Any]:
        try:
            return app.state.service.submit_answer(session_id, req.answer)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidSessionStateError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions")
    def list_sessions(limit: int = 25) -> list[dict[str, Any]]:
        return app.state.service.list_sessions(limit=limit)

    @app.post("/v1/arena/runs")
    def create_arena_run(req: ArenaRunRequest) -> dict[str, Any]:
        try:
            return app.state.service.run_arena_experiment(
                model=req.model,
                n_rounds=req.n_rounds,
                n_solvers=req.n_solvers,
                n_reviewers=req.n_reviewers,
                include_rounds=req.include_rounds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/arena/runs")
    def list_arena_runs(limit: int = 25) -> list[dict[str, Any]]:
        return app.state.service.list_arena_runs(limit=limit)

    @app.get("/v1/arena/runs/{run_id}")
    def get_arena_run(run_id: str, include_rounds: bool = True) -> dict[str, Any]:
        try:
            return app.state.service.get_arena_run(
                run_id,
                include_rounds=include_rounds,
            )
        except ArenaRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
