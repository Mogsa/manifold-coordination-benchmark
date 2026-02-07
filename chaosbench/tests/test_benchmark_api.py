"""Tests for the tool-callable benchmark API/session service."""

from __future__ import annotations

import asyncio

import numpy as np
import httpx

from chaosbench.api.app import create_app
from chaosbench.api.service import BenchmarkService
from chaosbench.arena.protocol import Proposal, Reputation, Review, RoundResult, SolveResult
from chaosbench.grammar.registry import create_atom


def _fake_arena_results() -> list[RoundResult]:
    proposal_1 = Proposal(
        proposer_id="proposer_0",
        family="logistic",
        params={"r": 3.891},
        question_type="classify",
        reasoning="challenging but valid",
        mock_answer="chaotic",
        mock_confidence=0.8,
    )
    round_1 = RoundResult(
        round_number=1,
        proposal=proposal_1,
        validation_passed=True,
        validation_details={"stage1_passed": True},
        solves=[
            SolveResult(
                solver_id="solver_0",
                answer="chaotic",
                explanation="positive lambda",
                confidence=0.8,
            ),
            SolveResult(
                solver_id="solver_1",
                answer="periodic",
                explanation="misread the signal",
                confidence=0.4,
            ),
        ],
        reviews=[
            Review(
                reviewer_id="reviewer_0",
                question_quality=5,
                question_reasoning="good discriminator",
                answer_ratings={"solver_0": 6, "solver_1": 2},
                confidence=4,
                correct_answer="chaotic",
            )
        ],
        consensus_answer="chaotic",
        math_ground_truth="chaotic",
        consensus_matches_math=True,
        reputation_updates={
            "proposer_0": Reputation(agent_id="proposer_0", propose_score=0.8),
            "solver_0": Reputation(agent_id="solver_0", solve_score=1.0),
            "reviewer_0": Reputation(agent_id="reviewer_0", review_score=1.0),
        },
    )

    proposal_2 = Proposal(
        proposer_id="proposer_0",
        family="rotation",
        params={"omega": 0.25},
        question_type="predict",
        reasoning="harder prediction variant",
        mock_answer="0.1,0.2",
        mock_confidence=0.5,
    )
    round_2 = RoundResult(
        round_number=2,
        proposal=proposal_2,
        validation_passed=False,
        validation_details={"stage1_passed": False, "reason": "failed gate"},
        consensus_answer=None,
        math_ground_truth=None,
        consensus_matches_math=None,
        reputation_updates={
            "proposer_0": Reputation(agent_id="proposer_0", propose_score=0.4),
        },
    )
    return [round_1, round_2]


def test_service_classify_session_lifecycle(tmp_path):
    db_path = tmp_path / "sessions.db"
    service = BenchmarkService(db_path=db_path, rng_seed=11)

    created = service.create_session(
        initial_points=12,
        question_types=["classify"],
        include_system_prompt=True,
    )

    assert created["status"] == "active"
    assert created["question_type"] == "classify"
    assert created["revealed_points"] == 12
    assert "SYSTEM PROMPT" in created["prompt"]["combined_text"]

    session_id = created["session_id"]
    expanded = service.request_more_data(session_id, n_points=7)
    assert expanded["revealed_points"] == 19
    assert expanded["granted_points"] == 7
    assert len(expanded["new_points"]) == 7

    hidden = service._store.get_session(session_id)
    assert hidden is not None
    answer = hidden["ground_truth"]["regime"]

    result = service.submit_answer(session_id, answer)
    assert result["status"] == "completed"
    assert result["score"]["raw_score"] == 1.0
    assert "SESSION COMPLETE" in result["result_text"]


def test_service_predict_scoring_with_exact_truth(tmp_path):
    db_path = tmp_path / "sessions_predict.db"
    service = BenchmarkService(db_path=db_path, rng_seed=23)

    created = service.create_session(
        initial_points=55,
        question_types=["predict"],
        include_system_prompt=False,
    )
    session_id = created["session_id"]

    hidden = service._store.get_session(session_id)
    assert hidden is not None

    horizon = int(hidden["question_params"]["horizon"])
    clean = np.asarray(hidden["clean_data"], dtype=float)
    revealed = int(hidden["revealed_points"])

    atom = create_atom(hidden["family"], hidden["params"])
    x = float(clean[revealed - 1])
    preds = []
    for _ in range(horizon):
        x = atom.iterate(x)
        preds.append(float(x))

    result = service.submit_answer(session_id, preds)
    assert result["status"] == "completed"
    assert result["score"]["raw_score"] == 1.0
    assert result["score"]["k_eff"] == horizon


def test_api_endpoints_round_trip(tmp_path):
    app = create_app(db_path=tmp_path / "api.db", seed=31)

    async def _exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/v1/sessions",
                json={
                    "initial_points": 10,
                    "question_types": ["identify"],
                    "include_system_prompt": True,
                },
            )
            assert created.status_code == 200
            payload = created.json()
            assert payload["status"] == "active"
            assert payload["question_type"] == "identify"

            session_id = payload["session_id"]

            data_resp = await client.post(
                f"/v1/sessions/{session_id}/request-data",
                json={"n_points": 5},
            )
            assert data_resp.status_code == 200
            data_payload = data_resp.json()
            assert data_payload["granted_points"] == 5
            assert len(data_payload["new_points"]) == 5

            submit = await client.post(
                f"/v1/sessions/{session_id}/submit-answer",
                json={"answer": "logistic"},
            )
            assert submit.status_code == 200
            submit_payload = submit.json()
            assert submit_payload["status"] == "completed"
            assert "raw_score" in submit_payload["score"]

            closed_data_req = await client.post(
                f"/v1/sessions/{session_id}/request-data",
                json={"n_points": 1},
            )
            assert closed_data_req.status_code == 400

    asyncio.run(_exercise())


def test_service_arena_run_persistence(tmp_path):
    service = BenchmarkService(db_path=tmp_path / "arena.db", rng_seed=7)
    service._execute_arena = lambda **kwargs: _fake_arena_results()  # type: ignore[method-assign]

    run = service.run_arena_experiment(
        model="mock/model",
        n_rounds=2,
        n_solvers=3,
        n_reviewers=2,
        include_rounds=True,
    )
    assert run["status"] == "completed"
    assert run["model"] == "mock/model"
    assert run["total_rounds"] == 2
    assert run["validated_rounds"] == 1
    assert run["consensus_correct"] == 1
    assert len(run["rounds"]) == 2

    loaded = service.get_arena_run(run["run_id"], include_rounds=True)
    assert loaded["run_id"] == run["run_id"]
    assert loaded["total_rounds"] == 2
    assert len(loaded["rounds"]) == 2

    listed = service.list_arena_runs(limit=10)
    assert listed
    assert listed[0]["run_id"] == run["run_id"]


def test_api_arena_endpoints(tmp_path):
    app = create_app(db_path=tmp_path / "arena_api.db", seed=41)
    app.state.service._execute_arena = lambda **kwargs: _fake_arena_results()

    async def _exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            create_resp = await client.post(
                "/v1/arena/runs",
                json={
                    "model": "mock/model",
                    "n_rounds": 2,
                    "n_solvers": 3,
                    "n_reviewers": 2,
                    "include_rounds": True,
                },
            )
            assert create_resp.status_code == 200
            created = create_resp.json()
            assert created["status"] == "completed"
            assert created["total_rounds"] == 2
            run_id = created["run_id"]

            list_resp = await client.get("/v1/arena/runs?limit=5")
            assert list_resp.status_code == 200
            listed = list_resp.json()
            assert listed
            assert listed[0]["run_id"] == run_id

            get_resp = await client.get(f"/v1/arena/runs/{run_id}?include_rounds=true")
            assert get_resp.status_code == 200
            fetched = get_resp.json()
            assert fetched["run_id"] == run_id
            assert len(fetched["rounds"]) == 2

    asyncio.run(_exercise())
