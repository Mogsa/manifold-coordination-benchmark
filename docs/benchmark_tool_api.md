# ChaosBench Tool API (No Copy/Paste Workflow)

This API is designed so LLM tools/connectors can run benchmark sessions end-to-end without manual copy/paste.

## Run locally

```bash
python -m chaosbench.api
```

Default URL: `http://127.0.0.1:8000`

OpenAPI schema:
- `GET /openapi.json`
- Swagger UI: `GET /docs`

## Session lifecycle endpoints

1. `POST /v1/sessions`
- Starts a benchmark session.
- Returns an LM-friendly prompt bundle and tool protocol.

Request example:

```json
{
  "initial_points": 40,
  "question_types": ["classify", "identify", "predict"],
  "include_system_prompt": true,
  "n_points": 200,
  "noise_std": 0.01
}
```

2. `GET /v1/sessions/{session_id}`
- Returns current public session state and current prompt text.

3. `POST /v1/sessions/{session_id}/request-data`
- Reveals additional datapoints.

Request example:

```json
{
  "n_points": 20
}
```

4. `POST /v1/sessions/{session_id}/submit-answer`
- Submits final answer, scores mathematically, and closes session.

Request example:

```json
{
  "answer": "chaotic"
}
```

5. `GET /v1/sessions`
- Lists recent sessions for audit/admin checks.

## Arena endpoints (propose -> solve -> review loop)

These endpoints run the existing ChaosBench arena pipeline using the same model
for proposer/solver/reviewer roles (current arena behavior).

1. `POST /v1/arena/runs`
- Starts and executes an arena run.
- Persists run + per-round payloads in SQLite.

Request example:

```json
{
  "model": "gemini/gemini-2.0-flash",
  "n_rounds": 5,
  "n_solvers": 3,
  "n_reviewers": 2,
  "include_rounds": true
}
```

2. `GET /v1/arena/runs`
- Lists recent arena runs.

3. `GET /v1/arena/runs/{run_id}`
- Returns one run, optionally including full round payloads.

## How to wire to LLM tools/connectors

Use three tool actions mapped to endpoints:

1. `start_session`
- calls `POST /v1/sessions`
- returns `session_id` and `prompt.combined_text`

2. `request_more_data`
- calls `POST /v1/sessions/{session_id}/request-data`
- returns `new_points` and updated prompt context

3. `submit_answer`
- calls `POST /v1/sessions/{session_id}/submit-answer`
- returns score and correct answer payload

This pattern works for:
- Claude remote MCP connector setups
- ChatGPT actions/connectors

## Ubuntu 24/7 hosting checklist

1. Run API under `systemd`.
2. Put reverse proxy in front (Caddy or Nginx).
3. Expose public HTTPS endpoint (Cloudflare Tunnel recommended for home NAT).
4. Restrict access with auth token/header at proxy layer.
5. Back up `chaosbench/data/benchmark_sessions.db` daily.

## Notes

- Sessions are persisted in SQLite (`chaosbench/data/benchmark_sessions.db`).
- Arena runs are persisted in SQLite tables `arena_runs` and `arena_rounds`.
- Predict scoring uses the existing ChaosBench verifier (`k_eff` contract).
- All scoring is deterministic for a given session state + submitted answer.
