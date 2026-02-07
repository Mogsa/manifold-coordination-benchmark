"""Fail-safe parsers for arena LLM outputs.

Key principle (same as agents/parsing.py): never crash on bad output.
Return defaults and let scoring handle garbage.
"""

from __future__ import annotations

import json
import re
from typing import Any

from chaosbench.agents.parsing import parse_confidence, parse_response
from chaosbench.arena.protocol import Proposal, Review, SolveResult
from chaosbench.grammar.registry import ATOM_REGISTRY


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text."""
    # Try the whole text first
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try nested brace matching first (handles {"params": {"r": 3.5}})
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = None

    # Fallback: flat { ... } block (no nested braces)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    """Clamp a value to [lo, hi], returning default on failure."""
    try:
        v = int(value)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _extract_json_array(text: str) -> list[dict] | None:
    """Try to extract a JSON array of objects from text."""
    # Try the whole text first
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Find bracketed array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_proposals(response: str, proposer_id: str = "proposer") -> list[Proposal]:
    """Parse a proposer's response into 1-3 Proposals.

    Tries JSON array first, then multiple JSON objects, then single proposal.
    Never returns empty — always falls back to at least one default.
    """
    # Try as JSON array
    items = _extract_json_array(response)
    if items and len(items) >= 1:
        proposals = []
        for i, item in enumerate(items[:3]):
            # Re-serialize each item and parse individually
            try:
                text = json.dumps(item)
                proposals.append(parse_proposal(text, proposer_id))
            except Exception:
                continue
        if proposals:
            return proposals

    # Try to find multiple JSON objects in the text
    proposals = []
    depth = 0
    start = None
    for i, c in enumerate(response):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = response[start : i + 1]
                try:
                    json.loads(chunk)  # validate it's valid JSON
                    proposals.append(parse_proposal(chunk, proposer_id))
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None
    if len(proposals) > 1:
        return proposals[:3]

    # Fallback: single proposal
    return [parse_proposal(response, proposer_id)]


def parse_proposal(response: str, proposer_id: str = "proposer") -> Proposal:
    """Parse a proposer's JSON response into a Proposal.

    Falls back to defaults on failure: logistic, r=3.5, classify.
    """
    data = _extract_json(response)

    if data and isinstance(data, dict):
        family = str(data.get("family", "logistic")).lower().strip()
        if family not in ATOM_REGISTRY:
            family = "logistic"

        # Parse params
        raw_params = data.get("params", {})
        if not isinstance(raw_params, dict):
            raw_params = {}
        params = {}
        spec = ATOM_REGISTRY[family]
        for pname, (lo, hi) in spec.param_ranges.items():
            val = raw_params.get(pname)
            if val is not None:
                try:
                    val = float(val)
                    params[pname] = max(lo, min(hi, val))
                except (TypeError, ValueError):
                    params[pname] = (lo + hi) / 2
            else:
                params[pname] = (lo + hi) / 2

        qt = str(data.get("question_type", "classify")).lower().strip()
        if qt not in ("classify", "identify", "predict"):
            qt = "classify"

        reasoning = str(data.get("reasoning", ""))[:500]
        mock_answer = str(data.get("mock_answer", ""))
        try:
            mock_confidence = max(0.0, min(1.0, float(data.get("mock_confidence", 0.5))))
        except (TypeError, ValueError):
            mock_confidence = 0.5

        return Proposal(
            proposer_id=proposer_id,
            family=family,
            params=params,
            question_type=qt,
            reasoning=reasoning,
            mock_answer=mock_answer,
            mock_confidence=mock_confidence,
        )

    # Total failure — return safe defaults
    return Proposal(
        proposer_id=proposer_id,
        family="logistic",
        params={"r": 3.5},
        question_type="classify",
        reasoning="(parse failure — using defaults)",
        mock_answer="periodic",
        mock_confidence=0.5,
    )


def parse_solve_result(
    response: str,
    solver_id: str,
    question_type: str,
    question_params: dict,
) -> SolveResult:
    """Parse a solver's response into a SolveResult.

    Delegates answer parsing to agents/parsing.parse_response().
    """
    answer = parse_response(response, question_type, question_params)
    confidence = parse_confidence(response)

    # Extract explanation
    explanation = ""
    exp_match = re.search(
        r"explanation:\s*(.+?)(?:\n\n|\nconfidence:|\nANSWER:|\Z)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if exp_match:
        explanation = exp_match.group(1).strip()[:500]

    return SolveResult(
        solver_id=solver_id,
        answer=answer,
        explanation=explanation,
        confidence=confidence,
    )


def parse_review(
    response: str,
    reviewer_id: str,
    solver_ids: list[str],
) -> Review:
    """Parse a reviewer's JSON response into a Review.

    Falls back to neutral defaults: quality=3, ratings=3, confidence=2.
    """
    data = _extract_json(response)

    if data and isinstance(data, dict):
        quality = _clamp(data.get("question_quality"), 1, 6, 3)
        reasoning = str(data.get("question_reasoning", ""))[:500]
        confidence = _clamp(data.get("confidence"), 1, 5, 2)

        raw_ratings = data.get("answer_ratings", {})
        if not isinstance(raw_ratings, dict):
            raw_ratings = {}

        ratings = {}
        for sid in solver_ids:
            ratings[sid] = _clamp(raw_ratings.get(sid), 1, 6, 3)

        correct_answer = data.get("correct_answer")

        return Review(
            reviewer_id=reviewer_id,
            question_quality=quality,
            question_reasoning=reasoning,
            answer_ratings=ratings,
            confidence=confidence,
            correct_answer=correct_answer,
        )

    # Try regex fallback
    quality = 3
    confidence = 2
    q_match = re.search(r"quality:\s*(\d)", response, re.IGNORECASE)
    if q_match:
        quality = _clamp(q_match.group(1), 1, 6, 3)
    c_match = re.search(r"confidence:\s*(\d)", response, re.IGNORECASE)
    if c_match:
        confidence = _clamp(c_match.group(1), 1, 5, 2)

    ratings = {sid: 3 for sid in solver_ids}

    return Review(
        reviewer_id=reviewer_id,
        question_quality=quality,
        question_reasoning="(parse failure)",
        answer_ratings=ratings,
        confidence=confidence,
    )
