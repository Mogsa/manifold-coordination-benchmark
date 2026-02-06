"""Parse LLM output into structured answers for ChaosBench v2.

Key principle: never crash on bad output. Return empty/default and let
verification score it as 0.
"""

from __future__ import annotations

import re

from chaosbench.agents.prompts import MVP_FAMILIES, MVP_REGIMES


def parse_classify(response: str) -> str:
    """Extract regime label from LLM response.

    Scans for known regime labels. Prefers ANSWER: line, falls back to
    last match in response body.
    """
    response_lower = response.lower()

    # Check longest labels first so "quasiperiodic" beats "periodic"
    labels_by_length = sorted(MVP_REGIMES, key=len, reverse=True)

    # Try ANSWER: line first
    answer_match = re.search(r"answer:\s*(.+)", response_lower)
    if answer_match:
        answer_line = answer_match.group(1).strip()
        for label in labels_by_length:
            if label in answer_line:
                return label

    # Fallback: last occurrence of any known label
    last_label = ""
    last_pos = -1
    for label in labels_by_length:
        pos = response_lower.rfind(label)
        if pos > last_pos:
            last_pos = pos
            last_label = label

    return last_label


def parse_identify(response: str) -> str:
    """Extract family name from LLM response.

    Same strategy as parse_classify: ANSWER: line first, then last match.
    """
    response_lower = response.lower()

    # Try ANSWER: line first
    answer_match = re.search(r"answer:\s*(.+)", response_lower)
    if answer_match:
        answer_line = answer_match.group(1).strip()
        for family in MVP_FAMILIES:
            if family in answer_line:
                return family

    # Fallback: last occurrence of any known family
    last_family = ""
    last_pos = -1
    for family in MVP_FAMILIES:
        pos = response_lower.rfind(family)
        if pos > last_pos:
            last_pos = pos
            last_family = family

    return last_family


def parse_predict(response: str, K: int) -> list[float]:
    """Extract K predicted numbers from LLM response.

    Strategy:
    1. Try ANSWER: line with comma-separated numbers
    2. Try any bracketed list of numbers
    3. Fall back to all floats found in response
    Pad/truncate to K. Return empty list on total failure.
    """
    # Strategy 1: ANSWER: line
    answer_match = re.search(r"answer:\s*(.+)", response, re.IGNORECASE)
    if answer_match:
        nums = _extract_floats(answer_match.group(1))
        if nums:
            return _pad_truncate(nums, K)

    # Strategy 2: bracketed list
    bracket_match = re.search(r"\[([^\]]+)\]", response)
    if bracket_match:
        nums = _extract_floats(bracket_match.group(1))
        if nums:
            return _pad_truncate(nums, K)

    # Strategy 3: all floats in response
    nums = _extract_floats(response)
    if nums:
        return _pad_truncate(nums, K)

    return []


def parse_confidence(response: str) -> float:
    """Extract confidence value from response, default 0.5."""
    match = re.search(r"confidence:\s*(-?[\d.]+)", response, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return 0.5


def parse_response(response: str, question_type: str, question_params: dict) -> object:
    """Dispatch to the appropriate parser based on question type."""
    if question_type == "classify":
        return parse_classify(response)
    if question_type == "identify":
        return parse_identify(response)
    if question_type == "predict":
        K = question_params.get("horizon", 10)
        return parse_predict(response, K)
    return ""


def _extract_floats(text: str) -> list[float]:
    """Extract all float-like numbers from a string."""
    # Match integers and decimals, including negatives
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    result = []
    for m in matches:
        try:
            result.append(float(m))
        except ValueError:
            continue
    return result


def _pad_truncate(values: list[float], K: int) -> list[float]:
    """Pad with last value or truncate to exactly K elements."""
    if len(values) >= K:
        return values[:K]
    if not values:
        return []
    # Pad with last value
    return values + [values[-1]] * (K - len(values))
