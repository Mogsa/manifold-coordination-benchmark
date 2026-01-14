"""
LLM utilities for the Manifold Benchmark suite.

This module provides shared LLM functionality using LiteLLM,
supporting 100+ LLM providers (OpenAI, Anthropic, Google, etc.).
"""

import os
import re
import time
from typing import List, Dict, Optional, Callable
from litellm import completion


# Default configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30


def set_api_key_for_model(model: str, api_key: str) -> None:
    """
    Set the appropriate environment variable for a model.

    Args:
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus', 'gemini/gemini-pro')
        api_key: API key to set
    """
    if 'gemini' in model.lower():
        os.environ['GEMINI_API_KEY'] = api_key
    elif 'claude' in model.lower():
        os.environ['ANTHROPIC_API_KEY'] = api_key
    elif 'gpt' in model.lower() or 'o1' in model.lower():
        os.environ['OPENAI_API_KEY'] = api_key


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 150,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> str:
    """
    Call an LLM using LiteLLM with retry logic and exponential backoff.

    Args:
        model: Model identifier (e.g., 'gpt-4', 'claude-3-5-sonnet-20241022')
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        on_retry: Optional callback called on retry with (exception, retry_count)

    Returns:
        Response text from the LLM

    Raises:
        Exception: If all retries are exhausted
    """
    retry_count = 0

    while True:
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content

        except Exception as e:
            if retry_count < max_retries:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** retry_count
                if on_retry:
                    on_retry(e, retry_count)
                else:
                    print(f"API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
            else:
                print(f"API failed after {max_retries} retries: {e}")
                raise


def parse_coordinate(
    response: str,
    domain_size: float = 10.0,
    coord_name: str = 'x'
) -> float:
    """
    Extract a coordinate value from an LLM response.

    Parsing priority:
    1. Look for "MY_POSITION: X" pattern (most reliable)
    2. Look for "x = X" or "y = X" patterns
    3. Look for "position: X" or "move to X" patterns
    4. Fall back to last valid number in range [0, domain_size]
    5. Last resort: last number (clamped)

    Args:
        response: Raw text response from LLM
        domain_size: Maximum valid coordinate value
        coord_name: Name of coordinate ('x' or 'y')

    Returns:
        Parsed coordinate value in [0, domain_size]

    Raises:
        ValueError: If no numbers found in response
    """
    def safe_float(s):
        try:
            s = s.rstrip('.')
            return float(s)
        except (ValueError, AttributeError):
            return None

    # Priority 1: Look for MY_POSITION: pattern
    my_pos_match = re.search(r'MY_POSITION:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if my_pos_match:
        value = safe_float(my_pos_match.group(1))
        if value is not None:
            return max(0.0, min(domain_size, value))

    # Priority 2: Look for explicit coordinate assignments
    coord_match = re.search(rf'{coord_name}\s*=\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if coord_match:
        value = safe_float(coord_match.group(1))
        if value is not None and 0 <= value <= domain_size:
            return value

    # Priority 3: Look for "move to X" or "position X" patterns
    move_match = re.search(
        r'(?:move to|position|choose|select|go to)\s*(\d+(?:\.\d+)?)',
        response, re.IGNORECASE
    )
    if move_match:
        value = safe_float(move_match.group(1))
        if value is not None and 0 <= value <= domain_size:
            return value

    # Priority 4: Find all properly formatted numbers
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)

    if not all_numbers:
        raise ValueError(f"Could not parse coordinate from: {response[:200]}")

    # Convert to floats
    float_numbers = []
    for n in all_numbers:
        val = safe_float(n)
        if val is not None:
            float_numbers.append(val)

    if not float_numbers:
        raise ValueError(f"Could not parse coordinate from: {response[:200]}")

    # Filter to valid domain range
    valid_numbers = [n for n in float_numbers if 0 <= n <= domain_size]

    # Prefer numbers that look like positions (not gradients)
    position_candidates = [n for n in valid_numbers if n >= 1.0 or n == 0.0]

    if position_candidates:
        return position_candidates[-1]
    elif valid_numbers:
        return valid_numbers[-1]
    else:
        # Last resort - clamp
        return max(0.0, min(domain_size, float_numbers[-1]))


def parse_position_list(
    response: str,
    n_positions: int = 20,
    domain_size: float = 10.0
) -> List[float]:
    """
    Parse a list of positions from an LLM response.

    Looks for patterns like:
    - POSITIONS: [1.0, 2.0, 3.0, ...]
    - PREDICTIONS: [1.0, 2.0, 3.0, ...]

    Args:
        response: Raw text response from LLM
        n_positions: Expected number of positions
        domain_size: Maximum valid coordinate value

    Returns:
        List of parsed positions, clamped to [0, domain_size]

    Raises:
        ValueError: If unable to parse enough positions
    """
    # Look for list pattern
    list_match = re.search(
        r'(?:POSITIONS|PREDICTIONS):\s*\[([^\]]+)\]',
        response, re.IGNORECASE
    )

    if list_match:
        numbers_str = list_match.group(1)
        # Extract all numbers from the list
        numbers = re.findall(r'(\d+(?:\.\d+)?)', numbers_str)
        positions = [max(0.0, min(domain_size, float(n))) for n in numbers]

        if len(positions) >= n_positions:
            return positions[:n_positions]

    # Fallback: find all numbers in response
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
    positions = []

    for n in all_numbers:
        try:
            val = float(n)
            if 0 <= val <= domain_size:
                positions.append(val)
        except ValueError:
            continue

    if len(positions) >= n_positions:
        return positions[:n_positions]

    raise ValueError(
        f"Could not parse {n_positions} positions from response. "
        f"Found {len(positions)}: {positions[:5]}..."
    )
