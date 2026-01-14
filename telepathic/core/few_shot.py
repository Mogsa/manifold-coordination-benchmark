"""
Few-shot example generation for the Telepathic benchmark.

Defines the examples shown to agents in few-shot condition.
These examples teach the token→function mappings.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .sampling import SAMPLE_X_VALUES, TEST_X_VALUES


@dataclass
class PrimitiveExample:
    """A few-shot example for a primitive function."""
    token: str
    function_name: str
    samples: List[Tuple[float, float]]  # (x, y) pairs
    test_examples: List[Tuple[float, float]]  # For Doer


@dataclass
class CompositionExample:
    """A few-shot example for a composition."""
    message: str  # e.g., "α-γ"
    tokens: List[str]
    description: str  # e.g., "sin(square(x))"
    samples: List[Tuple[float, float]]
    test_examples: List[Tuple[float, float]]


# =============================================================================
# FEW-SHOT PRIMITIVES (5 of 10)
# These are the token meanings taught in the few-shot prompt
# =============================================================================

def _compute_primitive_examples() -> Dict[str, PrimitiveExample]:
    """Compute the primitive few-shot examples with actual values."""

    primitives = {
        "α": {
            "name": "sin",
            "func": math.sin,
        },
        "β": {
            "name": "cos",
            "func": math.cos,
        },
        "γ": {
            "name": "square",
            "func": lambda x: x ** 2,
        },
        "δ": {
            "name": "abs",
            "func": abs,
        },
        "ε": {
            "name": "neg",
            "func": lambda x: -x,
        },
    }

    examples = {}
    for token, info in primitives.items():
        func = info["func"]
        samples = [(x, func(x)) for x in SAMPLE_X_VALUES]
        tests = [(x, func(x)) for x in TEST_X_VALUES]

        examples[token] = PrimitiveExample(
            token=token,
            function_name=info["name"],
            samples=samples,
            test_examples=tests
        )

    return examples


FEW_SHOT_PRIMITIVES: Dict[str, PrimitiveExample] = _compute_primitive_examples()


# =============================================================================
# FEW-SHOT COMPOSITIONS (2 only)
# These teach the composition syntax
# =============================================================================

def _compute_composition_examples() -> Dict[str, CompositionExample]:
    """Compute the composition few-shot examples."""

    compositions = {
        "α-γ": {
            "tokens": ["α", "γ"],
            "description": "sin(square(x))",
            "func": lambda x: math.sin(x ** 2),  # sin(x²)
        },
        "δ-α": {
            "tokens": ["δ", "α"],
            "description": "abs(sin(x))",
            "func": lambda x: abs(math.sin(x)),  # |sin(x)|
        },
    }

    examples = {}
    for message, info in compositions.items():
        func = info["func"]
        samples = [(x, func(x)) for x in SAMPLE_X_VALUES]
        tests = [(x, func(x)) for x in TEST_X_VALUES[:2]]  # Only 2 test examples

        examples[message] = CompositionExample(
            message=message,
            tokens=info["tokens"],
            description=info["description"],
            samples=samples,
            test_examples=tests
        )

    return examples


FEW_SHOT_COMPOSITIONS: Dict[str, CompositionExample] = _compute_composition_examples()


# =============================================================================
# HELD-OUT COMPOSITIONS (18 - these are tested in Phase 2)
# =============================================================================

def get_novel_compositions() -> List[Tuple[str, str]]:
    """
    Get the 18 novel compositions for Phase 2 testing.

    These are all pairs of {α, β, γ, δ, ε} minus the 2 shown in few-shot.

    Returns:
        List of (token1, token2) tuples representing compositions
    """
    mvp_tokens = ["α", "β", "γ", "δ", "ε"]
    shown_compositions = {("α", "γ"), ("δ", "α")}

    novel = []
    for t1 in mvp_tokens:
        for t2 in mvp_tokens:
            if t1 != t2:  # No self-composition
                pair = (t1, t2)
                if pair not in shown_compositions:
                    novel.append(pair)

    return novel


NOVEL_COMPOSITIONS: List[Tuple[str, str]] = get_novel_compositions()


# =============================================================================
# FORMATTING FOR PROMPTS
# =============================================================================

def format_seer_primitive_example(ex: PrimitiveExample) -> str:
    """Format a primitive example for the Seer prompt."""
    lines = ["Samples:"]
    for x, y in ex.samples:
        lines.append(f"  f({x:.1f}) = {y:.4f}")
    lines.append(f"MESSAGE: {ex.token}")
    return "\n".join(lines)


def format_seer_composition_example(ex: CompositionExample) -> str:
    """Format a composition example for the Seer prompt."""
    lines = ["Samples:"]
    for x, y in ex.samples:
        lines.append(f"  f({x:.1f}) = {y:.4f}")
    lines.append(f"MESSAGE: {ex.message}")
    lines.append(f"(This combines {' and '.join(ex.tokens)}: first apply {ex.tokens[-1]}, then apply {ex.tokens[0]} to the result)")
    return "\n".join(lines)


def format_doer_primitive_example(ex: PrimitiveExample) -> str:
    """Format a primitive example for the Doer prompt."""
    lines = [f"Message: {ex.token}"]
    for x, y in ex.test_examples:
        lines.append(f"Test input: {x} → Output: {y:.4f}")
    return "\n".join(lines)


def format_doer_composition_example(ex: CompositionExample) -> str:
    """Format a composition example for the Doer prompt."""
    lines = [f"Message: {ex.message}"]
    for x, y in ex.test_examples:
        lines.append(f"Test input: {x} → Output: {y:.4f}")
    lines.append(f"({ex.message} means: first apply {ex.tokens[-1]}, then apply {ex.tokens[0]} to that result)")
    return "\n".join(lines)


def get_all_seer_examples() -> str:
    """Get all few-shot examples formatted for Seer."""
    sections = []

    # Primitives
    for i, (token, ex) in enumerate(sorted(FEW_SHOT_PRIMITIVES.items()), 1):
        section = f"═══════════════════════════════════════════════════════════════════\nEXAMPLE {i} (primitive):\n"
        section += format_seer_primitive_example(ex)
        section += "\n═══════════════════════════════════════════════════════════════════"
        sections.append(section)

    # Compositions
    for i, (message, ex) in enumerate(sorted(FEW_SHOT_COMPOSITIONS.items()), len(FEW_SHOT_PRIMITIVES) + 1):
        section = f"═══════════════════════════════════════════════════════════════════\nEXAMPLE {i} (composition):\n"
        section += format_seer_composition_example(ex)
        section += "\n═══════════════════════════════════════════════════════════════════"
        sections.append(section)

    return "\n\n".join(sections)


def get_all_doer_examples() -> str:
    """Get all few-shot examples formatted for Doer."""
    sections = []

    # Primitives
    for i, (token, ex) in enumerate(sorted(FEW_SHOT_PRIMITIVES.items()), 1):
        section = f"═══════════════════════════════════════════════════════════════════\nEXAMPLE {i} (primitive {token}):\n"
        section += format_doer_primitive_example(ex)
        section += "\n═══════════════════════════════════════════════════════════════════"
        sections.append(section)

    # Compositions
    for i, (message, ex) in enumerate(sorted(FEW_SHOT_COMPOSITIONS.items()), len(FEW_SHOT_PRIMITIVES) + 1):
        section = f"═══════════════════════════════════════════════════════════════════\nEXAMPLE {i} (composition {message}):\n"
        section += format_doer_composition_example(ex)
        section += "\n═══════════════════════════════════════════════════════════════════"
        sections.append(section)

    return "\n\n".join(sections)
