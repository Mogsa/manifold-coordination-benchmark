"""
Function library for the Telepathic benchmark.

Defines primitive functions and composition for the communication game.
MVP uses 5 primitives (sin, cos, square, abs, neg) mapped to tokens (α, β, γ, δ, ε).
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Dict, Union


@dataclass
class PrimitiveFunction:
    """A single primitive function f: R -> R."""

    name: str
    token: str
    func: Callable[[float], float]
    description: str

    def __call__(self, x: float) -> float:
        """Evaluate function at x."""
        return self.func(x)

    def to_dict(self) -> dict:
        """Serialize (without the callable)."""
        return {
            "name": self.name,
            "token": self.token,
            "description": self.description,
        }


@dataclass
class ComposedFunction:
    """A composition of primitive functions."""

    primitives: List[str]  # Names, e.g., ["sin", "square"]
    tokens: List[str]      # Tokens, e.g., ["α", "γ"]

    def __call__(self, x: float) -> float:
        """
        Evaluate composition at x.

        Composition is right-to-left: ["sin", "square"] means sin(square(x)).
        """
        result = x
        for prim_name in reversed(self.primitives):
            result = PRIMITIVES[prim_name](result)
        return result

    @property
    def complexity(self) -> int:
        """Number of primitives in composition."""
        return len(self.primitives)

    @property
    def message(self) -> str:
        """Token message for this composition, e.g., 'α-γ'."""
        return "-".join(self.tokens)

    @property
    def description(self) -> str:
        """Generate description like 'sin(square(x))'."""
        inner = "x"
        for prim in reversed(self.primitives):
            inner = f"{prim}({inner})"
        return inner

    def to_dict(self) -> dict:
        return {
            "primitives": self.primitives,
            "tokens": self.tokens,
            "message": self.message,
            "description": self.description,
            "complexity": self.complexity,
        }


# =============================================================================
# MVP PRIMITIVES (5 functions, tokens α β γ δ ε)
# =============================================================================

PRIMITIVES: Dict[str, PrimitiveFunction] = {
    "sin": PrimitiveFunction(
        name="sin",
        token="α",
        func=lambda x: math.sin(x),
        description="Sine function"
    ),
    "cos": PrimitiveFunction(
        name="cos",
        token="β",
        func=lambda x: math.cos(x),
        description="Cosine function"
    ),
    "square": PrimitiveFunction(
        name="square",
        token="γ",
        func=lambda x: x ** 2,
        description="Square function (x²)"
    ),
    "abs": PrimitiveFunction(
        name="abs",
        token="δ",
        func=lambda x: abs(x),
        description="Absolute value (|x|)"
    ),
    "neg": PrimitiveFunction(
        name="neg",
        token="ε",
        func=lambda x: -x,
        description="Negation (-x)"
    ),
}

# Token -> Primitive name mapping
TOKEN_TO_NAME: Dict[str, str] = {p.token: p.name for p in PRIMITIVES.values()}
NAME_TO_TOKEN: Dict[str, str] = {p.name: p.token for p in PRIMITIVES.values()}


# =============================================================================
# HELD-OUT PRIMITIVES (for future extension, tokens ζ η θ ι κ)
# =============================================================================

HELD_OUT_PRIMITIVES: Dict[str, PrimitiveFunction] = {
    "sqrt": PrimitiveFunction(
        name="sqrt",
        token="ζ",
        func=lambda x: math.sqrt(abs(x)),  # Safe sqrt
        description="Square root (safe)"
    ),
    "exp": PrimitiveFunction(
        name="exp",
        token="η",
        func=lambda x: math.exp(min(x, 10)),  # Capped to avoid overflow
        description="Exponential (capped)"
    ),
    "log": PrimitiveFunction(
        name="log",
        token="θ",
        func=lambda x: math.log(abs(x) + 0.01),  # Safe log
        description="Logarithm (safe)"
    ),
    "relu": PrimitiveFunction(
        name="relu",
        token="ι",
        func=lambda x: max(0, x),
        description="ReLU activation"
    ),
    "sign": PrimitiveFunction(
        name="sign",
        token="κ",
        func=lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
        description="Sign function"
    ),
}


# =============================================================================
# COMPOSITION UTILITIES
# =============================================================================

def compose(*functions: Callable[[float], float]) -> Callable[[float], float]:
    """
    Compose functions right-to-left: (f ∘ g)(x) = f(g(x)).

    compose(f, g, h)(x) = f(g(h(x)))

    Args:
        *functions: Functions to compose

    Returns:
        Composed function
    """
    def composed(x: float) -> float:
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return composed


def compose_by_names(primitive_names: List[str]) -> ComposedFunction:
    """
    Create a composed function from primitive names.

    Args:
        primitive_names: List like ["sin", "square"] meaning sin(square(x))

    Returns:
        ComposedFunction instance
    """
    tokens = [PRIMITIVES[name].token for name in primitive_names]
    return ComposedFunction(primitives=primitive_names, tokens=tokens)


def compose_by_tokens(tokens: List[str]) -> ComposedFunction:
    """
    Create a composed function from tokens.

    Args:
        tokens: List like ["α", "γ"] meaning α(γ(x)) = sin(square(x))

    Returns:
        ComposedFunction instance
    """
    primitive_names = [TOKEN_TO_NAME[t] for t in tokens]
    return ComposedFunction(primitives=primitive_names, tokens=tokens)


def parse_message_to_function(message: str) -> Union[PrimitiveFunction, ComposedFunction]:
    """
    Parse a token message into an executable function.

    Args:
        message: Token string like "α" or "α-γ"

    Returns:
        PrimitiveFunction or ComposedFunction
    """
    tokens = message.split("-")

    if len(tokens) == 1:
        name = TOKEN_TO_NAME[tokens[0]]
        return PRIMITIVES[name]
    else:
        return compose_by_tokens(tokens)


def get_function_by_name(name: str) -> PrimitiveFunction:
    """Get a primitive by name."""
    return PRIMITIVES[name]


def get_function_by_token(token: str) -> PrimitiveFunction:
    """Get a primitive by token."""
    return PRIMITIVES[TOKEN_TO_NAME[token]]


def list_mvp_primitives() -> List[PrimitiveFunction]:
    """Return list of MVP primitives in token order."""
    return sorted(PRIMITIVES.values(), key=lambda p: p.token)
