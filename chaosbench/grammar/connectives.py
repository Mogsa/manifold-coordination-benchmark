"""
Grammar connectives — composing atoms into deeper systems.

AffineConjugacy wraps an atom f into h⁻¹ ∘ f ∘ h where h(x) = (x - b) / a.
Preserves all dynamics (Lyapunov, h_KS, regime) while shifting the domain.
"""

from typing import Tuple

import numpy as np

from chaosbench.grammar.atoms import Atom


class AffineConjugacy(Atom):
    """Affine conjugacy connective: y_{n+1} = a·f((y_n − b)/a) + b.

    Wraps an inner atom. The conjugated system has:
    - Same regime, Lyapunov, h_KS (dynamics preserved)
    - Different domain and numerical values (appearance changed)
    - grammar_depth = inner.grammar_depth + 1 (if inner tracked it)
    - IDENTIFY ground truth = inner atom's family
    - Agent gets NO hint that conjugacy was applied
    """

    def __init__(self, inner: Atom, a: float, b: float):
        if abs(a) < 1e-12:
            raise ValueError(f"Scale a must be non-zero, got {a}")
        self._inner = inner
        self._a = a
        self._b = b

    @property
    def inner_atom(self) -> Atom:
        """Access the wrapped atom (for factory/metadata)."""
        return self._inner

    @property
    def conjugacy_params(self) -> dict:
        return {"a": self._a, "b": self._b}

    @property
    def family(self) -> str:
        return self._inner.family

    @property
    def params(self) -> dict:
        return self._inner.params

    @property
    def domain(self) -> Tuple[float, float]:
        lo, hi = self._inner.domain
        # Transform domain: y = a*x + b
        y_lo = self._a * lo + self._b
        y_hi = self._a * hi + self._b
        return (min(y_lo, y_hi), max(y_lo, y_hi))

    def prepare(self, x0: float) -> None:
        """Delegate prepare to inner atom with inverse-transformed x0."""
        x0_inner = (x0 - self._b) / self._a
        self._inner.prepare(x0_inner)

    def iterate(self, y: float) -> float:
        """y_{n+1} = a·f((y_n − b)/a) + b."""
        x = (y - self._b) / self._a
        fx = self._inner.iterate(x)
        return self._a * fx + self._b

    def derivative(self, y: float) -> float:
        """Chain rule: dy'/dy = f'((y-b)/a) (scale factors cancel)."""
        x = (y - self._b) / self._a
        return self._inner.derivative(x)

    def lyapunov(self) -> float:
        """Lyapunov exponent is conjugacy-invariant."""
        return self._inner.lyapunov()

    def h_ks(self) -> float:
        """h_KS is conjugacy-invariant."""
        return self._inner.h_ks()

    def regime(self) -> str:
        """Regime is conjugacy-invariant."""
        return self._inner.regime()

    @property
    def name(self) -> str:
        return f"conj({self._inner.name}, a={self._a}, b={self._b})"
