"""
Dynamical system atoms — the primitive building blocks of ChaosBench v2.

Each Atom represents a single 1D iterated map with known analytical properties.
Atoms are composed via connectives (post-MVP) to create more complex systems.
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Tuple

import numpy as np

from chaosbench.core.lyapunov import compute_lyapunov_1d, compute_h_ks


class Atom(ABC):
    """Abstract base class for a 1D iterated map."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Family name (e.g. 'logistic', 'tent')."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """Parameter dict (e.g. {'r': 3.8})."""

    @property
    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        """Valid state-space interval [lo, hi]."""

    @abstractmethod
    def iterate(self, x: float) -> float:
        """One step: x_{n+1} = f(x_n)."""

    @abstractmethod
    def derivative(self, x: float) -> float:
        """df/dx at x."""

    @property
    def name(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.family}({param_str})"

    def trajectory(self, x0: float, n: int) -> np.ndarray:
        """Generate trajectory of length n starting from x0."""
        traj = np.empty(n)
        x = x0
        for i in range(n):
            traj[i] = x
            x = self.iterate(x)
        return traj

    def lyapunov(self) -> float:
        """Compute maximal Lyapunov exponent. Override for analytical values."""
        lo, hi = self.domain
        x0 = lo + 0.3 * (hi - lo)  # Avoid boundaries
        return compute_lyapunov_1d(self.iterate, self.derivative, x0)

    def h_ks(self) -> float:
        """Kolmogorov-Sinai entropy via Pesin's theorem."""
        lam = self.lyapunov()
        return compute_h_ks(np.array([lam]))

    @abstractmethod
    def regime(self) -> str:
        """Return 'fixed_point', 'periodic', 'chaotic', or 'quasiperiodic'."""


class LogisticAtom(Atom):
    """Logistic map: f(x) = r·x·(1-x), x ∈ [0,1]."""

    def __init__(self, r: float):
        if not 2.5 <= r <= 4.0:
            raise ValueError(f"r must be in [2.5, 4.0], got {r}")
        self._r = r

    @property
    def family(self) -> str:
        return "logistic"

    @property
    def params(self) -> dict:
        return {"r": self._r}

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def iterate(self, x: float) -> float:
        return self._r * x * (1.0 - x)

    def derivative(self, x: float) -> float:
        return self._r * (1.0 - 2.0 * x)

    def regime(self) -> str:
        r = self._r
        if r < 3.0:
            return "fixed_point"
        if r < 3.57:
            return "periodic"
        # In the chaotic band: compute numerically
        lam = self.lyapunov()
        return "chaotic" if lam > 0.01 else "periodic"


class TentAtom(Atom):
    """Tent map: f(x) = mu·min(x, 1-x), x ∈ [0,1].

    Hard cap mu ≤ 1.95 to avoid mu=2.0 degeneration (v1 bug).
    """

    def __init__(self, mu: float):
        if not 1.0 <= mu <= 1.95:
            raise ValueError(f"mu must be in [1.0, 1.95], got {mu}")
        self._mu = mu

    @property
    def family(self) -> str:
        return "tent"

    @property
    def params(self) -> dict:
        return {"mu": self._mu}

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def iterate(self, x: float) -> float:
        return self._mu * min(x, 1.0 - x)

    def derivative(self, x: float) -> float:
        return self._mu if x < 0.5 else -self._mu

    def lyapunov(self) -> float:
        # Analytical: λ = ln(mu) for mu > 1
        return float(np.log(self._mu))

    def h_ks(self) -> float:
        lam = self.lyapunov()
        return max(lam, 0.0)

    def regime(self) -> str:
        if self._mu <= 1.0:
            return "fixed_point"
        return "chaotic"


class DampedLinearAtom(Atom):
    """Damped linear map: f(x) = lam·x, x ∈ [-10,10].

    Always converges to fixed point at origin.
    """

    def __init__(self, lam: float):
        if not 0.0 < lam < 0.99:
            raise ValueError(f"lam must be in (0, 0.99), got {lam}")
        self._lam = lam

    @property
    def family(self) -> str:
        return "damped_linear"

    @property
    def params(self) -> dict:
        return {"lam": self._lam}

    @property
    def domain(self) -> Tuple[float, float]:
        return (-10.0, 10.0)

    def iterate(self, x: float) -> float:
        return self._lam * x

    def derivative(self, x: float) -> float:
        return self._lam

    def lyapunov(self) -> float:
        # Analytical: λ = ln(lam), always negative
        return float(np.log(self._lam))

    def h_ks(self) -> float:
        return 0.0

    def regime(self) -> str:
        return "fixed_point"


class RotationAtom(Atom):
    """Rotation map: f(x) = (x + omega) mod 1, x ∈ [0,1).

    Rational omega → periodic, irrational → quasiperiodic.
    """

    def __init__(self, omega: float):
        if not 0.0 < omega < 1.0:
            raise ValueError(f"omega must be in (0, 1), got {omega}")
        self._omega = omega

    @property
    def family(self) -> str:
        return "rotation"

    @property
    def params(self) -> dict:
        return {"omega": self._omega}

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def iterate(self, x: float) -> float:
        return (x + self._omega) % 1.0

    def derivative(self, x: float) -> float:
        return 1.0

    def lyapunov(self) -> float:
        return 0.0

    def h_ks(self) -> float:
        return 0.0

    def regime(self) -> str:
        frac = Fraction(self._omega).limit_denominator(1000)
        # If the fraction approximation is very close, treat as rational
        if abs(frac - self._omega) < 1e-9:
            return "periodic"
        return "quasiperiodic"
