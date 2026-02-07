"""
Dynamical system atoms — the primitive building blocks of ChaosBench v2.

Each Atom represents a single 1D iterated map with known analytical properties.
Atoms are composed via connectives (post-MVP) to create more complex systems.
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Tuple

import numpy as np

from chaosbench.core.lyapunov import (
    compute_lyapunov_1d,
    compute_lyapunov_spectrum,
    compute_h_ks,
)


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

    def prepare(self, x0: float) -> None:
        """Reset internal state before trajectory generation.

        No-op for 1D atoms. HenonAtom overrides to reset hidden y-component.
        """
        pass

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


class SineAtom(Atom):
    """Sine map: f(x) = a·sin(π·x), x ∈ [0, 1].

    Unimodal like logistic — acts as a confuser for IDENTIFY.
    """

    def __init__(self, a: float):
        if not 0.5 <= a <= 1.0:
            raise ValueError(f"a must be in [0.5, 1.0], got {a}")
        self._a = a

    @property
    def family(self) -> str:
        return "sine"

    @property
    def params(self) -> dict:
        return {"a": self._a}

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def iterate(self, x: float) -> float:
        return self._a * np.sin(np.pi * x)

    def derivative(self, x: float) -> float:
        return self._a * np.pi * np.cos(np.pi * x)

    def regime(self) -> str:
        lam = self.lyapunov()
        if lam > 0.01:
            return "chaotic"
        if lam < -0.01:
            # Check for fixed point vs periodic: iterate from mid-domain
            x = 0.5
            for _ in range(2000):
                x = self.iterate(x)
            x_prev = x
            x = self.iterate(x)
            if abs(x - x_prev) < 1e-10:
                return "fixed_point"
            return "periodic"
        return "periodic"


class CircleAtom(Atom):
    """Circle map: f(x) = (x + Ω − K/(2π)·sin(2πx)) mod 1, x ∈ [0, 1).

    Two parameters (K, Ω). Three regimes:
    - K < 1, Ω irrational → quasiperiodic
    - K < 1, Arnold tongues → mode-locked periodic
    - K > 1 → chaotic
    """

    def __init__(self, K: float, omega: float):
        if not 0.0 <= K <= 2.0:
            raise ValueError(f"K must be in [0.0, 2.0], got {K}")
        if not 0.0 < omega < 1.0:
            raise ValueError(f"omega must be in (0, 1), got {omega}")
        self._K = K
        self._omega = omega

    @property
    def family(self) -> str:
        return "circle"

    @property
    def params(self) -> dict:
        return {"K": self._K, "omega": self._omega}

    @property
    def domain(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def iterate(self, x: float) -> float:
        return (x + self._omega - self._K / (2 * np.pi) * np.sin(2 * np.pi * x)) % 1.0

    def derivative(self, x: float) -> float:
        return 1.0 - self._K * np.cos(2 * np.pi * x)

    def regime(self) -> str:
        if self._K > 1.0:
            lam = self.lyapunov()
            return "chaotic" if lam > 0.01 else "periodic"
        # K <= 1: quasiperiodic or mode-locked
        # Check winding number rationality via trajectory
        x = 0.5
        total_rotation = 0.0
        n = 5000
        for _ in range(1000):
            x = self.iterate(x)
        for _ in range(n):
            x_new = self.iterate(x)
            # Track unwrapped rotation
            dx = x_new - x + self._omega - self._K / (2 * np.pi) * np.sin(2 * np.pi * x)
            total_rotation += x_new - x
            if total_rotation < -0.5:
                total_rotation += 1.0
            x = x_new
        # Check if periodic by looking at orbit closure
        x_test = 0.3
        for _ in range(500):
            x_test = self.iterate(x_test)
        orbit = [x_test]
        for _ in range(2000):
            x_test = self.iterate(x_test)
            for prev in orbit:
                if abs(x_test - prev) < 1e-8:
                    return "periodic"
            orbit.append(x_test)
            if len(orbit) > 500:
                return "quasiperiodic"
        return "quasiperiodic"


class HenonAtom(Atom):
    """Hénon map (2D, projected to x): x' = 1 − a·x² + y, y' = b·x.

    First 2D system in ChaosBench. Internal _y state is hidden from the
    observation interface — the agent only sees 1D x-projections.
    """

    def __init__(self, a: float, b: float):
        if not 1.0 <= a <= 1.4:
            raise ValueError(f"a must be in [1.0, 1.4], got {a}")
        if not 0.2 <= b <= 0.4:
            raise ValueError(f"b must be in [0.2, 0.4], got {b}")
        self._a = a
        self._b = b
        self._y = 0.0

    @property
    def family(self) -> str:
        return "henon"

    @property
    def params(self) -> dict:
        return {"a": self._a, "b": self._b}

    @property
    def domain(self) -> Tuple[float, float]:
        # Hénon attractor x-range depends on params; conservative bound
        return (-1.5, 1.5)

    def prepare(self, x0: float) -> None:
        """Reset hidden y-component before trajectory generation."""
        self._y = 0.0

    def iterate(self, x: float) -> float:
        x_new = 1.0 - self._a * x * x + self._y
        self._y = self._b * x
        return x_new

    def derivative(self, x: float) -> float:
        # dx'/dx = -2ax (partial, for 1D Lyapunov fallback)
        return -2.0 * self._a * x

    def trajectory(self, x0: float, n: int) -> np.ndarray:
        """Generate trajectory with proper state reset."""
        self.prepare(x0)
        traj = np.empty(n)
        x = x0
        for i in range(n):
            traj[i] = x
            x = self.iterate(x)
        return traj

    def lyapunov(self) -> float:
        """Compute maximal Lyapunov exponent using 2D QR method."""
        def f_2d(state: np.ndarray) -> np.ndarray:
            x, y = state
            return np.array([1.0 - self._a * x * x + y, self._b * x])

        def jac_2d(state: np.ndarray) -> np.ndarray:
            x, _y = state
            return np.array([[-2.0 * self._a * x, 1.0], [self._b, 0.0]])

        x0_2d = np.array([0.1, 0.0])
        spectrum = compute_lyapunov_spectrum(f_2d, jac_2d, x0_2d, dim=2)
        return float(spectrum[0])

    def h_ks(self) -> float:
        """h_KS via Pesin's theorem on full 2D spectrum."""
        def f_2d(state: np.ndarray) -> np.ndarray:
            x, y = state
            return np.array([1.0 - self._a * x * x + y, self._b * x])

        def jac_2d(state: np.ndarray) -> np.ndarray:
            x, _y = state
            return np.array([[-2.0 * self._a * x, 1.0], [self._b, 0.0]])

        x0_2d = np.array([0.1, 0.0])
        spectrum = compute_lyapunov_spectrum(f_2d, jac_2d, x0_2d, dim=2)
        return compute_h_ks(spectrum)

    def regime(self) -> str:
        lam = self.lyapunov()
        if lam > 0.01:
            return "chaotic"
        if lam < -0.01:
            return "periodic"
        return "periodic"
