"""
DynamicalSystem wrapper — observation model and metadata for ChaosBench v2.

Wraps an Atom with:
- Reproducible noisy observations (seeded trajectory + Gaussian noise)
- Metadata extraction (regime, Lyapunov, h_KS, prediction horizon)
- Future trajectory generation for PREDICT scoring
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from chaosbench.grammar.atoms import Atom


@dataclass(frozen=True)
class SystemMetadata:
    """Static metadata about a dynamical system."""

    family: str
    params: dict
    regime: str
    h_ks: float
    lambda_max: float
    tau_lambda: float  # Lyapunov time = 1/|λ| (or inf if λ=0)
    dim: int
    grammar_depth: int
    domain: Tuple[float, float]


@dataclass
class Observation:
    """Noisy observation of a trajectory."""

    data: np.ndarray        # Noisy trajectory (what the agent sees)
    clean_data: np.ndarray  # Ground-truth trajectory (hidden)
    n_points: int
    noise_std: float
    stride: int
    x0: float
    seeds: dict             # {'ic_seed': ..., 'noise_seed': ...}


class DynamicalSystem:
    """Wraps an Atom with observation and metadata facilities."""

    def __init__(self, atom: Atom, grammar_depth: int = 0):
        self._atom = atom
        self._grammar_depth = grammar_depth
        self._metadata: Optional[SystemMetadata] = None

    @property
    def atom(self) -> Atom:
        return self._atom

    def metadata(self) -> SystemMetadata:
        """Compute and cache system metadata."""
        if self._metadata is not None:
            return self._metadata

        lam = self._atom.lyapunov()
        h = self._atom.h_ks()
        tau = 1.0 / abs(lam) if abs(lam) > 1e-12 else float("inf")

        self._metadata = SystemMetadata(
            family=self._atom.family,
            params=self._atom.params,
            regime=self._atom.regime(),
            h_ks=h,
            lambda_max=lam,
            tau_lambda=tau,
            dim=1,
            grammar_depth=self._grammar_depth,
            domain=self._atom.domain,
        )
        return self._metadata

    def observe(
        self,
        n_points: int = 200,
        noise_std: float = 0.01,
        stride: int = 1,
        ic_seed: int = 42,
        noise_seed: int = 123,
        burn_in: int = 500,
    ) -> Observation:
        """Generate a noisy observation of the system.

        1. Pick initial condition from domain using ic_seed
        2. Burn in for burn_in steps
        3. Record n_points values (every stride-th iterate)
        4. Add Gaussian noise using noise_seed
        """
        rng_ic = np.random.RandomState(ic_seed)
        lo, hi = self._atom.domain
        x0 = lo + rng_ic.random() * (hi - lo)

        # Reset internal state (needed for HenonAtom's hidden y-component)
        self._atom.prepare(x0)

        # Burn-in
        x = x0
        for _ in range(burn_in):
            x = self._atom.iterate(x)

        # Record clean trajectory
        clean = np.empty(n_points)
        for i in range(n_points):
            clean[i] = x
            for _ in range(stride):
                x = self._atom.iterate(x)

        # Add noise
        rng_noise = np.random.RandomState(noise_seed)
        noise = rng_noise.normal(0, noise_std, size=n_points)
        noisy = clean + noise

        return Observation(
            data=noisy,
            clean_data=clean,
            n_points=n_points,
            noise_std=noise_std,
            stride=stride,
            x0=x0,
            seeds={"ic_seed": ic_seed, "noise_seed": noise_seed},
        )

    def future_trajectory(self, observation: Observation, horizon: int) -> np.ndarray:
        """Continue the trajectory from the last clean observation value."""
        x = observation.clean_data[-1]
        future = np.empty(horizon)
        for i in range(horizon):
            x = self._atom.iterate(x)
            future[i] = x
        return future

    def prediction_horizon(self) -> int:
        """Suggested prediction horizon.

        Chaotic: ceil(5 * tau_lambda). Non-chaotic: 20.
        """
        meta = self.metadata()
        if meta.regime == "chaotic":
            return max(1, math.ceil(5.0 * meta.tau_lambda))
        return 20
