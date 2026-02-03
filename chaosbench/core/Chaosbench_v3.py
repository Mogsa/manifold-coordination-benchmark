#!/usr/bin/env python3
"""
ChaosBench v3: Information-Efficient Prediction on Chaotic Systems
===================================================================

A benchmark measuring how efficiently solvers extract predictive information
from partially observable chaotic dynamical systems.

Key features:
1. h_KS (Kolmogorov-Sinai entropy) as rigorous complexity measure
2. Observation cost λ quantifies "agency" - more agents = more parallel observation
3. Configurable difficulty weighting w(h_KS)
4. Both conditional (system family known) and unconditional settings
5. Discretized prediction for clean NLL computation

Author: ChaosBench Team
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import time
from collections import defaultdict

from .lyapunov import (
    compute_lyapunov_1d,
    compute_lyapunov_spectrum,
    compute_lyapunov_continuous,
    compute_h_ks,
)

np.random.seed(42)


# =============================================================================
# PART 1: CHAOTIC SYSTEMS
# =============================================================================

@dataclass
class ChaoticSystem:
    """A chaotic dynamical system with known h_KS."""
    name: str
    family: str
    dim: int
    h_ks: float  # Kolmogorov-Sinai entropy
    lyapunov_time: float  # τ_λ = 1/λ_max
    
    def step(self, x: np.ndarray) -> np.ndarray:
        """One iteration of the map."""
        raise NotImplementedError
    
    def trajectory(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Generate trajectory from initial condition."""
        traj = np.zeros((n_steps, self.dim))
        traj[0] = x0
        for i in range(1, n_steps):
            traj[i] = self.step(traj[i-1])
        return traj


class LogisticMap(ChaoticSystem):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n)"""

    def __init__(self, r: float = 4.0):
        self.r = r
        # Compute h_KS via Lyapunov exponent
        h_ks = self._compute_h_ks()
        super().__init__(
            name=f"logistic_r{r:.2f}",
            family="logistic",
            dim=1,
            h_ks=h_ks,
            lyapunov_time=1.0 / max(h_ks, 0.1)
        )

    def _f(self, x: float) -> float:
        """Map function for scalar input."""
        return self.r * x * (1 - x)

    def _df(self, x: float) -> float:
        """Derivative: f'(x) = r(1 - 2x)"""
        return self.r * (1 - 2 * x)

    def _compute_h_ks(self) -> float:
        """Compute h_KS via Lyapunov exponent."""
        # Use x0=0.3, not 0.5 which maps to the fixed point x=0 for r=4
        lyap = compute_lyapunov_1d(
            f=self._f,
            df=self._df,
            x0=0.3,
            n_iter=10000,
            n_burn=1000,
        )
        return max(0.0, lyap)

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.r * x * (1 - x)


class TentMap(ChaoticSystem):
    """Tent map: piecewise linear, exactly solvable h_KS = ln(μ)."""

    def __init__(self, mu: float = 2.0):
        self.mu = mu
        # Compute h_KS via Lyapunov exponent
        h_ks = self._compute_h_ks()
        super().__init__(
            name=f"tent_mu{mu:.2f}",
            family="tent",
            dim=1,
            h_ks=h_ks,
            lyapunov_time=1.0 / max(h_ks, 0.1)
        )

    def _f(self, x: float) -> float:
        """Map function for scalar input."""
        return self.mu * x if x < 0.5 else self.mu * (1 - x)

    def _df(self, x: float) -> float:
        """Derivative: |f'(x)| = μ (constant magnitude)."""
        # The absolute value is μ everywhere except at x=0.5
        return self.mu

    def _compute_h_ks(self) -> float:
        """Compute h_KS via Lyapunov exponent.

        For tent map, λ = ln(μ) exactly, but we compute numerically for consistency.
        """
        lyap = compute_lyapunov_1d(
            f=self._f,
            df=self._df,
            x0=0.3,
            n_iter=10000,
            n_burn=1000,
        )
        return max(0.0, lyap)

    def step(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.5, self.mu * x, self.mu * (1 - x))


class HenonMap(ChaoticSystem):
    """Hénon map: 2D quadratic map with classic parameters."""

    def __init__(self, a: float = 1.4, b: float = 0.3):
        self.a = a
        self.b = b
        # Compute h_KS via Lyapunov spectrum
        h_ks = self._compute_h_ks()
        super().__init__(
            name=f"henon_a{a:.2f}_b{b:.2f}",
            family="henon",
            dim=2,
            h_ks=h_ks,
            lyapunov_time=1.0 / max(h_ks, 0.1)
        )

    def _f(self, x: np.ndarray) -> np.ndarray:
        """Map function."""
        x_new = 1 - self.a * x[0]**2 + x[1]
        y_new = self.b * x[0]
        return np.clip(np.array([x_new, y_new]), -10, 10)

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian matrix: J = [[-2ax, 1], [b, 0]]"""
        return np.array([
            [-2 * self.a * x[0], 1.0],
            [self.b, 0.0]
        ])

    def _compute_h_ks(self) -> float:
        """Compute h_KS via Lyapunov spectrum using QR method."""
        lyap_spectrum = compute_lyapunov_spectrum(
            f=self._f,
            jacobian=self._jacobian,
            x0=np.array([0.0, 0.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )
        return compute_h_ks(lyap_spectrum)

    def step(self, x: np.ndarray) -> np.ndarray:
        x_new = 1 - self.a * x[0]**2 + x[1]
        y_new = self.b * x[0]
        # Clip to prevent divergence
        result = np.clip(np.array([x_new, y_new]), -10, 10)
        return result


class StandardMap(ChaoticSystem):
    """Chirikov standard map: canonical example of Hamiltonian chaos."""

    def __init__(self, K: float = 1.0):
        self.K = K
        # Compute h_KS via Lyapunov spectrum
        h_ks = self._compute_h_ks()
        super().__init__(
            name=f"standard_K{K:.2f}",
            family="standard",
            dim=2,
            h_ks=h_ks,
            lyapunov_time=1.0 / max(h_ks, 0.1)
        )

    def _f(self, x: np.ndarray) -> np.ndarray:
        """Map function."""
        p, q = x
        p_new = (p + self.K * np.sin(q)) % (2 * np.pi)
        q_new = (q + p_new) % (2 * np.pi)
        return np.array([p_new, q_new])

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian matrix: J = [[1, K cos(q)], [1, 1+K cos(q)]]"""
        _, q = x
        k_cos_q = self.K * np.cos(q)
        return np.array([
            [1.0, k_cos_q],
            [1.0, 1.0 + k_cos_q]
        ])

    def _compute_h_ks(self) -> float:
        """Compute h_KS via Lyapunov spectrum using QR method."""
        lyap_spectrum = compute_lyapunov_spectrum(
            f=self._f,
            jacobian=self._jacobian,
            x0=np.array([1.0, 1.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )
        return compute_h_ks(lyap_spectrum)

    def step(self, x: np.ndarray) -> np.ndarray:
        p, q = x
        p_new = (p + self.K * np.sin(q)) % (2 * np.pi)
        q_new = (q + p_new) % (2 * np.pi)
        return np.array([p_new, q_new])


class LorenzDisc(ChaoticSystem):
    """Discretized Lorenz system via Poincaré section."""

    def __init__(self, rho: float = 28.0, sigma: float = 10.0, beta: float = 8/3):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.dt = 0.01
        # Compute h_KS via Lyapunov spectrum of continuous flow
        h_ks = self._compute_h_ks()
        super().__init__(
            name=f"lorenz_rho{rho:.0f}",
            family="lorenz",
            dim=3,
            h_ks=h_ks,
            lyapunov_time=1.0 / max(h_ks, 0.1)
        )

    def _flow(self, x: np.ndarray) -> np.ndarray:
        """Lorenz flow: dx/dt = f(x)"""
        return np.array([
            self.sigma * (x[1] - x[0]),
            x[0] * (self.rho - x[2]) - x[1],
            x[0] * x[1] - self.beta * x[2]
        ])

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of Lorenz flow:
        J = [[-σ,      σ,   0  ],
             [ρ-z,    -1,  -x  ],
             [y,       x,  -β  ]]
        """
        return np.array([
            [-self.sigma, self.sigma, 0.0],
            [self.rho - x[2], -1.0, -x[0]],
            [x[1], x[0], -self.beta]
        ])

    def _compute_h_ks(self) -> float:
        """Compute h_KS via Lyapunov spectrum of continuous flow."""
        lyap_spectrum = compute_lyapunov_continuous(
            flow=self._flow,
            jacobian=self._jacobian,
            x0=np.array([1.0, 1.0, 1.0]),
            dim=3,
            dt=0.01,
            t_total=100.0,
            t_burn=10.0,
            ortho_time=1.0,
        )
        return compute_h_ks(lyap_spectrum)

    def _continuous_step(self, x: np.ndarray) -> np.ndarray:
        """RK4 integration step."""
        def f(x):
            return np.array([
                self.sigma * (x[1] - x[0]),
                x[0] * (self.rho - x[2]) - x[1],
                x[0] * x[1] - self.beta * x[2]
            ])
        k1 = f(x)
        k2 = f(x + 0.5 * self.dt * k1)
        k3 = f(x + 0.5 * self.dt * k2)
        k4 = f(x + self.dt * k3)
        return x + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, x: np.ndarray) -> np.ndarray:
        """10 RK4 steps = one 'discrete' step."""
        for _ in range(10):
            x = self._continuous_step(x)
        return x


def create_system_family() -> Dict[str, List[ChaoticSystem]]:
    """Create parameterized families of chaotic systems."""
    families = {
        "logistic": [LogisticMap(r) for r in np.linspace(3.57, 4.0, 8)],
        "tent": [TentMap(mu) for mu in np.linspace(1.5, 2.0, 6)],
        "henon": [HenonMap(a, 0.3) for a in np.linspace(1.2, 1.4, 5)],
        "standard": [StandardMap(K) for K in np.linspace(0.5, 5.0, 8)],
        "lorenz": [LorenzDisc(rho) for rho in np.linspace(20, 40, 5)],
    }
    return families


def create_chaotic_system(family: str = None) -> ChaoticSystem:
    """Create a provably chaotic system with no periodic windows.

    Only returns systems at parameters where chaos is mathematically guaranteed:
    - Logistic map: r=4.0 (fully chaotic, h_KS = ln(2) ≈ 0.693)
    - Tent map: μ=2.0 (fully chaotic, h_KS = ln(2) ≈ 0.693)

    These 1D maps at these exact parameters have:
    - No periodic windows
    - Positive Lyapunov exponents
    - Ergodic invariant measures

    Args:
        family: "logistic" or "tent". If None, randomly selects one.

    Returns:
        A ChaoticSystem instance guaranteed to be chaotic.

    Raises:
        ValueError: If family is not "logistic" or "tent".
    """
    chaotic_systems = {
        "logistic": LogisticMap(r=4.0),
        "tent": TentMap(mu=2.0),
    }

    if family is None:
        family = np.random.choice(list(chaotic_systems.keys()))

    if family not in chaotic_systems:
        raise ValueError(
            f"Unknown chaotic family: {family}. "
            f"Valid options: {list(chaotic_systems.keys())}"
        )

    return chaotic_systems[family]


def get_chaotic_systems() -> List[ChaoticSystem]:
    """Get all provably chaotic systems.

    Note: Tent map μ=2 is excluded due to numerical instability —
    floating-point precision causes trajectories to fall into the
    fixed point at x=0. Logistic r=4 is numerically stable.

    Returns list containing logistic r=4 only.
    """
    return [
        LogisticMap(r=4.0),
        # TentMap excluded: degenerates to fixed point due to float precision
    ]


# =============================================================================
# PART 2: DISCRETIZED STATE SPACE FOR CLEAN NLL
# =============================================================================

class DiscretizedSpace:
    """
    Discretize the state space into bins for clean probability computation.
    
    For 1D: n_bins bins in [0, 1] or appropriate range
    For 2D+: n_bins^d grid (can get expensive, keep d small)
    """
    
    def __init__(self, dim: int, n_bins: int = 20, bounds: Tuple[float, float] = (0, 1)):
        self.dim = dim
        self.n_bins = n_bins
        self.bounds = bounds
        self.n_states = n_bins ** dim
        
        # Create bin edges
        self.edges = np.linspace(bounds[0], bounds[1], n_bins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
    
    def state_to_bin(self, x: np.ndarray) -> int:
        """Convert continuous state to discrete bin index."""
        x = np.atleast_1d(x)
        # Clip to bounds
        x_clipped = np.clip(x, self.bounds[0] + 1e-10, self.bounds[1] - 1e-10)
        # Get bin indices for each dimension
        bins = np.digitize(x_clipped, self.edges[1:-1])
        # Convert to single index (row-major order)
        idx = 0
        for i, b in enumerate(bins):
            idx += b * (self.n_bins ** (self.dim - 1 - i))
        return int(idx)
    
    def bin_to_state(self, idx: int) -> np.ndarray:
        """Convert bin index to center of bin."""
        coords = []
        for d in range(self.dim):
            coords.append(self.centers[(idx // (self.n_bins ** (self.dim - 1 - d))) % self.n_bins])
        return np.array(coords)


# =============================================================================
# PART 3: TASK DEFINITION
# =============================================================================

@dataclass
class Task:
    """A prediction task."""
    task_id: int
    system: ChaoticSystem
    observations: np.ndarray  # Shape: (n_obs, dim) - what the solver sees
    obs_times: np.ndarray  # When each observation was taken
    true_future: np.ndarray  # Ground truth future state
    future_time: int  # How far ahead
    h_ks: float  # System complexity
    
    # For discretized evaluation
    discretizer: DiscretizedSpace = None
    true_bin: int = None
    
    # Metadata
    conditional: bool = True  # Is system family revealed?
    family: str = ""


@dataclass
class TaskConfig:
    """Configuration for task generation."""
    n_obs: int = 50  # Number of observations available
    obs_density: float = 1.0  # Fraction of timesteps observed (1.0 = all)
    horizon_lyapunov_multiplier: float = 1.5  # Horizon = k * lyapunov_time
    noise_std: float = 0.01  # Observation noise
    n_bins: int = 20  # Discretization resolution
    conditional: bool = True  # Reveal system family?
    min_h_ks: float = 0.1  # Minimum h_KS to include system (filters non-chaotic)


class TaskGenerator:
    """Generate prediction tasks from chaotic systems."""

    def __init__(self, config: TaskConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.families = create_system_family()

        # Filter out non-chaotic systems (h_KS below threshold)
        self.all_systems = [
            s for fam in self.families.values()
            for s in fam
            if s.h_ks >= self.config.min_h_ks
        ]

        if not self.all_systems:
            raise ValueError(f"No systems with h_KS >= {self.config.min_h_ks}")

        self.task_counter = 0
    
    def generate_task(self, system: ChaoticSystem = None) -> Task:
        """Generate a single task."""
        if system is None:
            system = self.rng.choice(self.all_systems)

        # Compute horizon from Lyapunov time
        horizon = max(1, int(self.config.horizon_lyapunov_multiplier * system.lyapunov_time))

        # Get appropriate bounds for discretization
        bounds = self._get_bounds(system)
        discretizer = DiscretizedSpace(system.dim, self.config.n_bins, bounds)

        # Generate trajectory
        x0 = self._get_initial_condition(system)
        n_available = int(self.config.n_obs / self.config.obs_density)
        total_steps = n_available + horizon + 200  # Buffer + burn-in
        traj = system.trajectory(x0, total_steps)

        # Burn-in to reach attractor
        burn_in = 100
        traj = traj[burn_in:]

        # Select observation times (with possible sparsity)
        obs_indices = self.rng.choice(n_available, size=self.config.n_obs, replace=False)
        obs_indices = np.sort(obs_indices)

        # Get observations with noise
        observations = traj[obs_indices] + self.rng.normal(0, self.config.noise_std,
                                                           (self.config.n_obs, system.dim))

        # Get true future state
        future_idx = n_available + horizon
        true_future = traj[future_idx]
        true_bin = discretizer.state_to_bin(true_future)

        self.task_counter += 1

        return Task(
            task_id=self.task_counter,
            system=system,
            observations=observations,
            obs_times=obs_indices,
            true_future=true_future,
            future_time=horizon,  # Now computed dynamically
            h_ks=system.h_ks,
            discretizer=discretizer,
            true_bin=true_bin,
            conditional=self.config.conditional,
            family=system.family if self.config.conditional else ""
        )
    
    def generate_batch(self, n_tasks: int, stratified: bool = True) -> List[Task]:
        """Generate a batch of tasks."""
        if stratified:
            # Stratify by h_KS
            sorted_systems = sorted(self.all_systems, key=lambda s: s.h_ks)
            n_per_stratum = n_tasks // 5
            tasks = []
            for i in range(5):
                start = int(i * len(sorted_systems) / 5)
                end = int((i + 1) * len(sorted_systems) / 5)
                stratum = sorted_systems[start:end]
                for _ in range(n_per_stratum):
                    tasks.append(self.generate_task(self.rng.choice(stratum)))
            # Fill remainder
            while len(tasks) < n_tasks:
                tasks.append(self.generate_task())
            return tasks
        else:
            return [self.generate_task() for _ in range(n_tasks)]
    
    def _get_bounds(self, system: ChaoticSystem) -> Tuple[float, float]:
        """Get appropriate state space bounds for system."""
        bounds_map = {
            "logistic": (0, 1),
            "tent": (0, 1),
            "henon": (-1.5, 1.5),
            "standard": (0, 2 * np.pi),
            "lorenz": (-30, 30),
        }
        return bounds_map.get(system.family, (-1, 1))
    
    def _get_initial_condition(self, system: ChaoticSystem) -> np.ndarray:
        """Get random initial condition on attractor."""
        bounds = self._get_bounds(system)
        return self.rng.uniform(bounds[0] * 0.5 + 0.25, bounds[1] * 0.5 + 0.25, system.dim)


# =============================================================================
# PART 4: SOLVERS
# =============================================================================

class Solver(ABC):
    """Base class for solvers."""
    
    @abstractmethod
    def predict(self, task: Task) -> np.ndarray:
        """
        Return probability distribution over discretized states.
        
        Output shape: (n_states,) summing to 1
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class UniformSolver(Solver):
    """Baseline: uniform distribution (maximum entropy guess)."""
    
    @property
    def name(self) -> str:
        return "Uniform"
    
    def predict(self, task: Task) -> np.ndarray:
        n_states = task.discretizer.n_states
        return np.ones(n_states) / n_states


class MeanSolver(Solver):
    """Baseline: predict mean of observations (mean reversion)."""
    
    @property
    def name(self) -> str:
        return "MeanReversion"
    
    def predict(self, task: Task) -> np.ndarray:
        mean_obs = np.mean(task.observations, axis=0)
        center_bin = task.discretizer.state_to_bin(mean_obs)
        
        # Put most mass on center bin, spread remainder
        n_states = task.discretizer.n_states
        probs = np.ones(n_states) * 0.01 / n_states
        probs[center_bin] = 0.99
        return probs / probs.sum()


class LastValueSolver(Solver):
    """Baseline: predict last observed value persists."""
    
    @property
    def name(self) -> str:
        return "LastValue"
    
    def predict(self, task: Task) -> np.ndarray:
        last_obs = task.observations[-1]
        last_bin = task.discretizer.state_to_bin(last_obs)
        
        n_states = task.discretizer.n_states
        probs = np.ones(n_states) * 0.01 / n_states
        probs[last_bin] = 0.99
        return probs / probs.sum()


class HistogramSolver(Solver):
    """Use empirical distribution from observations."""
    
    @property
    def name(self) -> str:
        return "Histogram"
    
    def predict(self, task: Task) -> np.ndarray:
        n_states = task.discretizer.n_states
        counts = np.ones(n_states)  # Laplace smoothing
        
        for obs in task.observations:
            bin_idx = task.discretizer.state_to_bin(obs)
            counts[bin_idx] += 1
        
        return counts / counts.sum()


class LinearExtrapolator(Solver):
    """Fit linear model to recent observations and extrapolate."""
    
    @property
    def name(self) -> str:
        return "LinearExtrap"
    
    def predict(self, task: Task) -> np.ndarray:
        # Use last 10 observations for linear fit
        n_fit = min(10, len(task.observations))
        recent_obs = task.observations[-n_fit:]
        recent_times = task.obs_times[-n_fit:]
        
        # Fit linear model per dimension
        future_time = task.obs_times[-1] + task.future_time
        predicted = np.zeros(task.system.dim)
        
        for d in range(task.system.dim):
            if len(recent_times) > 1:
                slope = np.polyfit(recent_times, recent_obs[:, d], 1)[0]
                predicted[d] = recent_obs[-1, d] + slope * task.future_time
            else:
                predicted[d] = recent_obs[-1, d]
        
        # Clip to bounds
        bounds = task.discretizer.bounds
        predicted = np.clip(predicted, bounds[0], bounds[1])
        
        pred_bin = task.discretizer.state_to_bin(predicted)
        
        # Gaussian-like spread around prediction
        n_states = task.discretizer.n_states
        probs = np.zeros(n_states)
        
        # Add mass to nearby bins
        spread = max(1, n_states // 20)
        for offset in range(-spread, spread + 1):
            idx = (pred_bin + offset) % n_states
            weight = np.exp(-0.5 * (offset / (spread/2))**2)
            probs[idx] += weight
        
        return probs / probs.sum()


class NearestNeighborSolver(Solver):
    """Find similar patterns in observation history."""
    
    @property
    def name(self) -> str:
        return "NearestNeighbor"
    
    def predict(self, task: Task) -> np.ndarray:
        obs = task.observations
        n_obs = len(obs)
        
        if n_obs < 5:
            return UniformSolver().predict(task)
        
        # Handle NaN/inf in observations
        if not np.all(np.isfinite(obs)):
            return UniformSolver().predict(task)
        
        # Current pattern: last 3 observations
        pattern_len = min(3, n_obs - 2)
        current_pattern = obs[-pattern_len:]
        
        # Find similar patterns in history
        best_matches = []
        for i in range(pattern_len, n_obs - 1):
            hist_pattern = obs[i-pattern_len:i]
            dist = np.linalg.norm(current_pattern - hist_pattern)
            if np.isfinite(dist):
                next_obs = obs[i]
                best_matches.append((dist, next_obs))
        
        if not best_matches:
            return UniformSolver().predict(task)
        
        # Weight by inverse distance
        best_matches.sort(key=lambda x: x[0])
        top_k = min(5, len(best_matches))
        
        n_states = task.discretizer.n_states
        probs = np.ones(n_states) * 0.001  # Smoothing
        
        for dist, next_obs in best_matches[:top_k]:
            bin_idx = task.discretizer.state_to_bin(next_obs)
            weight = 1.0 / (dist + 0.1)
            probs[bin_idx] += weight
        
        return probs / probs.sum()


class ConditionalSolver(Solver):
    """
    Uses system family information if available.
    Applies family-specific heuristics.
    """
    
    @property
    def name(self) -> str:
        return "Conditional"
    
    def predict(self, task: Task) -> np.ndarray:
        if not task.conditional or not task.family:
            return HistogramSolver().predict(task)
        
        family = task.family
        obs = task.observations
        
        n_states = task.discretizer.n_states
        
        if family == "logistic":
            # Logistic map is ergodic with known invariant density
            # Approximate with arcsine-like distribution
            centers = np.linspace(0.01, 0.99, n_states)
            probs = 1.0 / (np.pi * np.sqrt(centers * (1 - centers)) + 0.1)
            
        elif family == "tent":
            # Tent map has uniform invariant measure
            probs = np.ones(n_states)
            
        elif family == "lorenz":
            # Lorenz has butterfly-shaped attractor
            # Use histogram but weight recent observations more
            probs = np.ones(n_states)
            for i, o in enumerate(obs):
                weight = 1 + i / len(obs)  # Recent obs weighted more
                bin_idx = task.discretizer.state_to_bin(o)
                probs[bin_idx] += weight
                
        else:
            # Default to histogram
            probs = np.ones(n_states)
            for o in obs:
                bin_idx = task.discretizer.state_to_bin(o)
                probs[bin_idx] += 1
        
        return probs / probs.sum()


# =============================================================================
# PART 5: EVALUATION
# =============================================================================

@dataclass
class TaskResult:
    """Result of a solver on a single task."""
    task_id: int
    h_ks: float
    nll: float  # Negative log likelihood
    correct_bin: bool  # Did argmax(probs) == true_bin?
    solve_time: float  # Wall clock time
    family: str


@dataclass
class PhiPoint:
    """A point on the Φ(t) curve."""
    wall_time: float
    cumulative_score: float
    tasks_completed: int


class DifficultyWeighting:
    """Configurable difficulty weighting functions."""
    
    @staticmethod
    def constant(h_ks: float) -> float:
        """w(h) = 1: Pure throughput."""
        return 1.0
    
    @staticmethod
    def linear(h_ks: float) -> float:
        """w(h) = h: Original Φ(t)."""
        return h_ks
    
    @staticmethod
    def quadratic(h_ks: float) -> float:
        """w(h) = h²: Reward hard tasks."""
        return h_ks ** 2
    
    @staticmethod
    def log(h_ks: float) -> float:
        """w(h) = log(1+h): Diminishing returns."""
        return np.log(1 + h_ks)
    
    @staticmethod
    def inverse(h_ks: float) -> float:
        """w(h) = 1/h: Reward easy tasks (throughput focus)."""
        return 1.0 / max(h_ks, 0.1)


class Evaluator:
    """Evaluate solvers and track Φ(t) curves."""
    
    def __init__(self, 
                 weighting: Callable[[float], float] = DifficultyWeighting.linear,
                 obs_cost: float = 0.0):
        """
        Args:
            weighting: Function w(h_KS) for difficulty weighting
            obs_cost: Cost λ per observation (for agency analysis)
        """
        self.weighting = weighting
        self.obs_cost = obs_cost
    
    def evaluate_task(self, solver: Solver, task: Task) -> TaskResult:
        """Evaluate solver on a single task."""
        start = time.time()
        probs = solver.predict(task)
        elapsed = time.time() - start
        
        # Compute NLL
        true_prob = probs[task.true_bin]
        nll = -np.log(max(true_prob, 1e-10))
        
        # Check if argmax is correct
        pred_bin = np.argmax(probs)
        correct = (pred_bin == task.true_bin)
        
        return TaskResult(
            task_id=task.task_id,
            h_ks=task.h_ks,
            nll=nll,
            correct_bin=correct,
            solve_time=elapsed,
            family=task.family
        )
    
    def evaluate_batch(self, solver: Solver, tasks: List[Task]) -> Tuple[List[TaskResult], List[PhiPoint]]:
        """
        Evaluate solver on batch, tracking Φ(t).
        
        Returns:
            results: Per-task results
            phi_curve: Φ(t) curve points
        """
        results = []
        phi_curve = []
        
        cumulative_score = 0.0
        start_time = time.time()
        
        for task in tasks:
            result = self.evaluate_task(solver, task)
            results.append(result)
            
            # Score: weighted accuracy minus observation cost
            # Accuracy from NLL: exp(-NLL) — difficulty already captured by weighting
            accuracy = np.exp(-result.nll)
            task_score = self.weighting(result.h_ks) * accuracy
            
            # Subtract observation cost
            task_score -= self.obs_cost * len(task.observations)
            
            cumulative_score += max(0, task_score)
            
            phi_curve.append(PhiPoint(
                wall_time=time.time() - start_time,
                cumulative_score=cumulative_score,
                tasks_completed=len(results)
            ))
        
        return results, phi_curve
    
    def compute_metrics(self, results: List[TaskResult], phi_curve: List[PhiPoint]) -> Dict:
        """Compute summary metrics."""
        nlls = [r.nll for r in results]
        h_ks_values = [r.h_ks for r in results]
        
        # Accuracy by difficulty band
        easy = [r for r in results if r.h_ks < 0.3]
        medium = [r for r in results if 0.3 <= r.h_ks < 0.7]
        hard = [r for r in results if r.h_ks >= 0.7]
        
        metrics = {
            "mean_nll": np.mean(nlls),
            "median_nll": np.median(nlls),
            "accuracy": np.mean([r.correct_bin for r in results]),
            "accuracy_easy": np.mean([r.correct_bin for r in easy]) if easy else 0,
            "accuracy_medium": np.mean([r.correct_bin for r in medium]) if medium else 0,
            "accuracy_hard": np.mean([r.correct_bin for r in hard]) if hard else 0,
            "final_phi": phi_curve[-1].cumulative_score if phi_curve else 0,
            "total_time": phi_curve[-1].wall_time if phi_curve else 0,
            "tasks_per_second": len(results) / phi_curve[-1].wall_time if phi_curve else 0,
        }
        
        # AUC of Φ(t) curve
        if len(phi_curve) >= 2:
            times = [p.wall_time for p in phi_curve]
            scores = [p.cumulative_score for p in phi_curve]
            auc = np.trapezoid(scores, times)
            metrics["phi_auc"] = auc
        
        return metrics


# =============================================================================
# PART 6: ANALYSIS AND VISUALIZATION
# =============================================================================

def run_benchmark(solvers: List[Solver], 
                  n_tasks: int = 100,
                  config: TaskConfig = None,
                  weighting: Callable = DifficultyWeighting.linear) -> Dict:
    """
    Run full benchmark comparison.
    
    Returns dictionary with results for each solver.
    """
    if config is None:
        config = TaskConfig()
    
    generator = TaskGenerator(config)
    tasks = generator.generate_batch(n_tasks, stratified=True)
    evaluator = Evaluator(weighting=weighting)
    
    all_results = {}
    
    for solver in solvers:
        print(f"Evaluating {solver.name}...")
        results, phi_curve = evaluator.evaluate_batch(solver, tasks)
        metrics = evaluator.compute_metrics(results, phi_curve)
        
        all_results[solver.name] = {
            "results": results,
            "phi_curve": phi_curve,
            "metrics": metrics
        }
        
        print(f"  Mean NLL: {metrics['mean_nll']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Final Φ: {metrics['final_phi']:.2f}")
    
    return all_results, tasks


def plot_phi_curves(all_results: Dict, title: str = "Φ(t) Curves"):
    """Plot Φ(t) curves for all solvers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (name, data), color in zip(all_results.items(), colors):
        phi_curve = data["phi_curve"]
        times = [p.wall_time for p in phi_curve]
        scores = [p.cumulative_score for p in phi_curve]
        ax.plot(times, scores, label=name, color=color, linewidth=2)
    
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Φ(t) = Σ w(h_KS) · accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_nll_vs_difficulty(all_results: Dict, tasks: List[Task]):
    """Plot NLL vs h_KS for each solver."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(all_results.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        
        results = data["results"]
        h_ks = [r.h_ks for r in results]
        nlls = [r.nll for r in results]
        
        ax.scatter(h_ks, nlls, alpha=0.5, s=20)
        
        # Trend line
        z = np.polyfit(h_ks, nlls, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(h_ks), max(h_ks), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f"slope={z[0]:.2f}")
        
        ax.set_xlabel("h_KS")
        ax.set_ylabel("NLL")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(all_results), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle("NLL vs Task Difficulty (h_KS)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_accuracy_by_difficulty(all_results: Dict):
    """Bar chart of accuracy by difficulty band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    solvers = list(all_results.keys())
    x = np.arange(len(solvers))
    width = 0.25
    
    easy_acc = [all_results[s]["metrics"]["accuracy_easy"] for s in solvers]
    med_acc = [all_results[s]["metrics"]["accuracy_medium"] for s in solvers]
    hard_acc = [all_results[s]["metrics"]["accuracy_hard"] for s in solvers]
    
    bars1 = ax.bar(x - width, easy_acc, width, label="Easy (h_KS < 0.3)", color="green", alpha=0.7)
    bars2 = ax.bar(x, med_acc, width, label="Medium (0.3 ≤ h_KS < 0.7)", color="orange", alpha=0.7)
    bars3 = ax.bar(x + width, hard_acc, width, label="Hard (h_KS ≥ 0.7)", color="red", alpha=0.7)
    
    ax.set_ylabel("Accuracy (argmax = true bin)")
    ax.set_xlabel("Solver")
    ax.set_title("Accuracy by Difficulty Band")
    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_weighting_comparison(n_tasks: int = 100):
    """Compare different difficulty weightings."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    weightings = [
        ("Constant (throughput)", DifficultyWeighting.constant),
        ("Linear (balanced)", DifficultyWeighting.linear),
        ("Quadratic (hard focus)", DifficultyWeighting.quadratic),
        ("Inverse (easy focus)", DifficultyWeighting.inverse),
    ]
    
    solvers = [
        UniformSolver(),
        MeanSolver(),
        HistogramSolver(),
        LinearExtrapolator(),
    ]
    
    config = TaskConfig()
    generator = TaskGenerator(config, seed=42)
    tasks = generator.generate_batch(n_tasks, stratified=True)
    
    for ax, (w_name, w_func) in zip(axes.flatten(), weightings):
        evaluator = Evaluator(weighting=w_func)
        
        for solver in solvers:
            results, phi_curve = evaluator.evaluate_batch(solver, tasks)
            times = [p.wall_time for p in phi_curve]
            scores = [p.cumulative_score for p in phi_curve]
            ax.plot(times, scores, label=solver.name, linewidth=2)
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Φ(t)")
        ax.set_title(f"Weighting: {w_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Effect of Difficulty Weighting on Φ(t) Rankings", fontsize=14)
    plt.tight_layout()
    return fig


def plot_conditional_vs_unconditional():
    """Compare performance with and without system family information."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    solvers = [
        UniformSolver(),
        HistogramSolver(),
        ConditionalSolver(),
        LinearExtrapolator(),
    ]
    
    # Conditional (family known)
    config_cond = TaskConfig(conditional=True)
    generator_cond = TaskGenerator(config_cond, seed=42)
    tasks_cond = generator_cond.generate_batch(100, stratified=True)
    
    evaluator = Evaluator(weighting=DifficultyWeighting.linear)
    
    for solver in solvers:
        results, phi_curve = evaluator.evaluate_batch(solver, tasks_cond)
        times = [p.wall_time for p in phi_curve]
        scores = [p.cumulative_score for p in phi_curve]
        axes[0].plot(times, scores, label=solver.name, linewidth=2)
    
    axes[0].set_title("Conditional (System Family Known)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Φ(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Unconditional (family unknown)
    config_uncond = TaskConfig(conditional=False)
    generator_uncond = TaskGenerator(config_uncond, seed=42)
    tasks_uncond = generator_uncond.generate_batch(100, stratified=True)
    
    for solver in solvers:
        results, phi_curve = evaluator.evaluate_batch(solver, tasks_uncond)
        times = [p.wall_time for p in phi_curve]
        scores = [p.cumulative_score for p in phi_curve]
        axes[1].plot(times, scores, label=solver.name, linewidth=2)
    
    axes[1].set_title("Unconditional (System Family Unknown)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Φ(t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Conditional vs Unconditional Prediction", fontsize=14)
    plt.tight_layout()
    return fig


def plot_observability_analysis():
    """Analyze effect of observation density and count."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vary observation count
    obs_counts = [10, 25, 50, 100]
    evaluator = Evaluator(weighting=DifficultyWeighting.linear)
    
    solver = HistogramSolver()
    
    for n_obs in obs_counts:
        config = TaskConfig(n_obs=n_obs)
        generator = TaskGenerator(config, seed=42)
        tasks = generator.generate_batch(50, stratified=True)
        
        results, phi_curve = evaluator.evaluate_batch(solver, tasks)
        times = [p.wall_time for p in phi_curve]
        scores = [p.cumulative_score for p in phi_curve]
        axes[0].plot(times, scores, label=f"n_obs={n_obs}", linewidth=2)
    
    axes[0].set_title("Effect of Observation Count")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Φ(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Vary observation density (sparsity)
    densities = [0.25, 0.5, 0.75, 1.0]
    
    for density in densities:
        config = TaskConfig(obs_density=density, n_obs=50)
        generator = TaskGenerator(config, seed=42)
        tasks = generator.generate_batch(50, stratified=True)
        
        results, phi_curve = evaluator.evaluate_batch(solver, tasks)
        times = [p.wall_time for p in phi_curve]
        scores = [p.cumulative_score for p in phi_curve]
        axes[1].plot(times, scores, label=f"density={density}", linewidth=2)
    
    axes[1].set_title("Effect of Observation Density")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Φ(t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Observability Analysis (HistogramSolver)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_system_complexity_spectrum():
    """Visualize the h_KS spectrum of available systems."""
    families = create_system_family()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: h_KS distribution by family
    ax = axes[0]
    family_names = list(families.keys())
    positions = []
    labels = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(family_names)))
    
    for i, (fname, systems) in enumerate(families.items()):
        h_ks_values = [s.h_ks for s in systems]
        ax.scatter([i] * len(h_ks_values), h_ks_values, 
                   s=100, alpha=0.7, color=colors[i], label=fname)
        positions.append(i)
        labels.append(fname)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("h_KS (Kolmogorov-Sinai entropy)")
    ax.set_title("System Complexity by Family")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Right: Overall h_KS histogram
    ax = axes[1]
    all_h_ks = [s.h_ks for fam in families.values() for s in fam]
    ax.hist(all_h_ks, bins=15, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(all_h_ks), color="red", linestyle="--", 
               label=f"mean={np.mean(all_h_ks):.2f}")
    ax.set_xlabel("h_KS")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Task Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# PART 7: MAIN
# =============================================================================

def main():
    """Run full benchmark analysis."""
    print("=" * 60)
    print("ChaosBench v3: Information-Efficient Chaotic Prediction")
    print("=" * 60)

    # Use new defaults: scaled horizon, filtered systems
    config = TaskConfig(
        horizon_lyapunov_multiplier=1.5,  # ~1.5 Lyapunov times
        min_h_ks=0.1,  # Filter non-chaotic systems
    )

    # Create solvers
    solvers = [
        UniformSolver(),
        MeanSolver(),
        LastValueSolver(),
        HistogramSolver(),
        LinearExtrapolator(),
        NearestNeighborSolver(),
        ConditionalSolver(),
    ]

    # Run benchmark with explicit config
    print("\n1. Running main benchmark...")
    all_results, tasks = run_benchmark(solvers, n_tasks=100, config=config)
    
    # Generate plots
    print("\n2. Generating visualizations...")
    
    fig1 = plot_phi_curves(all_results, "Φ(t) Curves: Linear Difficulty Weighting")
    fig1.savefig("phi_curves.png", dpi=150, bbox_inches="tight")
    print("   Saved: phi_curves.png")
    
    fig2 = plot_nll_vs_difficulty(all_results, tasks)
    fig2.savefig("nll_vs_difficulty.png", dpi=150, bbox_inches="tight")
    print("   Saved: nll_vs_difficulty.png")
    
    fig3 = plot_accuracy_by_difficulty(all_results)
    fig3.savefig("accuracy_by_difficulty.png", dpi=150, bbox_inches="tight")
    print("   Saved: accuracy_by_difficulty.png")
    
    fig4 = plot_weighting_comparison(n_tasks=80)
    fig4.savefig("weighting_comparison.png", dpi=150, bbox_inches="tight")
    print("   Saved: weighting_comparison.png")
    
    fig5 = plot_conditional_vs_unconditional()
    fig5.savefig("conditional_comparison.png", dpi=150, bbox_inches="tight")
    print("   Saved: conditional_comparison.png")
    
    fig6 = plot_observability_analysis()
    fig6.savefig("observability_analysis.png", dpi=150, bbox_inches="tight")
    print("   Saved: observability_analysis.png")
    
    fig7 = plot_system_complexity_spectrum()
    fig7.savefig("complexity_spectrum.png", dpi=150, bbox_inches="tight")
    print("   Saved: complexity_spectrum.png")
    
    # Summary table
    print("\n3. Summary Metrics:")
    print("-" * 80)
    print(f"{'Solver':<18} {'Mean NLL':>10} {'Accuracy':>10} {'Easy Acc':>10} {'Hard Acc':>10} {'Φ(final)':>10}")
    print("-" * 80)
    
    for name, data in all_results.items():
        m = data["metrics"]
        print(f"{name:<18} {m['mean_nll']:>10.3f} {m['accuracy']:>10.1%} "
              f"{m['accuracy_easy']:>10.1%} {m['accuracy_hard']:>10.1%} {m['final_phi']:>10.2f}")
    
    print("-" * 80)
    
    # Key findings
    print("\n4. Key Findings:")
    
    # Best overall
    best_phi = max(all_results.items(), key=lambda x: x[1]["metrics"]["final_phi"])
    print(f"   - Best Φ(t): {best_phi[0]} ({best_phi[1]['metrics']['final_phi']:.2f})")
    
    # Best on hard tasks
    best_hard = max(all_results.items(), key=lambda x: x[1]["metrics"]["accuracy_hard"])
    print(f"   - Best on hard tasks: {best_hard[0]} ({best_hard[1]['metrics']['accuracy_hard']:.1%})")
    
    # Fastest
    fastest = max(all_results.items(), key=lambda x: x[1]["metrics"]["tasks_per_second"])
    print(f"   - Fastest throughput: {fastest[0]} ({fastest[1]['metrics']['tasks_per_second']:.1f} tasks/s)")
    
    plt.close("all")
    print("\n✓ Benchmark complete!")
    
    return all_results, tasks


if __name__ == "__main__":
    all_results, tasks = main()