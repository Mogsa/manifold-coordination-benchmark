"""
Lyapunov Exponent Computation for ChaosBench
=============================================

Computes Lyapunov exponents and Kolmogorov-Sinai entropy for chaotic systems.

Methods:
- 1D maps: Direct formula λ = lim (1/n) Σ ln|f'(x_i)|
- Multi-D maps: QR decomposition method for full Lyapunov spectrum
- Continuous flows: Integrate variational equations

References:
- Wolf et al. (1985) "Determining Lyapunov exponents from a time series"
- Pesin's theorem: h_KS = Σ λ_i for λ_i > 0
"""

import numpy as np
from typing import Callable, Optional, Tuple


def compute_lyapunov_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    n_iter: int = 10000,
    n_burn: int = 1000,
) -> float:
    """
    Compute Lyapunov exponent for a 1D map.

    Uses the direct formula: λ = lim (1/n) Σ ln|f'(x_i)|

    Args:
        f: The map function x_{n+1} = f(x_n)
        df: Derivative df/dx
        x0: Initial condition
        n_iter: Number of iterations for averaging
        n_burn: Burn-in iterations to reach attractor

    Returns:
        Lyapunov exponent λ (bits/iteration)
    """
    x = x0

    # Burn-in to reach attractor
    for _ in range(n_burn):
        x = f(x)
        # Escape detection
        if not np.isfinite(x) or abs(x) > 1e10:
            return 0.0

    # Accumulate log of derivatives
    log_sum = 0.0
    valid_count = 0

    for _ in range(n_iter):
        deriv = abs(df(x))
        if deriv > 1e-15:  # Avoid log(0)
            log_sum += np.log(deriv)
            valid_count += 1

        x = f(x)

        # Escape detection
        if not np.isfinite(x) or abs(x) > 1e10:
            break

    if valid_count == 0:
        return 0.0

    return log_sum / valid_count


def compute_lyapunov_spectrum(
    f: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dim: int,
    n_iter: int = 10000,
    n_burn: int = 1000,
    ortho_freq: int = 10,
) -> np.ndarray:
    """
    Compute full Lyapunov spectrum for a multi-dimensional map.

    Uses QR decomposition method:
    1. Evolve tangent vectors alongside trajectory
    2. Periodically orthonormalize via QR
    3. Accumulate log(diag(R)) → Lyapunov spectrum

    Args:
        f: The map function x_{n+1} = f(x_n)
        jacobian: Function returning Jacobian matrix at x
        x0: Initial condition (dim,)
        dim: State space dimension
        n_iter: Number of iterations for averaging
        n_burn: Burn-in iterations
        ortho_freq: How often to orthonormalize (every ortho_freq steps)

    Returns:
        Array of Lyapunov exponents, sorted descending
    """
    x = x0.copy()

    # Initialize tangent vectors as identity matrix
    Q = np.eye(dim)

    # Burn-in
    for _ in range(n_burn):
        x = f(x)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e10:
            return np.zeros(dim)

    # Accumulate Lyapunov sums
    lyap_sums = np.zeros(dim)
    n_ortho = 0

    for i in range(n_iter):
        # Get Jacobian at current point
        J = jacobian(x)

        # Evolve tangent vectors: Q_new = J @ Q
        Q = J @ Q

        # Periodically orthonormalize
        if (i + 1) % ortho_freq == 0:
            Q, R = np.linalg.qr(Q)

            # Accumulate log of diagonal elements
            diag_R = np.abs(np.diag(R))
            diag_R = np.maximum(diag_R, 1e-15)  # Avoid log(0)
            lyap_sums += np.log(diag_R)
            n_ortho += 1

        # Evolve state
        x = f(x)

        # Escape detection
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e10:
            break

    if n_ortho == 0:
        return np.zeros(dim)

    # Average over orthonormalizations (and account for ortho_freq steps each)
    lyapunov_exponents = lyap_sums / (n_ortho * ortho_freq)

    # Sort descending
    return np.sort(lyapunov_exponents)[::-1]


def compute_lyapunov_continuous(
    flow: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dim: int,
    dt: float = 0.01,
    t_total: float = 100.0,
    t_burn: float = 10.0,
    ortho_time: float = 1.0,
) -> np.ndarray:
    """
    Compute Lyapunov spectrum for a continuous flow using RK4.

    Integrates the variational equations alongside the trajectory.

    Args:
        flow: dx/dt = flow(x)
        jacobian: Jacobian of flow at x
        x0: Initial condition
        dim: State space dimension
        dt: Integration timestep
        t_total: Total integration time
        t_burn: Burn-in time
        ortho_time: Time between orthonormalizations

    Returns:
        Array of Lyapunov exponents, sorted descending
    """
    def rk4_step(x, f, h):
        """RK4 integration step."""
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        return x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def rk4_step_tangent(x, Y, f, jac, h):
        """RK4 step for both state and tangent vectors."""
        # State evolution
        k1_x = f(x)
        k2_x = f(x + 0.5 * h * k1_x)
        k3_x = f(x + 0.5 * h * k2_x)
        k4_x = f(x + h * k3_x)
        x_new = x + (h / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)

        # Tangent evolution: dY/dt = J(x) @ Y
        J = jac(x)
        k1_Y = J @ Y
        J2 = jac(x + 0.5 * h * k1_x)
        k2_Y = J2 @ (Y + 0.5 * h * k1_Y)
        J3 = jac(x + 0.5 * h * k2_x)
        k3_Y = J3 @ (Y + 0.5 * h * k2_Y)
        J4 = jac(x + h * k3_x)
        k4_Y = J4 @ (Y + h * k3_Y)
        Y_new = Y + (h / 6) * (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y)

        return x_new, Y_new

    x = x0.copy()

    # Burn-in
    n_burn_steps = int(t_burn / dt)
    for _ in range(n_burn_steps):
        x = rk4_step(x, flow, dt)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e10:
            return np.zeros(dim)

    # Initialize tangent vectors
    Y = np.eye(dim)

    # Orthonormalization schedule
    steps_per_ortho = int(ortho_time / dt)
    n_total_steps = int((t_total - t_burn) / dt)

    lyap_sums = np.zeros(dim)
    n_ortho = 0
    step_count = 0

    for _ in range(n_total_steps):
        x, Y = rk4_step_tangent(x, Y, flow, jacobian, dt)
        step_count += 1

        # Orthonormalize periodically
        if step_count >= steps_per_ortho:
            Q, R = np.linalg.qr(Y)
            Y = Q

            diag_R = np.abs(np.diag(R))
            diag_R = np.maximum(diag_R, 1e-15)
            lyap_sums += np.log(diag_R)
            n_ortho += 1
            step_count = 0

        # Escape detection
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e10:
            break

    if n_ortho == 0:
        return np.zeros(dim)

    # Divide by total time to get exponents in units of 1/time
    total_time = n_ortho * ortho_time
    lyapunov_exponents = lyap_sums / total_time

    return np.sort(lyapunov_exponents)[::-1]


def compute_h_ks(lyapunov_exponents: np.ndarray) -> float:
    """
    Compute Kolmogorov-Sinai entropy from Lyapunov exponents.

    By Pesin's theorem: h_KS = Σ λ_i for all λ_i > 0

    Args:
        lyapunov_exponents: Array of Lyapunov exponents

    Returns:
        Kolmogorov-Sinai entropy (bits per iteration/time)
    """
    positive_exponents = lyapunov_exponents[lyapunov_exponents > 0]
    return float(np.sum(positive_exponents))


# =============================================================================
# Known Values for Validation
# =============================================================================

# Literature values for validation (Wolf et al. 1985, etc.)
KNOWN_VALUES = {
    # Logistic map r=4: exactly ln(2)
    "logistic_r4": {
        "lambda": np.log(2),  # ≈ 0.693
        "h_ks": np.log(2),
        "tolerance": 0.01,
    },
    # Tent map μ=2: exactly ln(2)
    "tent_mu2": {
        "lambda": np.log(2),
        "h_ks": np.log(2),
        "tolerance": 0.01,
    },
    # Hénon map (a=1.4, b=0.3): λ1 ≈ 0.42, λ2 ≈ -1.62
    "henon_classic": {
        "lambda1": 0.42,
        "lambda2": -1.62,
        "h_ks": 0.42,
        "tolerance": 0.02,
    },
    # Lorenz system (σ=10, ρ=28, β=8/3): λ1 ≈ 0.906, λ2 ≈ 0, λ3 ≈ -14.57
    "lorenz_classic": {
        "lambda1": 0.906,
        "lambda2": 0.0,
        "lambda3": -14.57,
        "h_ks": 0.906,
        "tolerance": 0.05,
    },
}
