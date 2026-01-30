"""
Tests for Lyapunov exponent computation.

Validates against known literature values:
- Logistic map r=4: λ = ln(2) ≈ 0.693 (exact)
- Tent map μ=2: λ = ln(2) ≈ 0.693 (exact)
- Hénon map (a=1.4, b=0.3): λ1 ≈ 0.42 (Wolf et al. 1985)
- Lorenz system (σ=10, ρ=28, β=8/3): λ1 ≈ 0.906 (Wolf et al. 1985)
"""

import numpy as np
import pytest

from chaosbench.core.lyapunov import (
    compute_lyapunov_1d,
    compute_lyapunov_spectrum,
    compute_lyapunov_continuous,
    compute_h_ks,
    KNOWN_VALUES,
)


class TestLyapunov1D:
    """Tests for 1D Lyapunov exponent computation."""

    def test_logistic_map_r4(self):
        """Logistic map r=4 should have λ = ln(2) ≈ 0.693."""
        r = 4.0

        def f(x):
            return r * x * (1 - x)

        def df(x):
            return r * (1 - 2 * x)

        # Use x0=0.3 (not 0.5 which is the critical point where f'=0)
        lyap = compute_lyapunov_1d(f, df, x0=0.3, n_iter=10000, n_burn=1000)

        expected = np.log(2)
        tolerance = KNOWN_VALUES["logistic_r4"]["tolerance"]
        assert abs(lyap - expected) < tolerance, (
            f"Logistic r=4: expected λ ≈ {expected:.4f}, got {lyap:.4f}"
        )

    def test_logistic_map_r38(self):
        """Logistic map r=3.8 should have positive Lyapunov exponent."""
        r = 3.8

        def f(x):
            return r * x * (1 - x)

        def df(x):
            return r * (1 - 2 * x)

        lyap = compute_lyapunov_1d(f, df, x0=0.5, n_iter=10000, n_burn=1000)

        # r=3.8 is in chaotic regime, λ should be positive but less than ln(2)
        assert lyap > 0, f"Logistic r=3.8: expected λ > 0, got {lyap:.4f}"
        assert lyap < np.log(2), f"Logistic r=3.8: expected λ < ln(2), got {lyap:.4f}"

    def test_tent_map_mu2(self):
        """Tent map μ=2 should have λ = ln(2) ≈ 0.693."""
        mu = 2.0

        def f(x):
            return mu * x if x < 0.5 else mu * (1 - x)

        def df(x):
            return mu  # |f'| = μ everywhere

        lyap = compute_lyapunov_1d(f, df, x0=0.3, n_iter=10000, n_burn=1000)

        expected = np.log(2)
        tolerance = KNOWN_VALUES["tent_mu2"]["tolerance"]
        assert abs(lyap - expected) < tolerance, (
            f"Tent μ=2: expected λ ≈ {expected:.4f}, got {lyap:.4f}"
        )

    def test_tent_map_mu15(self):
        """Tent map μ=1.5 should have λ = ln(1.5) ≈ 0.405."""
        mu = 1.5

        def f(x):
            return mu * x if x < 0.5 else mu * (1 - x)

        def df(x):
            return mu

        lyap = compute_lyapunov_1d(f, df, x0=0.3, n_iter=10000, n_burn=1000)

        expected = np.log(1.5)
        assert abs(lyap - expected) < 0.02, (
            f"Tent μ=1.5: expected λ ≈ {expected:.4f}, got {lyap:.4f}"
        )


class TestLyapunovSpectrum:
    """Tests for multi-dimensional Lyapunov spectrum computation."""

    def test_henon_classic(self):
        """Hénon map (a=1.4, b=0.3) should have λ1 ≈ 0.42."""
        a, b = 1.4, 0.3

        def f(x):
            x_new = 1 - a * x[0] ** 2 + x[1]
            y_new = b * x[0]
            return np.clip(np.array([x_new, y_new]), -10, 10)

        def jacobian(x):
            return np.array([[-2 * a * x[0], 1.0], [b, 0.0]])

        lyap_spectrum = compute_lyapunov_spectrum(
            f=f,
            jacobian=jacobian,
            x0=np.array([0.0, 0.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )

        lambda1 = lyap_spectrum[0]
        expected = KNOWN_VALUES["henon_classic"]["lambda1"]
        tolerance = KNOWN_VALUES["henon_classic"]["tolerance"]

        assert abs(lambda1 - expected) < tolerance, (
            f"Hénon: expected λ1 ≈ {expected:.3f}, got {lambda1:.3f}"
        )

    def test_henon_spectrum_sum(self):
        """Hénon map should have sum of exponents ≈ ln|b| = ln(0.3)."""
        a, b = 1.4, 0.3

        def f(x):
            x_new = 1 - a * x[0] ** 2 + x[1]
            y_new = b * x[0]
            return np.clip(np.array([x_new, y_new]), -10, 10)

        def jacobian(x):
            return np.array([[-2 * a * x[0], 1.0], [b, 0.0]])

        lyap_spectrum = compute_lyapunov_spectrum(
            f=f,
            jacobian=jacobian,
            x0=np.array([0.0, 0.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )

        # For dissipative maps, sum of exponents = ln|det(J)| = ln|b|
        sum_lyap = np.sum(lyap_spectrum)
        expected_sum = np.log(abs(b))

        assert abs(sum_lyap - expected_sum) < 0.05, (
            f"Hénon sum: expected Σλ ≈ {expected_sum:.3f}, got {sum_lyap:.3f}"
        )

    def test_standard_map_k5(self):
        """Standard map K=5 should have positive λ1 (strongly chaotic)."""
        K = 5.0

        def f(x):
            p, q = x
            p_new = (p + K * np.sin(q)) % (2 * np.pi)
            q_new = (q + p_new) % (2 * np.pi)
            return np.array([p_new, q_new])

        def jacobian(x):
            _, q = x
            k_cos_q = K * np.cos(q)
            return np.array([[1.0, k_cos_q], [1.0, 1.0 + k_cos_q]])

        lyap_spectrum = compute_lyapunov_spectrum(
            f=f,
            jacobian=jacobian,
            x0=np.array([1.0, 1.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )

        lambda1 = lyap_spectrum[0]
        assert lambda1 > 0.5, f"Standard K=5: expected λ1 > 0.5, got {lambda1:.3f}"

    def test_standard_map_symplectic(self):
        """Standard map is symplectic: λ1 + λ2 ≈ 0."""
        K = 3.0

        def f(x):
            p, q = x
            p_new = (p + K * np.sin(q)) % (2 * np.pi)
            q_new = (q + p_new) % (2 * np.pi)
            return np.array([p_new, q_new])

        def jacobian(x):
            _, q = x
            k_cos_q = K * np.cos(q)
            return np.array([[1.0, k_cos_q], [1.0, 1.0 + k_cos_q]])

        lyap_spectrum = compute_lyapunov_spectrum(
            f=f,
            jacobian=jacobian,
            x0=np.array([1.0, 1.0]),
            dim=2,
            n_iter=10000,
            n_burn=1000,
        )

        sum_lyap = np.sum(lyap_spectrum)
        assert abs(sum_lyap) < 0.05, (
            f"Standard map symplectic: expected Σλ ≈ 0, got {sum_lyap:.4f}"
        )


class TestLyapunovContinuous:
    """Tests for continuous flow Lyapunov exponent computation."""

    def test_lorenz_classic(self):
        """Lorenz system (σ=10, ρ=28, β=8/3) should have λ1 ≈ 0.906."""
        sigma, rho, beta = 10.0, 28.0, 8 / 3

        def flow(x):
            return np.array([
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ])

        def jacobian(x):
            return np.array([
                [-sigma, sigma, 0.0],
                [rho - x[2], -1.0, -x[0]],
                [x[1], x[0], -beta],
            ])

        lyap_spectrum = compute_lyapunov_continuous(
            flow=flow,
            jacobian=jacobian,
            x0=np.array([1.0, 1.0, 1.0]),
            dim=3,
            dt=0.01,
            t_total=100.0,
            t_burn=10.0,
            ortho_time=1.0,
        )

        lambda1 = lyap_spectrum[0]
        expected = KNOWN_VALUES["lorenz_classic"]["lambda1"]
        tolerance = KNOWN_VALUES["lorenz_classic"]["tolerance"]

        assert abs(lambda1 - expected) < tolerance, (
            f"Lorenz: expected λ1 ≈ {expected:.3f}, got {lambda1:.3f}"
        )

    def test_lorenz_zero_exponent(self):
        """Lorenz system should have one exponent near zero (flow direction)."""
        sigma, rho, beta = 10.0, 28.0, 8 / 3

        def flow(x):
            return np.array([
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ])

        def jacobian(x):
            return np.array([
                [-sigma, sigma, 0.0],
                [rho - x[2], -1.0, -x[0]],
                [x[1], x[0], -beta],
            ])

        lyap_spectrum = compute_lyapunov_continuous(
            flow=flow,
            jacobian=jacobian,
            x0=np.array([1.0, 1.0, 1.0]),
            dim=3,
            dt=0.01,
            t_total=100.0,
            t_burn=10.0,
            ortho_time=1.0,
        )

        lambda2 = lyap_spectrum[1]
        assert abs(lambda2) < 0.1, (
            f"Lorenz: expected λ2 ≈ 0, got {lambda2:.3f}"
        )


class TestHKS:
    """Tests for Kolmogorov-Sinai entropy computation."""

    def test_h_ks_from_positive_exponents(self):
        """h_KS should be sum of positive Lyapunov exponents."""
        spectrum = np.array([0.5, 0.2, -0.3, -1.0])
        h_ks = compute_h_ks(spectrum)
        assert abs(h_ks - 0.7) < 1e-10, f"Expected h_KS = 0.7, got {h_ks}"

    def test_h_ks_single_positive(self):
        """h_KS with single positive exponent."""
        spectrum = np.array([0.42, -1.62])  # Hénon-like
        h_ks = compute_h_ks(spectrum)
        assert abs(h_ks - 0.42) < 1e-10, f"Expected h_KS = 0.42, got {h_ks}"

    def test_h_ks_all_negative(self):
        """h_KS should be 0 if all exponents are negative."""
        spectrum = np.array([-0.1, -0.5, -1.0])
        h_ks = compute_h_ks(spectrum)
        assert h_ks == 0.0, f"Expected h_KS = 0, got {h_ks}"


class TestSystemIntegration:
    """Integration tests with the ChaosBench system classes."""

    def test_logistic_system_h_ks(self):
        """LogisticMap should compute h_KS correctly."""
        from chaosbench.core.Chaosbench_v3 import LogisticMap

        system = LogisticMap(r=4.0)
        expected = np.log(2)
        tolerance = 0.02

        assert abs(system.h_ks - expected) < tolerance, (
            f"LogisticMap r=4: expected h_KS ≈ {expected:.3f}, got {system.h_ks:.3f}"
        )

    def test_tent_system_h_ks(self):
        """TentMap should compute h_KS correctly."""
        from chaosbench.core.Chaosbench_v3 import TentMap

        system = TentMap(mu=2.0)
        expected = np.log(2)
        tolerance = 0.02

        assert abs(system.h_ks - expected) < tolerance, (
            f"TentMap μ=2: expected h_KS ≈ {expected:.3f}, got {system.h_ks:.3f}"
        )

    def test_henon_system_h_ks(self):
        """HenonMap should compute h_KS correctly."""
        from chaosbench.core.Chaosbench_v3 import HenonMap

        system = HenonMap(a=1.4, b=0.3)
        expected = 0.42
        tolerance = 0.03

        assert abs(system.h_ks - expected) < tolerance, (
            f"HenonMap: expected h_KS ≈ {expected:.3f}, got {system.h_ks:.3f}"
        )

    def test_lorenz_system_h_ks(self):
        """LorenzDisc should compute h_KS correctly."""
        from chaosbench.core.Chaosbench_v3 import LorenzDisc

        system = LorenzDisc(rho=28.0)
        expected = 0.906
        tolerance = 0.1  # Continuous flows have more variance

        assert abs(system.h_ks - expected) < tolerance, (
            f"LorenzDisc: expected h_KS ≈ {expected:.3f}, got {system.h_ks:.3f}"
        )

    def test_h_ks_ordering(self):
        """Systems with higher chaos should have higher h_KS."""
        from chaosbench.core.Chaosbench_v3 import LogisticMap

        # r=3.6 is mildly chaotic, r=4.0 is maximally chaotic
        mild = LogisticMap(r=3.7)
        strong = LogisticMap(r=4.0)

        assert strong.h_ks > mild.h_ks, (
            f"Expected h_KS(r=4.0) > h_KS(r=3.7), got {strong.h_ks:.3f} vs {mild.h_ks:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
