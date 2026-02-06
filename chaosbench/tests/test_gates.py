"""Tests for chaosbench.validation.gates — Stage 1 hard gates."""

import numpy as np
import pytest

from chaosbench.grammar.atoms import LogisticAtom, TentAtom, DampedLinearAtom, RotationAtom
from chaosbench.validation.gates import (
    check_parameter_bounds,
    check_trajectory_stability,
    check_attractor_bounds,
    check_minimum_lyapunov,
    check_periodicity,
    check_permutation_entropy,
    check_autocorrelation,
    validate_stage1,
)


class TestParameterBounds:
    def test_valid(self):
        passed, _ = check_parameter_bounds("logistic", {"r": 3.5}, {"r": (2.5, 4.0)})
        assert passed

    def test_out_of_range(self):
        passed, reason = check_parameter_bounds("logistic", {"r": 5.0}, {"r": (2.5, 4.0)})
        assert not passed
        assert "outside" in reason

    def test_unknown_param(self):
        passed, _ = check_parameter_bounds("logistic", {"q": 1.0}, {"r": (2.5, 4.0)})
        assert not passed


class TestTrajectoryStability:
    def test_stable(self):
        traj = np.array([0.1, 0.5, 0.3, 0.7])
        passed, _ = check_trajectory_stability(traj)
        assert passed

    def test_nan(self):
        traj = np.array([0.1, np.nan, 0.3])
        passed, _ = check_trajectory_stability(traj)
        assert not passed

    def test_inf(self):
        traj = np.array([0.1, np.inf, 0.3])
        passed, _ = check_trajectory_stability(traj)
        assert not passed

    def test_too_large(self):
        traj = np.array([0.1, 1e11, 0.3])
        passed, _ = check_trajectory_stability(traj)
        assert not passed


class TestAttractorBounds:
    def test_within_domain(self):
        traj = np.array([0.1, 0.5, 0.9])
        passed, _ = check_attractor_bounds(traj, (0.0, 1.0))
        assert passed

    def test_below_domain(self):
        traj = np.array([-0.5, 0.5, 0.9])
        passed, _ = check_attractor_bounds(traj, (0.0, 1.0), margin=0.1)
        assert not passed

    def test_margin_respected(self):
        traj = np.array([-0.05, 0.5, 1.05])
        passed, _ = check_attractor_bounds(traj, (0.0, 1.0), margin=0.1)
        assert passed


class TestMinimumLyapunov:
    def test_positive(self):
        passed, _ = check_minimum_lyapunov(0.5)
        assert passed

    def test_below_threshold(self):
        passed, _ = check_minimum_lyapunov(0.005)
        assert not passed


class TestPeriodicity:
    def test_chaotic_not_periodic(self):
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 8000)
        passed, _ = check_periodicity(traj)
        assert passed

    def test_periodic_detected(self):
        # Period-2 orbit
        traj = np.tile([0.3, 0.7], 500)
        passed, reason = check_periodicity(traj, max_period=100)
        assert not passed
        assert "period" in reason.lower()

    def test_insufficient_length_fails(self):
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 2000)
        passed, reason = check_periodicity(traj)
        assert not passed
        assert "insufficient" in reason.lower()


class TestPermutationEntropy:
    def test_chaotic_high_pe(self):
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 2000)
        passed, _ = check_permutation_entropy(traj)
        assert passed

    def test_constant_low_pe(self):
        traj = np.ones(200)
        passed, _ = check_permutation_entropy(traj, threshold=0.5)
        assert not passed

    def test_short_trajectory(self):
        traj = np.array([1.0, 2.0])
        passed, _ = check_permutation_entropy(traj, order=5)
        assert not passed


class TestAutocorrelation:
    def test_chaotic_passes(self):
        """Chaotic logistic should pass ACF gate (threshold=0.95)."""
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 5000)[1000:]
        passed, _ = check_autocorrelation(traj)
        assert passed

    def test_constant_fails(self):
        """Zero variance should fail."""
        traj = np.ones(100)
        passed, _ = check_autocorrelation(traj)
        assert not passed

    def test_perfect_correlation_fails(self):
        """Near-perfect ACF should fail — catches periodic mislabelling."""
        # Slowly growing linear sequence has ACF ≈ 1
        traj = np.linspace(0, 1, 200)
        passed, _ = check_autocorrelation(traj)
        assert not passed


class TestValidateStage1:
    def test_chaotic_passes(self):
        atom = LogisticAtom(r=3.891)
        traj = atom.trajectory(0.3, 8000)[1000:]  # Burn in
        lam = atom.lyapunov()
        passed, results = validate_stage1(
            trajectory=traj,
            family="logistic",
            params={"r": 3.891},
            valid_ranges={"r": (2.5, 4.0)},
            domain=(0.0, 1.0),
            regime="chaotic",
            lambda_max=lam,
        )
        assert passed
        # Chaotic should have all 7 gates
        gate_names = [name for name, _, _ in results]
        assert "permutation_entropy" in gate_names
        assert "autocorrelation" in gate_names

    def test_quasiperiodic_skips_chaotic_only_gates(self):
        atom = RotationAtom(omega=0.381966011)
        traj = atom.trajectory(0.3, 200)
        passed, results = validate_stage1(
            trajectory=traj,
            family="rotation",
            params={"omega": 0.381966011},
            valid_ranges={"omega": (0.0, 1.0)},
            domain=(0.0, 1.0),
            regime="quasiperiodic",
            lambda_max=atom.lyapunov(),
        )
        assert passed
        gate_names = [name for name, _, _ in results]
        assert "minimum_lyapunov" not in gate_names
        assert "periodicity" not in gate_names
        assert "autocorrelation" not in gate_names
        assert "permutation_entropy" not in gate_names

    def test_fixed_point_fewer_gates(self):
        atom = DampedLinearAtom(lam=0.5)
        traj = atom.trajectory(5.0, 200)
        passed, results = validate_stage1(
            trajectory=traj,
            family="damped_linear",
            params={"lam": 0.5},
            valid_ranges={"lam": (0.0, 0.99)},
            domain=(-10.0, 10.0),
            regime="fixed_point",
            lambda_max=atom.lyapunov(),
        )
        assert passed
        gate_names = [name for name, _, _ in results]
        # Fixed point should NOT have chaotic-only gates
        assert "minimum_lyapunov" not in gate_names
        assert "periodicity" not in gate_names
        assert "autocorrelation" not in gate_names
        assert "permutation_entropy" not in gate_names
