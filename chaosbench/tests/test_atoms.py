"""Tests for chaosbench.grammar.atoms — all 4 MVP atoms."""

import numpy as np
import pytest

from chaosbench.grammar.atoms import (
    Atom,
    DampedLinearAtom,
    LogisticAtom,
    RotationAtom,
    TentAtom,
)


# ── LogisticAtom ──────────────────────────────────────────────────────────

class TestLogisticAtom:
    def test_iterate(self):
        atom = LogisticAtom(r=4.0)
        assert atom.iterate(0.5) == pytest.approx(1.0)
        assert atom.iterate(0.25) == pytest.approx(0.75)

    def test_derivative(self):
        atom = LogisticAtom(r=4.0)
        assert atom.derivative(0.5) == pytest.approx(0.0)
        assert atom.derivative(0.0) == pytest.approx(4.0)

    def test_domain(self):
        atom = LogisticAtom(r=3.5)
        assert atom.domain == (0.0, 1.0)

    def test_family(self):
        atom = LogisticAtom(r=3.5)
        assert atom.family == "logistic"

    def test_params(self):
        atom = LogisticAtom(r=3.5)
        assert atom.params == {"r": 3.5}

    def test_lyapunov_r4(self):
        """Logistic r=4 should have λ ≈ ln(2)."""
        atom = LogisticAtom(r=4.0)
        lam = atom.lyapunov()
        assert lam == pytest.approx(np.log(2), abs=0.02)

    def test_regime_fixed_point(self):
        atom = LogisticAtom(r=2.73)
        assert atom.regime() == "fixed_point"

    def test_regime_periodic(self):
        atom = LogisticAtom(r=3.236)
        assert atom.regime() == "periodic"

    def test_regime_chaotic(self):
        """r=3.891 is verified chaotic (NOT in periodic window)."""
        atom = LogisticAtom(r=3.891)
        assert atom.regime() == "chaotic"

    def test_param_validation_low(self):
        with pytest.raises(ValueError):
            LogisticAtom(r=2.0)

    def test_param_validation_high(self):
        with pytest.raises(ValueError):
            LogisticAtom(r=4.1)

    def test_trajectory_shape(self):
        atom = LogisticAtom(r=3.5)
        traj = atom.trajectory(0.5, 100)
        assert traj.shape == (100,)
        assert traj[0] == 0.5

    def test_name(self):
        atom = LogisticAtom(r=3.5)
        assert atom.name == "logistic(r=3.5)"


# ── TentAtom ──────────────────────────────────────────────────────────────

class TestTentAtom:
    def test_iterate(self):
        atom = TentAtom(mu=1.5)
        assert atom.iterate(0.3) == pytest.approx(0.45)
        assert atom.iterate(0.8) == pytest.approx(0.3)  # 1.5 * min(0.8, 0.2)

    def test_derivative(self):
        atom = TentAtom(mu=1.5)
        assert atom.derivative(0.3) == pytest.approx(1.5)
        assert atom.derivative(0.7) == pytest.approx(-1.5)

    def test_lyapunov_exact(self):
        """Tent map: λ = ln(mu) exactly."""
        for mu in [1.237, 1.574, 1.891]:
            atom = TentAtom(mu=mu)
            assert atom.lyapunov() == pytest.approx(np.log(mu), abs=1e-10)

    def test_h_ks(self):
        atom = TentAtom(mu=1.5)
        assert atom.h_ks() == pytest.approx(np.log(1.5), abs=1e-10)

    def test_regime_chaotic(self):
        atom = TentAtom(mu=1.5)
        assert atom.regime() == "chaotic"

    def test_regime_fixed_point(self):
        atom = TentAtom(mu=1.0)
        assert atom.regime() == "fixed_point"

    def test_mu_cap(self):
        """mu=2.0 should be rejected (v1 degeneration bug)."""
        with pytest.raises(ValueError):
            TentAtom(mu=2.0)

    def test_domain(self):
        atom = TentAtom(mu=1.5)
        assert atom.domain == (0.0, 1.0)


# ── DampedLinearAtom ──────────────────────────────────────────────────────

class TestDampedLinearAtom:
    def test_iterate(self):
        atom = DampedLinearAtom(lam=0.5)
        assert atom.iterate(2.0) == pytest.approx(1.0)

    def test_derivative(self):
        atom = DampedLinearAtom(lam=0.5)
        assert atom.derivative(999.0) == pytest.approx(0.5)

    def test_lyapunov_exact(self):
        """λ = ln(lam), always negative."""
        atom = DampedLinearAtom(lam=0.5)
        assert atom.lyapunov() == pytest.approx(np.log(0.5), abs=1e-10)

    def test_h_ks_zero(self):
        atom = DampedLinearAtom(lam=0.5)
        assert atom.h_ks() == 0.0

    def test_regime(self):
        atom = DampedLinearAtom(lam=0.5)
        assert atom.regime() == "fixed_point"

    def test_domain(self):
        atom = DampedLinearAtom(lam=0.5)
        assert atom.domain == (-10.0, 10.0)

    def test_param_validation(self):
        with pytest.raises(ValueError):
            DampedLinearAtom(lam=0.0)
        with pytest.raises(ValueError):
            DampedLinearAtom(lam=1.0)
        with pytest.raises(ValueError):
            DampedLinearAtom(lam=-0.1)


# ── RotationAtom ──────────────────────────────────────────────────────────

class TestRotationAtom:
    def test_iterate(self):
        atom = RotationAtom(omega=0.25)
        assert atom.iterate(0.0) == pytest.approx(0.25)
        assert atom.iterate(0.9) == pytest.approx(0.15)

    def test_derivative(self):
        atom = RotationAtom(omega=0.25)
        assert atom.derivative(0.0) == 1.0

    def test_lyapunov_zero(self):
        atom = RotationAtom(omega=0.25)
        assert atom.lyapunov() == 0.0

    def test_h_ks_zero(self):
        atom = RotationAtom(omega=0.25)
        assert atom.h_ks() == 0.0

    def test_regime_periodic(self):
        """omega=0.25 is rational (1/4), so periodic."""
        atom = RotationAtom(omega=0.25)
        assert atom.regime() == "periodic"

    def test_regime_quasiperiodic(self):
        """Golden ratio related omega is irrational."""
        atom = RotationAtom(omega=0.381966011)
        assert atom.regime() == "quasiperiodic"

    def test_domain(self):
        atom = RotationAtom(omega=0.25)
        assert atom.domain == (0.0, 1.0)

    def test_param_validation(self):
        with pytest.raises(ValueError):
            RotationAtom(omega=0.0)
        with pytest.raises(ValueError):
            RotationAtom(omega=1.0)


# ── Cross-cutting ─────────────────────────────────────────────────────────

class TestAtomBase:
    def test_trajectory_base_class(self):
        """trajectory() is implemented on the ABC via iterate()."""
        atom = LogisticAtom(r=3.5)
        traj = atom.trajectory(0.5, 10)
        assert len(traj) == 10
        # Verify first step
        assert traj[1] == pytest.approx(atom.iterate(0.5))

    def test_all_atoms_are_atoms(self):
        atoms = [
            LogisticAtom(r=3.5),
            TentAtom(mu=1.5),
            DampedLinearAtom(lam=0.5),
            RotationAtom(omega=0.25),
        ]
        for a in atoms:
            assert isinstance(a, Atom)
