"""Tests for chaosbench.grammar.atoms — all 4 MVP atoms."""

import numpy as np
import pytest

from chaosbench.grammar.atoms import (
    Atom,
    CircleAtom,
    DampedLinearAtom,
    HenonAtom,
    LogisticAtom,
    RotationAtom,
    SineAtom,
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


# ── SineAtom ─────────────────────────────────────────────────────────────

class TestSineAtom:
    def test_iterate(self):
        atom = SineAtom(a=1.0)
        # sin(π·0.5) = 1.0, so f(0.5) = 1.0·1.0 = 1.0
        assert atom.iterate(0.5) == pytest.approx(1.0)
        # sin(π·0) = 0, so f(0) = 0
        assert atom.iterate(0.0) == pytest.approx(0.0)

    def test_derivative(self):
        atom = SineAtom(a=1.0)
        # d/dx[a·sin(πx)] = a·π·cos(πx)
        # At x=0: π·cos(0) = π
        assert atom.derivative(0.0) == pytest.approx(np.pi)
        # At x=0.5: π·cos(π/2) = 0
        assert atom.derivative(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_domain(self):
        atom = SineAtom(a=0.8)
        assert atom.domain == (0.0, 1.0)

    def test_family(self):
        atom = SineAtom(a=0.8)
        assert atom.family == "sine"

    def test_params(self):
        atom = SineAtom(a=0.8)
        assert atom.params == {"a": 0.8}

    def test_regime_chaotic(self):
        atom = SineAtom(a=0.97)
        assert atom.regime() == "chaotic"

    def test_regime_periodic_or_fixed(self):
        atom = SineAtom(a=0.65)
        regime = atom.regime()
        assert regime in ("periodic", "fixed_point")

    def test_param_validation(self):
        with pytest.raises(ValueError):
            SineAtom(a=0.3)
        with pytest.raises(ValueError):
            SineAtom(a=1.1)

    def test_trajectory_shape(self):
        atom = SineAtom(a=0.8)
        traj = atom.trajectory(0.5, 100)
        assert traj.shape == (100,)
        assert traj[0] == 0.5


# ── CircleAtom ───────────────────────────────────────────────────────────

class TestCircleAtom:
    def test_iterate_K0(self):
        """With K=0, circle map reduces to rotation."""
        atom = CircleAtom(K=0.0, omega=0.25)
        assert atom.iterate(0.0) == pytest.approx(0.25)
        assert atom.iterate(0.9) == pytest.approx(0.15)

    def test_derivative(self):
        atom = CircleAtom(K=0.5, omega=0.3)
        # d/dx = 1 - K·cos(2πx); at x=0: 1 - 0.5·1 = 0.5
        assert atom.derivative(0.0) == pytest.approx(0.5)

    def test_domain(self):
        atom = CircleAtom(K=0.5, omega=0.3)
        assert atom.domain == (0.0, 1.0)

    def test_family(self):
        atom = CircleAtom(K=0.5, omega=0.3)
        assert atom.family == "circle"

    def test_params(self):
        atom = CircleAtom(K=0.5, omega=0.3)
        assert atom.params == {"K": 0.5, "omega": 0.3}

    def test_regime_quasiperiodic(self):
        atom = CircleAtom(K=0.5, omega=0.382)
        assert atom.regime() == "quasiperiodic"

    def test_regime_chaotic(self):
        atom = CircleAtom(K=1.15, omega=0.618)
        lam = atom.lyapunov()
        # K > 1 should give positive Lyapunov (chaotic)
        assert lam > 0.0
        assert atom.regime() == "chaotic"

    def test_param_validation(self):
        with pytest.raises(ValueError):
            CircleAtom(K=-0.1, omega=0.3)
        with pytest.raises(ValueError):
            CircleAtom(K=0.5, omega=0.0)
        with pytest.raises(ValueError):
            CircleAtom(K=0.5, omega=1.0)

    def test_trajectory_in_domain(self):
        atom = CircleAtom(K=0.5, omega=0.382)
        traj = atom.trajectory(0.3, 200)
        assert np.all(traj >= 0.0)
        assert np.all(traj < 1.0)


# ── HenonAtom ────────────────────────────────────────────────────────────

class TestHenonAtom:
    def test_iterate(self):
        atom = HenonAtom(a=1.4, b=0.3)
        atom.prepare(0.0)
        # x' = 1 - 1.4·0² + 0 = 1.0, y' = 0.3·0 = 0
        assert atom.iterate(0.0) == pytest.approx(1.0)

    def test_prepare_resets_y(self):
        atom = HenonAtom(a=1.4, b=0.3)
        atom.prepare(0.5)
        assert atom._y == 0.0

    def test_domain(self):
        atom = HenonAtom(a=1.3, b=0.3)
        assert atom.domain == (-1.5, 1.5)

    def test_family(self):
        atom = HenonAtom(a=1.3, b=0.3)
        assert atom.family == "henon"

    def test_params(self):
        atom = HenonAtom(a=1.3, b=0.3)
        assert atom.params == {"a": 1.3, "b": 0.3}

    def test_lyapunov_classic(self):
        """Classic Hénon (a=1.4, b=0.3): λ1 ≈ 0.42."""
        atom = HenonAtom(a=1.4, b=0.3)
        lam = atom.lyapunov()
        assert lam == pytest.approx(0.42, abs=0.05)

    def test_regime_chaotic(self):
        atom = HenonAtom(a=1.35, b=0.30)
        assert atom.regime() == "chaotic"

    def test_regime_periodic(self):
        atom = HenonAtom(a=1.07, b=0.28)
        assert atom.regime() == "periodic"

    def test_param_validation(self):
        with pytest.raises(ValueError):
            HenonAtom(a=0.5, b=0.3)
        with pytest.raises(ValueError):
            HenonAtom(a=1.3, b=0.1)

    def test_trajectory_shape(self):
        atom = HenonAtom(a=1.3, b=0.3)
        traj = atom.trajectory(0.0, 100)
        assert traj.shape == (100,)
        assert traj[0] == 0.0

    def test_trajectory_bounded(self):
        """Chaotic Hénon should stay bounded."""
        atom = HenonAtom(a=1.35, b=0.30)
        traj = atom.trajectory(0.0, 1000)
        assert np.all(np.isfinite(traj))
        assert np.all(np.abs(traj) < 2.0)


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
            SineAtom(a=0.8),
            CircleAtom(K=0.5, omega=0.3),
            HenonAtom(a=1.3, b=0.3),
        ]
        for a in atoms:
            assert isinstance(a, Atom)

    def test_prepare_noop_for_1d(self):
        """prepare() should be a no-op for standard 1D atoms."""
        atom = LogisticAtom(r=3.5)
        atom.prepare(0.5)  # Should not raise
