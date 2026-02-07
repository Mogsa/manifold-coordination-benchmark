"""Tests for chaosbench.grammar.connectives — AffineConjugacy."""

import numpy as np
import pytest

from chaosbench.grammar.atoms import (
    Atom,
    LogisticAtom,
    TentAtom,
    RotationAtom,
    SineAtom,
)
from chaosbench.grammar.connectives import AffineConjugacy


class TestAffineConjugacy:
    def test_is_atom(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert isinstance(conj, Atom)

    def test_preserves_family(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.family == "logistic"

    def test_preserves_params(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.params == {"r": 3.891}

    def test_preserves_regime(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.regime() == inner.regime()

    def test_preserves_lyapunov(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.lyapunov() == pytest.approx(inner.lyapunov())

    def test_preserves_h_ks(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.h_ks() == pytest.approx(inner.h_ks())

    def test_transforms_domain(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        # inner domain = (0, 1), so conjugated = (0*1.7+0.5, 1*1.7+0.5) = (0.5, 2.2)
        lo, hi = conj.domain
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(2.2)

    def test_iterate_round_trip(self):
        """Conjugated iterate should equal a·f((y-b)/a) + b."""
        inner = LogisticAtom(r=3.891)
        a, b = 1.7, 0.5
        conj = AffineConjugacy(inner, a=a, b=b)

        y = 1.0  # Some point in the conjugated domain
        x = (y - b) / a
        expected = a * inner.iterate(x) + b
        assert conj.iterate(y) == pytest.approx(expected)

    def test_derivative_equals_inner(self):
        """Conjugacy preserves derivative (scale factors cancel in chain rule)."""
        inner = LogisticAtom(r=3.891)
        a, b = 1.7, 0.5
        conj = AffineConjugacy(inner, a=a, b=b)

        y = 1.0
        x = (y - b) / a
        assert conj.derivative(y) == pytest.approx(inner.derivative(x))

    def test_trajectory_in_conjugated_domain(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        lo, hi = conj.domain
        traj = conj.trajectory(1.0, 200)
        # All points should be in conjugated domain (with small margin)
        assert np.all(traj >= lo - 0.1)
        assert np.all(traj <= hi + 0.1)

    def test_trajectory_differs_from_inner(self):
        """Conjugated trajectory should have different numerical values."""
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        inner_traj = inner.trajectory(0.5, 50)
        conj_traj = conj.trajectory(1.0, 50)
        # Should not be equal (different domains, different starting points)
        assert not np.allclose(inner_traj, conj_traj)

    def test_zero_scale_rejected(self):
        inner = LogisticAtom(r=3.5)
        with pytest.raises(ValueError):
            AffineConjugacy(inner, a=0.0, b=0.5)

    def test_inner_atom_accessible(self):
        inner = LogisticAtom(r=3.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.inner_atom is inner

    def test_name(self):
        inner = LogisticAtom(r=3.5)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert "conj" in conj.name
        assert "logistic" in conj.name

    def test_works_with_tent(self):
        inner = TentAtom(mu=1.891)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.regime() == "chaotic"
        assert conj.lyapunov() == pytest.approx(inner.lyapunov())

    def test_works_with_rotation(self):
        inner = RotationAtom(omega=0.381966011)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.regime() == "quasiperiodic"
        assert conj.lyapunov() == 0.0

    def test_works_with_sine(self):
        inner = SineAtom(a=0.97)
        conj = AffineConjugacy(inner, a=1.7, b=0.5)
        assert conj.regime() == inner.regime()
        assert conj.lyapunov() == pytest.approx(inner.lyapunov())
