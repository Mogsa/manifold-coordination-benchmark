"""Tests for chaosbench.grammar.system — DynamicalSystem wrapper."""

import numpy as np
import pytest

from chaosbench.grammar.atoms import LogisticAtom, TentAtom, DampedLinearAtom
from chaosbench.grammar.system import DynamicalSystem


class TestDynamicalSystem:
    def test_metadata_cached(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        m1 = sys.metadata()
        m2 = sys.metadata()
        assert m1 is m2  # Same object (cached)

    def test_metadata_fields(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        meta = sys.metadata()
        assert meta.family == "logistic"
        assert meta.params == {"r": 3.891}
        assert meta.regime == "chaotic"
        assert meta.dim == 1
        assert meta.grammar_depth == 0
        assert meta.domain == (0.0, 1.0)
        assert meta.lambda_max > 0
        assert meta.h_ks > 0
        assert meta.tau_lambda > 0 and meta.tau_lambda < float("inf")

    def test_observe_shape(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs = sys.observe(n_points=100, noise_std=0.01)
        assert obs.data.shape == (100,)
        assert obs.clean_data.shape == (100,)
        assert obs.n_points == 100

    def test_observe_noisy_differs_from_clean(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs = sys.observe(noise_std=0.05)
        assert not np.allclose(obs.data, obs.clean_data)

    def test_observe_zero_noise(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs = sys.observe(noise_std=0.0)
        assert np.allclose(obs.data, obs.clean_data)

    def test_seed_reproducibility(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs1 = sys.observe(ic_seed=42, noise_seed=99)
        obs2 = sys.observe(ic_seed=42, noise_seed=99)
        np.testing.assert_array_equal(obs1.data, obs2.data)
        np.testing.assert_array_equal(obs1.clean_data, obs2.clean_data)

    def test_different_seeds_differ(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs1 = sys.observe(ic_seed=42, noise_seed=99)
        obs2 = sys.observe(ic_seed=43, noise_seed=99)
        assert not np.allclose(obs1.clean_data, obs2.clean_data)

    def test_future_trajectory(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        obs = sys.observe(n_points=50)
        future = sys.future_trajectory(obs, 10)
        assert future.shape == (10,)
        # First future value should be iterate(last clean value)
        expected_first = atom.iterate(obs.clean_data[-1])
        assert future[0] == pytest.approx(expected_first)

    def test_prediction_horizon_chaotic(self):
        atom = LogisticAtom(r=3.891)
        sys = DynamicalSystem(atom)
        h = sys.prediction_horizon()
        assert h >= 1
        # Should be ceil(5 * tau_lambda)
        meta = sys.metadata()
        import math
        assert h == math.ceil(5.0 * meta.tau_lambda)

    def test_prediction_horizon_non_chaotic(self):
        atom = DampedLinearAtom(lam=0.5)
        sys = DynamicalSystem(atom)
        assert sys.prediction_horizon() == 20

    def test_observe_seeds_stored(self):
        atom = TentAtom(mu=1.5)
        sys = DynamicalSystem(atom)
        obs = sys.observe(ic_seed=7, noise_seed=13)
        assert obs.seeds == {"ic_seed": 7, "noise_seed": 13}
