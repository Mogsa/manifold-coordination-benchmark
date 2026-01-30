"""Tests for parameter fitting."""
import pytest
import numpy as np

from chaosbench.core.fitting import fit_model, FitResult


class TestFitModel:
    def test_fit_logistic_recovers_r(self):
        """Fitting logistic data should recover r parameter."""
        # Generate observations from logistic r=3.85
        true_r = 3.85
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_r * x[i-1] * (1 - x[i-1])

        result = fit_model("logistic", x)

        assert isinstance(result, FitResult)
        assert abs(result.params["r"] - true_r) < 0.1  # Within 0.1 of true
        assert result.mae < 0.05

    def test_fit_tent_recovers_mu(self):
        """Fitting tent data should recover mu parameter."""
        true_mu = 1.85
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_mu * x[i-1] if x[i-1] < 0.5 else true_mu * (1 - x[i-1])

        result = fit_model("tent", x)

        assert abs(result.params["mu"] - true_mu) < 0.1
        assert result.mae < 0.05

    def test_fit_includes_prediction(self):
        """Fit result includes predicted x_50."""
        true_r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = true_r * x[i-1] * (1 - x[i-1])

        result = fit_model("logistic", x)

        assert result.predicted_next is not None
        # Prediction should be reasonable (in [0, 1] for logistic)
        assert 0 <= result.predicted_next <= 1

    def test_unknown_family_raises(self):
        """Unknown family should raise ValueError."""
        x = np.random.rand(50)
        with pytest.raises(ValueError, match="Unknown model family"):
            fit_model("unknown_family", x)

    def test_fit_henon_recovers_params(self):
        """Fitting henon data should recover a and b parameters."""
        true_a, true_b = 1.4, 0.3
        # Generate 2D Henon trajectory (interleaved x, y)
        # obs = [x0, y0, x1, y1, ..., x24, y24] = 50 values
        obs = np.zeros(50)
        x, y = 0.1, 0.0
        for i in range(25):
            obs[2*i] = x
            obs[2*i + 1] = y
            x_new = 1 - true_a * x**2 + y
            y_new = true_b * x
            x, y = x_new, y_new

        result = fit_model("henon", obs)

        assert abs(result.params["a"] - true_a) < 0.1
        assert abs(result.params["b"] - true_b) < 0.1
        assert result.mae < 0.05
