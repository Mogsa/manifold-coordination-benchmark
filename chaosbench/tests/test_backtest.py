"""Tests for backtest function."""
import pytest
import numpy as np

from chaosbench.core.backtest import backtest_model, BacktestResult


class TestBacktestModel:
    def test_perfect_logistic_fit(self):
        """Perfect model should have near-zero MAE."""
        # Generate observations from logistic r=3.9
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.9}, x)

        assert isinstance(result, BacktestResult)
        assert result.mae < 0.01  # Near-perfect fit
        assert result.predicted_next is not None

    def test_wrong_params_high_mae(self):
        """Wrong parameters should have high MAE."""
        # Generate from r=3.9, test with r=3.5
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.5}, x)

        assert result.mae > 0.05  # Poor fit (significantly worse than perfect)

    def test_predicts_next_value(self):
        """Result includes prediction for x_50."""
        r = 3.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = r * x[i-1] * (1 - x[i-1])

        result = backtest_model("logistic", {"r": 3.9}, x)

        # Predicted x_50 should be r * x_49 * (1 - x_49)
        expected = r * x[-1] * (1 - x[-1])
        assert abs(result.predicted_next - expected) < 0.001

    def test_tent_map_backtest(self):
        """Backtest works for tent map."""
        mu = 1.9
        x = np.zeros(50)
        x[0] = 0.3
        for i in range(1, 50):
            x[i] = mu * x[i-1] if x[i-1] < 0.5 else mu * (1 - x[i-1])

        result = backtest_model("tent", {"mu": 1.9}, x)

        assert result.mae < 0.01
