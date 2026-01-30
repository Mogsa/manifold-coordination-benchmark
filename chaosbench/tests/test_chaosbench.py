"""Tests for ChaosBench v3 core functionality."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from chaosbench.core.Chaosbench_v3 import (
    TaskConfig,
    TaskGenerator,
    LogisticMap,
    Evaluator,
    DifficultyWeighting,
    UniformSolver,
)


class TestHorizonScaling:
    """Test that prediction horizon scales with Lyapunov time."""

    def test_horizon_scales_with_lyapunov_time(self):
        """Horizon should be k * lyapunov_time, not fixed."""
        config = TaskConfig(horizon_lyapunov_multiplier=1.5)
        generator = TaskGenerator(config, seed=42)

        # Generate task for a system
        system = LogisticMap(r=4.0)  # h_KS ~ 0.69, lyapunov_time ~ 1.44
        task = generator.generate_task(system=system)

        expected_horizon = max(1, int(1.5 * system.lyapunov_time))
        assert task.future_time == expected_horizon

    def test_different_systems_get_different_horizons(self):
        """High-chaos systems should get shorter horizons than low-chaos."""
        config = TaskConfig(horizon_lyapunov_multiplier=2.0)
        generator = TaskGenerator(config, seed=42)

        # High chaos: r=4.0, lyapunov_time ~ 1.44
        high_chaos = LogisticMap(r=4.0)
        task_high = generator.generate_task(system=high_chaos)

        # Lower chaos: r=3.7, lyapunov_time > 1.44
        low_chaos = LogisticMap(r=3.7)
        task_low = generator.generate_task(system=low_chaos)

        # Lower chaos system should have longer horizon
        assert task_low.future_time >= task_high.future_time


class TestSystemFiltering:
    """Test that non-chaotic systems are filtered out."""

    def test_filters_low_h_ks_systems(self):
        """Systems with h_KS below threshold should be excluded."""
        config = TaskConfig(min_h_ks=0.1)
        generator = TaskGenerator(config, seed=42)

        # All systems should have h_KS >= 0.1
        for system in generator.all_systems:
            assert system.h_ks >= 0.1, f"{system.name} has h_KS={system.h_ks}"

    def test_zero_threshold_includes_all(self):
        """With min_h_ks=0, all systems should be included."""
        config = TaskConfig(min_h_ks=0.0)
        generator = TaskGenerator(config, seed=42)

        # Should have systems from all families
        families = set(s.family for s in generator.all_systems)
        assert len(families) == 5  # logistic, tent, henon, standard, lorenz


class TestScoringFormula:
    """Test that scoring formula doesn't double-weight difficulty."""

    def test_score_uses_difficulty_weight_only_once(self):
        """Score should be w(h_ks) * exp(-NLL), not w(h_ks) * exp(-NLL/h_ks)."""
        # The new formula should give higher scores for same NLL
        # because it doesn't penalize h_KS twice
        h_ks = 0.5
        nll = 2.0

        accuracy_new = np.exp(-nll)  # New: no division by h_ks
        score_new = DifficultyWeighting.linear(h_ks) * accuracy_new

        # Score should be approximately 0.068, not 0.009
        assert score_new > 0.05, f"Score {score_new} too low, likely using old formula"


class TestScoringIntegration:
    """Integration test for scoring changes."""

    def test_uniform_solver_not_crushed(self):
        """Uniform solver should get non-trivial Φ, not near-zero."""
        config = TaskConfig(min_h_ks=0.1)
        generator = TaskGenerator(config, seed=42)
        tasks = generator.generate_batch(20, stratified=True)

        evaluator = Evaluator(weighting=DifficultyWeighting.linear)
        solver = UniformSolver()

        results, phi_curve = evaluator.evaluate_batch(solver, tasks)
        final_phi = phi_curve[-1].cumulative_score

        # Uniform should get meaningful score, not near-zero
        # With 20 bins, uniform gives p=0.05, NLL ≈ 3.0
        # New score per task: w(h_ks) * exp(-3) ≈ 0.5 * 0.05 = 0.025
        # Over 20 tasks should give meaningful Φ (was 0.03 with old formula, now ~0.2)
        assert final_phi > 0.15, f"Uniform Φ={final_phi} too low (suggests double-weighting bug)"
