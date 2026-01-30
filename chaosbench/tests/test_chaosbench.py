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
