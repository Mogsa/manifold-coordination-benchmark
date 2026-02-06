"""
Tests for evaluation harness (Phase 5).

These tests verify batch evaluation, statistical analysis, and baseline comparisons.
"""

import pytest
import os
import shutil
import json
from pathlib import Path

from coordination.experiments.eval import (
    run_evaluation, TEST_SURFACES, create_agent_from_config
)
from coordination.experiments.analysis import (
    load_results, compute_summary_stats, compare_agents, generate_report,
    plot_results, plot_comparison_matrix
)
from coordination.experiments.baselines import (
    run_random_baseline, run_greedy_baseline, run_all_baselines
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test results."""
    output_dir = tmp_path / "test_results"
    output_dir.mkdir()
    yield str(output_dir)
    # Cleanup handled by tmp_path fixture


class TestBatchEvaluation:
    """Tests for batch evaluation runner."""

    def test_create_random_agent(self):
        """Test creating random agent from config."""
        config = {"type": "random", "seed": 42}
        agent = create_agent_from_config(config, role='A')
        assert agent.role == 'A'
        assert hasattr(agent, 'decide_position')

    def test_create_greedy_agent(self):
        """Test creating greedy agent from config."""
        config = {"type": "greedy", "step_size": 1.0}
        agent = create_agent_from_config(config, role='B')
        assert agent.role == 'B'
        assert hasattr(agent, 'decide_position')

    def test_test_surfaces_defined(self):
        """Test that pre-defined surfaces are available."""
        assert len(TEST_SURFACES) > 0
        assert "single_peak_center" in TEST_SURFACES
        assert "two_peaks_clear" in TEST_SURFACES

    def test_run_evaluation_basic(self, temp_output_dir):
        """Test basic batch evaluation with minimal settings."""
        surfaces = ["single_peak_center"]
        agent_configs = [{"type": "random", "seed": 42, "name": "Random"}]

        results = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=2,
            output_dir=temp_output_dir,
            n_turns=3,  # Short episodes for testing
            verbose=False
        )

        assert results['total_runs'] == 2  # 1 surface × 1 agent × 2 trials
        assert results['completed'] == 2
        assert results['failed'] == 0
        assert len(results['episodes']) == 2

    def test_run_evaluation_multiple_agents(self, temp_output_dir):
        """Test evaluation with multiple agent types."""
        surfaces = ["single_peak_center"]
        agent_configs = [
            {"type": "random", "seed": 42, "name": "Random"},
            {"type": "greedy", "step_size": 1.0, "name": "Greedy"}
        ]

        results = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=2,
            output_dir=temp_output_dir,
            n_turns=3,
            verbose=False
        )

        assert results['total_runs'] == 4  # 1 surface × 2 agents × 2 trials
        assert results['completed'] == 4

    def test_run_evaluation_multiple_surfaces(self, temp_output_dir):
        """Test evaluation with multiple surfaces."""
        surfaces = ["single_peak_center", "two_peaks_clear"]
        agent_configs = [{"type": "random", "seed": 42, "name": "Random"}]

        results = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=2,
            output_dir=temp_output_dir,
            n_turns=3,
            verbose=False
        )

        assert results['total_runs'] == 4  # 2 surfaces × 1 agent × 2 trials
        assert results['completed'] == 4

    def test_evaluation_saves_results(self, temp_output_dir):
        """Test that results are properly saved."""
        surfaces = ["single_peak_center"]
        agent_configs = [{"type": "random", "seed": 42, "name": "Random"}]

        results = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=2,
            output_dir=temp_output_dir,
            n_turns=3,
            verbose=False
        )

        # Check that output directory has results
        output_path = Path(temp_output_dir)
        assert output_path.exists()

        # Check for summary file
        summary_file = output_path / "evaluation_summary.json"
        assert summary_file.exists()

        # Check that result files were created
        json_files = list(output_path.rglob("*.json"))
        assert len(json_files) >= 3  # 2 episode results + 1 summary


class TestStatisticalAnalysis:
    """Tests for statistical analysis functions."""

    @pytest.fixture
    def sample_results(self, temp_output_dir):
        """Create sample results for analysis testing."""
        surfaces = ["single_peak_center"]
        agent_configs = [
            {"type": "random", "seed": 42, "name": "Random"},
            {"type": "greedy", "step_size": 1.0, "name": "Greedy"}
        ]

        run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=3,
            output_dir=temp_output_dir,
            n_turns=3,
            verbose=False
        )

        return temp_output_dir

    def test_load_results(self, sample_results):
        """Test loading results from directory."""
        results = load_results(sample_results)
        assert len(results) > 0
        assert all('score' in r for r in results)

    def test_compute_summary_stats(self, sample_results):
        """Test computing summary statistics."""
        results = load_results(sample_results)
        summary = compute_summary_stats(results)

        assert len(summary) > 0
        assert 'agent_name' in summary.columns
        assert 'surface_name' in summary.columns
        assert 'mean_score' in summary.columns
        assert 'std_score' in summary.columns
        assert 'n_trials' in summary.columns

    def test_compare_agents(self, sample_results):
        """Test statistical comparison between agents."""
        results = load_results(sample_results)
        comparison = compare_agents(results, "Random", "Greedy")

        assert 'mean_diff' in comparison
        assert 'p_value' in comparison
        assert 'effect_size' in comparison
        assert 'significant' in comparison
        assert comparison['agent_a'] == "Random"
        assert comparison['agent_b'] == "Greedy"

    def test_generate_report(self, sample_results, tmp_path):
        """Test report generation."""
        report_path = tmp_path / "report.txt"
        report = generate_report(sample_results, output_path=str(report_path))

        assert len(report) > 0
        assert "EVALUATION REPORT" in report
        assert report_path.exists()

    def test_plot_results(self, sample_results, tmp_path):
        """Test results visualization."""
        results = load_results(sample_results)
        summary = compute_summary_stats(results)

        plot_path = tmp_path / "plot.png"
        fig = plot_results(summary, save_path=str(plot_path))

        assert fig is not None
        assert plot_path.exists()

    def test_plot_comparison_matrix(self, sample_results, tmp_path):
        """Test comparison matrix visualization."""
        results = load_results(sample_results)

        plot_path = tmp_path / "heatmap.png"
        fig = plot_comparison_matrix(results, save_path=str(plot_path))

        assert fig is not None
        assert plot_path.exists()


class TestBaselineComparison:
    """Tests for baseline comparison tools."""

    def test_run_random_baseline(self, temp_output_dir):
        """Test running random baseline."""
        surfaces = ["single_peak_center"]

        results = run_random_baseline(
            surfaces=surfaces,
            n_trials=2,
            output_dir=temp_output_dir,
            seed=42
        )

        assert results['completed'] == 2
        assert results['failed'] == 0

    def test_run_greedy_baseline(self, temp_output_dir):
        """Test running greedy baseline."""
        surfaces = ["single_peak_center"]

        results = run_greedy_baseline(
            surfaces=surfaces,
            n_trials=2,
            output_dir=temp_output_dir,
            step_size=1.0
        )

        assert results['completed'] == 2
        assert results['failed'] == 0

    def test_run_all_baselines(self, temp_output_dir):
        """Test running all baselines together."""
        surfaces = ["single_peak_center"]

        results = run_all_baselines(
            surfaces=surfaces,
            n_trials=2,
            output_dir=temp_output_dir
        )

        # 1 surface × 2 baselines (random + greedy) × 2 trials
        assert results['total_runs'] == 4
        assert results['completed'] == 4


class TestIntegration:
    """Integration tests for full evaluation pipeline."""

    def test_full_pipeline(self, temp_output_dir):
        """Test complete evaluation pipeline from run to analysis."""
        # Step 1: Run evaluation
        surfaces = ["single_peak_center", "two_peaks_clear"]
        agent_configs = [
            {"type": "random", "seed": 42, "name": "Random"},
            {"type": "greedy", "step_size": 1.0, "name": "Greedy"}
        ]

        eval_results = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=2,
            output_dir=temp_output_dir,
            n_turns=3,
            verbose=False
        )

        # Verify evaluation completed
        assert eval_results['completed'] == 8  # 2 surfaces × 2 agents × 2 trials

        # Step 2: Load results
        results = load_results(temp_output_dir)
        assert len(results) == 8

        # Step 3: Compute statistics
        summary = compute_summary_stats(results)
        assert len(summary) == 4  # 2 agents × 2 surfaces

        # Step 4: Compare agents
        comparison = compare_agents(results, "Random", "Greedy")
        assert comparison is not None

        # Step 5: Generate report
        report = generate_report(temp_output_dir)
        assert "EVALUATION REPORT" in report

    def test_reproducibility(self, temp_output_dir):
        """Test that same configuration produces consistent results."""
        surfaces = ["single_peak_center"]
        agent_configs = [{"type": "random", "seed": 42, "name": "Random"}]

        # Run twice with same seed
        results1 = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=1,
            output_dir=temp_output_dir + "/run1",
            n_turns=3,
            verbose=False
        )

        results2 = run_evaluation(
            surfaces=surfaces,
            agent_configs=agent_configs,
            n_trials=1,
            output_dir=temp_output_dir + "/run2",
            n_turns=3,
            verbose=False
        )

        # Load scores
        episodes1 = results1['episodes']
        episodes2 = results2['episodes']

        # Random agent with same seed should produce same trajectory
        # (Note: due to how randomness works, positions should be identical)
        assert episodes1[0]['score'] == episodes2[0]['score']
