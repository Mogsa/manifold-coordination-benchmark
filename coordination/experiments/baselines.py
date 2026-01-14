"""
Baseline comparison tools.

This module provides convenient functions for running baseline agents
and comparing them to LLM agents.
"""

import os
from typing import List, Dict, Optional
import pandas as pd

from coordination.experiments.eval import run_evaluation, TEST_SURFACES
from coordination.experiments.analysis import (
    load_results, compute_summary_stats, compare_agents, generate_report
)


def run_random_baseline(
    surfaces: Optional[List[str]] = None,
    n_trials: int = 10,
    output_dir: str = "results/random_baseline",
    seed: int = 42
) -> Dict:
    """
    Run random baseline agent on all specified surfaces.

    Args:
        surfaces: List of surface names (if None, uses all test surfaces)
        n_trials: Number of trials per surface
        output_dir: Directory to save results
        seed: Random seed for reproducibility

    Returns:
        Evaluation results dictionary
    """
    if surfaces is None:
        surfaces = list(TEST_SURFACES.keys())

    agent_configs = [
        {"type": "random", "seed": seed, "name": "Random Baseline"}
    ]

    print(f"\nRunning Random Baseline on {len(surfaces)} surfaces...")
    results = run_evaluation(
        surfaces=surfaces,
        agent_configs=agent_configs,
        n_trials=n_trials,
        output_dir=output_dir
    )

    return results


def run_greedy_baseline(
    surfaces: Optional[List[str]] = None,
    n_trials: int = 10,
    output_dir: str = "results/greedy_baseline",
    step_size: float = 1.0
) -> Dict:
    """
    Run greedy gradient-following baseline on all specified surfaces.

    Args:
        surfaces: List of surface names (if None, uses all test surfaces)
        n_trials: Number of trials per surface
        output_dir: Directory to save results
        step_size: Step size for gradient following

    Returns:
        Evaluation results dictionary
    """
    if surfaces is None:
        surfaces = list(TEST_SURFACES.keys())

    agent_configs = [
        {"type": "greedy", "step_size": step_size, "name": "Greedy Baseline"}
    ]

    print(f"\nRunning Greedy Baseline on {len(surfaces)} surfaces...")
    results = run_evaluation(
        surfaces=surfaces,
        agent_configs=agent_configs,
        n_trials=n_trials,
        output_dir=output_dir
    )

    return results


def run_all_baselines(
    surfaces: Optional[List[str]] = None,
    n_trials: int = 10,
    output_dir: str = "results/baselines_comparison",
    random_seed: int = 42,
    greedy_step_size: float = 1.0
) -> Dict:
    """
    Run both random and greedy baselines.

    Args:
        surfaces: List of surface names (if None, uses all test surfaces)
        n_trials: Number of trials per surface per agent
        output_dir: Directory to save results
        random_seed: Random seed for random baseline
        greedy_step_size: Step size for greedy baseline

    Returns:
        Evaluation results dictionary
    """
    if surfaces is None:
        surfaces = list(TEST_SURFACES.keys())

    agent_configs = [
        {"type": "random", "seed": random_seed, "name": "Random Baseline"},
        {"type": "greedy", "step_size": greedy_step_size, "name": "Greedy Baseline"}
    ]

    print(f"\nRunning All Baselines on {len(surfaces)} surfaces...")
    results = run_evaluation(
        surfaces=surfaces,
        agent_configs=agent_configs,
        n_trials=n_trials,
        output_dir=output_dir
    )

    return results


def compare_to_baselines(
    llm_results_dir: str,
    baseline_results_dir: str,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare LLM agent results to baseline results.

    Args:
        llm_results_dir: Directory containing LLM agent results
        baseline_results_dir: Directory containing baseline results
        output_file: Optional path to save comparison table

    Returns:
        DataFrame with comparison statistics
    """
    # Load results
    llm_results = load_results(llm_results_dir)
    baseline_results = load_results(baseline_results_dir)

    # Combine
    all_results = llm_results + baseline_results

    # Compute summary stats
    summary = compute_summary_stats(all_results)

    # Print comparison
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    print(summary.to_string(index=False))
    print("\n")

    # Perform statistical comparisons
    agents = summary['agent_name'].unique()

    if len(agents) >= 2:
        print("Statistical Comparisons:")
        print("-" * 70)

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                try:
                    comparison = compare_agents(all_results, agents[i], agents[j])
                    print(f"\n{agents[i]} vs {agents[j]}:")
                    print(f"  Mean scores: {comparison['mean_a']:.3f} vs {comparison['mean_b']:.3f}")
                    print(f"  Difference: {comparison['mean_diff']:.3f}")
                    print(f"  p-value: {comparison['p_value']:.4f} "
                          f"({'significant' if comparison['significant'] else 'not significant'})")
                    print(f"  Effect size: {comparison['effect_size']:.3f}")
                except Exception as e:
                    print(f"Error comparing {agents[i]} vs {agents[j]}: {e}")

    if output_file:
        summary.to_csv(output_file, index=False)
        print(f"\nComparison table saved to: {output_file}")

    return summary


def generate_baseline_report(
    results_dir: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive report comparing baseline performance.

    Args:
        results_dir: Directory containing baseline results
        output_path: Optional path to save report

    Returns:
        Report as string
    """
    report = generate_report(results_dir, output_path)
    return report


def create_comparison_table(
    results_dirs: Dict[str, str],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table from multiple result directories.

    Args:
        results_dirs: Dictionary mapping agent names to result directories
                     e.g., {"Random": "results/random", "Greedy": "results/greedy"}
        output_file: Optional path to save table

    Returns:
        DataFrame with comparison across all agents
    """
    all_results = []

    for agent_name, results_dir in results_dirs.items():
        results = load_results(results_dir)
        # Ensure agent names are consistent
        for result in results:
            result['agent_name'] = agent_name
        all_results.extend(results)

    summary = compute_summary_stats(all_results)

    # Pivot table for better readability
    pivot = summary.pivot_table(
        index='surface_name',
        columns='agent_name',
        values='mean_score'
    )

    print("\n" + "="*70)
    print("AGENT COMPARISON TABLE")
    print("="*70)
    print(pivot.to_string())
    print("\n")

    if output_file:
        pivot.to_csv(output_file)
        print(f"Table saved to: {output_file}")

    return pivot


if __name__ == "__main__":
    """Example usage of baseline comparison tools."""
    import sys

    # Default surfaces
    surfaces = ["single_peak_center", "two_peaks_clear", "three_peaks"]

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "random":
            print("Running random baseline...")
            run_random_baseline(surfaces=surfaces, n_trials=5)

        elif command == "greedy":
            print("Running greedy baseline...")
            run_greedy_baseline(surfaces=surfaces, n_trials=5)

        elif command == "all":
            print("Running all baselines...")
            run_all_baselines(surfaces=surfaces, n_trials=5)

        elif command == "compare":
            if len(sys.argv) < 4:
                print("Usage: python baselines.py compare <llm_dir> <baseline_dir>")
            else:
                llm_dir = sys.argv[2]
                baseline_dir = sys.argv[3]
                compare_to_baselines(llm_dir, baseline_dir)

        else:
            print(f"Unknown command: {command}")
            print("Available commands: random, greedy, all, compare")

    else:
        print("Baseline Comparison Tools")
        print("=" * 70)
        print("\nUsage:")
        print("  python baselines.py random         # Run random baseline")
        print("  python baselines.py greedy         # Run greedy baseline")
        print("  python baselines.py all            # Run both baselines")
        print("  python baselines.py compare <llm_dir> <baseline_dir>  # Compare results")
        print("\nExample:")
        print("  python baselines.py all")
        print("  python baselines.py compare results/llm results/baselines")
