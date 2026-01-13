"""
Statistical analysis tools for evaluation results.

This module provides functions for computing summary statistics,
comparing agents, and visualizing results.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_results(results_dir: str) -> List[Dict]:
    """
    Load all episode results from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of result dictionaries
    """
    results = []
    results_path = Path(results_dir)

    # Look for all JSON files in subdirectories
    for json_file in results_path.rglob("*.json"):
        # Skip the summary file
        if json_file.name == "evaluation_summary.json":
            continue

        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def compute_summary_stats(results: List[Dict]) -> pd.DataFrame:
    """
    Compute summary statistics for evaluation results.

    Args:
        results: List of episode result dictionaries

    Returns:
        DataFrame with summary statistics per condition
        Columns: agent_name, surface_name, mean_score, std_score,
                 min_score, max_score, n_trials
    """
    # Extract relevant data
    data = []
    for result in results:
        data.append({
            'agent_name': result.get('agent_name', result.get('agent_config', {}).get('name', 'unknown')),
            'surface_name': result.get('surface_name', 'unknown'),
            'score': result.get('score', 0.0),
            'final_position': result.get('final_decision', [0, 0]),
            'n_turns': len(result.get('turns', []))
        })

    df = pd.DataFrame(data)

    # Compute summary statistics grouped by agent and surface
    summary = df.groupby(['agent_name', 'surface_name']).agg({
        'score': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()

    # Flatten column names
    summary.columns = ['agent_name', 'surface_name', 'mean_score', 'std_score',
                      'min_score', 'max_score', 'n_trials']

    # Sort by agent and mean score
    summary = summary.sort_values(['agent_name', 'mean_score'], ascending=[True, False])

    return summary


def compare_agents(results: List[Dict], agent_a: str, agent_b: str,
                  surface: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare two agents statistically.

    Args:
        results: List of episode results
        agent_a: Name of first agent
        agent_b: Name of second agent
        surface: Optional surface name to filter by (if None, uses all surfaces)

    Returns:
        Dictionary with comparison statistics:
        - mean_diff: Difference in mean scores (a - b)
        - t_statistic: t-test statistic
        - p_value: p-value from t-test
        - effect_size: Cohen's d effect size
        - significant: Whether difference is significant at p<0.05
    """
    # Extract scores for each agent
    scores_a = []
    scores_b = []

    for result in results:
        agent_name = result.get('agent_name', result.get('agent_config', {}).get('name', ''))
        surface_name = result.get('surface_name', '')

        # Filter by surface if specified
        if surface is not None and surface_name != surface:
            continue

        if agent_name == agent_a:
            scores_a.append(result.get('score', 0.0))
        elif agent_name == agent_b:
            scores_b.append(result.get('score', 0.0))

    if len(scores_a) == 0 or len(scores_b) == 0:
        raise ValueError(f"No results found for agents {agent_a} and/or {agent_b}")

    # Compute statistics
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)

    # t-test
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(scores_a) - 1) * std_a**2 +
                          (len(scores_b) - 1) * std_b**2) /
                         (len(scores_a) + len(scores_b) - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    return {
        'agent_a': agent_a,
        'agent_b': agent_b,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'std_a': std_a,
        'std_b': std_b,
        'n_a': len(scores_a),
        'n_b': len(scores_b),
        'mean_diff': mean_a - mean_b,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'significant': p_value < 0.05
    }


def plot_results(summary: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization of summary statistics.

    Args:
        summary: DataFrame from compute_summary_stats()
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart with error bars
    agents = summary['agent_name'].unique()
    surfaces = summary['surface_name'].unique()

    x = np.arange(len(surfaces))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_data = summary[summary['agent_name'] == agent]
        means = []
        stds = []

        for surface in surfaces:
            surface_data = agent_data[agent_data['surface_name'] == surface]
            if len(surface_data) > 0:
                means.append(surface_data['mean_score'].values[0])
                stds.append(surface_data['std_score'].values[0])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(agents)/2 + 0.5) * width
        ax1.bar(x + offset, means, width, label=agent, yerr=stds, capsize=5)

    ax1.set_xlabel('Surface')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Agent Performance by Surface')
    ax1.set_xticks(x)
    ax1.set_xticklabels(surfaces, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Box plot
    data_for_box = []
    labels_for_box = []

    for agent in agents:
        agent_scores = []
        for _, row in summary[summary['agent_name'] == agent].iterrows():
            # Approximate individual scores from summary stats
            # (This is a simplified version; ideally we'd have raw data)
            agent_scores.extend([row['mean_score']] * int(row['n_trials']))

        data_for_box.append(agent_scores)
        labels_for_box.append(agent)

    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    ax2.set_ylabel('Score')
    ax2.set_title('Score Distribution by Agent Type')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_matrix(results: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap comparing all agent pairs across surfaces.

    Args:
        results: List of episode results
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Get unique agents and surfaces
    agents = sorted(set(r.get('agent_name', 'unknown') for r in results))
    surfaces = sorted(set(r.get('surface_name', 'unknown') for r in results))

    # Create matrix of mean scores
    matrix = np.zeros((len(agents), len(surfaces)))

    for i, agent in enumerate(agents):
        for j, surface in enumerate(surfaces):
            scores = [r['score'] for r in results
                     if r.get('agent_name') == agent and r.get('surface_name') == surface]
            if scores:
                matrix[i, j] = np.mean(scores)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(surfaces)))
    ax.set_yticks(np.arange(len(agents)))
    ax.set_xticklabels(surfaces, rotation=45, ha='right')
    ax.set_yticklabels(agents)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(agents)):
        for j in range(len(surfaces)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Agent Performance Heatmap')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_report(results_dir: str, output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive text report of evaluation results.

    Args:
        results_dir: Directory containing results
        output_path: Optional path to save report (if None, returns as string)

    Returns:
        Report as string
    """
    results = load_results(results_dir)
    summary = compute_summary_stats(results)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTotal episodes: {len(results)}")
    report_lines.append(f"Results directory: {results_dir}\n")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 70)
    report_lines.append(summary.to_string(index=False))

    # Agent comparisons
    agents = summary['agent_name'].unique()
    if len(agents) >= 2:
        report_lines.append("\n" + "=" * 70)
        report_lines.append("PAIRWISE COMPARISONS")
        report_lines.append("=" * 70)

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                try:
                    comparison = compare_agents(results, agents[i], agents[j])
                    report_lines.append(f"\n{agents[i]} vs {agents[j]}:")
                    report_lines.append(f"  Mean difference: {comparison['mean_diff']:.3f}")
                    report_lines.append(f"  t-statistic: {comparison['t_statistic']:.3f}")
                    report_lines.append(f"  p-value: {comparison['p_value']:.4f}")
                    report_lines.append(f"  Effect size (Cohen's d): {comparison['effect_size']:.3f}")
                    report_lines.append(f"  Significant: {'Yes' if comparison['significant'] else 'No'}")
                except Exception as e:
                    report_lines.append(f"\nError comparing {agents[i]} vs {agents[j]}: {e}")

    report_lines.append("\n" + "=" * 70)

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


if __name__ == "__main__":
    """Example usage of analysis tools."""
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results/example_evaluation"

    print(f"Analyzing results from: {results_dir}\n")

    # Load and analyze
    results = load_results(results_dir)
    print(f"Loaded {len(results)} episodes")

    # Compute summary stats
    summary = compute_summary_stats(results)
    print("\nSummary Statistics:")
    print(summary)

    # Generate report
    report = generate_report(results_dir)
    print("\n" + report)

    # Create visualizations
    fig1 = plot_results(summary, save_path=f"{results_dir}/analysis_plots.png")
    fig2 = plot_comparison_matrix(results, save_path=f"{results_dir}/comparison_heatmap.png")

    print(f"\nVisualizations saved to {results_dir}/")
