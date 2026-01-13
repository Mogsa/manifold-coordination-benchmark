"""
Batch evaluation harness for running multiple experiments.

This module provides functions for running systematic evaluations across
multiple surfaces, agent configurations, and trials.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.episode import Episode
from manifold_benchmark.agents.random_agent import RandomAgent
from manifold_benchmark.agents.greedy_agent import GreedyAgent
from manifold_benchmark.agents.llm_agent import LLMAgent
from manifold_benchmark.experiments.runner import EpisodeRunner
from manifold_benchmark.experiments.logger import ResultLogger


# Pre-defined test surfaces from PLAN.md Section 7.4
TEST_SURFACES = {
    "single_peak_center": {
        "peaks": [{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.5}],
        "optimal": (5.0, 5.0),
        "difficulty": 1
    },
    "single_peak_corner": {
        "peaks": [{"cx": 8.0, "cy": 8.0, "height": 1.0, "sigma": 1.2}],
        "optimal": (8.0, 8.0),
        "difficulty": 1
    },
    "two_peaks_clear": {
        "peaks": [
            {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},
            {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}
        ],
        "optimal": (7.5, 7.5),
        "difficulty": 2
    },
    "two_peaks_close": {
        "peaks": [
            {"cx": 3.0, "cy": 7.0, "height": 0.92, "sigma": 1.0},
            {"cx": 7.0, "cy": 3.0, "height": 1.00, "sigma": 1.0}
        ],
        "optimal": (7.0, 3.0),
        "difficulty": 3
    },
    "three_peaks": {
        "peaks": [
            {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
            {"cx": 8.0, "cy": 2.0, "height": 0.7, "sigma": 1.0},
            {"cx": 5.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
        ],
        "optimal": (5.0, 8.0),
        "difficulty": 4
    }
}


def create_agent_from_config(config: dict, role: str):
    """
    Create an agent instance from configuration dictionary.

    Args:
        config: Agent configuration with 'type' key and optional parameters
        role: 'A' or 'B'

    Returns:
        Agent instance
    """
    from manifold_benchmark.agents.random_agent import RandomAgent
    from manifold_benchmark.agents.greedy_agent import GreedyAgent
    from manifold_benchmark.agents.llm_agent import LLMAgent

    agent_type = config.get('type', 'random').lower()

    if agent_type == 'random':
        seed = config.get('seed', None)
        return RandomAgent(role=role, seed=seed)

    elif agent_type == 'greedy':
        step_size = config.get('step_size', 1.0)
        return GreedyAgent(role=role, step_size=step_size)

    elif agent_type == 'llm':
        model = config.get('model', 'gpt-4')
        api_key = config.get('api_key', None)
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 500)

        # Construct system prompt path based on role
        system_prompt_path = f"manifold_benchmark/prompts/agent_{role.lower()}_system.txt"

        return LLMAgent(
            role=role,
            model=model,
            api_key=api_key,
            system_prompt_path=system_prompt_path,
            temperature=temperature,
            max_tokens=max_tokens
        )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_evaluation(
    surfaces: list,
    agent_configs: list,
    n_trials: int = 5,
    output_dir: str = "results",
    n_turns: int = 10,
    verbose: bool = True
):
    """
    Run batch evaluation across multiple surfaces and agent configurations.

    Args:
        surfaces: List of surface names (keys from TEST_SURFACES) or surface config dicts
        agent_configs: List of agent configuration dicts with format:
                      [{"type": "random", "seed": 42, "name": "Random Baseline"},
                       {"type": "greedy", "step_size": 1.0, "name": "Greedy"},
                       {"type": "llm", "model": "gpt-4", "name": "GPT-4"}]
        n_trials: Number of trials per (surface, agent_config) combination
        output_dir: Base directory for saving results
        n_turns: Number of turns per episode
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation results and metadata
    """
    from manifold_benchmark.core.surface import Surface
    from manifold_benchmark.core.episode import Episode
    from tqdm import tqdm
    import os
    import time

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result logger
    logger = ResultLogger(output_dir)

    # Calculate total runs
    total_runs = len(surfaces) * len(agent_configs) * n_trials

    if verbose:
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION")
        print(f"{'='*60}")
        print(f"Surfaces: {len(surfaces)}")
        print(f"Agent configurations: {len(agent_configs)}")
        print(f"Trials per combination: {n_trials}")
        print(f"Total episodes to run: {total_runs}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

    # Track results
    results = {
        "total_runs": total_runs,
        "completed": 0,
        "failed": 0,
        "results_dir": output_dir,
        "episodes": [],
        "start_time": time.time()
    }

    # Progress bar
    pbar = tqdm(total=total_runs, desc="Running experiments", disable=not verbose)

    # Iterate through all combinations
    for surface_spec in surfaces:
        # Create surface
        if isinstance(surface_spec, str):
            # Surface name from TEST_SURFACES
            if surface_spec not in TEST_SURFACES:
                raise ValueError(f"Unknown surface: {surface_spec}. "
                               f"Available: {list(TEST_SURFACES.keys())}")
            surface_config = TEST_SURFACES[surface_spec]
            surface_name = surface_spec
        else:
            # Direct surface configuration
            surface_config = surface_spec
            surface_name = surface_config.get('name', 'custom_surface')

        surface = Surface(surface_config['peaks'])

        for agent_config in agent_configs:
            agent_name = agent_config.get('name', agent_config['type'])

            for trial in range(n_trials):
                try:
                    # Create fresh agents for this trial
                    agent_a = create_agent_from_config(agent_config, role='A')
                    agent_b = create_agent_from_config(agent_config, role='B')

                    # Create episode
                    episode = Episode(surface, n_turns=n_turns)

                    # Create runner
                    runner = EpisodeRunner(episode, agent_a, agent_b)

                    # Run episode
                    pbar.set_description(f"Running: {surface_name} | {agent_name} | Trial {trial+1}/{n_trials}")
                    result = runner.run_episode()

                    # Add metadata
                    result['surface_name'] = surface_name
                    result['surface_config'] = surface_config
                    result['agent_config'] = agent_config
                    result['agent_name'] = agent_name
                    result['trial'] = trial + 1

                    # Save result
                    run_id = logger.save_result(result)
                    result['run_id'] = run_id

                    # Track
                    results['episodes'].append({
                        'run_id': run_id,
                        'surface': surface_name,
                        'agent': agent_name,
                        'trial': trial + 1,
                        'score': result['score'],
                        'final_position': result['final_decision']
                    })
                    results['completed'] += 1

                except Exception as e:
                    if verbose:
                        tqdm.write(f"ERROR: {surface_name} | {agent_name} | Trial {trial+1}: {str(e)}")
                    results['failed'] += 1

                finally:
                    pbar.update(1)

    pbar.close()

    # Finalize
    results['end_time'] = time.time()
    results['duration_seconds'] = results['end_time'] - results['start_time']

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Completed: {results['completed']}/{total_runs}")
        print(f"Failed: {results['failed']}")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}\n")

    # Save summary
    import json
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    """Example usage of batch evaluation."""

    # Define surfaces to test
    surfaces = [
        "single_peak_center",
        "two_peaks_clear",
        "three_peaks"
    ]

    # Define agent configurations
    agent_configs = [
        {"type": "random", "seed": 42, "name": "Random Baseline"},
        {"type": "greedy", "step_size": 1.0, "name": "Greedy Gradient"}
    ]

    # Run evaluation
    results = run_evaluation(
        surfaces=surfaces,
        agent_configs=agent_configs,
        n_trials=5,
        output_dir="results/example_evaluation",
        n_turns=10
    )

    print(f"\nResults summary:")
    print(f"Total episodes: {results['total_runs']}")
    print(f"Successful: {results['completed']}")
    print(f"Failed: {results['failed']}")
