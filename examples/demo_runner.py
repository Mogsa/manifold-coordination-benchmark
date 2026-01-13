"""
Demo script showing how to use the EpisodeRunner.

This demonstrates running a complete episode with two agents.
"""

from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.episode import Episode
from manifold_benchmark.agents.random_agent import RandomAgent
from manifold_benchmark.agents.greedy_agent import GreedyAgent
from manifold_benchmark.experiments.runner import EpisodeRunner


def main():
    """Run a demo episode."""
    print("=" * 60)
    print("Manifold Coordination Benchmark - Episode Runner Demo")
    print("=" * 60)
    print()

    # Create a surface with two peaks
    print("Creating surface with two peaks...")
    surface = Surface([
        {"cx": 2.5, "cy": 2.5, "height": 0.6, "sigma": 1.2},  # Smaller peak
        {"cx": 7.5, "cy": 7.5, "height": 1.0, "sigma": 1.2}   # Global maximum
    ])
    x_opt, y_opt, f_opt = surface.get_optimal()
    print(f"  Global maximum: ({x_opt:.2f}, {y_opt:.2f}) with value {f_opt:.3f}")
    print()

    # Create episode
    print("Creating episode with 5 turns...")
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=5)
    print(f"  Starting position: {episode.current_position}")
    print()

    # Create agents
    print("Creating agents...")
    agent_a = RandomAgent(role='A', seed=42)
    agent_b = GreedyAgent(role='B', step_size=1.0)
    print("  Agent A: RandomAgent (controls x)")
    print("  Agent B: GreedyAgent (controls y)")
    print()

    # Create runner
    runner = EpisodeRunner(episode, agent_a, agent_b)

    # Run episode
    print("Running episode...")
    print("-" * 60)
    result = runner.run_episode()

    # Display results
    print()
    print("Episode Complete!")
    print("=" * 60)
    print()
    print(f"Turns executed: {len(result['turns'])}")
    print()
    print("Trajectory:")
    for turn in result['turns']:
        x, y = turn['position_after']
        value = surface.evaluate(x, y)
        print(f"  Turn {turn['turn']}: ({x:.2f}, {y:.2f}) -> f = {value:.3f}")

    print()
    print("Final Decision:")
    x_final, y_final = result['final_decision']
    final_value = surface.evaluate(x_final, y_final)
    print(f"  Position: ({x_final:.2f}, {y_final:.2f})")
    print(f"  Value: {final_value:.3f}")
    print(f"  Score: {result['score']:.3f} (1.0 = optimal)")
    print()
    print(f"Optimal position: ({x_opt:.2f}, {y_opt:.2f})")
    print(f"Distance to optimal: {((x_final - x_opt)**2 + (y_final - y_opt)**2)**0.5:.2f}")
    print()


if __name__ == "__main__":
    main()
