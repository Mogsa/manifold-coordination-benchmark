#!/usr/bin/env python3
"""Run a metacognitive agent session on ChaosBench."""
import argparse
from pathlib import Path
import json

from chaosbench.legacy_v0.experiments.session import SessionRunner, SessionConfig
from chaosbench.legacy_v0.agents.metacognitive_agent import MetacognitiveAgent
from chaosbench.visualization import plot_phi_curve, plot_session_summary, plot_task


def main():
    parser = argparse.ArgumentParser(description="Run metacognitive agent on ChaosBench")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--timeout", type=int, default=300, help="Session timeout in seconds")
    parser.add_argument("--output", default="session_output", help="Output directory")
    parser.add_argument("--conditional", action="store_true", help="Reveal system family")
    parser.add_argument("--scaffolded", action="store_true", help="Enable HYPOTHESIZE/FIT actions")
    # Difficulty settings
    parser.add_argument("--horizon", type=float, default=1.5, help="Horizon multiplier (higher=harder, default 1.5)")
    parser.add_argument("--noise", type=float, default=0.01, help="Observation noise std (higher=harder, default 0.01)")
    # Visualization settings
    parser.add_argument("--save-task-plots", action="store_true", help="Save individual task plots with bins")
    parser.add_argument("--max-task-plots", type=int, default=10, help="Max number of task plots to save (default 10)")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Create agent and runner
    agent = MetacognitiveAgent(model=args.model, scaffolded=args.scaffolded)
    config = SessionConfig(
        n_tasks=args.n_tasks,
        timeout_seconds=args.timeout,
        conditional=args.conditional,
        scaffolded=args.scaffolded,
        horizon_multiplier=args.horizon,
        noise_std=args.noise,
    )
    runner = SessionRunner(config)

    print(f"Running {args.n_tasks} tasks with {args.model}...")
    print(f"Timeout: {args.timeout}s, Conditional: {args.conditional}, Scaffolded: {args.scaffolded}")
    print(f"Difficulty: horizon={args.horizon}x, noise={args.noise}")
    print("-" * 50)

    # Run session
    result = runner.run(agent)

    # Print summary
    print("-" * 50)
    print(f"Tasks completed: {result.tasks_completed}")
    print(f"Final Phi: {result.final_phi:.2f}")
    print(f"Total time: {result.total_time:.1f}s")
    print(f"Tasks/second: {result.tasks_completed / result.total_time:.2f}")

    # Save outputs
    trace_path = output_dir / "trace.md"
    trace_path.write_text(result.trace.to_markdown())
    print(f"Trace saved to: {trace_path}")

    learnings_path = output_dir / "learnings.md"
    learnings_path.write_text(result.final_learnings)
    print(f"Learnings saved to: {learnings_path}")

    # Save Phi(t) curve as JSON
    phi_path = output_dir / "phi_curve.json"
    phi_data = [
        {"time": p.wall_time, "phi": p.cumulative_phi, "tasks": p.tasks_completed}
        for p in result.phi_curve
    ]
    phi_path.write_text(json.dumps(phi_data, indent=2))
    print(f"Phi(t) curve saved to: {phi_path}")

    # Generate visualizations
    print("Generating visualizations...")

    # Plot Φ(n) curve
    phi_plot_path = output_dir / "phi_curve.png"
    plot_phi_curve(
        phi_data,
        title=f"Learning Curve - {args.model}",
        save_path=str(phi_plot_path),
        show=False,
    )
    print(f"Φ(n) plot saved to: {phi_plot_path}")

    # Extract per-task scores from trace
    scores = [task.final_score for task in result.trace.tasks]

    # Plot session summary
    summary_plot_path = output_dir / "session_summary.png"
    plot_session_summary(
        phi_data,
        scores,
        title=f"Session Summary - {args.model} ({'Scaffolded' if args.scaffolded else 'MVP'})",
        save_path=str(summary_plot_path),
        show=False,
    )
    print(f"Summary plot saved to: {summary_plot_path}")

    # Generate individual task plots if requested
    if args.save_task_plots:
        task_plots_dir = output_dir / "task_plots"
        task_plots_dir.mkdir(exist_ok=True)

        n_plots = min(args.max_task_plots, len(result.tasks))
        print(f"Generating {n_plots} task plots...")

        for i in range(n_plots):
            task = result.tasks[i]
            task_trace = result.trace.tasks[i]

            # Get prediction and actual from trace feedback
            prediction = None
            actual = None
            for turn in task_trace.turns:
                if turn.feedback is not None:
                    prediction = turn.feedback.prediction
                    actual = turn.feedback.actual
                    break

            if actual is None:
                # Fallback: get actual from task
                actual = float(task.true_future[0]) if task.true_future.ndim > 0 else float(task.true_future)

            # Determine bounds based on system family
            bounds_map = {
                "logistic": (0.0, 1.0),
                "tent": (0.0, 1.0),
                "henon": (-1.5, 1.5),
                "standard": (0.0, 6.28),
                "lorenz": (-30.0, 30.0),
            }
            bounds = bounds_map.get(task.system.family, (0.0, 1.0))

            plot_path = task_plots_dir / f"task_{i+1:02d}_{task.system.family}.png"
            plot_task(
                observations=task.observations.flatten(),
                actual=actual,
                prediction=prediction,
                uncertainty=0.1,  # Default sigma
                title=f"Task {i+1}: {task.system.family} (h_KS={task.h_ks:.3f})",
                save_path=str(plot_path),
                show=False,
                n_bins=20,
                bounds=bounds,
            )

        print(f"Task plots saved to: {task_plots_dir}/")


if __name__ == "__main__":
    main()
