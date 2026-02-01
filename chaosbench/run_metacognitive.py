#!/usr/bin/env python3
"""Run a metacognitive agent session on ChaosBench."""
import argparse
from pathlib import Path
import json

from chaosbench.experiments.session import SessionRunner, SessionConfig
from chaosbench.agents.metacognitive_agent import MetacognitiveAgent


def main():
    parser = argparse.ArgumentParser(description="Run metacognitive agent on ChaosBench")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--timeout", type=int, default=300, help="Session timeout in seconds")
    parser.add_argument("--output", default="session_output", help="Output directory")
    parser.add_argument("--conditional", action="store_true", help="Reveal system family")
    parser.add_argument("--scaffolded", action="store_true", help="Enable HYPOTHESIZE/FIT actions")
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
    )
    runner = SessionRunner(config)

    print(f"Running {args.n_tasks} tasks with {args.model}...")
    print(f"Timeout: {args.timeout}s, Conditional: {args.conditional}, Scaffolded: {args.scaffolded}")
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


if __name__ == "__main__":
    main()
