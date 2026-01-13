"""
Simple transcript viewer for episode results.

Usage:
    python -m manifold_benchmark.experiments.transcript results/path/to/run.json
    python -m manifold_benchmark.experiments.transcript results/path/to/run.json --turn 3
"""

import json
import sys
import argparse
from typing import Dict, Any, Optional


def format_slice(slice_data: list, axis: str) -> str:
    """Format slice data as a simple ASCII representation."""
    if not slice_data:
        return "  (no data)"

    lines = []
    for sample in slice_data:
        coord = sample.get(axis, 0.0)
        val = sample.get('value', 0.0)
        # Simple bar visualization
        bar_len = int(val * 20)
        bar = '#' * bar_len
        lines.append(f"    {axis}={coord:.1f}: {val:.3f} {bar}")
    return '\n'.join(lines)


def print_turn(turn_data: Dict[str, Any], turn_num: int) -> None:
    """Print a single turn in readable format."""
    pos_before = turn_data.get('position_before', [0, 0])
    pos_after = turn_data.get('position_after', [0, 0])

    print(f"\n{'='*60}")
    print(f"TURN {turn_num}")
    print(f"{'='*60}")
    print(f"Position: ({pos_before[0]:.2f}, {pos_before[1]:.2f})")

    # Agent A observation
    obs_a = turn_data.get('observation_a', {})
    print(f"\n--- AGENT A (controls X) ---")
    print(f"  Value here: {obs_a.get('value_at_position', 0):.3f}")
    print(f"  Gradient df/dx: {obs_a.get('gradient_x', 0):.3f}")
    print(f"  X-slice:")
    print(format_slice(obs_a.get('slice', []), 'x'))

    # Agent B observation
    obs_b = turn_data.get('observation_b', {})
    print(f"\n--- AGENT B (controls Y) ---")
    print(f"  Value here: {obs_b.get('value_at_position', 0):.3f}")
    print(f"  Gradient df/dy: {obs_b.get('gradient_y', 0):.3f}")
    print(f"  Y-slice:")
    print(format_slice(obs_b.get('slice', []), 'y'))

    # Messages
    msg_a = turn_data.get('message_a', '')
    msg_b = turn_data.get('message_b', '')

    print(f"\n--- COMMUNICATION ---")
    if msg_a:
        print(f"  Agent A says: {msg_a}")
    else:
        print(f"  Agent A: (no message)")

    if msg_b:
        print(f"  Agent B says: {msg_b}")
    else:
        print(f"  Agent B: (no message)")

    # Reasoning (if captured)
    reasoning_a = turn_data.get('reasoning_a', '')
    reasoning_b = turn_data.get('reasoning_b', '')

    if reasoning_a or reasoning_b:
        print(f"\n--- REASONING ---")
        if reasoning_a:
            print(f"  Agent A thinks: {reasoning_a}")
        if reasoning_b:
            print(f"  Agent B thinks: {reasoning_b}")

    # Decisions
    dec_a = turn_data.get('decision_a', 0)
    dec_b = turn_data.get('decision_b', 0)

    print(f"\n--- DECISIONS ---")
    print(f"  Agent A decides: x = {dec_a:.2f}")
    print(f"  Agent B decides: y = {dec_b:.2f}")
    print(f"  -> Move to: ({pos_after[0]:.2f}, {pos_after[1]:.2f})")


def print_final(result: Dict[str, Any]) -> None:
    """Print final decision and score."""
    print(f"\n{'='*60}")
    print(f"FINAL DECISION")
    print(f"{'='*60}")

    final_pos = result.get('final_position', [0, 0])
    final_dec = result.get('final_decision', [0, 0])
    optimal = result.get('optimal_position', [0, 0])
    score = result.get('score', 0)

    # Final observations
    obs_a = result.get('final_observation_a', {})
    obs_b = result.get('final_observation_b', {})

    print(f"\nFinal position after exploration: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"Value at final position: {obs_a.get('value_at_position', 0):.3f}")

    # Final messages
    msg_a = result.get('final_message_a', '')
    msg_b = result.get('final_message_b', '')

    if msg_a or msg_b:
        print(f"\n--- FINAL COMMUNICATION ---")
        if msg_a:
            print(f"  Agent A: {msg_a}")
        if msg_b:
            print(f"  Agent B: {msg_b}")

    # Final reasoning
    reasoning_a = result.get('final_reasoning_a', '')
    reasoning_b = result.get('final_reasoning_b', '')

    if reasoning_a or reasoning_b:
        print(f"\n--- FINAL REASONING ---")
        if reasoning_a:
            print(f"  Agent A thinks: {reasoning_a}")
        if reasoning_b:
            print(f"  Agent B thinks: {reasoning_b}")

    print(f"\n--- FINAL ANSWER ---")
    print(f"  Agents' answer: ({final_dec[0]:.2f}, {final_dec[1]:.2f})")
    print(f"  Optimal was:    ({optimal[0]:.2f}, {optimal[1]:.2f})")
    print(f"\n  SCORE: {score:.3f} ({score*100:.1f}%)")


def print_header(result: Dict[str, Any]) -> None:
    """Print episode header info."""
    print(f"{'='*60}")
    print(f"EPISODE TRANSCRIPT")
    print(f"{'='*60}")

    agent_a = result.get('agent_a_type', 'Unknown')
    agent_b = result.get('agent_b_type', 'Unknown')
    n_turns = result.get('n_turns', 10)
    initial = result.get('initial_position', [5, 5])

    print(f"Agent A: {agent_a}")
    print(f"Agent B: {agent_b}")
    print(f"Turns: {n_turns}")
    print(f"Starting position: ({initial[0]:.2f}, {initial[1]:.2f})")

    # Surface info
    surface = result.get('surface_config', {})
    peaks = surface.get('peaks', [])
    if peaks:
        print(f"Surface: {len(peaks)} peaks")
        for i, peak in enumerate(peaks):
            print(f"  Peak {i+1}: ({peak['cx']:.1f}, {peak['cy']:.1f}) height={peak['height']:.1f}")


def view_transcript(filepath: str, turn: Optional[int] = None) -> None:
    """
    View an episode transcript.

    Args:
        filepath: Path to JSON result file
        turn: Optional specific turn to view (None = all turns)
    """
    with open(filepath, 'r') as f:
        result = json.load(f)

    print_header(result)

    turns = result.get('turns', [])

    if turn is not None:
        # Show specific turn
        if 1 <= turn <= len(turns):
            print_turn(turns[turn - 1], turn)
        else:
            print(f"Error: Turn {turn} not found (episode has {len(turns)} turns)")
    else:
        # Show all turns
        for i, turn_data in enumerate(turns):
            print_turn(turn_data, i + 1)

        # Show final decision
        print_final(result)


def main():
    parser = argparse.ArgumentParser(description='View episode transcript')
    parser.add_argument('filepath', help='Path to JSON result file')
    parser.add_argument('--turn', '-t', type=int, help='View specific turn only')

    args = parser.parse_args()
    view_transcript(args.filepath, args.turn)


if __name__ == '__main__':
    main()
