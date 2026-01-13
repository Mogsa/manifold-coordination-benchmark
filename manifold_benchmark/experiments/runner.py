"""
Episode runner module for the Manifold Coordination Benchmark.

This module orchestrates the execution of episodes, coordinating observations,
communication, and decisions between agents.
"""

from typing import Dict, Any
from manifold_benchmark.core.episode import Episode
from manifold_benchmark.agents.base import BaseAgent


class EpisodeRunner:
    """
    Orchestrates the execution of episodes with two agents.

    The runner handles the critical sequencing of:
    1. Observations - giving each agent their view
    2. Communication - exchanging messages between agents
    3. Decisions - collecting position choices
    4. Movement - updating the episode state
    """

    def __init__(self, episode: Episode, agent_a: BaseAgent, agent_b: BaseAgent):
        """
        Initialize an episode runner.

        Args:
            episode: The Episode to run
            agent_a: Agent controlling x-coordinate (role='A')
            agent_b: Agent controlling y-coordinate (role='B')
        """
        self.episode = episode
        self.agent_a = agent_a
        self.agent_b = agent_b

        # Validate agent roles
        if agent_a.role != 'A':
            raise ValueError(f"agent_a must have role='A', got role='{agent_a.role}'")
        if agent_b.role != 'B':
            raise ValueError(f"agent_b must have role='B', got role='{agent_b.role}'")

        # Transcript storage
        self.transcript = []

    def run_turn(self, turn_number: int) -> Dict[str, Any]:
        """
        Execute a single turn of the episode.

        Turn sequence:
        1. Get current position
        2. Generate observations for both agents
        3. Deliver observations to agents
        4. Exchange messages (with alternating speaker order)
        5. Agents decide new positions
        6. Update episode state

        Args:
            turn_number: Current turn number (1-indexed)

        Returns:
            Dictionary containing turn data:
                - turn: Turn number
                - position_before: Position before move
                - observation_a: Agent A's observation
                - observation_b: Agent B's observation
                - message_a: Agent A's message
                - message_b: Agent B's message
                - decision_a: Agent A's position decision
                - decision_b: Agent B's position decision
                - position_after: Position after move
        """
        # Get position before this turn's move
        position_before = self.episode.current_position
        x_current, y_current = position_before

        # STEP 1: OBSERVE
        # Generate observations at current position
        obs_a = self.episode.observation_generator.generate_observation_a(x_current, y_current)
        obs_b = self.episode.observation_generator.generate_observation_b(x_current, y_current)

        # Deliver observations to agents
        self.agent_a.receive_observation(obs_a)
        self.agent_b.receive_observation(obs_b)

        # STEP 2: COMMUNICATE
        # Alternate who speaks first for fairness
        # Odd turns (1,3,5,7,9): A speaks first
        # Even turns (2,4,6,8,10): B speaks first
        if turn_number % 2 == 1:
            # A speaks first
            message_a = self.agent_a.generate_message()
            self.agent_b.receive_message(message_a)
            message_b = self.agent_b.generate_message()
            self.agent_a.receive_message(message_b)
        else:
            # B speaks first
            message_b = self.agent_b.generate_message()
            self.agent_a.receive_message(message_b)
            message_a = self.agent_a.generate_message()
            self.agent_b.receive_message(message_a)

        # STEP 3: DECIDE
        # Both agents decide their new position
        decision_a = self.agent_a.decide_position()
        decision_b = self.agent_b.decide_position()

        # STEP 4: MOVE
        # Update episode with new position (this increments turn counter)
        obs_a_new, obs_b_new = self.episode.step(decision_a, decision_b)
        position_after = (decision_a, decision_b)

        # Capture reasoning if available (LLM agents store this)
        reasoning_a = getattr(self.agent_a, 'last_reasoning', '')
        reasoning_b = getattr(self.agent_b, 'last_reasoning', '')

        # Record turn data
        turn_data = {
            "turn": turn_number,
            "position_before": position_before,
            "observation_a": obs_a,
            "observation_b": obs_b,
            "message_a": message_a,
            "message_b": message_b,
            "reasoning_a": reasoning_a,
            "reasoning_b": reasoning_b,
            "decision_a": decision_a,
            "decision_b": decision_b,
            "position_after": position_after
        }

        # Add to transcript
        self.transcript.append(turn_data)

        return turn_data

    def run_episode(self) -> Dict[str, Any]:
        """
        Run a complete episode with all turns plus final decision.

        Returns:
            Dictionary containing complete episode results:
                - surface_config: Surface configuration
                - initial_position: Starting position
                - n_turns: Number of turns
                - turns: List of turn data dictionaries
                - final_position: Final position after exploration
                - final_decision: Agents' final answer for maximum
                - score: Normalized score [0, 1]
                - optimal_position: True optimal position
                - optimal_value: True optimal value
        """
        # Reset agents at start of episode
        self.agent_a.reset()
        self.agent_b.reset()
        self.transcript = []

        # Get initial position
        initial_position = self.episode.current_position

        # Run all turns
        for turn_num in range(1, self.episode.n_turns + 1):
            turn_data = self.run_turn(turn_num)
            # Note: run_turn already appends to transcript

        # FINAL DECISION
        # After all exploration turns, agents make final decision
        final_position = self.episode.current_position
        x_final, y_final = final_position

        # Generate final observations at current position
        obs_a_final = self.episode.observation_generator.generate_observation_a(x_final, y_final)
        obs_b_final = self.episode.observation_generator.generate_observation_b(x_final, y_final)

        # Deliver final observations
        self.agent_a.receive_observation(obs_a_final)
        self.agent_b.receive_observation(obs_b_final)

        # Final message exchange (A speaks first by default)
        message_a_final = self.agent_a.generate_message()
        self.agent_b.receive_message(message_a_final)
        message_b_final = self.agent_b.generate_message()
        self.agent_a.receive_message(message_b_final)

        # Get final decisions
        x_final_decision = self.agent_a.final_decision()
        y_final_decision = self.agent_b.final_decision()
        final_decision = (x_final_decision, y_final_decision)

        # Capture final reasoning
        final_reasoning_a = getattr(self.agent_a, 'last_reasoning', '')
        final_reasoning_b = getattr(self.agent_b, 'last_reasoning', '')

        # Calculate score
        score = self.episode.get_score(x_final_decision, y_final_decision)

        # Get optimal position for reference
        x_opt, y_opt, f_opt = self.episode.surface.get_optimal()

        # Compile complete episode result
        result = {
            "surface_config": self.episode.surface.to_dict(),
            "initial_position": initial_position,
            "n_turns": self.episode.n_turns,
            "turns": self.transcript,
            "final_position": final_position,  # Position after last exploration turn
            "final_observation_a": obs_a_final,
            "final_observation_b": obs_b_final,
            "final_message_a": message_a_final,
            "final_message_b": message_b_final,
            "final_reasoning_a": final_reasoning_a,
            "final_reasoning_b": final_reasoning_b,
            "final_decision": final_decision,  # Agents' final answer
            "score": score,
            "optimal_position": (x_opt, y_opt),
            "optimal_value": f_opt,
            "agent_a_type": type(self.agent_a).__name__,
            "agent_b_type": type(self.agent_b).__name__
        }

        return result

    def get_transcript(self) -> list:
        """
        Get the current episode transcript.

        Returns:
            List of turn data dictionaries
        """
        return self.transcript.copy()
