"""
Tests for the Episode Runner module.

These tests verify the critical orchestration logic that coordinates
observations, communication, and decisions between agents.
"""

import pytest
from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.episode import Episode
from manifold_benchmark.agents.random_agent import RandomAgent
from manifold_benchmark.agents.greedy_agent import GreedyAgent
from manifold_benchmark.experiments.runner import EpisodeRunner


class MockAgent:
    """Mock agent for testing runner logic."""

    def __init__(self, role: str):
        self.role = role
        self.observation_history = []
        self.message_history = []
        self.observations_received = []
        self.messages_received = []
        self.generate_message_calls = 0
        self.decide_position_calls = 0
        self.final_decision_calls = 0

    def receive_observation(self, observation: dict):
        self.observation_history.append(observation)
        self.observations_received.append(observation)

    def receive_message(self, message: str):
        self.message_history.append(message)
        self.messages_received.append(message)

    def generate_message(self) -> str:
        self.generate_message_calls += 1
        return f"Message from {self.role} (call {self.generate_message_calls})"

    def decide_position(self) -> float:
        self.decide_position_calls += 1
        # Return a deterministic position for testing
        return 5.0 + self.decide_position_calls * 0.1

    def final_decision(self) -> float:
        self.final_decision_calls += 1
        return 7.5

    def reset(self):
        self.observation_history = []
        self.message_history = []
        self.observations_received = []
        self.messages_received = []
        self.generate_message_calls = 0
        self.decide_position_calls = 0
        self.final_decision_calls = 0


def test_runner_initialization():
    """Test that runner initializes correctly."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, n_turns=3)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')

    runner = EpisodeRunner(episode, agent_a, agent_b)

    assert runner.episode == episode
    assert runner.agent_a == agent_a
    assert runner.agent_b == agent_b
    assert runner.transcript == []


def test_runner_validates_agent_roles():
    """Test that runner validates agent roles."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)
    agent_a = MockAgent('A')
    agent_b_wrong = MockAgent('A')  # Wrong role!

    with pytest.raises(ValueError, match="agent_b must have role='B'"):
        EpisodeRunner(episode, agent_a, agent_b_wrong)


def test_run_turn_sequence():
    """Test that a single turn executes in correct sequence."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    # Run turn 1
    turn_data = runner.run_turn(turn_number=1)

    # Verify agents received observations
    assert len(agent_a.observations_received) == 1
    assert len(agent_b.observations_received) == 1

    # Verify agents generated messages
    assert agent_a.generate_message_calls == 1
    assert agent_b.generate_message_calls == 1

    # Verify agents received each other's messages
    assert len(agent_a.messages_received) == 1
    assert len(agent_b.messages_received) == 1

    # Verify agents made decisions
    assert agent_a.decide_position_calls == 1
    assert agent_b.decide_position_calls == 1

    # Verify turn data structure
    assert turn_data["turn"] == 1
    assert "position_before" in turn_data
    assert "observation_a" in turn_data
    assert "observation_b" in turn_data
    assert "message_a" in turn_data
    assert "message_b" in turn_data
    assert "decision_a" in turn_data
    assert "decision_b" in turn_data
    assert "position_after" in turn_data


def test_speaker_alternation():
    """Test that agents alternate who speaks first."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, n_turns=4)

    # Create agents that track message order
    class OrderTrackingAgent(MockAgent):
        def __init__(self, role: str):
            super().__init__(role)
            self.message_order = []

        def generate_message(self) -> str:
            msg = super().generate_message()
            self.message_order.append(('sent', msg))
            return msg

        def receive_message(self, message: str):
            super().receive_message(message)
            self.message_order.append(('received', message))

    agent_a = OrderTrackingAgent('A')
    agent_b = OrderTrackingAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    # Turn 1 (odd): A should speak first
    runner.run_turn(1)
    # A sends, B receives, B sends, A receives
    assert agent_a.message_order[0][0] == 'sent'
    assert agent_b.message_order[0][0] == 'received'
    assert agent_b.message_order[1][0] == 'sent'
    assert agent_a.message_order[1][0] == 'received'

    # Clear for turn 2
    agent_a.message_order = []
    agent_b.message_order = []

    # Turn 2 (even): B should speak first
    runner.run_turn(2)
    # B sends, A receives, A sends, B receives
    assert agent_b.message_order[0][0] == 'sent'
    assert agent_a.message_order[0][0] == 'received'
    assert agent_a.message_order[1][0] == 'sent'
    assert agent_b.message_order[1][0] == 'received'


def test_run_episode_complete():
    """Test that a complete episode runs correctly."""
    surface = Surface([{"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    result = runner.run_episode()

    # Verify result structure
    assert "surface_config" in result
    assert "initial_position" in result
    assert "n_turns" in result
    assert "turns" in result
    assert "final_decision" in result
    assert "score" in result
    assert "optimal_position" in result

    # Verify correct number of turns
    assert len(result["turns"]) == 3
    assert result["n_turns"] == 3

    # Verify agents were called correct number of times
    # Each turn: 1 observation, 1 message, 1 decision
    # Plus final: 1 observation, 1 message, 1 final decision
    assert len(agent_a.observations_received) == 4  # 3 turns + 1 final
    assert len(agent_b.observations_received) == 4
    assert agent_a.generate_message_calls == 4  # 3 turns + 1 final
    assert agent_b.generate_message_calls == 4
    assert agent_a.decide_position_calls == 3  # Only during turns
    assert agent_b.decide_position_calls == 3
    assert agent_a.final_decision_calls == 1  # Only at end
    assert agent_b.final_decision_calls == 1

    # Verify score is computed
    assert 0.0 <= result["score"] <= 1.0


def test_episode_position_updates():
    """Test that episode position updates correctly each turn."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    # Turn 1
    turn1 = runner.run_turn(1)
    assert turn1["position_before"] == (5.0, 5.0)
    assert turn1["position_after"] == (turn1["decision_a"], turn1["decision_b"])
    assert episode.current_position == turn1["position_after"]

    # Turn 2
    turn2 = runner.run_turn(2)
    assert turn2["position_before"] == turn1["position_after"]
    assert turn2["position_after"] == (turn2["decision_a"], turn2["decision_b"])
    assert episode.current_position == turn2["position_after"]


def test_run_episode_with_real_agents():
    """Test episode with real Random and Greedy agents."""
    surface = Surface([{"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=5)
    agent_a = RandomAgent(role='A', seed=42)
    agent_b = GreedyAgent(role='B', step_size=1.0)
    runner = EpisodeRunner(episode, agent_a, agent_b)

    result = runner.run_episode()

    # Verify episode completed
    assert len(result["turns"]) == 5
    assert result["final_decision"] is not None
    assert 0.0 <= result["score"] <= 1.0

    # Verify all turns have required data
    for turn in result["turns"]:
        assert "turn" in turn
        assert "observation_a" in turn
        assert "observation_b" in turn
        assert "message_a" in turn
        assert "message_b" in turn
        assert "decision_a" in turn
        assert "decision_b" in turn


def test_agents_reset_at_episode_start():
    """Test that agents are reset at the start of each episode."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode1 = Episode(surface, n_turns=2)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')

    # Run first episode
    runner1 = EpisodeRunner(episode1, agent_a, agent_b)
    runner1.run_episode()

    # Agents should have accumulated history
    assert len(agent_a.observations_received) > 0

    # Run second episode with same agents
    episode2 = Episode(surface, n_turns=2)
    runner2 = EpisodeRunner(episode2, agent_a, agent_b)

    # Before run_episode, manually reset to test
    agent_a.reset()
    agent_b.reset()

    # History should be cleared
    assert len(agent_a.observations_received) == 0
    assert len(agent_b.observations_received) == 0


def test_final_decision_separate_from_last_turn():
    """Test that final decision is separate from last turn decision."""
    surface = Surface([{"cx": 7.0, "cy": 7.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, n_turns=2)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    result = runner.run_episode()

    # Last turn position
    last_turn = result["turns"][-1]
    last_turn_position = last_turn["position_after"]

    # Final decision (made after last turn)
    final_decision = result["final_decision"]

    # They can be different (agents get one more observation and message exchange)
    # Final decision should be from final_decision(), not decide_position()
    assert final_decision == (7.5, 7.5)  # MockAgent.final_decision returns 7.5


def test_transcript_accumulation():
    """Test that transcript accumulates turn data correctly."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, n_turns=3)
    agent_a = MockAgent('A')
    agent_b = MockAgent('B')
    runner = EpisodeRunner(episode, agent_a, agent_b)

    # Run turns manually
    runner.run_turn(1)
    assert len(runner.transcript) == 1

    runner.run_turn(2)
    assert len(runner.transcript) == 2

    runner.run_turn(3)
    assert len(runner.transcript) == 3

    # Verify turn numbers
    assert runner.transcript[0]["turn"] == 1
    assert runner.transcript[1]["turn"] == 2
    assert runner.transcript[2]["turn"] == 3
