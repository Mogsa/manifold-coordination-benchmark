"""
Test suite for agent implementations in the Manifold Coordination Benchmark.

This module tests:
- BaseAgent (abstract class behavior)
- RandomAgent (bounds, determinism)
- GreedyAgent (gradient following, bounds)
- LLMAgent (coordinate parsing, observation formatting)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from manifold_benchmark.agents.base import BaseAgent
from manifold_benchmark.agents.random_agent import RandomAgent
from manifold_benchmark.agents.greedy_agent import GreedyAgent
from manifold_benchmark.agents.llm_agent import LLMAgent


# =============================================================================
# BaseAgent Tests
# =============================================================================

class TestBaseAgent:
    """Tests for the abstract BaseAgent class."""

    def test_base_agent_cannot_be_instantiated(self):
        """BaseAgent is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(role='A')

    def test_base_agent_abstract_methods(self):
        """BaseAgent requires implementation of abstract methods."""
        # Create a minimal subclass that doesn't implement abstract methods
        class IncompleteAgent(BaseAgent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent(role='A')


# =============================================================================
# RandomAgent Tests
# =============================================================================

class TestRandomAgent:
    """Tests for the RandomAgent class."""

    def test_random_agent_bounds(self):
        """Random agent stays within bounds [0, domain_size]."""
        agent = RandomAgent(role='A', seed=42)
        for _ in range(100):
            pos = agent.decide_position()
            assert 0 <= pos <= 10, f"Position {pos} out of bounds [0, 10]"
            final_pos = agent.final_decision()
            assert 0 <= final_pos <= 10, f"Final position {final_pos} out of bounds [0, 10]"

    def test_random_agent_deterministic(self):
        """Random agent is deterministic with same seed."""
        agent1 = RandomAgent(role='A', seed=42)
        agent2 = RandomAgent(role='A', seed=42)

        for _ in range(10):
            pos1 = agent1.decide_position()
            pos2 = agent2.decide_position()
            assert pos1 == pos2, "Same seed should produce same sequence"

    def test_random_agent_different_seeds(self):
        """Random agent produces different sequences with different seeds."""
        agent1 = RandomAgent(role='A', seed=42)
        agent2 = RandomAgent(role='A', seed=123)

        positions1 = [agent1.decide_position() for _ in range(5)]
        positions2 = [agent2.decide_position() for _ in range(5)]

        # Very unlikely to be the same with different seeds
        assert positions1 != positions2, "Different seeds should produce different sequences"

    def test_random_agent_ignores_observations(self):
        """Random agent ignores observations (doesn't use them)."""
        agent = RandomAgent(role='A', seed=42)

        # Generate positions before observations
        positions_before = [agent.decide_position() for _ in range(5)]

        # Reset and add observations
        agent2 = RandomAgent(role='A', seed=42)
        agent2.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": 0.3,
            "slice": []
        })

        # Generate same positions (observations have no effect)
        for i in range(5):
            pos = agent2.decide_position()
            assert pos == positions_before[i], "Observations should not affect random agent"

    def test_random_agent_custom_domain_size(self):
        """Random agent respects custom domain size."""
        agent = RandomAgent(role='A', seed=42, domain_size=5.0)
        for _ in range(50):
            pos = agent.decide_position()
            assert 0 <= pos <= 5.0, f"Position {pos} out of custom bounds [0, 5]"

    def test_random_agent_no_communication(self):
        """Random agent generates empty messages (no communication)."""
        agent = RandomAgent(role='A', seed=42)
        message = agent.generate_message()
        assert message == "", "Random agent should not communicate"


# =============================================================================
# GreedyAgent Tests
# =============================================================================

class TestGreedyAgent:
    """Tests for the GreedyAgent class."""

    def test_greedy_agent_follows_positive_gradient(self):
        """Greedy agent moves in positive gradient direction."""
        agent = GreedyAgent(role='A', step_size=1.0)

        # Positive gradient should increase position
        agent.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": 0.5,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos > 5.0, "Positive gradient should increase position"
        assert new_pos == 6.0, f"Expected 6.0 with step_size=1.0, got {new_pos}"

    def test_greedy_agent_follows_negative_gradient(self):
        """Greedy agent moves in negative gradient direction."""
        agent = GreedyAgent(role='A', step_size=1.0)

        # Negative gradient should decrease position
        agent.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": -0.5,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos < 5.0, "Negative gradient should decrease position"
        assert new_pos == 4.0, f"Expected 4.0 with step_size=1.0, got {new_pos}"

    def test_greedy_agent_zero_gradient(self):
        """Greedy agent stays put with zero gradient."""
        agent = GreedyAgent(role='A', step_size=1.0)

        agent.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": 0.0,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos == 5.0, "Zero gradient should keep position unchanged"

    def test_greedy_agent_respects_upper_bound(self):
        """Greedy agent clamps to upper domain bound."""
        agent = GreedyAgent(role='A', step_size=2.0)

        # Near upper bound with positive gradient
        agent.receive_observation({
            "position": {"x": 9.5, "y": 5.0},
            "value_at_position": 0.8,
            "gradient_x": 0.5,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos == 10.0, f"Position should be clamped to 10.0, got {new_pos}"

    def test_greedy_agent_respects_lower_bound(self):
        """Greedy agent clamps to lower domain bound."""
        agent = GreedyAgent(role='A', step_size=2.0)

        # Near lower bound with negative gradient
        agent.receive_observation({
            "position": {"x": 0.5, "y": 5.0},
            "value_at_position": 0.3,
            "gradient_x": -0.5,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos == 0.0, f"Position should be clamped to 0.0, got {new_pos}"

    def test_greedy_agent_role_b_uses_gradient_y(self):
        """Greedy agent B uses gradient_y, not gradient_x."""
        agent = GreedyAgent(role='B', step_size=1.0)

        agent.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": 1.0,  # Should be ignored for role B
            "gradient_y": -0.5,  # Should be used
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos == 4.0, f"Agent B should use gradient_y, expected 4.0, got {new_pos}"

    def test_greedy_agent_custom_step_size(self):
        """Greedy agent respects custom step size."""
        agent = GreedyAgent(role='A', step_size=0.5)

        agent.receive_observation({
            "position": {"x": 5.0, "y": 5.0},
            "value_at_position": 0.5,
            "gradient_x": 0.3,
            "slice": []
        })
        new_pos = agent.decide_position()
        assert new_pos == 5.5, f"Expected 5.5 with step_size=0.5, got {new_pos}"

    def test_greedy_agent_no_observation_returns_current(self):
        """Greedy agent returns current position with no observations."""
        agent = GreedyAgent(role='A', step_size=1.0)
        pos = agent.decide_position()
        assert pos == 5.0, "Should return initial position with no observations"

    def test_greedy_agent_no_communication(self):
        """Greedy agent generates empty messages (no communication)."""
        agent = GreedyAgent(role='A', step_size=1.0)
        message = agent.generate_message()
        assert message == "", "Greedy agent should not communicate"

    def test_greedy_agent_final_decision(self):
        """Greedy agent final_decision returns current position."""
        agent = GreedyAgent(role='A', step_size=1.0)

        agent.receive_observation({
            "position": {"x": 7.0, "y": 5.0},
            "gradient_x": 0.5,
            "slice": []
        })
        agent.decide_position()  # Updates current_position to 7.0, then moves

        final = agent.final_decision()
        # After decide_position with positive gradient from 7.0, current_position is used
        assert final == 7.0, f"Final decision should be current_position (7.0), got {final}"


# =============================================================================
# LLMAgent Tests (without API calls)
# =============================================================================

class TestLLMAgentParsing:
    """Tests for LLMAgent coordinate parsing logic."""

    @pytest.fixture
    def mock_llm_agent(self):
        """Create an LLMAgent with mocked API client."""
        with patch.object(LLMAgent, '_init_api_client'):
            with patch.object(LLMAgent, '_load_system_prompt'):
                agent = LLMAgent(role='A', model='gpt-4')
                agent.system_prompt = "You are a test agent."
                agent.api_type = 'openai'
                agent.client = MagicMock()
                return agent

    def test_coordinate_parsing_simple(self, mock_llm_agent):
        """Parse simple number: '7.5' -> 7.5"""
        result = mock_llm_agent._parse_coordinate("7.5")
        assert result == 7.5

    def test_coordinate_parsing_integer(self, mock_llm_agent):
        """Parse integer: '7' -> 7.0"""
        result = mock_llm_agent._parse_coordinate("7")
        assert result == 7.0

    def test_coordinate_parsing_sentence(self, mock_llm_agent):
        """Parse from sentence: 'My answer is 7.5' -> 7.5"""
        result = mock_llm_agent._parse_coordinate("My answer is 7.5")
        assert result == 7.5

    def test_coordinate_parsing_multiple_numbers(self, mock_llm_agent):
        """Parse with multiple numbers - takes LAST: 'gradient 0.12, move to 6.8' -> 6.8"""
        result = mock_llm_agent._parse_coordinate("gradient 0.12, move to 6.8")
        assert result == 6.8

    def test_coordinate_parsing_complex_response(self, mock_llm_agent):
        """Parse complex response with multiple numbers."""
        response = "The current value is 0.5, gradient is 0.12. I'll move to 7.2."
        result = mock_llm_agent._parse_coordinate(response)
        assert result == 7.2

    def test_coordinate_parsing_approximately(self, mock_llm_agent):
        """Parse 'approximately 7 to 8, say 7.5' -> 7.5"""
        result = mock_llm_agent._parse_coordinate("approximately 7 to 8, say 7.5")
        assert result == 7.5

    def test_coordinate_parsing_equals_format(self, mock_llm_agent):
        """Parse 'x=6.5' -> 6.5"""
        result = mock_llm_agent._parse_coordinate("x=6.5")
        assert result == 6.5

    def test_coordinate_clamping_high(self, mock_llm_agent):
        """Values above domain_size are clamped to domain_size."""
        result = mock_llm_agent._parse_coordinate("My answer is 15.5")
        assert result == 10.0

    def test_coordinate_clamping_low(self, mock_llm_agent):
        """Negative values are clamped to 0."""
        result = mock_llm_agent._parse_coordinate("My answer is -2.5")
        assert result == 0.0

    def test_coordinate_parsing_no_numbers_raises(self, mock_llm_agent):
        """Raise ValueError if no numbers found."""
        with pytest.raises(ValueError):
            mock_llm_agent._parse_coordinate("I don't know what position to choose")

    def test_coordinate_parsing_decimal_without_leading_zero(self, mock_llm_agent):
        """Parse decimal without leading zero: '.5' -> 0.5"""
        result = mock_llm_agent._parse_coordinate("about .5")
        assert result == 0.5

    def test_coordinate_parsing_negative_clamped(self, mock_llm_agent):
        """Negative coordinates are clamped to 0."""
        result = mock_llm_agent._parse_coordinate("-3.5")
        assert result == 0.0


class TestLLMAgentObservationFormatting:
    """Tests for LLMAgent observation formatting."""

    @pytest.fixture
    def mock_llm_agent_a(self):
        """Create an LLMAgent (role A) with mocked API client."""
        with patch.object(LLMAgent, '_init_api_client'):
            with patch.object(LLMAgent, '_load_system_prompt'):
                agent = LLMAgent(role='A', model='gpt-4')
                agent.system_prompt = "You are a test agent."
                agent.api_type = 'openai'
                agent.client = MagicMock()
                return agent

    @pytest.fixture
    def mock_llm_agent_b(self):
        """Create an LLMAgent (role B) with mocked API client."""
        with patch.object(LLMAgent, '_init_api_client'):
            with patch.object(LLMAgent, '_load_system_prompt'):
                agent = LLMAgent(role='B', model='gpt-4')
                agent.system_prompt = "You are a test agent."
                agent.api_type = 'openai'
                agent.client = MagicMock()
                return agent

    def test_observation_formatting_includes_position(self, mock_llm_agent_a):
        """Formatted observation includes position."""
        obs = {
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.12,
            "slice": []
        }
        formatted = mock_llm_agent_a._format_observation(obs)
        assert "5.00" in formatted
        assert "3.00" in formatted

    def test_observation_formatting_includes_value(self, mock_llm_agent_a):
        """Formatted observation includes surface value."""
        obs = {
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.567,
            "gradient_x": 0.12,
            "slice": []
        }
        formatted = mock_llm_agent_a._format_observation(obs)
        assert "0.567" in formatted

    def test_observation_formatting_agent_a_gradient(self, mock_llm_agent_a):
        """Agent A observation includes gradient_x."""
        obs = {
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.123,
            "slice": []
        }
        formatted = mock_llm_agent_a._format_observation(obs)
        assert "0.123" in formatted
        assert "x-direction" in formatted or "df/dx" in formatted

    def test_observation_formatting_agent_b_gradient(self, mock_llm_agent_b):
        """Agent B observation includes gradient_y."""
        obs = {
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_y": 0.456,
            "slice": []
        }
        formatted = mock_llm_agent_b._format_observation(obs)
        assert "0.456" in formatted
        assert "y-direction" in formatted or "df/dy" in formatted

    def test_observation_formatting_includes_slice(self, mock_llm_agent_a):
        """Formatted observation includes slice data."""
        obs = {
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.12,
            "slice": [
                {"x": 4.0, "value": 0.4},
                {"x": 5.0, "value": 0.5},
                {"x": 6.0, "value": 0.6}
            ]
        }
        formatted = mock_llm_agent_a._format_observation(obs)
        assert "4.00" in formatted
        assert "0.400" in formatted
        assert "x-slice" in formatted or "x=" in formatted


class TestLLMAgentPromptBuilding:
    """Tests for LLMAgent prompt building."""

    @pytest.fixture
    def mock_llm_agent(self):
        """Create an LLMAgent with mocked API client."""
        with patch.object(LLMAgent, '_init_api_client'):
            with patch.object(LLMAgent, '_load_system_prompt'):
                agent = LLMAgent(role='A', model='gpt-4')
                agent.system_prompt = "You are a test agent."
                agent.api_type = 'openai'
                agent.client = MagicMock()
                return agent

    def test_prompt_building_includes_system_message(self, mock_llm_agent):
        """Prompt includes system message for OpenAI."""
        messages = mock_llm_agent._build_prompt()
        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == "You are a test agent."

    def test_prompt_building_with_observations(self, mock_llm_agent):
        """Prompt includes observation history."""
        mock_llm_agent.receive_observation({
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.12,
            "slice": []
        })
        messages = mock_llm_agent._build_prompt()
        # Should have system + 1 observation
        assert len(messages) == 2
        assert "5.00" in messages[1]['content']

    def test_prompt_building_with_decision_request(self, mock_llm_agent):
        """Prompt includes decision request when specified."""
        mock_llm_agent.receive_observation({
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.12,
            "slice": []
        })
        messages = mock_llm_agent._build_prompt(include_decision=True)
        # Should have system + observation + decision request
        assert len(messages) == 3
        last_message = messages[-1]['content']
        assert "x-coordinate" in last_message
        assert "0" in last_message and "10" in last_message


class TestLLMAgentReset:
    """Tests for LLMAgent reset functionality."""

    @pytest.fixture
    def mock_llm_agent(self):
        """Create an LLMAgent with mocked API client."""
        with patch.object(LLMAgent, '_init_api_client'):
            with patch.object(LLMAgent, '_load_system_prompt'):
                agent = LLMAgent(role='A', model='gpt-4')
                agent.system_prompt = "You are a test agent."
                agent.api_type = 'openai'
                agent.client = MagicMock()
                return agent

    def test_reset_clears_histories(self, mock_llm_agent):
        """Reset clears observation and message histories."""
        mock_llm_agent.receive_observation({
            "position": {"x": 5.0, "y": 3.0},
            "value_at_position": 0.5,
            "gradient_x": 0.12,
            "slice": []
        })
        mock_llm_agent.receive_message("Test message")

        assert len(mock_llm_agent.observation_history) == 1
        assert len(mock_llm_agent.message_history) == 1

        mock_llm_agent.reset()

        assert len(mock_llm_agent.observation_history) == 0
        assert len(mock_llm_agent.message_history) == 0

    def test_reset_restores_position(self, mock_llm_agent):
        """Reset restores current position to center."""
        mock_llm_agent.current_position = 8.5
        mock_llm_agent.reset()
        assert mock_llm_agent.current_position == 5.0


class TestLLMAgentAPIDetection:
    """Tests for LLMAgent API type detection."""

    def test_openai_gpt4_detection(self):
        """Detect OpenAI API for gpt-4 model."""
        # Create mock openai module
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.object(LLMAgent, '_load_system_prompt'):
            with patch.dict('sys.modules', {'openai': mock_openai}):
                agent = LLMAgent(role='A', model='gpt-4', api_key='test-key')
                assert agent.api_type == 'openai'
                mock_openai.OpenAI.assert_called_once()

    def test_openai_gpt4o_detection(self):
        """Detect OpenAI API for gpt-4o model."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.object(LLMAgent, '_load_system_prompt'):
            with patch.dict('sys.modules', {'openai': mock_openai}):
                agent = LLMAgent(role='A', model='gpt-4o', api_key='test-key')
                assert agent.api_type == 'openai'

    def test_openai_o1_detection(self):
        """Detect OpenAI API for o1 model."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.object(LLMAgent, '_load_system_prompt'):
            with patch.dict('sys.modules', {'openai': mock_openai}):
                agent = LLMAgent(role='A', model='o1-preview', api_key='test-key')
                assert agent.api_type == 'openai'

    def test_anthropic_claude_detection(self):
        """Detect Anthropic API for claude model."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()

        with patch.object(LLMAgent, '_load_system_prompt'):
            with patch.dict('sys.modules', {'anthropic': mock_anthropic}):
                agent = LLMAgent(role='A', model='claude-3-opus-20240229', api_key='test-key')
                assert agent.api_type == 'anthropic'
                mock_anthropic.Anthropic.assert_called_once()

    def test_unknown_model_raises(self):
        """Raise ValueError for unknown model."""
        with patch.object(LLMAgent, '_load_system_prompt'):
            with pytest.raises(ValueError) as exc_info:
                LLMAgent(role='A', model='unknown-model', api_key='test-key')
            assert "Unknown model type" in str(exc_info.value)
