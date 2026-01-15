"""
Tests for Telepathic benchmark agents.

Tests random baselines, oracle agents, and parsing logic.
LLM integration tests are marked with pytest.mark.slow.
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from telepathic.agents import (
    SeerInput,
    SeerOutput,
    DoerInput,
    DoerOutput,
    RandomSeerAgent,
    RandomDoerAgent,
    OracleSeerAgent,
    OracleDoerAgent,
    LLMSeerAgent,
    LLMDoerAgent,
)
from telepathic.core.protocol import MVP_VOCABULARY


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_seer_input():
    """Sample input for Seer agents."""
    return SeerInput(
        samples=[
            (-2.0, 4.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 4.0),
        ],
        vocabulary=MVP_VOCABULARY,
        max_tokens=5
    )


@pytest.fixture
def sample_doer_input():
    """Sample input for Doer agents."""
    return DoerInput(
        message="γ",
        test_input=1.5,
        vocabulary=MVP_VOCABULARY
    )


# =============================================================================
# Random Agent Tests
# =============================================================================

class TestRandomSeerAgent:
    """Tests for RandomSeerAgent."""

    def test_generates_valid_message(self, sample_seer_input):
        """Random Seer generates messages from vocabulary."""
        agent = RandomSeerAgent(seed=42)
        output = agent.generate_message(sample_seer_input)

        assert isinstance(output, SeerOutput)
        assert output.message is not None

        # Check all tokens are from vocabulary
        tokens = output.message.split("-")
        for token in tokens:
            assert token in MVP_VOCABULARY

    def test_reproducible_with_seed(self, sample_seer_input):
        """Same seed produces same output."""
        agent1 = RandomSeerAgent(seed=42)
        agent2 = RandomSeerAgent(seed=42)

        out1 = agent1.generate_message(sample_seer_input)
        out2 = agent2.generate_message(sample_seer_input)

        assert out1.message == out2.message

    def test_respects_max_tokens(self, sample_seer_input):
        """Message doesn't exceed max tokens."""
        agent = RandomSeerAgent(seed=42, max_composition_depth=3)

        for _ in range(10):
            output = agent.generate_message(sample_seer_input)
            tokens = output.message.split("-")
            assert len(tokens) <= 3

    def test_reset_restores_state(self, sample_seer_input):
        """Reset restores random state."""
        agent = RandomSeerAgent(seed=42)

        out1 = agent.generate_message(sample_seer_input)
        out2 = agent.generate_message(sample_seer_input)

        # After reset, should get same sequence
        agent.reset()
        out3 = agent.generate_message(sample_seer_input)

        assert out1.message == out3.message

    def test_get_config(self):
        """Config returns correct info."""
        agent = RandomSeerAgent(seed=123, max_composition_depth=4)
        config = agent.get_config()

        assert config["type"] == "RandomSeerAgent"
        assert config["seed"] == 123
        assert config["max_composition_depth"] == 4


class TestRandomDoerAgent:
    """Tests for RandomDoerAgent."""

    def test_generates_prediction(self, sample_doer_input):
        """Random Doer generates numerical prediction."""
        agent = RandomDoerAgent(seed=42)
        output = agent.generate_prediction(sample_doer_input)

        assert isinstance(output, DoerOutput)
        assert isinstance(output.prediction, float)

    def test_respects_output_range(self, sample_doer_input):
        """Predictions stay within range."""
        agent = RandomDoerAgent(seed=42, output_range=(-5.0, 5.0))

        for _ in range(20):
            output = agent.generate_prediction(sample_doer_input)
            assert -5.0 <= output.prediction <= 5.0

    def test_reproducible_with_seed(self, sample_doer_input):
        """Same seed produces same output."""
        agent1 = RandomDoerAgent(seed=42)
        agent2 = RandomDoerAgent(seed=42)

        out1 = agent1.generate_prediction(sample_doer_input)
        out2 = agent2.generate_prediction(sample_doer_input)

        assert out1.prediction == out2.prediction


# =============================================================================
# Oracle Agent Tests
# =============================================================================

class TestOracleSeerAgent:
    """Tests for OracleSeerAgent."""

    def test_returns_correct_message(self, sample_seer_input):
        """Oracle returns the message it was given."""
        agent = OracleSeerAgent(correct_message="α-γ")
        output = agent.generate_message(sample_seer_input)

        assert output.message == "α-γ"
        assert output.parse_error is False

    def test_get_config(self):
        """Config includes correct message."""
        agent = OracleSeerAgent(correct_message="β")
        config = agent.get_config()

        assert config["type"] == "OracleSeerAgent"
        assert config["correct_message"] == "β"


class TestOracleDoerAgent:
    """Tests for OracleDoerAgent."""

    def test_computes_correct_output(self, sample_doer_input):
        """Oracle computes correct function output."""
        # Square function
        agent = OracleDoerAgent(compute_func=lambda x: x ** 2)
        output = agent.generate_prediction(sample_doer_input)

        expected = 1.5 ** 2  # 2.25
        assert abs(output.prediction - expected) < 0.0001

    def test_handles_different_inputs(self):
        """Oracle works with various inputs."""
        agent = OracleDoerAgent(compute_func=math.sin)

        for x in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            input = DoerInput(message="α", test_input=x, vocabulary=MVP_VOCABULARY)
            output = agent.generate_prediction(input)
            assert abs(output.prediction - math.sin(x)) < 0.0001


# =============================================================================
# LLM Agent Parsing Tests (No actual LLM calls)
# =============================================================================

class TestLLMSeerParsing:
    """Test LLMSeerAgent parsing logic without making LLM calls."""

    def test_parse_simple_message(self):
        """Parses simple MESSAGE: token format."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)
        agent.protocol = MagicMock()
        agent.protocol.vocabulary = MVP_VOCABULARY

        # Mock validate_message to return True for valid tokens
        with patch('telepathic.agents.seer.validate_message') as mock_validate:
            mock_validate.return_value = True

            result = agent._parse_message("I think this is sin.\nMESSAGE: α")
            assert result == "α"

    def test_parse_composition_message(self):
        """Parses composition MESSAGE: token-token format."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)
        agent.protocol = MagicMock()

        with patch('telepathic.agents.seer.validate_message') as mock_validate:
            mock_validate.return_value = True

            result = agent._parse_message("MESSAGE: α-γ")
            assert result == "α-γ"

    def test_parse_strips_punctuation(self):
        """Removes trailing punctuation."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)

        with patch('telepathic.agents.seer.validate_message') as mock_validate:
            mock_validate.return_value = True

            result = agent._parse_message("MESSAGE: β.")
            assert result == "β"

    def test_parse_handles_quotes(self):
        """Handles quoted messages."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)

        with patch('telepathic.agents.seer.validate_message') as mock_validate:
            mock_validate.return_value = True

            result = agent._parse_message('MESSAGE: "γ"')
            assert result == "γ"

    def test_parse_returns_none_for_invalid(self):
        """Returns None when no valid message found."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)

        with patch('telepathic.agents.seer.validate_message') as mock_validate:
            mock_validate.return_value = False

            result = agent._parse_message("I don't know what to say")
            assert result is None

    def test_parse_salvages_greek_letters(self):
        """Extracts Greek letters from malformed messages."""
        agent = LLMSeerAgent.__new__(LLMSeerAgent)

        def mock_validate(msg):
            # Only accept clean messages
            return msg in ["α", "β", "γ", "α-γ", "β-δ"]

        with patch('telepathic.agents.seer.validate_message', side_effect=mock_validate):
            # MESSAGE has extra stuff but contains valid tokens
            result = agent._parse_message("MESSAGE: α then γ together")
            assert result == "α-γ"


class TestLLMDoerParsing:
    """Test LLMDoerAgent parsing logic without making LLM calls."""

    def test_parse_simple_prediction(self):
        """Parses PREDICTION: number format."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("PREDICTION: 0.4794")
        assert abs(result - 0.4794) < 0.0001

    def test_parse_negative_prediction(self):
        """Parses negative numbers."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("PREDICTION: -1.5")
        assert abs(result - (-1.5)) < 0.0001

    def test_parse_integer_prediction(self):
        """Parses integer predictions."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("PREDICTION: 4")
        assert result == 4.0

    def test_parse_with_trailing_text(self):
        """Handles text after number."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("PREDICTION: 2.25 (approximately)")
        assert abs(result - 2.25) < 0.0001

    def test_parse_fallback_to_equals(self):
        """Falls back to finding = pattern."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("The answer = 3.14")
        assert abs(result - 3.14) < 0.0001

    def test_parse_fallback_to_last_number(self):
        """Falls back to last number in response."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("First I compute 0.5 squared = 0.25, so the answer is 0.25")
        assert abs(result - 0.25) < 0.0001

    def test_parse_returns_none_for_no_numbers(self):
        """Returns None when no numbers found."""
        agent = LLMDoerAgent.__new__(LLMDoerAgent)

        result = agent._parse_prediction("I don't know")
        assert result is None


# =============================================================================
# Integration Tests (Require LLM - marked slow)
# =============================================================================

@pytest.mark.slow
class TestLLMSeerIntegration:
    """Integration tests for LLMSeerAgent (requires API key)."""

    def test_generates_message_for_primitive(self, sample_seer_input):
        """LLM Seer can identify a primitive function."""
        agent = LLMSeerAgent(model="gemini/gemini-2.5-flash", verbose=True)
        output = agent.generate_message(sample_seer_input)

        assert output.message is not None
        # Should recognize this as γ (square)
        print(f"Seer message: {output.message}")


@pytest.mark.slow
class TestLLMDoerIntegration:
    """Integration tests for LLMDoerAgent (requires API key)."""

    def test_generates_prediction_for_primitive(self, sample_doer_input):
        """LLM Doer can compute primitive output."""
        agent = LLMDoerAgent(model="gemini/gemini-2.5-flash", verbose=True)
        output = agent.generate_prediction(sample_doer_input)

        assert output.prediction is not None
        # γ at x=1.5 should be 2.25
        print(f"Doer prediction: {output.prediction}")
