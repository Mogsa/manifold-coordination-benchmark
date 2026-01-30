"""Integration test for full metacognitive session."""
import pytest
from unittest.mock import patch

from chaosbench.experiments.session import SessionRunner, SessionConfig
from chaosbench.agents.metacognitive_agent import MetacognitiveAgent


@pytest.mark.integration
class TestFullSession:
    @patch('chaosbench.agents.metacognitive_agent.call_llm')
    def test_full_session_with_mock_llm(self, mock_llm):
        """Run a full session with mocked LLM responses."""
        # Simulate agent behavior: predict, learn, move on
        responses = [
            'Analyzing pattern...\n{"action": "PREDICT", "value": 0.5}',
            'Score was low, let me try again\n{"action": "PREDICT", "value": 0.3}',
            'Recording insight\n{"action": "WRITE", "text": "## Logistic Maps\\nThey oscillate"}',
            'Moving on\n{"action": "MOVE_ON"}',
        ] * 10  # Repeat for multiple tasks

        mock_llm.side_effect = responses

        agent = MetacognitiveAgent(model="gemini/gemini-2.0-flash")
        config = SessionConfig(n_tasks=3, timeout_seconds=60)
        runner = SessionRunner(config)

        result = runner.run(agent)

        assert result.tasks_completed == 3
        assert result.final_phi > 0
        assert "Logistic Maps" in result.final_learnings
        assert len(result.trace.tasks) == 3
