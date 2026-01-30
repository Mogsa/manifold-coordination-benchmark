"""Tests for learnings manager."""
import pytest
from chaosbench.agents.learnings import LearningsManager


class TestLearningsManager:
    def test_initial_state_empty(self):
        lm = LearningsManager()
        assert lm.content == "# My Learnings\n"

    def test_write_appends(self):
        lm = LearningsManager()
        lm.write("First insight")
        assert "First insight" in lm.content
        lm.write("Second insight")
        assert "First insight" in lm.content
        assert "Second insight" in lm.content

    def test_delete_section(self):
        lm = LearningsManager()
        lm.write("## Section A\nContent A")
        lm.write("## Section B\nContent B")
        assert "Section A" in lm.content
        assert "Section B" in lm.content

        lm.delete("## Section A")
        assert "Section A" not in lm.content
        assert "Section B" in lm.content

    def test_delete_nonexistent_section_no_error(self):
        lm = LearningsManager()
        lm.write("Some content")
        lm.delete("## Nonexistent")  # Should not raise
        assert "Some content" in lm.content

    def test_reset(self):
        lm = LearningsManager()
        lm.write("Some content")
        lm.reset()
        assert lm.content == "# My Learnings\n"

    def test_token_count(self):
        lm = LearningsManager()
        assert lm.token_count() > 0  # Header has tokens
        lm.write("A" * 1000)
        assert lm.token_count() > 100  # Rough estimate
