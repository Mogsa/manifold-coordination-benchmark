"""Manages persistent learnings for metacognitive agent."""
import re


class LearningsManager:
    """Manages the agent's persistent learnings notepad."""

    HEADER = "# My Learnings\n"

    def __init__(self):
        self._content = self.HEADER

    @property
    def content(self) -> str:
        return self._content

    def write(self, text: str) -> None:
        """Append text to learnings."""
        self._content += f"\n{text}\n"

    def delete(self, section: str) -> None:
        """Delete a section by its header.

        Removes from the header to the next header of same or higher level,
        or to end of content.
        """
        # Escape regex special chars in section header
        escaped = re.escape(section)
        # Match section header to next same-or-higher level header or end
        pattern = rf'{escaped}.*?(?=\n#{{1,2}} |\Z)'
        self._content = re.sub(pattern, '', self._content, flags=re.DOTALL)

    def reset(self) -> None:
        """Reset to initial state."""
        self._content = self.HEADER

    def token_count(self) -> int:
        """Rough token estimate (chars / 4)."""
        return len(self._content) // 4
