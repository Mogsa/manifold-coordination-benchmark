"""Concrete LLM agent for ChaosBench v2.

Wraps shared.llm_utils.call_llm with ChaosBench prompt formatting and parsing.
"""

from __future__ import annotations

from chaosbench.agents.parsing import parse_confidence, parse_response
from chaosbench.agents.prompts import format_problem_prompt, format_system_prompt
from chaosbench.agents.protocol import Solution, TaskResult
from chaosbench.problems.factory import Problem
from shared.llm_utils import call_llm


class LLMAgent:
    """Agent that sends problems to an LLM and parses responses."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def agent_id(self) -> str:
        return self.model

    def solve(
        self, problem: Problem, task_history: list[TaskResult] | None = None
    ) -> Solution:
        messages = [
            {"role": "system", "content": format_system_prompt()},
            {"role": "user", "content": format_problem_prompt(problem, task_history)},
        ]
        response = call_llm(
            self.model,
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = parse_response(
            response, problem.question_type.value, problem.question_params
        )
        confidence = parse_confidence(response)
        return Solution(
            answer=answer, confidence=confidence, reasoning_trace=response
        )
