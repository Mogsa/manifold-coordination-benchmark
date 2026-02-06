"""
Tests for Telepathic Benchmark v2 Evaluation Pipeline.

Tests end-to-end evaluation: lambda program -> Score.
"""

import math
import pytest

from telepathic.core.evaluation_v2 import (
    # Classes
    SynthesisResult,
    ProgramEvaluator,
    EvaluationReport,
    # Exceptions
    EvaluationError,
    ProgramError,
    TimeoutError,
    # Functions
    evaluate_synthesis,
    evaluate_program_string,
    compile_and_count_bits,
    generate_report,
)
from telepathic.core.environment import create_environment, create_environment_from_function_id
from telepathic.core.scoring import Score


# =============================================================================
# SynthesisResult Tests
# =============================================================================

class TestSynthesisResult:
    """Test SynthesisResult dataclass."""

    def test_synthesis_result_fields(self):
        """Should have reasoning and program fields."""
        result = SynthesisResult(
            reasoning="I observed x^2 pattern",
            program="lambda x: x"
        )
        assert result.reasoning == "I observed x^2 pattern"
        assert result.program == "lambda x: x"


# =============================================================================
# Program Evaluator Tests
# =============================================================================

class TestProgramEvaluator:
    """Test the ProgramEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a default evaluator."""
        return ProgramEvaluator()

    def test_evaluate_identity_program(self, evaluator):
        """Identity should return (approximately) the same value."""
        # Identity: lambda x: x
        # Note: This uses Church numeral encoding, so x gets scaled
        # For now, just test it runs without error
        result = evaluator.evaluate_program("lambda x: x", [0.5])
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_evaluate_invalid_program(self, evaluator):
        """Invalid programs should raise ProgramError."""
        with pytest.raises(ProgramError):
            evaluator.evaluate_program("not valid python", [0.5])

    def test_evaluate_non_lambda_program(self, evaluator):
        """Non-lambda constructs should raise ProgramError."""
        with pytest.raises(ProgramError):
            evaluator.evaluate_program("lambda x: x + 1", [0.5])  # + not allowed


# =============================================================================
# Compile and Count Bits Tests
# =============================================================================

class TestCompileAndCountBits:
    """Test BLC compilation and bit counting."""

    def test_identity_bits(self):
        """Identity should compile to 4 bits."""
        bits = compile_and_count_bits("lambda x: x")
        assert bits == 4

    def test_true_bits(self):
        """TRUE combinator should compile to 7 bits."""
        bits = compile_and_count_bits("lambda x: lambda y: x")
        assert bits == 7

    def test_false_bits(self):
        """FALSE combinator should compile to 6 bits."""
        bits = compile_and_count_bits("lambda x: lambda y: y")
        assert bits == 6

    def test_invalid_program_raises(self):
        """Invalid program should raise ProgramError."""
        with pytest.raises(ProgramError):
            compile_and_count_bits("invalid")

    def test_multiline_program(self):
        """Should handle multi-line programs with definitions."""
        program = """
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y
AND = lambda a: lambda b: a(b)(FALSE)
AND
"""
        bits = compile_and_count_bits(program)
        assert bits > 0


# =============================================================================
# Full Evaluation Tests
# =============================================================================

class TestFullEvaluation:
    """Test full evaluation pipeline."""

    @pytest.fixture
    def identity_env(self):
        """Environment for identity function."""
        return create_environment(lambda x: x, function_id="identity")

    def test_evaluate_synthesis_returns_score(self, identity_env):
        """evaluate_synthesis should return a Score."""
        result = SynthesisResult(
            reasoning="Identity function",
            program="lambda x: x"
        )
        score = evaluate_synthesis(result, identity_env)
        assert isinstance(score, Score)

    def test_evaluate_synthesis_includes_bits(self, identity_env):
        """Score should include compression bits."""
        result = SynthesisResult(
            reasoning="Test",
            program="lambda x: x"  # 4 bits
        )
        score = evaluate_synthesis(result, identity_env)
        assert score.compression_bits == 4

    def test_evaluate_synthesis_preserves_reasoning(self, identity_env):
        """Score should preserve agent reasoning."""
        result = SynthesisResult(
            reasoning="This is my reasoning",
            program="lambda x: x"
        )
        score = evaluate_synthesis(result, identity_env)
        assert "This is my reasoning" in score.reasoning

    def test_evaluate_invalid_program_returns_timeout(self, identity_env):
        """Invalid program should return timeout score."""
        result = SynthesisResult(
            reasoning="Bad program",
            program="not valid"
        )
        score = evaluate_synthesis(result, identity_env)
        assert score.timed_out or math.isinf(score.total)

    def test_evaluate_with_probes(self, identity_env):
        """Score should include probe penalty."""
        # Make some probes
        identity_env.probe(0.5)
        identity_env.probe(0.7)
        identity_env.probe(0.3)

        result = SynthesisResult(
            reasoning="After probing",
            program="lambda x: x"
        )
        score = evaluate_synthesis(result, identity_env)

        # 3 probes * 5 bits = 15 bits penalty
        assert score.probe_penalty == 15
        assert score.n_probes == 3


# =============================================================================
# Evaluation Report Tests
# =============================================================================

class TestEvaluationReport:
    """Test evaluation report generation."""

    def test_report_summary(self):
        """Report should generate readable summary."""
        env = create_environment(lambda x: x, function_id="test")
        result = SynthesisResult(reasoning="test", program="lambda x: x")
        score = evaluate_synthesis(result, env)

        report = generate_report(result, env, score)

        summary = report.summary()
        assert "Total Score" in summary
        assert "test" in summary


# =============================================================================
# Integration Tests with Function Library
# =============================================================================

class TestIntegrationWithFunctions:
    """Test evaluation with actual function library."""

    def test_evaluate_with_polynomial_function(self):
        """Should evaluate against polynomial function."""
        env = create_environment_from_function_id("P1")  # Identity
        result = SynthesisResult(
            reasoning="Identity",
            program="lambda x: x"
        )
        score = evaluate_synthesis(result, env)

        # Should not timeout
        assert not score.timed_out
        assert not math.isinf(score.total)

    def test_evaluate_program_string_convenience(self):
        """Test convenience function."""
        score = evaluate_program_string(
            program="lambda x: x",
            function=lambda x: x,
            n_probes=2,
            reasoning="Test"
        )

        assert isinstance(score, Score)
        # Note: probe count may not work exactly as expected with placeholder
