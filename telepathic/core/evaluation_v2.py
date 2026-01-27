"""
Evaluation Pipeline for Telepathic Benchmark v2.

End-to-end evaluation: lambda program → Score

Integrates:
- Lambda parser (Phase 1)
- BLC compiler (Phase 1)
- BLC interpreter (Phase 1)
- Environment (Phase 2)
- Scoring (Phase 3)

Checkpoint 3.6: Result reporting and export
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Callable

from .lambda_parser import (
    parse,
    parse_with_definitions,
    to_debruijn,
    ParseError,
    ConversionError,
)
from .blc_compiler import compile_to_blc, CompileError
from .blc_interpreter import (
    evaluate,
    evaluate_church_application,
    int_to_church,
    church_to_int,
    TimeoutError as InterpreterTimeoutError,
    DEFAULT_TIMEOUT,
)
from .scoring import Score, compute_score, SCALE
from .environment import Environment


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_REDUCTIONS = DEFAULT_TIMEOUT  # 10M beta reductions


# =============================================================================
# Synthesis Result (From spec Section 9.2)
# =============================================================================

@dataclass
class SynthesisResult:
    """
    Result from agent synthesis phase.

    Contains both the program and reasoning for research analysis.
    """
    reasoning: str  # Agent's thought process (for dissertation)
    program: str  # Pure λ-calculus program


# =============================================================================
# Evaluation Errors
# =============================================================================

class EvaluationError(Exception):
    """Base class for evaluation errors."""
    pass


class ProgramError(EvaluationError):
    """Error in the program (parse, compile, or runtime)."""
    pass


class TimeoutError(EvaluationError):
    """Program execution exceeded timeout."""
    pass


# =============================================================================
# Program Evaluator
# =============================================================================

class ProgramEvaluator:
    """
    Evaluates lambda calculus programs on numerical inputs.

    Uses fixed-point arithmetic with SCALE=1000.
    """

    def __init__(
        self,
        scale: int = SCALE,
        max_reductions: int = DEFAULT_MAX_REDUCTIONS
    ):
        """
        Initialize evaluator.

        Args:
            scale: Fixed-point scaling factor.
            max_reductions: Maximum beta reductions before timeout.
        """
        self.scale = scale
        self.max_reductions = max_reductions

    def evaluate_program(
        self,
        program_source: str,
        inputs: List[float]
    ) -> List[float]:
        """
        Evaluate a lambda program on a list of inputs.

        Pipeline:
        1. Parse lambda program
        2. Convert to De Bruijn
        3. For each input:
           a. Scale to integer
           b. Encode as Church numeral
           c. Apply program
           d. Decode result
           e. Unscale to float

        Args:
            program_source: Lambda calculus source code.
            inputs: List of x values in (0, 1].

        Returns:
            List of output values.

        Raises:
            ProgramError: If parsing or compilation fails.
            TimeoutError: If execution exceeds limit.
        """
        # Parse the program
        try:
            if '\n' in program_source or '=' in program_source:
                # Multi-line program with definitions
                named_ast = parse_with_definitions(program_source)
            else:
                # Single expression
                named_ast = parse(program_source)
        except ParseError as e:
            raise ProgramError(f"Parse error: {e}")

        # Convert to De Bruijn indices
        try:
            db_term = to_debruijn(named_ast)
        except ConversionError as e:
            raise ProgramError(f"Conversion error: {e}")

        # Evaluate on each input
        outputs = []
        for x in inputs:
            y = self._evaluate_single(db_term, x)
            outputs.append(y)

        return outputs

    def _evaluate_single(self, program, x: float) -> float:
        """
        Evaluate program on a single input using Church numerals.

        Args:
            program: De Bruijn term (the function).
            x: Input value in (0, 1].

        Returns:
            Output value.

        Raises:
            TimeoutError: If execution exceeds limit.
        """
        # Scale input to integer
        x_int = round(x * self.scale)

        try:
            # Apply program to Church-encoded input
            result_int = evaluate_church_application(
                program, x_int, self.max_reductions
            )
        except InterpreterTimeoutError:
            raise TimeoutError(f"Execution exceeded {self.max_reductions} reductions")

        if result_int is None:
            # Result wasn't a valid Church numeral
            # This could happen with non-standard encodings
            raise ProgramError("Result is not a valid Church numeral")

        # Unscale to float
        return result_int / self.scale


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_synthesis(
    result: SynthesisResult,
    environment: Environment,
    scale: int = SCALE,
    max_reductions: int = DEFAULT_MAX_REDUCTIONS
) -> Score:
    """
    Evaluate a synthesis result and compute its score.

    Complete pipeline:
    1. Parse lambda program → AST
    2. Convert to De Bruijn indices
    3. Compile to BLC
    4. Count bits
    5. Evaluate on test points
    6. Compute error penalty
    7. Return Score

    Args:
        result: Agent's synthesis result (reasoning + program).
        environment: The benchmark environment with test points.
        scale: Fixed-point scaling factor.
        max_reductions: Maximum beta reductions.

    Returns:
        Score with all components.
    """
    # Extract probe count before evaluation
    n_probes = environment.get_probe_count()

    # Parse the program
    try:
        if '\n' in result.program or '=' in result.program:
            named_ast = parse_with_definitions(result.program)
        else:
            named_ast = parse(result.program)
    except ParseError as e:
        return Score.timeout_score(
            reasoning=result.reasoning,
            n_probes=n_probes
        )

    # Convert to De Bruijn
    try:
        db_term = to_debruijn(named_ast)
    except ConversionError as e:
        return Score.timeout_score(
            reasoning=result.reasoning,
            n_probes=n_probes
        )

    # Compile to BLC
    try:
        blc = compile_to_blc(db_term)
    except CompileError as e:
        return Score.timeout_score(
            reasoning=result.reasoning,
            n_probes=n_probes
        )

    # Get test points
    test_points = environment.get_test_points()

    # Evaluate program on test points
    evaluator = ProgramEvaluator(scale=scale, max_reductions=max_reductions)
    try:
        predictions = evaluator.evaluate_program(result.program, test_points)
    except (ProgramError, TimeoutError):
        return Score.timeout_score(
            reasoning=result.reasoning,
            n_probes=n_probes
        )

    # Compute errors
    errors = environment.evaluate(predictions)

    # Compute and return score
    return compute_score(
        blc_program=blc,
        errors=errors,
        n_probes=n_probes,
        reasoning=result.reasoning,
        program=result.program
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_program_string(
    program: str,
    function: Callable[[float], float],
    n_probes: int = 0,
    reasoning: str = ""
) -> Score:
    """
    Quick evaluation of a program against a function.

    Useful for testing without full environment setup.

    Args:
        program: Lambda calculus source.
        function: Ground truth function.
        n_probes: Number of probes (for scoring).
        reasoning: Agent reasoning (for logging).

    Returns:
        Score.
    """
    from .environment import create_environment

    env = create_environment(function)
    # Simulate probes by adding to count
    for _ in range(n_probes):
        env.probes.append(None)  # Placeholder

    result = SynthesisResult(reasoning=reasoning, program=program)
    return evaluate_synthesis(result, env)


def compile_and_count_bits(program: str) -> int:
    """
    Compile a program to BLC and count bits.

    Args:
        program: Lambda calculus source.

    Returns:
        Number of bits in BLC encoding.

    Raises:
        ProgramError: If parsing or compilation fails.
    """
    try:
        if '\n' in program or '=' in program:
            named_ast = parse_with_definitions(program)
        else:
            named_ast = parse(program)
        db_term = to_debruijn(named_ast)
        blc = compile_to_blc(db_term)
        return len(blc)
    except (ParseError, ConversionError, CompileError) as e:
        raise ProgramError(f"Compilation error: {e}")


# =============================================================================
# Result Analysis
# =============================================================================

@dataclass
class EvaluationReport:
    """Detailed report from evaluation."""

    score: Score
    function_id: Optional[str]
    program_bits: int
    predictions: List[float]
    ground_truth: List[float]
    test_points: List[float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Evaluation Report for {self.function_id or 'unknown function'}")
        lines.append("=" * 50)
        lines.append(f"Total Score: {self.score.total:.2f} bits")
        lines.append(f"  - Program:  {self.score.compression_bits} bits")
        lines.append(f"  - Error:    {self.score.error_penalty:.2f} bits")
        lines.append(f"  - Probes:   {self.score.probe_penalty} bits")
        lines.append("")

        if self.score.errors:
            avg_err = sum(self.score.errors) / len(self.score.errors)
            max_err = max(self.score.errors)
            lines.append(f"Error Statistics:")
            lines.append(f"  - Average:  {avg_err:.4f}")
            lines.append(f"  - Maximum:  {max_err:.4f}")

        return "\n".join(lines)


def generate_report(
    result: SynthesisResult,
    environment: Environment,
    score: Score
) -> EvaluationReport:
    """
    Generate a detailed evaluation report.

    Args:
        result: Agent's synthesis result.
        environment: Benchmark environment.
        score: Computed score.

    Returns:
        EvaluationReport with full details.
    """
    test_points = environment.get_test_points()
    ground_truth = [environment.function(x) for x in test_points]

    # Re-evaluate to get predictions (may timeout)
    try:
        evaluator = ProgramEvaluator()
        predictions = evaluator.evaluate_program(result.program, test_points)
    except (ProgramError, TimeoutError):
        predictions = [float('nan')] * len(test_points)

    return EvaluationReport(
        score=score,
        function_id=environment.function_id,
        program_bits=score.compression_bits,
        predictions=predictions,
        ground_truth=ground_truth,
        test_points=test_points,
    )
