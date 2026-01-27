"""
Binary Lambda Calculus (BLC) Interpreter for Telepathic Benchmark v2.

Executes lambda calculus terms with beta reduction and timeout protection.

Checkpoint 1.4: BLC interpreter with timeout

Execution model (from spec Section 6.7):
    Reduction: Normal order (leftmost-outermost)
    Timeout:   10,000,000 beta reductions
    Result:    Final lambda term or TIMEOUT_ERROR
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from .lambda_parser import DBTerm, DBVar, DBAbs, DBApp


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT = 10_000_000  # Max beta reductions before timeout


# =============================================================================
# Exceptions
# =============================================================================

class InterpreterError(Exception):
    """Base class for interpreter errors."""
    pass


class TimeoutError(InterpreterError):
    """Raised when execution exceeds the beta reduction limit."""
    pass


class ReductionError(InterpreterError):
    """Raised when reduction fails (e.g., malformed term)."""
    pass


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing a lambda term."""
    term: DBTerm
    reductions: int
    timed_out: bool = False

    def is_normal_form(self) -> bool:
        """Check if the result is in normal form (no more reductions possible)."""
        return not self.timed_out


# =============================================================================
# Substitution (Core of Beta Reduction)
# =============================================================================

def shift(term: DBTerm, d: int, c: int = 0) -> DBTerm:
    """
    Shift free variables in a term.

    When substituting under a binder, we need to increment free variable
    indices to avoid capture. This implements the shifting operation.

    Args:
        term: The term to shift.
        d: The amount to shift (can be negative for unshifting).
        c: The cutoff - variables with index >= c are shifted.

    Returns:
        DBTerm: The shifted term.
    """
    if isinstance(term, DBVar):
        if term.index >= c:
            return DBVar(term.index + d)
        return term

    elif isinstance(term, DBAbs):
        # Under a binder, increase cutoff
        return DBAbs(shift(term.body, d, c + 1))

    elif isinstance(term, DBApp):
        return DBApp(shift(term.func, d, c), shift(term.arg, d, c))

    else:
        raise ReductionError(f"Unknown term type: {type(term)}")


def substitute(term: DBTerm, j: int, replacement: DBTerm) -> DBTerm:
    """
    Substitute replacement for variable j in term.

    [j → replacement]term

    Args:
        term: The term to substitute in.
        j: The variable index to replace.
        replacement: The term to substitute.

    Returns:
        DBTerm: The result of substitution.
    """
    if isinstance(term, DBVar):
        if term.index == j:
            return replacement
        return term

    elif isinstance(term, DBAbs):
        # When going under a binder:
        # 1. Increment j (the target variable is now one level deeper)
        # 2. Shift free variables in replacement up by 1
        shifted_replacement = shift(replacement, 1, 0)
        return DBAbs(substitute(term.body, j + 1, shifted_replacement))

    elif isinstance(term, DBApp):
        return DBApp(
            substitute(term.func, j, replacement),
            substitute(term.arg, j, replacement)
        )

    else:
        raise ReductionError(f"Unknown term type: {type(term)}")


# =============================================================================
# Beta Reduction
# =============================================================================

def is_value(term: DBTerm) -> bool:
    """Check if a term is a value (lambda abstraction)."""
    return isinstance(term, DBAbs)


def beta_reduce_once(term: DBTerm) -> tuple[DBTerm, bool]:
    """
    Perform one step of normal-order beta reduction.

    Normal order: reduce the leftmost-outermost redex first.

    Args:
        term: The term to reduce.

    Returns:
        tuple[DBTerm, bool]: (reduced term, whether a reduction occurred)
    """
    if isinstance(term, DBVar):
        # Variables can't be reduced
        return term, False

    elif isinstance(term, DBAbs):
        # Reduce inside the body
        new_body, reduced = beta_reduce_once(term.body)
        if reduced:
            return DBAbs(new_body), True
        return term, False

    elif isinstance(term, DBApp):
        # Check if this is a redex: (λ.M) N
        if isinstance(term.func, DBAbs):
            # Beta reduction: (λ.M) N → M[0 := N] then shift down
            body = term.func.body
            arg = term.arg
            # Substitute arg for variable 0 in body
            result = substitute(body, 0, shift(arg, 1, 0))
            # Shift down to account for removed binder
            result = shift(result, -1, 0)
            return result, True

        # Try reducing the function first (normal order)
        new_func, reduced = beta_reduce_once(term.func)
        if reduced:
            return DBApp(new_func, term.arg), True

        # Then try reducing the argument
        new_arg, reduced = beta_reduce_once(term.arg)
        if reduced:
            return DBApp(term.func, new_arg), True

        return term, False

    else:
        raise ReductionError(f"Unknown term type: {type(term)}")


def reduce_to_normal_form(
    term: DBTerm,
    max_reductions: int = DEFAULT_TIMEOUT
) -> ExecutionResult:
    """
    Reduce a term to normal form using normal-order reduction.

    Args:
        term: The term to reduce.
        max_reductions: Maximum number of beta reductions allowed.

    Returns:
        ExecutionResult: The result with final term and statistics.

    Raises:
        TimeoutError: If max_reductions is exceeded.
    """
    current = term
    reductions = 0

    while reductions < max_reductions:
        new_term, reduced = beta_reduce_once(current)
        if not reduced:
            # Normal form reached
            return ExecutionResult(
                term=current,
                reductions=reductions,
                timed_out=False
            )
        current = new_term
        reductions += 1

    # Timeout
    return ExecutionResult(
        term=current,
        reductions=reductions,
        timed_out=True
    )


# =============================================================================
# Church Numeral Utilities
# =============================================================================

def int_to_church(n: int) -> DBTerm:
    """
    Convert an integer to a Church numeral.

    Church numeral n = λf.λx. f^n(x)
    - 0 = λf.λx. x
    - 1 = λf.λx. f(x)
    - 2 = λf.λx. f(f(x))

    Args:
        n: Non-negative integer.

    Returns:
        DBTerm: Church numeral as De Bruijn term.
    """
    if n < 0:
        raise ValueError("Church numerals must be non-negative")

    # Build f^n(x) where f is index 1, x is index 0
    body: DBTerm = DBVar(0)  # Start with x
    for _ in range(n):
        body = DBApp(DBVar(1), body)  # Apply f

    # Wrap in λf.λx.
    return DBAbs(DBAbs(body))


def church_to_int(term: DBTerm) -> int | None:
    """
    Convert a Church numeral back to an integer.

    Returns None if the term is not a valid Church numeral.

    Args:
        term: A lambda term (possibly a Church numeral).

    Returns:
        int | None: The integer value, or None if not a Church numeral.
    """
    # Church numeral has form λf.λx. f^n(x)
    if not isinstance(term, DBAbs):
        return None
    if not isinstance(term.body, DBAbs):
        return None

    inner = term.body.body

    # Count applications of f (index 1) to x (index 0)
    count = 0
    while isinstance(inner, DBApp):
        if not isinstance(inner.func, DBVar) or inner.func.index != 1:
            return None
        inner = inner.arg
        count += 1

    # Should end with x (index 0)
    if not isinstance(inner, DBVar) or inner.index != 0:
        return None

    return count


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate(term: DBTerm, max_reductions: int = DEFAULT_TIMEOUT) -> DBTerm:
    """
    Evaluate a term to normal form, raising TimeoutError if it takes too long.

    Args:
        term: The term to evaluate.
        max_reductions: Maximum beta reductions.

    Returns:
        DBTerm: The normal form.

    Raises:
        TimeoutError: If evaluation times out.
    """
    result = reduce_to_normal_form(term, max_reductions)
    if result.timed_out:
        raise TimeoutError(
            f"Evaluation exceeded {max_reductions} reductions"
        )
    return result.term


def evaluate_church_application(
    func: DBTerm,
    arg_int: int,
    max_reductions: int = DEFAULT_TIMEOUT
) -> int | None:
    """
    Apply a function to a Church numeral and interpret result as integer.

    Args:
        func: The function to apply.
        arg_int: The argument as an integer (converted to Church numeral).
        max_reductions: Maximum beta reductions.

    Returns:
        int | None: The result as integer, or None if not a Church numeral.

    Raises:
        TimeoutError: If evaluation times out.
    """
    arg = int_to_church(arg_int)
    application = DBApp(func, arg)
    result = evaluate(application, max_reductions)
    return church_to_int(result)


# =============================================================================
# Parsing + Evaluation Convenience
# =============================================================================

def run(source: str, max_reductions: int = DEFAULT_TIMEOUT) -> DBTerm:
    """
    Parse a Python lambda expression and evaluate it.

    Args:
        source: Python lambda expression.
        max_reductions: Maximum beta reductions.

    Returns:
        DBTerm: The normal form.
    """
    from .lambda_parser import parse_to_debruijn
    term = parse_to_debruijn(source)
    return evaluate(term, max_reductions)


def run_program(source: str, max_reductions: int = DEFAULT_TIMEOUT) -> DBTerm:
    """
    Parse a multi-line program and evaluate it.

    Args:
        source: Multi-line Python with definitions.
        max_reductions: Maximum beta reductions.

    Returns:
        DBTerm: The normal form.
    """
    from .lambda_parser import parse_program_to_debruijn
    term = parse_program_to_debruijn(source)
    return evaluate(term, max_reductions)
