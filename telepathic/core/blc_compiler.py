"""
Binary Lambda Calculus (BLC) Compiler for Telepathic Benchmark v2.

Compiles De Bruijn indexed lambda terms to BLC bit strings.

Checkpoint 1.3: BLC compiler (AST → binary string)

BLC Encoding Rules (from spec Section 6.6):
    Abstraction:  λM     → 00 ++ blc(M)
    Application:  M N    → 01 ++ blc(M) ++ blc(N)
    Variable:     i      → 1^(i+1) ++ 0   (i+1 ones, then a zero)

Examples:
    Identity (λx.x):     λ.0  → 00 10           = 4 bits
    True (λx.λy.x):      λ.λ.1 → 00 00 110      = 7 bits
    False (λx.λy.y):     λ.λ.0 → 00 00 10       = 6 bits
"""

from __future__ import annotations
from .lambda_parser import DBTerm, DBVar, DBAbs, DBApp


class CompileError(Exception):
    """Raised when BLC compilation fails."""
    pass


def compile_to_blc(term: DBTerm) -> str:
    """
    Compile a De Bruijn term to a BLC bit string.

    Args:
        term: De Bruijn indexed lambda term.

    Returns:
        str: Binary string (e.g., "0010" for identity).

    Examples:
        >>> compile_to_blc(DBVar(0))  # Just variable 0
        '10'
        >>> compile_to_blc(DBAbs(DBVar(0)))  # λ.0 (identity)
        '0010'
        >>> compile_to_blc(DBAbs(DBAbs(DBVar(1))))  # λ.λ.1 (true)
        '0000110'
        >>> compile_to_blc(DBAbs(DBAbs(DBVar(0))))  # λ.λ.0 (false)
        '000010'
    """
    if isinstance(term, DBVar):
        # Variable: i → 1^(i+1) ++ 0
        # Index 0 → "10", Index 1 → "110", Index 2 → "1110", etc.
        return "1" * (term.index + 1) + "0"

    elif isinstance(term, DBAbs):
        # Abstraction: λM → 00 ++ blc(M)
        return "00" + compile_to_blc(term.body)

    elif isinstance(term, DBApp):
        # Application: M N → 01 ++ blc(M) ++ blc(N)
        return "01" + compile_to_blc(term.func) + compile_to_blc(term.arg)

    else:
        raise CompileError(f"Unknown term type: {type(term)}")


def bit_length(term: DBTerm) -> int:
    """
    Get the BLC bit length of a term without building the string.

    This is slightly more efficient for just counting bits.

    Args:
        term: De Bruijn indexed lambda term.

    Returns:
        int: Number of bits in BLC encoding.
    """
    if isinstance(term, DBVar):
        return term.index + 2  # i+1 ones + 1 zero

    elif isinstance(term, DBAbs):
        return 2 + bit_length(term.body)  # 00 + body

    elif isinstance(term, DBApp):
        return 2 + bit_length(term.func) + bit_length(term.arg)  # 01 + func + arg

    else:
        raise CompileError(f"Unknown term type: {type(term)}")


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_from_source(source: str) -> str:
    """
    Parse Python lambda and compile directly to BLC.

    Args:
        source: Python lambda expression string.

    Returns:
        str: BLC bit string.

    Example:
        >>> compile_from_source("lambda x: x")
        '0010'
        >>> compile_from_source("lambda x: lambda y: x")
        '0000110'
    """
    from .lambda_parser import parse_to_debruijn
    return compile_to_blc(parse_to_debruijn(source))


def compile_program(source: str) -> str:
    """
    Parse multi-line program with definitions and compile to BLC.

    Args:
        source: Multi-line Python with assignments.

    Returns:
        str: BLC bit string.

    Example:
        >>> code = '''
        ... TRUE = lambda x: lambda y: x
        ... FALSE = lambda x: lambda y: y
        ... AND = lambda a: lambda b: a(b)(FALSE)
        ... AND
        ... '''
        >>> len(compile_program(code))  # Some number of bits
    """
    from .lambda_parser import parse_program_to_debruijn
    return compile_to_blc(parse_program_to_debruijn(source))


def bits_from_source(source: str) -> int:
    """
    Get BLC bit count for a Python lambda expression.

    Args:
        source: Python lambda expression string.

    Returns:
        int: Number of bits.

    Example:
        >>> bits_from_source("lambda x: x")
        4
        >>> bits_from_source("lambda x: lambda y: x")
        7
    """
    from .lambda_parser import parse_to_debruijn
    return bit_length(parse_to_debruijn(source))
