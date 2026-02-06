"""
Tests for BLC Compiler (Checkpoint 1.3).

Tests the compilation of De Bruijn indexed lambda terms to
Binary Lambda Calculus bit strings.

BLC Encoding Rules:
    Abstraction:  λM     → 00 ++ blc(M)
    Application:  M N    → 01 ++ blc(M) ++ blc(N)
    Variable:     i      → 1^(i+1) ++ 0

Examples from spec:
    Identity (λx.x):     00 10           = 4 bits
    True (λx.λy.x):      00 00 110       = 7 bits
    False (λx.λy.y):     00 00 10        = 6 bits
"""

import pytest
from telepathic.core.lambda_parser import (
    parse_to_debruijn,
    DBVar, DBAbs, DBApp,
)
from telepathic.core.blc_compiler import (
    compile_to_blc,
    bit_length,
    compile_from_source,
    bits_from_source,
    CompileError,
)


# =============================================================================
# Variable Encoding Tests
# =============================================================================

class TestVariableEncoding:
    """Test encoding of De Bruijn variables."""

    def test_var_0(self):
        """Variable 0 → 10"""
        result = compile_to_blc(DBVar(0))
        assert result == "10"

    def test_var_1(self):
        """Variable 1 → 110"""
        result = compile_to_blc(DBVar(1))
        assert result == "110"

    def test_var_2(self):
        """Variable 2 → 1110"""
        result = compile_to_blc(DBVar(2))
        assert result == "1110"

    def test_var_5(self):
        """Variable 5 → 1111110"""
        result = compile_to_blc(DBVar(5))
        assert result == "1111110"


# =============================================================================
# Abstraction Encoding Tests
# =============================================================================

class TestAbstractionEncoding:
    """Test encoding of lambda abstractions."""

    def test_identity(self):
        """Identity (λ.0) → 00 10 = 4 bits"""
        # λx.x in De Bruijn is λ.0
        term = DBAbs(DBVar(0))
        result = compile_to_blc(term)
        assert result == "0010"
        assert len(result) == 4

    def test_nested_abstraction(self):
        """λ.λ.0 → 00 00 10 = 6 bits"""
        # λx.λy.y in De Bruijn
        term = DBAbs(DBAbs(DBVar(0)))
        result = compile_to_blc(term)
        assert result == "000010"
        assert len(result) == 6


# =============================================================================
# Application Encoding Tests
# =============================================================================

class TestApplicationEncoding:
    """Test encoding of function applications."""

    def test_simple_application(self):
        """(0 0) → 01 10 10"""
        term = DBApp(DBVar(0), DBVar(0))
        result = compile_to_blc(term)
        assert result == "011010"

    def test_nested_application(self):
        """(0 (0 0)) → 01 10 01 10 10"""
        inner = DBApp(DBVar(0), DBVar(0))
        outer = DBApp(DBVar(0), inner)
        result = compile_to_blc(outer)
        assert result == "0110011010"


# =============================================================================
# Spec Examples (Critical Tests)
# =============================================================================

class TestSpecExamples:
    """Test the exact examples from the spec (Section 6.6)."""

    def test_identity_spec(self):
        """Identity (λx.x) → "0010" = 4 bits"""
        result = compile_from_source("lambda x: x")
        assert result == "0010"
        assert len(result) == 4

    def test_true_spec(self):
        """True (λx.λy.x) → "0000110" = 7 bits"""
        result = compile_from_source("lambda x: lambda y: x")
        assert result == "0000110"
        assert len(result) == 7

    def test_false_spec(self):
        """False (λx.λy.y) → "000010" = 6 bits"""
        result = compile_from_source("lambda x: lambda y: y")
        assert result == "000010"
        assert len(result) == 6


# =============================================================================
# Bit Length Tests
# =============================================================================

class TestBitLength:
    """Test efficient bit counting without string building."""

    def test_identity_length(self):
        """Identity is 4 bits."""
        term = parse_to_debruijn("lambda x: x")
        assert bit_length(term) == 4

    def test_true_length(self):
        """TRUE is 7 bits."""
        term = parse_to_debruijn("lambda x: lambda y: x")
        assert bit_length(term) == 7

    def test_false_length(self):
        """FALSE is 6 bits."""
        term = parse_to_debruijn("lambda x: lambda y: y")
        assert bit_length(term) == 6

    def test_length_matches_compile(self):
        """bit_length should match len(compile_to_blc())."""
        test_cases = [
            "lambda x: x",
            "lambda x: lambda y: x",
            "lambda x: lambda y: y",
            "lambda f: lambda x: f(x)",
            "lambda f: lambda x: f(f(x))",
            "lambda m: lambda n: lambda f: m(n(f))",
        ]
        for source in test_cases:
            term = parse_to_debruijn(source)
            assert bit_length(term) == len(compile_to_blc(term)), f"Mismatch for: {source}"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test compile_from_source and bits_from_source."""

    def test_compile_from_source(self):
        """compile_from_source works end-to-end."""
        assert compile_from_source("lambda x: x") == "0010"
        assert compile_from_source("lambda x: lambda y: x") == "0000110"

    def test_bits_from_source(self):
        """bits_from_source returns correct count."""
        assert bits_from_source("lambda x: x") == 4
        assert bits_from_source("lambda x: lambda y: x") == 7
        assert bits_from_source("lambda x: lambda y: y") == 6


# =============================================================================
# Church Numerals
# =============================================================================

class TestChurchNumerals:
    """Test BLC encoding of Church numerals."""

    def test_church_zero(self):
        """Church 0 = λf.λx.x (same as FALSE)."""
        result = compile_from_source("lambda f: lambda x: x")
        assert result == "000010"  # Same as FALSE
        assert len(result) == 6

    def test_church_one(self):
        """Church 1 = λf.λx.f(x)."""
        result = compile_from_source("lambda f: lambda x: f(x)")
        # λ.λ.(1 0) → 00 00 01 110 10
        assert result == "00000111010"
        assert len(result) == 11

    def test_church_two(self):
        """Church 2 = λf.λx.f(f(x))."""
        result = compile_from_source("lambda f: lambda x: f(f(x))")
        # λ.λ.App(1, App(1, 0))
        # App encoding: 01 + func + arg
        # = 00 00 01 110 (01 110 10) = 0000011100111010
        assert result == "0000011100111010"
        assert len(result) == 16


# =============================================================================
# Combinators
# =============================================================================

class TestCombinators:
    """Test BLC encoding of standard combinators."""

    def test_succ(self):
        """SUCC = λn.λf.λx.f(n(f)(x))"""
        blc = compile_from_source("lambda n: lambda f: lambda x: f(n(f)(x))")
        # Should be a valid bit string
        assert all(c in "01" for c in blc)
        # Size should be reasonable
        assert 20 < len(blc) < 50

    def test_mul(self):
        """MUL = λm.λn.λf.m(n(f))"""
        blc = compile_from_source("lambda m: lambda n: lambda f: m(n(f))")
        assert all(c in "01" for c in blc)
        # MUL is pretty compact
        assert len(blc) < 30

    def test_add(self):
        """ADD = λm.λn.λf.λx.m(f)(n(f)(x))"""
        blc = compile_from_source("lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))")
        assert all(c in "01" for c in blc)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_deeply_nested(self):
        """Test deeply nested abstractions."""
        # λa.λb.λc.λd.λe.a
        source = "lambda a: lambda b: lambda c: lambda d: lambda e: a"
        blc = compile_from_source(source)
        # 5 abstractions (00 each) + variable 4 (11110)
        assert blc.startswith("0000000000")  # 5 × "00"
        assert "111110" in blc  # Index 4

    def test_complex_application(self):
        """Test complex nested applications."""
        # ((λx.x)(λy.y))
        source = "lambda z: z"  # We need to parse an application
        # Actually let's test a combinator that has applications
        source = "lambda f: lambda x: f(f(f(x)))"  # Church 3
        blc = compile_from_source(source)
        assert all(c in "01" for c in blc)
