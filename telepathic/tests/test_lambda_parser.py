"""
Tests for Lambda Parser (Checkpoints 1.1 and 1.2).

Tests:
- Parsing Python lambda syntax to AST
- De Bruijn index conversion
- Error handling for invalid syntax
"""

import pytest
from telepathic.core.lambda_parser import (
    parse, parse_with_definitions, parse_to_debruijn, parse_program_to_debruijn,
    to_debruijn,
    Var, Abs, App,
    DBVar, DBAbs, DBApp,
    ParseError, ConversionError,
)


# =============================================================================
# Checkpoint 1.1: Lambda Parser Tests
# =============================================================================

class TestParsing:
    """Test parsing Python lambda syntax to named AST."""

    def test_identity(self):
        """Parse identity function: λx.x"""
        result = parse("lambda x: x")
        assert isinstance(result, Abs)
        assert result.param == "x"
        assert isinstance(result.body, Var)
        assert result.body.name == "x"

    def test_true_combinator(self):
        """Parse TRUE/K combinator: λx.λy.x"""
        result = parse("lambda x: lambda y: x")
        assert isinstance(result, Abs)
        assert result.param == "x"
        assert isinstance(result.body, Abs)
        assert result.body.param == "y"
        assert isinstance(result.body.body, Var)
        assert result.body.body.name == "x"

    def test_false_combinator(self):
        """Parse FALSE combinator: λx.λy.y"""
        result = parse("lambda x: lambda y: y")
        assert isinstance(result, Abs)
        assert result.param == "x"
        assert isinstance(result.body, Abs)
        assert result.body.param == "y"
        assert isinstance(result.body.body, Var)
        assert result.body.body.name == "y"

    def test_application(self):
        """Parse function application: λf.λx.f(x)"""
        result = parse("lambda f: lambda x: f(x)")
        assert isinstance(result, Abs)
        assert result.param == "f"
        inner = result.body
        assert isinstance(inner, Abs)
        assert inner.param == "x"
        app = inner.body
        assert isinstance(app, App)
        assert isinstance(app.func, Var) and app.func.name == "f"
        assert isinstance(app.arg, Var) and app.arg.name == "x"

    def test_nested_application(self):
        """Parse nested application: λf.λx.f(f(x))"""
        result = parse("lambda f: lambda x: f(f(x))")
        assert isinstance(result, Abs)
        inner = result.body
        assert isinstance(inner, Abs)
        outer_app = inner.body
        assert isinstance(outer_app, App)
        assert isinstance(outer_app.func, Var) and outer_app.func.name == "f"
        inner_app = outer_app.arg
        assert isinstance(inner_app, App)
        assert isinstance(inner_app.func, Var) and inner_app.func.name == "f"
        assert isinstance(inner_app.arg, Var) and inner_app.arg.name == "x"

    def test_succ_combinator(self):
        """Parse SUCC combinator: λn.λf.λx.f(n(f)(x))"""
        result = parse("lambda n: lambda f: lambda x: f(n(f)(x))")
        assert isinstance(result, Abs)
        assert result.param == "n"

    def test_mul_combinator(self):
        """Parse MUL combinator: λm.λn.λf.m(n(f))"""
        result = parse("lambda m: lambda n: lambda f: m(n(f))")
        assert isinstance(result, Abs)
        assert result.param == "m"


class TestParsingWithDefinitions:
    """Test parsing multi-line programs with variable definitions."""

    def test_simple_definition(self):
        """Parse program with one definition."""
        code = """
TRUE = lambda x: lambda y: x
TRUE
"""
        result = parse_with_definitions(code)
        # Should be fully inlined λx.λy.x
        assert isinstance(result, Abs)
        assert result.param == "x"
        assert isinstance(result.body, Abs)
        assert result.body.param == "y"
        assert isinstance(result.body.body, Var)
        assert result.body.body.name == "x"

    def test_definition_with_application(self):
        """Parse program using defined term in application."""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
SUCC(ZERO)
"""
        result = parse_with_definitions(code)
        # Should be application of SUCC to ZERO, fully inlined
        assert isinstance(result, App)

    def test_multiple_definitions(self):
        """Parse program with multiple definitions."""
        code = """
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y
AND = lambda a: lambda b: a(b)(FALSE)
AND
"""
        result = parse_with_definitions(code)
        # AND with FALSE inlined
        assert isinstance(result, Abs)
        assert result.param == "a"


class TestParseErrors:
    """Test error handling for invalid lambda syntax."""

    def test_reject_arithmetic(self):
        """Reject arithmetic operators."""
        with pytest.raises(ParseError, match="Binary operators not allowed"):
            parse("lambda x: x + 1")

    def test_reject_multiplication(self):
        """Reject multiplication."""
        with pytest.raises(ParseError, match="Binary operators not allowed"):
            parse("lambda x: x * 2")

    def test_reject_conditional(self):
        """Reject conditional expressions."""
        with pytest.raises(ParseError, match="Conditional expressions not allowed"):
            parse("lambda x: x if x else x")

    def test_reject_literal(self):
        """Reject numeric literals."""
        with pytest.raises(ParseError, match="Literals not allowed"):
            parse("lambda x: 5")

    def test_reject_multi_arg_lambda(self):
        """Reject multi-argument lambda."""
        with pytest.raises(ParseError, match="exactly one argument"):
            parse("lambda x, y: x")

    def test_reject_multi_arg_call(self):
        """Reject multi-argument function call."""
        with pytest.raises(ParseError, match="exactly one argument"):
            parse("lambda f: f(1, 2)")

    def test_reject_kwargs(self):
        """Reject keyword arguments."""
        with pytest.raises(ParseError, match="Keyword arguments not allowed"):
            parse("lambda f: f(x=1)")


# =============================================================================
# Checkpoint 1.2: De Bruijn Conversion Tests
# =============================================================================

class TestDeBruijnConversion:
    """Test conversion from named variables to De Bruijn indices."""

    def test_identity(self):
        """Identity: λx.x → λ.0"""
        result = parse_to_debruijn("lambda x: x")
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBVar)
        assert result.body.index == 0

    def test_true_combinator(self):
        """TRUE: λx.λy.x → λ.λ.1"""
        result = parse_to_debruijn("lambda x: lambda y: x")
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBAbs)
        assert isinstance(result.body.body, DBVar)
        assert result.body.body.index == 1  # x is one binder away

    def test_false_combinator(self):
        """FALSE: λx.λy.y → λ.λ.0"""
        result = parse_to_debruijn("lambda x: lambda y: y")
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBAbs)
        assert isinstance(result.body.body, DBVar)
        assert result.body.body.index == 0  # y is innermost

    def test_application_indices(self):
        """Application: λf.λx.f(x) → λ.λ.(1 0)"""
        result = parse_to_debruijn("lambda f: lambda x: f(x)")
        assert isinstance(result, DBAbs)
        inner = result.body
        assert isinstance(inner, DBAbs)
        app = inner.body
        assert isinstance(app, DBApp)
        assert isinstance(app.func, DBVar) and app.func.index == 1  # f
        assert isinstance(app.arg, DBVar) and app.arg.index == 0   # x

    def test_church_two(self):
        """Church 2: λf.λx.f(f(x)) → λ.λ.(1 (1 0))"""
        result = parse_to_debruijn("lambda f: lambda x: f(f(x))")
        assert isinstance(result, DBAbs)
        inner = result.body
        assert isinstance(inner, DBAbs)
        outer_app = inner.body
        assert isinstance(outer_app, DBApp)
        assert isinstance(outer_app.func, DBVar) and outer_app.func.index == 1
        inner_app = outer_app.arg
        assert isinstance(inner_app, DBApp)
        assert isinstance(inner_app.func, DBVar) and inner_app.func.index == 1
        assert isinstance(inner_app.arg, DBVar) and inner_app.arg.index == 0

    def test_shadowing(self):
        """Shadowing: λx.λx.x → λ.λ.0 (inner x shadows outer)"""
        result = parse_to_debruijn("lambda x: lambda x: x")
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBAbs)
        assert isinstance(result.body.body, DBVar)
        assert result.body.body.index == 0  # Refers to inner x

    def test_deep_nesting(self):
        """Deep nesting: λa.λb.λc.a → λ.λ.λ.2"""
        result = parse_to_debruijn("lambda a: lambda b: lambda c: a")
        assert isinstance(result, DBAbs)
        inner1 = result.body
        assert isinstance(inner1, DBAbs)
        inner2 = inner1.body
        assert isinstance(inner2, DBAbs)
        var = inner2.body
        assert isinstance(var, DBVar)
        assert var.index == 2  # a is two binders away


class TestDeBruijnErrors:
    """Test error handling for De Bruijn conversion."""

    def test_free_variable(self):
        """Reject free variables."""
        term = Abs("x", Var("y"))  # y is free
        with pytest.raises(ConversionError, match="Free variable"):
            to_debruijn(term)

    def test_free_variable_in_body(self):
        """Reject free variables in application."""
        term = Abs("x", App(Var("x"), Var("z")))  # z is free
        with pytest.raises(ConversionError, match="Free variable"):
            to_debruijn(term)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end parsing and conversion tests."""

    def test_succ_combinator(self):
        """SUCC: λn.λf.λx.f(n(f)(x))"""
        result = parse_to_debruijn("lambda n: lambda f: lambda x: f(n(f)(x))")
        # Should have structure: λ.λ.λ.(1 ((2 1) 0))
        assert isinstance(result, DBAbs)  # λn
        assert isinstance(result.body, DBAbs)  # λf
        assert isinstance(result.body.body, DBAbs)  # λx
        body = result.body.body.body
        assert isinstance(body, DBApp)  # f(...)
        assert isinstance(body.func, DBVar)
        assert body.func.index == 1  # f

    def test_program_with_church_numerals(self):
        """Parse program with Church numeral definitions."""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
ONE
"""
        result = parse_program_to_debruijn(code)
        # ONE = SUCC(ZERO) should be a big application term
        assert isinstance(result, DBApp)

    def test_repr_methods(self):
        """Test string representations."""
        term = parse("lambda x: lambda y: x")
        assert "λ" in repr(term)

        db_term = to_debruijn(term)
        assert "λ" in repr(db_term)
