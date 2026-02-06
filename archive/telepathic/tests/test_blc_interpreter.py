"""
Tests for BLC Interpreter (Checkpoint 1.4).

Tests beta reduction, Church numeral evaluation, and timeout handling.

Execution model:
    Reduction: Normal order (leftmost-outermost)
    Timeout:   10,000,000 beta reductions
"""

import pytest
from telepathic.core.lambda_parser import (
    parse_to_debruijn, parse_program_to_debruijn,
    DBVar, DBAbs, DBApp,
)
from telepathic.core.blc_interpreter import (
    shift, substitute,
    beta_reduce_once, reduce_to_normal_form, evaluate,
    int_to_church, church_to_int,
    evaluate_church_application,
    run, run_program,
    TimeoutError, ExecutionResult,
)


# =============================================================================
# Shifting Tests
# =============================================================================

class TestShifting:
    """Test the shift operation for De Bruijn terms."""

    def test_shift_var_below_cutoff(self):
        """Variables below cutoff are not shifted."""
        term = DBVar(0)
        result = shift(term, 1, 1)  # Shift by 1 with cutoff 1
        assert result.index == 0

    def test_shift_var_at_cutoff(self):
        """Variables at or above cutoff are shifted."""
        term = DBVar(1)
        result = shift(term, 1, 1)  # Shift by 1 with cutoff 1
        assert result.index == 2

    def test_shift_abstraction(self):
        """Shifting under abstraction increases cutoff."""
        # λ.1 → shift by 1 with cutoff 0 → λ.2
        term = DBAbs(DBVar(1))
        result = shift(term, 1, 0)
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBVar)
        assert result.body.index == 2

    def test_shift_preserves_bound_vars(self):
        """Bound variables (index < cutoff under binders) stay bound."""
        # λ.0 (identity) → shift by 1 → λ.0 (still identity)
        term = DBAbs(DBVar(0))
        result = shift(term, 1, 0)
        assert isinstance(result.body, DBVar)
        assert result.body.index == 0  # Still bound to this lambda


# =============================================================================
# Substitution Tests
# =============================================================================

class TestSubstitution:
    """Test the substitution operation."""

    def test_substitute_matching_var(self):
        """Substitute matching variable."""
        term = DBVar(0)
        replacement = DBVar(1)
        result = substitute(term, 0, replacement)
        assert isinstance(result, DBVar)
        assert result.index == 1

    def test_substitute_non_matching_var(self):
        """Non-matching variable unchanged."""
        term = DBVar(1)
        replacement = DBVar(0)
        result = substitute(term, 0, replacement)
        assert isinstance(result, DBVar)
        assert result.index == 1

    def test_substitute_in_application(self):
        """Substitution distributes over application."""
        # (0 0)[0 := 1] → (1 1)
        term = DBApp(DBVar(0), DBVar(0))
        replacement = DBVar(1)
        result = substitute(term, 0, replacement)
        assert isinstance(result, DBApp)
        assert result.func.index == 1
        assert result.arg.index == 1


# =============================================================================
# Beta Reduction Tests
# =============================================================================

class TestBetaReduction:
    """Test single-step beta reduction."""

    def test_reduce_value(self):
        """Values (abstractions) don't reduce at top level."""
        term = DBAbs(DBVar(0))  # λ.0 (identity)
        result, reduced = beta_reduce_once(term)
        assert not reduced
        assert result == term

    def test_reduce_simple_redex(self):
        """Reduce (λ.0) arg → arg."""
        # (λx.x) y → y
        identity = DBAbs(DBVar(0))
        arg = DBVar(0)
        redex = DBApp(identity, arg)
        result, reduced = beta_reduce_once(redex)
        assert reduced
        assert isinstance(result, DBVar)
        assert result.index == 0

    def test_reduce_true_applied(self):
        """TRUE applied to two args: (λx.λy.x) a b → a."""
        true = parse_to_debruijn("lambda x: lambda y: x")
        # Apply to two dummy args (we'll use variables for testing)
        # In a closed context, we'd apply to closed terms
        # Let's test with nested applications
        first_app = DBApp(true, DBVar(0))  # TRUE a
        result, reduced = beta_reduce_once(first_app)
        assert reduced
        # Result should be λy.a where a is now free (shifted)


# =============================================================================
# Church Numerals
# =============================================================================

class TestChurchNumerals:
    """Test Church numeral creation and interpretation."""

    def test_int_to_church_zero(self):
        """0 → λf.λx.x"""
        zero = int_to_church(0)
        assert isinstance(zero, DBAbs)
        assert isinstance(zero.body, DBAbs)
        assert isinstance(zero.body.body, DBVar)
        assert zero.body.body.index == 0  # x

    def test_int_to_church_one(self):
        """1 → λf.λx.f(x)"""
        one = int_to_church(1)
        assert isinstance(one, DBAbs)
        assert isinstance(one.body, DBAbs)
        body = one.body.body
        assert isinstance(body, DBApp)
        assert body.func.index == 1  # f
        assert body.arg.index == 0   # x

    def test_int_to_church_two(self):
        """2 → λf.λx.f(f(x))"""
        two = int_to_church(2)
        body = two.body.body
        assert isinstance(body, DBApp)
        assert body.func.index == 1  # f
        inner = body.arg
        assert isinstance(inner, DBApp)
        assert inner.func.index == 1  # f
        assert inner.arg.index == 0   # x

    def test_church_to_int_zero(self):
        """λf.λx.x → 0"""
        zero = int_to_church(0)
        assert church_to_int(zero) == 0

    def test_church_to_int_one(self):
        """λf.λx.f(x) → 1"""
        one = int_to_church(1)
        assert church_to_int(one) == 1

    def test_church_to_int_five(self):
        """Church 5 → 5"""
        five = int_to_church(5)
        assert church_to_int(five) == 5

    def test_church_roundtrip(self):
        """int → Church → int preserves value."""
        for n in range(10):
            church = int_to_church(n)
            assert church_to_int(church) == n

    def test_church_to_int_invalid(self):
        """Non-Church terms return None."""
        # Identity is not a Church numeral
        identity = DBAbs(DBVar(0))
        assert church_to_int(identity) is None

        # Wrong structure
        wrong = DBAbs(DBAbs(DBApp(DBVar(0), DBVar(1))))  # λ.λ.(0 1) not Church
        assert church_to_int(wrong) is None


# =============================================================================
# Full Reduction Tests
# =============================================================================

class TestFullReduction:
    """Test reduction to normal form."""

    def test_identity_already_normal(self):
        """Identity is already in normal form."""
        identity = parse_to_debruijn("lambda x: x")
        result = reduce_to_normal_form(identity)
        assert result.reductions == 0
        assert not result.timed_out
        assert isinstance(result.term, DBAbs)

    def test_identity_application(self):
        """(λx.x) (λy.y) → λy.y"""
        # Parse both identities
        id1 = parse_to_debruijn("lambda x: x")
        id2 = parse_to_debruijn("lambda y: y")
        app = DBApp(id1, id2)
        result = reduce_to_normal_form(app)
        assert result.reductions == 1
        # Result should be identity
        assert isinstance(result.term, DBAbs)
        assert isinstance(result.term.body, DBVar)
        assert result.term.body.index == 0

    def test_true_applied_twice(self):
        """(TRUE a) b → a (where a, b are closed terms)."""
        # Use Church numerals as closed terms
        true = parse_to_debruijn("lambda x: lambda y: x")
        zero = int_to_church(0)
        one = int_to_church(1)
        # TRUE ZERO ONE → ZERO
        app1 = DBApp(true, zero)
        app2 = DBApp(app1, one)
        result = reduce_to_normal_form(app2)
        assert not result.timed_out
        # Should reduce to zero
        assert church_to_int(result.term) == 0

    def test_false_applied_twice(self):
        """(FALSE a) b → b."""
        false = parse_to_debruijn("lambda x: lambda y: y")
        zero = int_to_church(0)
        one = int_to_church(1)
        # FALSE ZERO ONE → ONE
        app1 = DBApp(false, zero)
        app2 = DBApp(app1, one)
        result = reduce_to_normal_form(app2)
        assert not result.timed_out
        assert church_to_int(result.term) == 1


# =============================================================================
# Church Arithmetic Tests
# =============================================================================

class TestChurchArithmetic:
    """Test arithmetic with Church numerals."""

    def test_succ_zero(self):
        """SUCC(ZERO) → ONE"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
SUCC(ZERO)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=1000)
        assert not result.timed_out
        assert church_to_int(result.term) == 1

    def test_succ_one(self):
        """SUCC(ONE) → TWO"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
SUCC(ONE)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=10000)
        assert not result.timed_out
        assert church_to_int(result.term) == 2

    def test_add_one_one(self):
        """ADD ONE ONE → TWO"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))
ADD(ONE)(ONE)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=10000)
        assert not result.timed_out
        assert church_to_int(result.term) == 2

    def test_mul_two_three(self):
        """MUL TWO THREE → SIX"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
TWO = SUCC(ONE)
THREE = SUCC(TWO)
MUL = lambda m: lambda n: lambda f: m(n(f))
MUL(TWO)(THREE)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=100000)
        assert not result.timed_out
        assert church_to_int(result.term) == 6


# =============================================================================
# Timeout Tests
# =============================================================================

class TestTimeout:
    """Test timeout handling."""

    def test_timeout_on_infinite_loop(self):
        """Omega combinator should timeout."""
        # Ω = (λx.x(x))(λx.x(x)) - infinite loop
        # We'll construct it manually
        omega_inner = DBAbs(DBApp(DBVar(0), DBVar(0)))  # λx.x(x)
        omega = DBApp(omega_inner, omega_inner)
        result = reduce_to_normal_form(omega, max_reductions=100)
        assert result.timed_out
        assert result.reductions == 100

    def test_evaluate_raises_on_timeout(self):
        """evaluate() raises TimeoutError."""
        omega_inner = DBAbs(DBApp(DBVar(0), DBVar(0)))
        omega = DBApp(omega_inner, omega_inner)
        with pytest.raises(TimeoutError):
            evaluate(omega, max_reductions=100)

    def test_small_timeout_for_testing(self):
        """Small timeout works for quick tests."""
        identity = parse_to_debruijn("lambda x: x")
        result = evaluate(identity, max_reductions=10)
        assert isinstance(result, DBAbs)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test run() and run_program()."""

    def test_run_identity(self):
        """run() evaluates a simple expression."""
        result = run("lambda x: x")
        assert isinstance(result, DBAbs)
        assert isinstance(result.body, DBVar)
        assert result.body.index == 0

    def test_run_program(self):
        """run_program() handles definitions."""
        code = """
TRUE = lambda x: lambda y: x
TRUE
"""
        result = run_program(code)
        assert isinstance(result, DBAbs)

    def test_evaluate_church_application(self):
        """Test evaluate_church_application helper."""
        # Identity applied to 5 → 5
        identity = parse_to_debruijn("lambda x: x")
        result = evaluate_church_application(identity, 5, max_reductions=1000)
        assert result == 5


# =============================================================================
# Integration: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Test the complete pipeline from source to evaluated result."""

    def test_square_function(self):
        """Test SQUARE = λx. MUL(x)(x)"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
TWO = SUCC(ONE)
THREE = SUCC(TWO)
MUL = lambda m: lambda n: lambda f: m(n(f))
SQUARE = lambda x: MUL(x)(x)
SQUARE(THREE)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=1000000)
        assert not result.timed_out, f"Timed out after {result.reductions} reductions"
        assert church_to_int(result.term) == 9

    def test_composition(self):
        """Test function composition: (f ∘ g)(x) = f(g(x))"""
        code = """
ZERO = lambda f: lambda x: x
SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
ONE = SUCC(ZERO)
COMPOSE = lambda f: lambda g: lambda x: f(g(x))
COMPOSE(SUCC)(SUCC)(ONE)
"""
        term = parse_program_to_debruijn(code)
        result = reduce_to_normal_form(term, max_reductions=100000)
        assert not result.timed_out
        # SUCC(SUCC(ONE)) = 3
        assert church_to_int(result.term) == 3
