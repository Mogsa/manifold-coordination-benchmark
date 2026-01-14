"""
Tests for the Telepathic benchmark core engine.
"""

import math
import pytest

from telepathic.core.functions import (
    PRIMITIVES,
    TOKEN_TO_NAME,
    NAME_TO_TOKEN,
    compose,
    compose_by_names,
    compose_by_tokens,
    parse_message_to_function,
)
from telepathic.core.sampling import (
    SAMPLE_X_VALUES,
    TEST_X_VALUES,
    SampleGenerator,
    verify_no_overlap,
)
from telepathic.core.protocol import (
    Protocol,
    validate_message,
    parse_message,
)
from telepathic.core.evaluation import (
    Evaluator,
    score_prediction,
    compute_phi,
)
from telepathic.core.few_shot import (
    FEW_SHOT_PRIMITIVES,
    FEW_SHOT_COMPOSITIONS,
    NOVEL_COMPOSITIONS,
)


# =============================================================================
# FUNCTION TESTS
# =============================================================================

class TestPrimitives:
    """Tests for primitive functions."""

    def test_sin_evaluates_correctly(self):
        """Sin function evaluates correctly."""
        f = PRIMITIVES["sin"]
        assert abs(f(0) - 0.0) < 0.001
        assert abs(f(math.pi/2) - 1.0) < 0.001

    def test_cos_evaluates_correctly(self):
        """Cos function evaluates correctly."""
        f = PRIMITIVES["cos"]
        assert abs(f(0) - 1.0) < 0.001
        assert abs(f(math.pi) - (-1.0)) < 0.001

    def test_square_evaluates_correctly(self):
        """Square function evaluates correctly."""
        f = PRIMITIVES["square"]
        assert f(2) == 4
        assert f(-3) == 9
        assert f(0) == 0

    def test_abs_evaluates_correctly(self):
        """Abs function evaluates correctly."""
        f = PRIMITIVES["abs"]
        assert f(2) == 2
        assert f(-3) == 3
        assert f(0) == 0

    def test_neg_evaluates_correctly(self):
        """Neg function evaluates correctly."""
        f = PRIMITIVES["neg"]
        assert f(2) == -2
        assert f(-3) == 3
        assert f(0) == 0

    def test_token_mappings(self):
        """Token mappings are correct."""
        assert TOKEN_TO_NAME["α"] == "sin"
        assert TOKEN_TO_NAME["β"] == "cos"
        assert TOKEN_TO_NAME["γ"] == "square"
        assert TOKEN_TO_NAME["δ"] == "abs"
        assert TOKEN_TO_NAME["ε"] == "neg"

        assert NAME_TO_TOKEN["sin"] == "α"
        assert NAME_TO_TOKEN["cos"] == "β"


class TestComposition:
    """Tests for function composition."""

    def test_composition_order(self):
        """Composition applies functions right-to-left."""
        # sin(x²) at x=0 should be sin(0) = 0
        f = compose(PRIMITIVES["sin"].func, PRIMITIVES["square"].func)
        assert abs(f(0) - 0.0) < 0.001

        # At x=√(π/2), x² = π/2, sin(π/2) = 1
        x = math.sqrt(math.pi/2)
        assert abs(f(x) - 1.0) < 0.001

    def test_compose_by_names(self):
        """compose_by_names creates correct ComposedFunction."""
        cf = compose_by_names(["sin", "square"])
        assert cf.primitives == ["sin", "square"]
        assert cf.tokens == ["α", "γ"]
        assert cf.message == "α-γ"
        assert cf.complexity == 2

        # Evaluate: sin(2²) = sin(4)
        expected = math.sin(4)
        assert abs(cf(2) - expected) < 0.001

    def test_compose_by_tokens(self):
        """compose_by_tokens creates correct ComposedFunction."""
        cf = compose_by_tokens(["δ", "α"])
        assert cf.primitives == ["abs", "sin"]
        assert cf.message == "δ-α"

        # Evaluate: |sin(-2)|
        expected = abs(math.sin(-2))
        assert abs(cf(-2) - expected) < 0.001

    def test_parse_message_primitive(self):
        """parse_message_to_function handles single token."""
        f = parse_message_to_function("α")
        assert abs(f(0) - 0.0) < 0.001  # sin(0) = 0

    def test_parse_message_composition(self):
        """parse_message_to_function handles composition."""
        f = parse_message_to_function("α-γ")
        # sin(square(2)) = sin(4)
        expected = math.sin(4)
        assert abs(f(2) - expected) < 0.001


# =============================================================================
# SAMPLING TESTS
# =============================================================================

class TestSampling:
    """Tests for sample generation."""

    def test_sample_count(self):
        """Generates correct number of samples."""
        gen = SampleGenerator(PRIMITIVES["sin"])
        samples = gen.generate_samples()
        assert len(samples.samples) == 5

    def test_test_count(self):
        """Generates correct number of test points."""
        gen = SampleGenerator(PRIMITIVES["sin"])
        tests = gen.generate_tests()
        assert len(tests) == 3

    def test_no_overlap(self):
        """Sample and test points don't overlap."""
        assert verify_no_overlap()
        sample_set = set(SAMPLE_X_VALUES)
        test_set = set(TEST_X_VALUES)
        assert sample_set.isdisjoint(test_set)

    def test_sample_values_correct(self):
        """Sample values are computed correctly."""
        gen = SampleGenerator(PRIMITIVES["square"])
        samples = gen.generate_samples()

        # Check specific values for square
        sample_dict = {s.x: s.y for s in samples.samples}
        assert sample_dict[-2.0] == 4.0
        assert sample_dict[-1.0] == 1.0
        assert sample_dict[0.0] == 0.0
        assert sample_dict[1.0] == 1.0
        assert sample_dict[2.0] == 4.0


# =============================================================================
# PROTOCOL TESTS
# =============================================================================

class TestProtocol:
    """Tests for communication protocol."""

    def test_valid_single_token(self):
        """Single token message is valid."""
        assert validate_message("α") is True
        assert validate_message("β") is True
        assert validate_message("ε") is True

    def test_valid_composition(self):
        """Composition message is valid."""
        assert validate_message("α-β") is True
        assert validate_message("α-β-γ") is True
        assert validate_message("α-β-γ-δ-ε") is True  # Max 5 tokens

    def test_invalid_token(self):
        """Unknown token rejected."""
        assert validate_message("x") is False
        assert validate_message("α-x") is False
        assert validate_message("sin") is False

    def test_too_long(self):
        """Message exceeding max length rejected."""
        assert validate_message("α-β-γ-δ-ε-α") is False  # 6 tokens

    def test_empty_invalid(self):
        """Empty message is invalid."""
        assert validate_message("") is False
        assert validate_message("   ") is False

    def test_parse_message(self):
        """Message parsing works."""
        assert parse_message("α") == ["α"]
        assert parse_message("α-β-γ") == ["α", "β", "γ"]

    def test_protocol_instance(self):
        """Protocol instance methods work."""
        p = Protocol()
        assert p.is_primitive("α") is True
        assert p.is_primitive("α-β") is False
        assert p.is_composition("α-β") is True
        assert p.is_composition("α") is False
        assert p.count_tokens("α-β-γ") == 3


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Tests for evaluation logic."""

    def test_exact_match(self):
        """Exact prediction scores as success."""
        score = score_prediction(predicted=1.0, expected=1.0)
        assert score.success is True

    def test_within_tolerance(self):
        """Prediction within 1% scores as success."""
        score = score_prediction(predicted=1.005, expected=1.0)
        assert score.success is True

        score = score_prediction(predicted=0.995, expected=1.0)
        assert score.success is True

    def test_outside_tolerance(self):
        """Prediction outside 1% scores as failure."""
        score = score_prediction(predicted=1.02, expected=1.0)
        assert score.success is False

        score = score_prediction(predicted=0.98, expected=1.0)
        assert score.success is False

    def test_near_zero_absolute(self):
        """Near-zero uses absolute tolerance."""
        score = score_prediction(predicted=0.005, expected=0.0)
        assert score.success is True

        score = score_prediction(predicted=0.02, expected=0.0)
        assert score.success is False

    def test_phi_calculation(self):
        """Phi metric computed correctly."""
        assert compute_phi(1.0, 1.0) == 1.0
        assert compute_phi(1.0, 0.5) == 0.5
        assert compute_phi(0.8, 0.4) == 0.5
        assert compute_phi(0.0, 0.5) == 0.0  # Avoid division by zero


# =============================================================================
# FEW-SHOT TESTS
# =============================================================================

class TestFewShot:
    """Tests for few-shot example generation."""

    def test_primitive_count(self):
        """Correct number of primitive examples."""
        assert len(FEW_SHOT_PRIMITIVES) == 5

    def test_composition_count(self):
        """Correct number of composition examples."""
        assert len(FEW_SHOT_COMPOSITIONS) == 2

    def test_novel_compositions_count(self):
        """Correct number of novel compositions for testing."""
        # 5 tokens, all pairs minus self-compositions (5*4=20) minus 2 shown = 18
        assert len(NOVEL_COMPOSITIONS) == 18

    def test_primitive_samples_correct(self):
        """Primitive examples have correct sample values."""
        sin_ex = FEW_SHOT_PRIMITIVES["α"]
        assert sin_ex.function_name == "sin"

        # Check sin(0) = 0
        sample_dict = dict(sin_ex.samples)
        assert abs(sample_dict[0.0] - 0.0) < 0.001

    def test_composition_samples_correct(self):
        """Composition examples have correct sample values."""
        comp = FEW_SHOT_COMPOSITIONS["α-γ"]  # sin(x²)
        sample_dict = dict(comp.samples)

        # sin(0²) = sin(0) = 0
        assert abs(sample_dict[0.0] - 0.0) < 0.001

        # sin(1²) = sin(1) ≈ 0.8415
        assert abs(sample_dict[1.0] - math.sin(1)) < 0.001

    def test_novel_excludes_shown(self):
        """Novel compositions exclude those shown in few-shot."""
        shown = {("α", "γ"), ("δ", "α")}
        for comp in NOVEL_COMPOSITIONS:
            assert comp not in shown
