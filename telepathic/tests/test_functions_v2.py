"""
Tests for Telepathic Benchmark v2 Function Library.

Tests all 4 tiers of functions are correctly defined and work.
"""

import math
import pytest

from telepathic.core.functions_v2 import (
    # Tier imports
    TIER_1_POLYNOMIALS,
    TIER_2_TRANSCENDENTALS,
    TIER_3_COMPOSITIONS,
    TIER_4_INCOMPRESSIBLE,
    ALL_FUNCTIONS,
    # Individual function access
    get_function,
    get_functions_by_tier,
    list_all_functions,
    list_function_ids,
    # Metadata
    TIER_NAMES,
    TIER_DESCRIPTIONS,
    TestFunction,
)


# =============================================================================
# Tier 1: Polynomials
# =============================================================================

class TestTier1Polynomials:
    """Test polynomial functions."""

    def test_tier1_has_5_functions(self):
        """Tier 1 should have exactly 5 polynomial functions."""
        assert len(TIER_1_POLYNOMIALS) == 5

    def test_p1_identity(self):
        """P1: f(x) = x"""
        f = get_function("P1")
        assert f.tier == 1
        assert f.name == "identity"
        assert f(0.5) == 0.5
        assert f(1.0) == 1.0
        assert f(0.123) == pytest.approx(0.123)

    def test_p2_square(self):
        """P2: f(x) = x²"""
        f = get_function("P2")
        assert f.tier == 1
        assert f.name == "square"
        assert f(0.5) == pytest.approx(0.25)
        assert f(1.0) == 1.0
        assert f(0.1) == pytest.approx(0.01)

    def test_p3_cube(self):
        """P3: f(x) = x³"""
        f = get_function("P3")
        assert f.tier == 1
        assert f.name == "cube"
        assert f(0.5) == pytest.approx(0.125)
        assert f(1.0) == 1.0
        assert f(0.2) == pytest.approx(0.008)

    def test_p4_linear(self):
        """P4: f(x) = 2x + 1"""
        f = get_function("P4")
        assert f.tier == 1
        assert f.name == "linear"
        assert f(0.5) == pytest.approx(2.0)
        assert f(1.0) == pytest.approx(3.0)
        assert f(0.0) == pytest.approx(1.0)

    def test_p5_quadratic(self):
        """P5: f(x) = (x+1)²"""
        f = get_function("P5")
        assert f.tier == 1
        assert f.name == "quadratic"
        assert f(0.0) == pytest.approx(1.0)
        assert f(1.0) == pytest.approx(4.0)
        assert f(0.5) == pytest.approx(2.25)


# =============================================================================
# Tier 2: Transcendentals
# =============================================================================

class TestTier2Transcendentals:
    """Test transcendental functions."""

    def test_tier2_has_4_functions(self):
        """Tier 2 should have exactly 4 transcendental functions."""
        assert len(TIER_2_TRANSCENDENTALS) == 4

    def test_t1_sin(self):
        """T1: f(x) = sin(x)"""
        f = get_function("T1")
        assert f.tier == 2
        assert f.name == "sin"
        assert f(0.0) == pytest.approx(0.0)
        assert f(math.pi / 2) == pytest.approx(1.0)
        assert f(0.5) == pytest.approx(math.sin(0.5))

    def test_t2_cos(self):
        """T2: f(x) = cos(x)"""
        f = get_function("T2")
        assert f.tier == 2
        assert f.name == "cos"
        assert f(0.0) == pytest.approx(1.0)
        assert f(math.pi / 2) == pytest.approx(0.0, abs=1e-10)
        assert f(0.5) == pytest.approx(math.cos(0.5))

    def test_t3_exp(self):
        """T3: f(x) = exp(x)"""
        f = get_function("T3")
        assert f.tier == 2
        assert f.name == "exp"
        assert f(0.0) == pytest.approx(1.0)
        assert f(1.0) == pytest.approx(math.e)
        assert f(0.5) == pytest.approx(math.exp(0.5))

    def test_t4_ln1p(self):
        """T4: f(x) = ln(1+x)"""
        f = get_function("T4")
        assert f.tier == 2
        assert f.name == "ln1p"
        assert f(0.0) == pytest.approx(0.0)
        assert f(math.e - 1) == pytest.approx(1.0)
        assert f(0.5) == pytest.approx(math.log(1.5))


# =============================================================================
# Tier 3: Compositions
# =============================================================================

class TestTier3Compositions:
    """Test composition functions."""

    def test_tier3_has_3_functions(self):
        """Tier 3 should have exactly 3 composition functions."""
        assert len(TIER_3_COMPOSITIONS) == 3

    def test_c1_sin_of_square(self):
        """C1: f(x) = sin(x²)"""
        f = get_function("C1")
        assert f.tier == 3
        assert f.name == "sin_of_square"
        assert f(0.0) == pytest.approx(0.0)
        assert f(1.0) == pytest.approx(math.sin(1.0))
        assert f(0.5) == pytest.approx(math.sin(0.25))

    def test_c2_gaussian(self):
        """C2: f(x) = exp(-x²)"""
        f = get_function("C2")
        assert f.tier == 3
        assert f.name == "gaussian"
        assert f(0.0) == pytest.approx(1.0)
        assert f(1.0) == pytest.approx(math.exp(-1))
        assert f(0.5) == pytest.approx(math.exp(-0.25))

    def test_c3_pythagorean_identity(self):
        """C3: f(x) = sin²(x) + cos²(x) = 1"""
        f = get_function("C3")
        assert f.tier == 3
        assert f.name == "pythagorean_identity"
        # Should always return 1.0 regardless of x
        assert f(0.0) == pytest.approx(1.0)
        assert f(0.5) == pytest.approx(1.0)
        assert f(1.0) == pytest.approx(1.0)
        assert f(0.123) == pytest.approx(1.0)


# =============================================================================
# Tier 4: Incompressible
# =============================================================================

class TestTier4Incompressible:
    """Test incompressible (control) functions."""

    def test_tier4_has_2_functions(self):
        """Tier 4 should have exactly 2 incompressible functions."""
        assert len(TIER_4_INCOMPRESSIBLE) == 2

    def test_r1_random_lookup(self):
        """R1: Random lookup table - deterministic for same x."""
        f = get_function("R1")
        assert f.tier == 4
        assert f.name == "random_lookup"

        # Should return values in [0, 1]
        y1 = f(0.5)
        assert 0.0 <= y1 <= 1.0

        # Should be deterministic for same input
        y2 = f(0.5)
        assert y1 == y2

        # Different inputs should (likely) give different outputs
        y3 = f(0.3)
        assert 0.0 <= y3 <= 1.0

    def test_r2_pseudo_random(self):
        """R2: Pseudo-random hash - deterministic for same x."""
        f = get_function("R2")
        assert f.tier == 4
        assert f.name == "pseudo_random"

        # Should return values in [0, 1]
        y1 = f(0.5)
        assert 0.0 <= y1 <= 1.0

        # Should be deterministic for same input
        y2 = f(0.5)
        assert y1 == y2

        # Different inputs should give different outputs
        y3 = f(0.3)
        assert 0.0 <= y3 <= 1.0
        # Very unlikely to be equal
        assert y1 != y3


# =============================================================================
# Registry Functions
# =============================================================================

class TestRegistry:
    """Test function registry operations."""

    def test_all_functions_count(self):
        """Total should be 5 + 4 + 3 + 2 = 14 functions."""
        assert len(ALL_FUNCTIONS) == 14

    def test_get_function_by_id(self):
        """Should retrieve functions by ID."""
        assert get_function("P1").name == "identity"
        assert get_function("T1").name == "sin"
        assert get_function("C1").name == "sin_of_square"
        assert get_function("R1").name == "random_lookup"

    def test_get_function_invalid_id(self):
        """Should raise KeyError for invalid ID."""
        with pytest.raises(KeyError):
            get_function("INVALID")

    def test_get_functions_by_tier(self):
        """Should filter functions by tier."""
        tier1 = get_functions_by_tier(1)
        assert len(tier1) == 5
        assert all(f.tier == 1 for f in tier1)

        tier2 = get_functions_by_tier(2)
        assert len(tier2) == 4
        assert all(f.tier == 2 for f in tier2)

        tier3 = get_functions_by_tier(3)
        assert len(tier3) == 3
        assert all(f.tier == 3 for f in tier3)

        tier4 = get_functions_by_tier(4)
        assert len(tier4) == 2
        assert all(f.tier == 4 for f in tier4)

    def test_list_all_functions(self):
        """Should return all functions."""
        funcs = list_all_functions()
        assert len(funcs) == 14
        assert all(isinstance(f, TestFunction) for f in funcs)

    def test_list_function_ids(self):
        """Should return all function IDs."""
        ids = list_function_ids()
        assert len(ids) == 14
        assert "P1" in ids
        assert "T1" in ids
        assert "C1" in ids
        assert "R1" in ids


# =============================================================================
# TestFunction Dataclass
# =============================================================================

class TestTestFunctionDataclass:
    """Test the TestFunction dataclass."""

    def test_callable(self):
        """TestFunction should be callable."""
        f = get_function("P1")
        assert callable(f)
        assert f(0.5) == 0.5

    def test_to_dict(self):
        """Should serialize to dict without callable."""
        f = get_function("P1")
        d = f.to_dict()
        assert d["id"] == "P1"
        assert d["name"] == "identity"
        assert d["tier"] == 1
        assert d["formula"] == "f(x) = x"
        assert "func" not in d  # Should not include callable

    def test_all_functions_have_required_fields(self):
        """All functions should have required fields."""
        for fid, f in ALL_FUNCTIONS.items():
            assert f.id == fid
            assert f.name
            assert f.tier in [1, 2, 3, 4]
            assert f.formula
            assert callable(f.func)
            assert f.description


# =============================================================================
# Domain Tests
# =============================================================================

class TestDomain:
    """Test functions work on the benchmark domain (0, 1]."""

    def test_all_functions_work_on_domain(self):
        """All functions should work on (0, 1]."""
        test_points = [0.001, 0.1, 0.25, 0.5, 0.75, 1.0]

        for fid, f in ALL_FUNCTIONS.items():
            for x in test_points:
                try:
                    y = f(x)
                    assert not math.isnan(y), f"{fid} returned NaN at x={x}"
                    assert not math.isinf(y), f"{fid} returned Inf at x={x}"
                except Exception as e:
                    pytest.fail(f"{fid} raised exception at x={x}: {e}")


# =============================================================================
# Metadata Tests
# =============================================================================

class TestMetadata:
    """Test tier metadata."""

    def test_tier_names(self):
        """All tiers should have names."""
        assert TIER_NAMES[1] == "Polynomials"
        assert TIER_NAMES[2] == "Transcendentals"
        assert TIER_NAMES[3] == "Compositions"
        assert TIER_NAMES[4] == "Incompressible"

    def test_tier_descriptions(self):
        """All tiers should have descriptions."""
        assert len(TIER_DESCRIPTIONS) == 4
        for tier in [1, 2, 3, 4]:
            assert tier in TIER_DESCRIPTIONS
            assert len(TIER_DESCRIPTIONS[tier]) > 0
