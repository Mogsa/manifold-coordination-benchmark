"""Tests for model factory."""
import pytest
import numpy as np

from chaosbench.core.models import create_model, MODEL_PARAMS


class TestCreateModel:
    def test_create_logistic(self):
        """Create logistic map with r parameter."""
        model = create_model("logistic", {"r": 3.9})
        assert model.family == "logistic"
        assert model.r == 3.9

    def test_create_tent(self):
        """Create tent map with mu parameter."""
        model = create_model("tent", {"mu": 1.8})
        assert model.family == "tent"
        assert model.mu == 1.8

    def test_unknown_family_raises(self):
        """Unknown family raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model family"):
            create_model("unknown", {})

    def test_model_params_lists_families(self):
        """MODEL_PARAMS contains all supported families."""
        assert "logistic" in MODEL_PARAMS
        assert "tent" in MODEL_PARAMS
        assert MODEL_PARAMS["logistic"] == ["r"]
        assert MODEL_PARAMS["tent"] == ["mu"]
