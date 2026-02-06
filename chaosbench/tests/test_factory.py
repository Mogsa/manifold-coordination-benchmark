"""Tests for chaosbench.problems.factory — problem generation."""

import numpy as np
import pytest

from chaosbench.problems.factory import (
    Problem,
    QuestionType,
    create_problem,
)


class TestCreateProblem:
    def test_classify_problem(self):
        p = create_problem("logistic", {"r": 3.891}, QuestionType.CLASSIFY)
        assert p.question_type == QuestionType.CLASSIFY
        assert p.ground_truth.regime is not None
        assert p.ground_truth.regime == "chaotic"
        assert p.observations.shape == (200,)

    def test_identify_problem(self):
        p = create_problem("tent", {"mu": 1.5}, QuestionType.IDENTIFY)
        assert p.question_type == QuestionType.IDENTIFY
        assert p.ground_truth.family == "tent"

    def test_predict_problem(self):
        p = create_problem("logistic", {"r": 3.891}, QuestionType.PREDICT)
        assert p.question_type == QuestionType.PREDICT
        assert p.ground_truth.future_values is not None
        assert len(p.ground_truth.future_values) > 0
        assert "horizon" in p.question_params

    def test_deterministic_ids(self):
        """Same inputs → same problem_id."""
        p1 = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)
        p2 = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)
        assert p1.problem_id == p2.problem_id

    def test_different_questions_different_ids(self):
        p1 = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)
        p2 = create_problem("logistic", {"r": 3.5}, QuestionType.PREDICT)
        assert p1.problem_id != p2.problem_id

    def test_different_observation_configs_different_ids(self):
        p1 = create_problem(
            "logistic",
            {"r": 3.5},
            QuestionType.CLASSIFY,
            n_points=200,
            noise_std=0.01,
            stride=1,
            burn_in=500,
        )
        p2 = create_problem(
            "logistic",
            {"r": 3.5},
            QuestionType.CLASSIFY,
            n_points=220,
            noise_std=0.02,
            stride=2,
            burn_in=600,
        )
        assert p1.problem_id != p2.problem_id

    def test_difficulty_positive(self):
        p = create_problem("logistic", {"r": 3.891}, QuestionType.CLASSIFY)
        assert p.difficulty > 0

    def test_metadata_shown_to_agent(self):
        p = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)
        assert "domain" in p.metadata
        assert "n_points" in p.metadata
        assert "noise_std" in p.metadata
        assert "observation_mode" in p.metadata

    def test_observation_mode_clean_vs_noisy(self):
        clean = create_problem(
            "logistic", {"r": 3.5}, QuestionType.CLASSIFY, noise_std=0.0
        )
        noisy = create_problem(
            "logistic", {"r": 3.5}, QuestionType.CLASSIFY, noise_std=0.02
        )
        assert clean.metadata["observation_mode"] == "clean"
        assert noisy.metadata["observation_mode"] == "noisy"

    def test_system_metadata_hidden(self):
        p = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)
        assert p.system_metadata.family == "logistic"

    def test_custom_n_points(self):
        p = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY, n_points=50)
        assert p.observations.shape == (50,)

    def test_all_families(self):
        """Every family should produce a valid problem."""
        configs = [
            ("logistic", {"r": 3.5}),
            ("tent", {"mu": 1.5}),
            ("damped_linear", {"lam": 0.5}),
            ("rotation", {"omega": 0.25}),
        ]
        for family, params in configs:
            p = create_problem(family, params, QuestionType.CLASSIFY)
            assert isinstance(p, Problem)
