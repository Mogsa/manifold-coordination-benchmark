"""Tests for chaosbench.problems.bank — mini-bank generation."""

import json
import os
import tempfile

import numpy as np
import pytest

from chaosbench.problems.bank import (
    build_and_freeze_track_banks,
    freeze_bank,
    generate_mini_bank,
    generate_track_banks,
    load_bank,
    validate_and_filter_bank,
    validate_problem,
)
from chaosbench.problems.factory import QuestionType


class TestGenerateMiniBank:
    def test_count(self):
        """4 families × 3 params × 3 question types = 36 problems."""
        bank = generate_mini_bank()
        assert len(bank) == 36

    def test_all_families_present(self):
        bank = generate_mini_bank()
        families = {p.system_metadata.family for p in bank}
        assert families == {"logistic", "tent", "damped_linear", "rotation"}

    def test_all_question_types_present(self):
        bank = generate_mini_bank()
        types = {p.question_type for p in bank}
        assert types == {QuestionType.CLASSIFY, QuestionType.IDENTIFY, QuestionType.PREDICT}

    def test_predict_only_slice(self):
        bank = generate_mini_bank(question_types=[QuestionType.PREDICT])
        assert len(bank) == 12  # 4 families × 3 params × 1 type
        assert {p.question_type for p in bank} == {QuestionType.PREDICT}

    def test_no_textbook_params(self):
        """Verify no textbook parameter values (r=4, mu=2, etc.)."""
        bank = generate_mini_bank()
        for p in bank:
            params = p.system_metadata.params
            if "r" in params:
                assert params["r"] != 4.0, "r=4.0 is textbook"
                assert params["r"] != 3.0, "r=3.0 is textbook bifurcation"
            if "mu" in params:
                assert params["mu"] != 2.0, "mu=2.0 is textbook"
                assert params["mu"] != 1.0, "mu=1.0 is textbook boundary"


class TestValidateProblem:
    def test_chaotic_logistic_passes(self):
        from chaosbench.problems.factory import create_problem
        p = create_problem("logistic", {"r": 3.891}, QuestionType.CLASSIFY)
        passed, details = validate_problem(p)
        assert details["stage1_passed"]

    def test_damped_linear_passes(self):
        from chaosbench.problems.factory import create_problem
        p = create_problem("damped_linear", {"lam": 0.5}, QuestionType.CLASSIFY)
        passed, details = validate_problem(p)
        assert details["stage1_passed"]


class TestValidateAndFilter:
    def test_some_pass(self):
        bank = generate_mini_bank()
        valid = validate_and_filter_bank(bank)
        assert len(valid) > 0
        assert len(valid) <= len(bank)

    def test_quasiperiodic_regime_survives(self):
        bank = generate_mini_bank()
        valid = validate_and_filter_bank(bank)
        regimes = {p.system_metadata.regime for p in valid}
        assert "quasiperiodic" in regimes


class TestTrackBanks:
    def test_generate_track_banks_default_excludes_identify(self):
        tracks = generate_track_banks()
        for problems in tracks.values():
            assert QuestionType.IDENTIFY not in {p.question_type for p in problems}

    def test_generate_track_banks_clean_and_noisy(self):
        tracks = generate_track_banks(
            question_types=[QuestionType.PREDICT],
            noisy_noise_stds=[0.01, 0.02],
        )
        assert set(tracks.keys()) == {"clean", "noisy_sigma_0p010", "noisy_sigma_0p020"}
        assert len(tracks["clean"]) == 12
        assert all(p.metadata["observation_mode"] == "clean" for p in tracks["clean"])
        assert all(p.metadata["noise_std"] == 0.0 for p in tracks["clean"])
        assert all(
            p.metadata["observation_mode"] == "noisy"
            for p in tracks["noisy_sigma_0p010"]
        )

    def test_build_and_freeze_track_banks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_and_freeze_track_banks(
                output_dir=tmpdir,
                question_types=[QuestionType.PREDICT],
                noisy_noise_stds=[0.01],
            )
            assert set(summary.keys()) == {"clean", "noisy_sigma_0p010"}
            for track, info in summary.items():
                assert os.path.exists(info["path"])
                loaded = load_bank(info["path"])
                assert loaded["track"] == track
                assert loaded["version"] == "2.0"


class TestFreezeAndLoad:
    def test_roundtrip(self):
        bank = generate_mini_bank()
        valid = validate_and_filter_bank(bank)
        assert len(valid) > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_bank.json")
            freeze_bank(valid, path)

            # File exists and is valid JSON
            assert os.path.exists(path)
            loaded = load_bank(path)
            assert loaded["version"] == "2.0"
            assert loaded["n_problems"] == len(valid)
            assert len(loaded["problems"]) == len(valid)

            # Check a problem roundtripped correctly
            p0 = loaded["problems"][0]
            assert "problem_id" in p0
            assert "question_type" in p0
            assert "observations" in p0
            assert isinstance(p0["observations"], list)

    def test_freeze_creates_directory(self):
        from chaosbench.problems.factory import create_problem
        p = create_problem("logistic", {"r": 3.5}, QuestionType.CLASSIFY)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "bank.json")
            freeze_bank([p], path)
            assert os.path.exists(path)
