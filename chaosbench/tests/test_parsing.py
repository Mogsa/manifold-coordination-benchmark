"""Tests for ChaosBench v2 response parsing."""

import pytest

from chaosbench.agents.parsing import (
    parse_classify,
    parse_confidence,
    parse_identify,
    parse_predict,
    parse_response,
)


# --- CLASSIFY ---

class TestParseClassify:
    def test_answer_line(self):
        resp = "The system shows...\nANSWER: chaotic"
        assert parse_classify(resp) == "chaotic"

    def test_answer_line_fixed_point(self):
        resp = "Converges to a single value.\nANSWER: fixed_point"
        assert parse_classify(resp) == "fixed_point"

    def test_answer_line_periodic(self):
        resp = "Oscillates regularly.\nAnswer: periodic"
        assert parse_classify(resp) == "periodic"

    def test_answer_line_quasiperiodic(self):
        resp = "Almost periodic but never repeats.\nANSWER: quasiperiodic"
        assert parse_classify(resp) == "quasiperiodic"

    def test_fallback_to_body(self):
        resp = "This looks chaotic because the variance is high."
        assert parse_classify(resp) == "chaotic"

    def test_last_match_wins_in_body(self):
        resp = "Could be periodic but actually looks chaotic."
        assert parse_classify(resp) == "chaotic"

    def test_answer_line_beats_body(self):
        resp = "Reasoning mentions chaotic...\nANSWER: periodic"
        assert parse_classify(resp) == "periodic"

    def test_empty_response(self):
        assert parse_classify("") == ""

    def test_no_match(self):
        assert parse_classify("I have no idea what this is.") == ""

    def test_case_insensitive(self):
        assert parse_classify("ANSWER: CHAOTIC") == "chaotic"


# --- IDENTIFY ---

class TestParseIdentify:
    def test_answer_line(self):
        resp = "The return map is parabolic.\nANSWER: logistic"
        assert parse_identify(resp) == "logistic"

    def test_tent(self):
        resp = "V-shaped return map.\nANSWER: tent"
        assert parse_identify(resp) == "tent"

    def test_damped_linear(self):
        resp = "Exponential decay.\nANSWER: damped_linear"
        assert parse_identify(resp) == "damped_linear"

    def test_rotation(self):
        resp = "Irrational rotation.\nANSWER: rotation"
        assert parse_identify(resp) == "rotation"

    def test_fallback(self):
        resp = "This is clearly a tent map based on the shape."
        assert parse_identify(resp) == "tent"

    def test_empty(self):
        assert parse_identify("") == ""

    def test_no_match(self):
        assert parse_identify("Sine wave maybe?") == ""


# --- PREDICT ---

class TestParsePredict:
    def test_answer_line_csv(self):
        resp = "ANSWER: 0.5, 0.6, 0.7, 0.8, 0.9"
        assert parse_predict(resp, 5) == [0.5, 0.6, 0.7, 0.8, 0.9]

    def test_bracketed_list(self):
        resp = "My prediction: [0.1, 0.2, 0.3]"
        assert parse_predict(resp, 3) == [0.1, 0.2, 0.3]

    def test_truncate(self):
        resp = "ANSWER: 1.0, 2.0, 3.0, 4.0, 5.0"
        assert parse_predict(resp, 3) == [1.0, 2.0, 3.0]

    def test_pad(self):
        resp = "ANSWER: 0.5, 0.6"
        result = parse_predict(resp, 5)
        assert len(result) == 5
        assert result[:2] == [0.5, 0.6]
        assert result[2:] == [0.6, 0.6, 0.6]  # Padded with last value

    def test_negative_numbers(self):
        resp = "ANSWER: -0.5, 0.3, -1.2"
        assert parse_predict(resp, 3) == [-0.5, 0.3, -1.2]

    def test_fallback_to_body_floats(self):
        resp = "I think the next values will be around 0.5 and 0.6 and 0.7"
        result = parse_predict(resp, 3)
        assert len(result) == 3
        assert 0.5 in result

    def test_empty_response(self):
        assert parse_predict("", 5) == []

    def test_no_numbers(self):
        assert parse_predict("I cannot predict.", 5) == []


# --- CONFIDENCE ---

class TestParseConfidence:
    def test_explicit(self):
        resp = "ANSWER: chaotic\nconfidence: 0.85"
        assert parse_confidence(resp) == 0.85

    def test_caps(self):
        assert parse_confidence("Confidence: 0.7") == 0.7

    def test_clamp_high(self):
        assert parse_confidence("confidence: 1.5") == 1.0

    def test_clamp_low(self):
        assert parse_confidence("confidence: -0.5") == 0.0

    def test_default(self):
        assert parse_confidence("no confidence here") == 0.5

    def test_empty(self):
        assert parse_confidence("") == 0.5


# --- DISPATCHER ---

class TestParseResponse:
    def test_classify_dispatch(self):
        assert parse_response("ANSWER: chaotic", "classify", {}) == "chaotic"

    def test_identify_dispatch(self):
        assert parse_response("ANSWER: tent", "identify", {}) == "tent"

    def test_predict_dispatch(self):
        result = parse_response("ANSWER: 0.1, 0.2, 0.3", "predict", {"horizon": 3})
        assert result == [0.1, 0.2, 0.3]

    def test_unknown_type(self):
        assert parse_response("whatever", "bogus", {}) == ""
