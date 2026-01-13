"""
Tests for the Episode class.

Tests verify episode initialization, turn progression, scoring,
and history tracking.
"""

import pytest
from manifold_benchmark.core.surface import Surface
from manifold_benchmark.core.observation import ObservationGenerator
from manifold_benchmark.core.episode import Episode


def test_episode_initialization():
    """Episode starts at correct position."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=10)
    assert episode.current_position == (5.0, 5.0)
    assert episode.turn_number == 0


def test_episode_step():
    """Step updates position and returns observations."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=10)
    obs_a, obs_b = episode.step(6.0, 7.0)
    assert episode.current_position == (6.0, 7.0)
    assert episode.turn_number == 1
    assert "gradient_x" in obs_a
    assert "gradient_y" in obs_b


def test_episode_completion():
    """Episode is done after N turns."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)
    for i in range(3):
        assert not episode.is_done()
        episode.step(5.0, 5.0)
    assert episode.is_done()


def test_perfect_score():
    """Score is 1.0 when finding exact optimal."""
    surface = Surface([{"cx": 7.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)
    score = episode.get_score(7.0, 8.0)
    assert abs(score - 1.0) < 0.01


def test_score_at_suboptimal_position():
    """Score is less than 1.0 at suboptimal position."""
    surface = Surface([{"cx": 7.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)

    # Far from optimal
    score_far = episode.get_score(0.0, 0.0)
    assert score_far < 0.5

    # Near optimal (distance ~0.707 from peak)
    score_near = episode.get_score(6.5, 7.5)
    assert 0.7 < score_near < 1.0


def test_history_tracking():
    """History is recorded correctly."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(5.0, 5.0), n_turns=3)

    # Take 3 steps
    episode.step(5.5, 5.5)
    episode.step(6.0, 6.0)
    episode.step(6.5, 6.5)

    # Check history
    history = episode.get_history()
    assert len(history) == 3

    # Check first turn
    assert history[0]["turn"] == 1
    assert history[0]["position"] == (5.5, 5.5)
    assert "observation_a" in history[0]
    assert "observation_b" in history[0]

    # Check last turn
    assert history[2]["turn"] == 3
    assert history[2]["position"] == (6.5, 6.5)


def test_get_state():
    """get_state returns current position and turn."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(3.0, 4.0), n_turns=10)

    state = episode.get_state()
    assert state["position"] == (3.0, 4.0)
    assert state["turn_number"] == 0

    # After one step
    episode.step(5.0, 6.0)
    state = episode.get_state()
    assert state["position"] == (5.0, 6.0)
    assert state["turn_number"] == 1


def test_custom_observation_generator():
    """Episode accepts custom observation generator."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    obs_gen = ObservationGenerator(surface, radius=2.0, n_samples=15)
    episode = Episode(surface, observation_generator=obs_gen)

    obs_a, obs_b = episode.step(5.0, 5.0)

    # Should use custom parameters
    assert len(obs_a["slice"]) == 15
    assert len(obs_b["slice"]) == 15


def test_default_initial_position():
    """Episode uses default initial position (5.0, 5.0)."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)
    assert episode.current_position == (5.0, 5.0)


def test_default_n_turns():
    """Episode uses default n_turns=10."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)

    # Should complete after 10 turns
    for i in range(10):
        assert not episode.is_done()
        episode.step(5.0, 5.0)
    assert episode.is_done()


def test_turn_number_increments():
    """Turn number increments correctly."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, n_turns=5)

    assert episode.turn_number == 0

    for expected_turn in range(1, 6):
        episode.step(5.0, 5.0)
        assert episode.turn_number == expected_turn


def test_observations_match_position():
    """Observations reflect the new position after step."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface, initial_position=(3.0, 3.0))

    obs_a, obs_b = episode.step(7.0, 8.0)

    # Observations should be at the new position (7.0, 8.0)
    assert obs_a["position"]["x"] == 7.0
    assert obs_a["position"]["y"] == 8.0
    assert obs_b["position"]["x"] == 7.0
    assert obs_b["position"]["y"] == 8.0


def test_score_with_multiple_peaks():
    """Score works correctly with multiple peaks."""
    surface = Surface([
        {"cx": 2.0, "cy": 2.0, "height": 0.5, "sigma": 1.0},
        {"cx": 8.0, "cy": 8.0, "height": 1.0, "sigma": 1.0}
    ])
    episode = Episode(surface)

    # At global maximum
    score_optimal = episode.get_score(8.0, 8.0)
    assert abs(score_optimal - 1.0) < 0.01

    # At secondary peak
    score_secondary = episode.get_score(2.0, 2.0)
    assert 0.4 < score_secondary < 0.6


def test_history_is_copy():
    """get_history returns a copy, not reference."""
    surface = Surface([{"cx": 5.0, "cy": 5.0, "height": 1.0, "sigma": 1.0}])
    episode = Episode(surface)

    episode.step(5.0, 5.0)
    history1 = episode.get_history()

    episode.step(6.0, 6.0)
    history2 = episode.get_history()

    # First history should not have been modified
    assert len(history1) == 1
    assert len(history2) == 2
