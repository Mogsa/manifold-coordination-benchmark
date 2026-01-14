"""
Tests for the Result Logger module.
"""

import pytest
import os
import json
import tempfile
import shutil
from coordination.experiments.logger import ResultLogger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test results."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def logger(temp_dir):
    """Create a ResultLogger with temporary directory."""
    return ResultLogger(output_dir=temp_dir)


@pytest.fixture
def sample_result():
    """Create a sample episode result."""
    return {
        "surface_config": {"name": "test_surface"},
        "agent_a_type": "RandomAgent",
        "agent_b_type": "GreedyAgent",
        "n_turns": 5,
        "score": 0.75,
        "final_decision": [7.0, 8.0]
    }


def test_logger_creates_output_dir():
    """Test that logger creates output directory if it doesn't exist."""
    temp_path = tempfile.mkdtemp()
    output_dir = os.path.join(temp_path, "new_results")

    assert not os.path.exists(output_dir)
    logger = ResultLogger(output_dir=output_dir)
    assert os.path.exists(output_dir)

    # Cleanup
    shutil.rmtree(temp_path)


def test_save_result(logger, sample_result, temp_dir):
    """Test saving a result to JSON."""
    filepath = logger.save_result(sample_result)

    # Check file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.json')

    # Check file contains correct data
    with open(filepath, 'r') as f:
        loaded = json.load(f)

    assert loaded["agent_a_type"] == "RandomAgent"
    assert loaded["score"] == 0.75
    assert "id" in loaded
    assert "timestamp" in loaded


def test_save_result_with_custom_filename(logger, sample_result, temp_dir):
    """Test saving with custom filename."""
    filepath = logger.save_result(sample_result, filename="custom_name")

    assert os.path.basename(filepath) == "custom_name.json"


def test_save_result_adds_id_and_timestamp(logger, sample_result):
    """Test that save_result adds ID and timestamp."""
    filepath = logger.save_result(sample_result)
    loaded = logger.load_result(os.path.basename(filepath)[:-5])

    assert "id" in loaded
    assert "timestamp" in loaded
    assert loaded["id"].startswith("run_")


def test_load_result(logger, sample_result):
    """Test loading a result from JSON."""
    # Save first
    filepath = logger.save_result(sample_result, filename="test_load")

    # Load
    loaded = logger.load_result("test_load")

    assert loaded["agent_a_type"] == "RandomAgent"
    assert loaded["score"] == 0.75


def test_load_result_with_json_extension(logger, sample_result):
    """Test loading works with .json extension."""
    logger.save_result(sample_result, filename="test")

    # Should work with or without .json
    loaded1 = logger.load_result("test")
    loaded2 = logger.load_result("test.json")

    assert loaded1 == loaded2


def test_load_result_nonexistent_raises(logger):
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        logger.load_result("nonexistent")


def test_load_all_results(logger, sample_result):
    """Test loading all results."""
    # Save multiple results
    result1 = sample_result.copy()
    result1["score"] = 0.5
    result2 = sample_result.copy()
    result2["score"] = 0.8
    result3 = sample_result.copy()
    result3["score"] = 0.9

    logger.save_result(result1, filename="result1")
    logger.save_result(result2, filename="result2")
    logger.save_result(result3, filename="result3")

    # Load all
    results = logger.load_all_results()

    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert 0.5 in scores
    assert 0.8 in scores
    assert 0.9 in scores


def test_list_results(logger, sample_result):
    """Test listing result filenames."""
    logger.save_result(sample_result, filename="result1")
    logger.save_result(sample_result, filename="result2")

    filenames = logger.list_results()

    assert len(filenames) == 2
    assert "result1" in filenames
    assert "result2" in filenames
    # Should not include .json extension
    assert not any(f.endswith('.json') for f in filenames)


def test_filter_results_by_agent_type(logger):
    """Test filtering results by agent type."""
    result1 = {"agent_a_type": "RandomAgent", "agent_b_type": "GreedyAgent", "score": 0.5}
    result2 = {"agent_a_type": "GreedyAgent", "agent_b_type": "GreedyAgent", "score": 0.7}
    result3 = {"agent_a_type": "RandomAgent", "agent_b_type": "RandomAgent", "score": 0.3}

    logger.save_result(result1, filename="r1")
    logger.save_result(result2, filename="r2")
    logger.save_result(result3, filename="r3")

    # Filter by agent_a_type
    filtered = logger.filter_results(agent_a_type="RandomAgent")
    assert len(filtered) == 2

    # Filter by agent_b_type
    filtered = logger.filter_results(agent_b_type="GreedyAgent")
    assert len(filtered) == 2


def test_filter_results_by_score(logger):
    """Test filtering results by score range."""
    result1 = {"score": 0.3, "agent_a_type": "RandomAgent", "agent_b_type": "RandomAgent"}
    result2 = {"score": 0.7, "agent_a_type": "GreedyAgent", "agent_b_type": "GreedyAgent"}
    result3 = {"score": 0.9, "agent_a_type": "LLMAgent", "agent_b_type": "LLMAgent"}

    logger.save_result(result1, filename="r1")
    logger.save_result(result2, filename="r2")
    logger.save_result(result3, filename="r3")

    # Filter by min_score
    filtered = logger.filter_results(min_score=0.5)
    assert len(filtered) == 2
    assert all(r["score"] >= 0.5 for r in filtered)

    # Filter by max_score
    filtered = logger.filter_results(max_score=0.8)
    assert len(filtered) == 2
    assert all(r["score"] <= 0.8 for r in filtered)

    # Filter by range
    filtered = logger.filter_results(min_score=0.5, max_score=0.8)
    assert len(filtered) == 1
    assert filtered[0]["score"] == 0.7


def test_filter_results_by_surface_name(logger):
    """Test filtering results by surface name."""
    result1 = {"surface_config": {"name": "single_peak"}, "score": 0.5}
    result2 = {"surface_config": {"name": "two_peaks"}, "score": 0.7}
    result3 = {"surface_config": {"name": "single_peak"}, "score": 0.9}

    logger.save_result(result1, filename="r1")
    logger.save_result(result2, filename="r2")
    logger.save_result(result3, filename="r3")

    filtered = logger.filter_results(surface_name="single_peak")
    assert len(filtered) == 2


def test_get_summary_stats(logger):
    """Test getting summary statistics."""
    result1 = {"agent_a_type": "RandomAgent", "agent_b_type": "GreedyAgent", "score": 0.5}
    result2 = {"agent_a_type": "RandomAgent", "agent_b_type": "GreedyAgent", "score": 0.7}
    result3 = {"agent_a_type": "GreedyAgent", "agent_b_type": "GreedyAgent", "score": 0.9}

    logger.save_result(result1, filename="r1")
    logger.save_result(result2, filename="r2")
    logger.save_result(result3, filename="r3")

    stats = logger.get_summary_stats()

    assert stats["total_runs"] == 3
    assert "RandomAgent + GreedyAgent" in stats["agent_types"]
    assert stats["agent_types"]["RandomAgent + GreedyAgent"] == 2
    assert stats["score_stats"]["mean"] == pytest.approx(0.7, abs=0.01)
    assert stats["score_stats"]["min"] == 0.5
    assert stats["score_stats"]["max"] == 0.9


def test_get_summary_stats_empty(logger):
    """Test summary stats with no results."""
    stats = logger.get_summary_stats()

    assert stats["total_runs"] == 0
    assert stats["agent_types"] == {}
    assert stats["score_stats"] == {}


def test_delete_result(logger, sample_result):
    """Test deleting a result file."""
    logger.save_result(sample_result, filename="to_delete")

    assert "to_delete" in logger.list_results()

    deleted = logger.delete_result("to_delete")
    assert deleted is True
    assert "to_delete" not in logger.list_results()


def test_delete_result_nonexistent(logger):
    """Test deleting nonexistent file returns False."""
    deleted = logger.delete_result("nonexistent")
    assert deleted is False


def test_clear_all_results(logger, sample_result):
    """Test clearing all results."""
    logger.save_result(sample_result, filename="r1")
    logger.save_result(sample_result, filename="r2")
    logger.save_result(sample_result, filename="r3")

    assert len(logger.list_results()) == 3

    count = logger.clear_all_results(confirm=True)
    assert count == 3
    assert len(logger.list_results()) == 0


def test_clear_all_results_requires_confirmation(logger, sample_result):
    """Test that clear_all_results requires confirmation."""
    logger.save_result(sample_result, filename="r1")

    with pytest.raises(ValueError, match="Must pass confirm=True"):
        logger.clear_all_results(confirm=False)

    # File should still exist
    assert len(logger.list_results()) == 1
