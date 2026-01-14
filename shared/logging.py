"""
Result logging utilities for the Manifold Benchmark suite.

This module provides base logging functionality that can be used
by both coordination and temporal benchmarks.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


def generate_result_id(prefix: str = "run") -> str:
    """
    Generate a unique result ID with timestamp.

    Args:
        prefix: Prefix for the ID (e.g., 'run', 'episode', 'trial')

    Returns:
        Unique ID string like 'run_20260113_143022_abc123'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_suffix = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{unique_suffix}"


def save_json_result(
    result: Dict[str, Any],
    output_dir: str,
    filename: Optional[str] = None,
    id_prefix: str = "run"
) -> str:
    """
    Save a result dictionary to a JSON file.

    Automatically adds 'id' and 'timestamp' fields if not present.

    Args:
        result: Result dictionary to save
        output_dir: Directory to save to (created if doesn't exist)
        filename: Optional custom filename (without extension)
        id_prefix: Prefix for auto-generated ID

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add ID and timestamp if not present
    if "id" not in result:
        result["id"] = generate_result_id(id_prefix)

    if "timestamp" not in result:
        result["timestamp"] = datetime.now().isoformat()

    # Generate filename if not provided
    if filename is None:
        filename = f"{result['id']}.json"
    elif not filename.endswith('.json'):
        filename = f"{filename}.json"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    return filepath


def load_json_result(filepath: str) -> Dict[str, Any]:
    """
    Load a result dictionary from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded result dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_json_results(directory: str) -> List[Dict[str, Any]]:
    """
    Load all JSON result files from a directory.

    Args:
        directory: Directory containing JSON files

    Returns:
        List of result dictionaries
    """
    results = []

    if not os.path.exists(directory):
        return results

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            try:
                filepath = os.path.join(directory, filename)
                result = load_json_result(filepath)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    return results


def list_json_results(directory: str) -> List[str]:
    """
    List all JSON result filenames in a directory.

    Args:
        directory: Directory to list

    Returns:
        Sorted list of filenames (without .json extension)
    """
    filenames = []

    if not os.path.exists(directory):
        return filenames

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filenames.append(filename[:-5])

    return sorted(filenames)


class BaseResultLogger:
    """
    Base class for result logging.

    Provides common functionality for saving/loading JSON results.
    Subclass this for benchmark-specific logging.
    """

    def __init__(self, output_dir: str = "results", id_prefix: str = "run"):
        """
        Initialize the logger.

        Args:
            output_dir: Directory to save results
            id_prefix: Prefix for auto-generated IDs
        """
        self.output_dir = output_dir
        self.id_prefix = id_prefix
        os.makedirs(output_dir, exist_ok=True)

    def save_result(
        self,
        result: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Save a result to JSON file.

        Args:
            result: Result dictionary
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        return save_json_result(
            result, self.output_dir, filename, self.id_prefix
        )

    def load_result(self, filename: str) -> Dict[str, Any]:
        """
        Load a result from JSON file.

        Args:
            filename: Filename (with or without .json extension)

        Returns:
            Result dictionary
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        filepath = os.path.join(self.output_dir, filename)
        return load_json_result(filepath)

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all results from output directory."""
        return load_all_json_results(self.output_dir)

    def list_results(self) -> List[str]:
        """List all result filenames."""
        return list_json_results(self.output_dir)

    def delete_result(self, filename: str) -> bool:
        """
        Delete a result file.

        Args:
            filename: Filename (with or without .json extension)

        Returns:
            True if deleted, False if didn't exist
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
