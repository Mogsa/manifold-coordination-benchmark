"""
Result logger module for the Manifold Coordination Benchmark.

This module handles saving and loading episode results to/from JSON files.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


class ResultLogger:
    """
    Handles saving and loading episode results.

    Results are saved as JSON files with timestamps and unique IDs.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the result logger.

        Args:
            output_dir: Directory to save results (created if doesn't exist)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_result(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save an episode result to JSON file.

        Args:
            result: Episode result dictionary from EpisodeRunner.run_episode()
            filename: Optional custom filename (without extension)

        Returns:
            Path to saved file
        """
        # Generate unique ID and timestamp if not in result
        if "id" not in result:
            result["id"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()

        # Generate filename if not provided
        if filename is None:
            filename = f"{result['id']}.json"
        elif not filename.endswith('.json'):
            filename = f"{filename}.json"

        # Full path
        filepath = os.path.join(self.output_dir, filename)

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        return filepath

    def load_result(self, filename: str) -> Dict[str, Any]:
        """
        Load an episode result from JSON file.

        Args:
            filename: Filename (with or without .json extension)

        Returns:
            Episode result dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Add .json if not present
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'r') as f:
            result = json.load(f)

        return result

    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all results from the output directory.

        Returns:
            List of episode result dictionaries
        """
        results = []

        # Get all JSON files
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                try:
                    result = self.load_result(filename)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        return results

    def list_results(self) -> List[str]:
        """
        List all result filenames in the output directory.

        Returns:
            List of filenames (without .json extension)
        """
        filenames = []

        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                filenames.append(filename[:-5])  # Remove .json extension

        return sorted(filenames)

    def filter_results(
        self,
        agent_a_type: Optional[str] = None,
        agent_b_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        surface_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and filter results based on criteria.

        Args:
            agent_a_type: Filter by Agent A type (e.g., "RandomAgent")
            agent_b_type: Filter by Agent B type
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            surface_name: Filter by surface name

        Returns:
            List of filtered episode result dictionaries
        """
        results = self.load_all_results()
        filtered = []

        for result in results:
            # Check filters
            if agent_a_type and result.get("agent_a_type") != agent_a_type:
                continue
            if agent_b_type and result.get("agent_b_type") != agent_b_type:
                continue
            if min_score is not None and result.get("score", 0) < min_score:
                continue
            if max_score is not None and result.get("score", 0) > max_score:
                continue
            if surface_name and result.get("surface_config", {}).get("name") != surface_name:
                continue

            filtered.append(result)

        return filtered

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all results.

        Returns:
            Dictionary with summary statistics:
                - total_runs: Total number of episodes
                - agent_types: Count of each agent type combination
                - score_stats: Mean, min, max scores
        """
        results = self.load_all_results()

        if not results:
            return {
                "total_runs": 0,
                "agent_types": {},
                "score_stats": {}
            }

        # Count agent type combinations
        agent_types = {}
        scores = []

        for result in results:
            agent_a = result.get("agent_a_type", "Unknown")
            agent_b = result.get("agent_b_type", "Unknown")
            combo = f"{agent_a} + {agent_b}"
            agent_types[combo] = agent_types.get(combo, 0) + 1

            if "score" in result:
                scores.append(result["score"])

        # Compute score statistics
        score_stats = {}
        if scores:
            score_stats = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }

        return {
            "total_runs": len(results),
            "agent_types": agent_types,
            "score_stats": score_stats
        }

    def delete_result(self, filename: str) -> bool:
        """
        Delete a result file.

        Args:
            filename: Filename (with or without .json extension)

        Returns:
            True if file was deleted, False if file didn't exist
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def clear_all_results(self, confirm: bool = False) -> int:
        """
        Delete all result files in the output directory.

        Args:
            confirm: Must be True to actually delete files (safety check)

        Returns:
            Number of files deleted
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to delete all results")

        count = 0
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.output_dir, filename)
                os.remove(filepath)
                count += 1

        return count
