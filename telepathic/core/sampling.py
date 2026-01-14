"""
Sample generation for the Telepathic benchmark.

Generates input-output samples for the Seer and test points for evaluation.
Fixed points ensure consistency across trials.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Callable

from .functions import PrimitiveFunction, ComposedFunction


# Fixed sample points (what the Seer sees)
SAMPLE_X_VALUES: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Fixed test points (what the Doer must compute)
TEST_X_VALUES: List[float] = [-1.5, 0.5, 1.5]


@dataclass
class Sample:
    """A single input-output sample."""
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class SampleSet:
    """Collection of samples for a function."""
    samples: List[Sample]
    function_name: str

    def to_list(self) -> List[Tuple[float, float]]:
        """Return as list of (x, y) tuples."""
        return [(s.x, s.y) for s in self.samples]

    def format_for_prompt(self) -> str:
        """Format samples as text for LLM prompt."""
        lines = []
        for s in self.samples:
            lines.append(f"  f({s.x:.1f}) = {s.y:.4f}")
        return "\n".join(lines)


class SampleGenerator:
    """Generates input-output samples for functions."""

    def __init__(
        self,
        function: Union[PrimitiveFunction, ComposedFunction, Callable[[float], float]],
        function_name: str = "unknown"
    ):
        """
        Initialize generator.

        Args:
            function: The function to sample (callable)
            function_name: Name for logging
        """
        self.function = function
        self.function_name = function_name

    def generate_samples(self) -> SampleSet:
        """
        Generate training samples at fixed points.

        Returns:
            SampleSet with 5 samples at [-2, -1, 0, 1, 2]
        """
        samples = []
        for x in SAMPLE_X_VALUES:
            y = self.function(x)
            samples.append(Sample(x=x, y=y))
        return SampleSet(samples=samples, function_name=self.function_name)

    def generate_tests(self) -> List[Tuple[float, float]]:
        """
        Generate test cases at fixed points.

        Returns:
            List of (x, y) tuples at [-1.5, 0.5, 1.5]
        """
        return [(x, self.function(x)) for x in TEST_X_VALUES]

    def generate_single_test(self, test_index: int = 0) -> Tuple[float, float]:
        """
        Generate a single test case.

        Args:
            test_index: Which test point (0, 1, or 2)

        Returns:
            (x, y) tuple
        """
        x = TEST_X_VALUES[test_index]
        return (x, self.function(x))


def generate_samples_for_function(
    function: Union[PrimitiveFunction, ComposedFunction],
) -> SampleSet:
    """
    Convenience function to generate samples.

    Args:
        function: Function to sample

    Returns:
        SampleSet
    """
    if isinstance(function, PrimitiveFunction):
        name = function.name
    elif isinstance(function, ComposedFunction):
        name = function.description
    else:
        name = "unknown"

    gen = SampleGenerator(function, name)
    return gen.generate_samples()


def verify_no_overlap() -> bool:
    """Verify that sample and test points don't overlap."""
    sample_set = set(SAMPLE_X_VALUES)
    test_set = set(TEST_X_VALUES)
    return sample_set.isdisjoint(test_set)
