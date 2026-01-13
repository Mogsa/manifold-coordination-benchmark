"""
Surface module for the Manifold Coordination Benchmark.

This module defines 2D surfaces as sums of Gaussian peaks for the benchmark.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize


class Surface:
    """A 2D surface defined by Gaussian peaks."""

    def __init__(self, peaks: List[dict], domain_size: float = 10.0):
        """
        Initialize a surface with multiple Gaussian peaks.

        Args:
            peaks: List of peak definitions, each with keys:
                   - cx: float, x-coordinate of peak center
                   - cy: float, y-coordinate of peak center
                   - height: float, peak height (typically 0-1)
                   - sigma: float, peak width (standard deviation)
            domain_size: Size of domain [0, domain_size] × [0, domain_size]
        """
        self.peaks = peaks
        self.domain_size = domain_size

        # Validate peak definitions
        for i, peak in enumerate(peaks):
            required_keys = {'cx', 'cy', 'height', 'sigma'}
            if not all(key in peak for key in required_keys):
                raise ValueError(
                    f"Peak {i} missing required keys. Expected {required_keys}, "
                    f"got {set(peak.keys())}"
                )

    def evaluate(self, x: float, y: float) -> float:
        """
        Evaluate surface at point (x, y).

        Args:
            x: x-coordinate
            y: y-coordinate

        Returns:
            Surface value f(x, y) as sum of all Gaussian peaks
        """
        value = 0.0
        for peak in self.peaks:
            value += self._gaussian_peak(x, y, peak)
        return value

    def _gaussian_peak(self, x: float, y: float, peak: dict) -> float:
        """
        Evaluate single Gaussian peak at point (x, y).

        Args:
            x: x-coordinate
            y: y-coordinate
            peak: Peak definition dict

        Returns:
            Gaussian peak value at (x, y)
        """
        cx = peak['cx']
        cy = peak['cy']
        height = peak['height']
        sigma = peak['sigma']

        return height * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    def gradient(self, x: float, y: float, eps: float = 0.001) -> Tuple[float, float]:
        """
        Compute gradient (∂f/∂x, ∂f/∂y) at point (x, y) using numerical differentiation.

        Args:
            x: x-coordinate
            y: y-coordinate
            eps: Small perturbation for numerical differentiation

        Returns:
            Tuple (∂f/∂x, ∂f/∂y) representing the gradient
        """
        # Compute partial derivative with respect to x
        df_dx = (self.evaluate(x + eps, y) - self.evaluate(x - eps, y)) / (2 * eps)

        # Compute partial derivative with respect to y
        df_dy = (self.evaluate(x, y + eps) - self.evaluate(x, y - eps)) / (2 * eps)

        return (df_dx, df_dy)

    def get_optimal(self) -> Tuple[float, float, float]:
        """
        Find the global maximum of the surface.

        Uses scipy.optimize.minimize to find the maximum by minimizing the negative
        of the surface function.

        Returns:
            Tuple (x_opt, y_opt, f_opt) where:
                - x_opt: x-coordinate of global maximum
                - y_opt: y-coordinate of global maximum
                - f_opt: surface value at global maximum
        """
        # Define objective function (negative for minimization)
        def objective(p):
            return -self.evaluate(p[0], p[1])

        # Try multiple starting points to avoid local maxima
        best_result = None
        best_value = float('-inf')

        # Starting points: center + each peak center
        starting_points = [(self.domain_size / 2, self.domain_size / 2)]
        for peak in self.peaks:
            starting_points.append((peak['cx'], peak['cy']))

        # Optimize from each starting point
        for x0, y0 in starting_points:
            result = minimize(
                objective,
                x0=[x0, y0],
                bounds=[(0, self.domain_size), (0, self.domain_size)],
                method='L-BFGS-B'
            )

            if -result.fun > best_value:
                best_value = -result.fun
                best_result = result

        return (best_result.x[0], best_result.x[1], -best_result.fun)

    def to_dict(self) -> dict:
        """
        Serialize surface configuration to dictionary.

        Returns:
            Dictionary containing peaks and domain_size
        """
        return {
            'peaks': self.peaks,
            'domain_size': self.domain_size
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'Surface':
        """
        Deserialize surface from configuration dictionary.

        Args:
            config: Dictionary with 'peaks' and 'domain_size' keys

        Returns:
            Surface instance
        """
        return cls(
            peaks=config['peaks'],
            domain_size=config.get('domain_size', 10.0)
        )
