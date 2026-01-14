"""
Gaussian peak utilities for the Manifold Benchmark suite.

This module contains shared functions for computing Gaussian peaks
used by both the coordination and temporal benchmarks.
"""

import numpy as np
from typing import Union


def gaussian_peak_1d(x: float, cx: float, height: float, sigma: float) -> float:
    """
    Evaluate a 1D Gaussian peak at position x.

    Args:
        x: Position to evaluate
        cx: Center of the peak
        height: Peak height (amplitude)
        sigma: Peak width (standard deviation)

    Returns:
        Value of the Gaussian at position x
    """
    return height * np.exp(-((x - cx) ** 2) / (2 * sigma ** 2))


def gaussian_peak_2d(
    x: float, y: float, cx: float, cy: float, height: float, sigma: float
) -> float:
    """
    Evaluate a 2D Gaussian peak at position (x, y).

    Args:
        x: x-coordinate
        y: y-coordinate
        cx: x-coordinate of peak center
        cy: y-coordinate of peak center
        height: Peak height (amplitude)
        sigma: Peak width (standard deviation)

    Returns:
        Value of the Gaussian at position (x, y)
    """
    return height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def gaussian_peak_from_dict_2d(x: float, y: float, peak: dict) -> float:
    """
    Evaluate a 2D Gaussian peak at (x, y) using peak definition dict.

    Args:
        x: x-coordinate
        y: y-coordinate
        peak: Peak definition with keys: cx, cy, height, sigma

    Returns:
        Value of the Gaussian at position (x, y)
    """
    return gaussian_peak_2d(
        x, y, peak['cx'], peak['cy'], peak['height'], peak['sigma']
    )


def gaussian_peak_from_dict_1d(x: float, peak: dict) -> float:
    """
    Evaluate a 1D Gaussian peak at x using peak definition dict.

    Args:
        x: Position to evaluate
        peak: Peak definition with keys: cx, height, sigma

    Returns:
        Value of the Gaussian at position x
    """
    return gaussian_peak_1d(x, peak['cx'], peak['height'], peak['sigma'])
