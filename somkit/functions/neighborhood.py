from typing import Callable

import numpy as np


def gaussian(radius: float, distance: np.ndarray, neighborhood_radius: float) -> float:
    return np.exp(-(distance**2) / (2 * neighborhood_radius**2))


def mexican_hat(radius: float, distance: float, neighborhood_radius: float) -> float:
    return (1 - (distance**2) / (neighborhood_radius**2)) * np.exp(
        -(distance**2) / (2 * neighborhood_radius**2)
    )


def bubble(
    radius: np.float64, distance: np.ndarray, neighborhood_radius: float
) -> float:
    return np.where(distance <= radius, 1, 0)


def cone(radius: float, distance: np.ndarray, neighborhood_radius: float) -> float:
    return np.where(distance <= radius, 1 - (distance / radius), 0)
