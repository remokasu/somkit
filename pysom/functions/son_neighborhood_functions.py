from typing import Callable

import numpy as np


def gaussian(radius: float, distance: np.ndarray, neighborhood_width: float) -> float:
    return np.exp(-distance ** 2 / (2 * neighborhood_width ** 2))


def mexican_hat(radius: float, distance: float, neighborhood_width: float) -> float:
    return (1 - (distance ** 2) / (neighborhood_width ** 2)) * np.exp(-distance ** 2 / (2 * neighborhood_width ** 2))


def bubble(radius: np.float64, distance: np.ndarray, neighborhood_width: float) -> float:
    return (distance <= radius).astype(float)


def cone(radius: float, distance: np.ndarray, neighborhood_width: float) -> float:
    return np.where(distance <= radius, 1 - (distance / radius), 0)


def create_neighborhood_function(name: str) -> Callable[[float, float, float], float]:
    if name.lower() == "gaussian":
        return gaussian
    elif name.lower() == "mexican_hat":
        return mexican_hat
    elif name.lower() == "bubble":
        return bubble
    elif name.lower() == "cone":
        return cone
    else:
        raise ValueError(f"Invalid neighborhood function name: {name}")
