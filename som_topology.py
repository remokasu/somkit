from __future__ import annotations

from typing import Tuple

import numpy as np


class SOMTopology:
    def __init__(self, topology: str | callable):
        self.topology_function = self._get_topology_function(topology)

    def _get_topology_function(self, topology: str | callable) -> callable:
        if callable(topology):
            return topology
        elif topology == 'rectangular':
            return self._rectangular_topology
        elif topology == 'hexagonal':
            return self._hexagonal_topology
        elif topology == 'circular':
            return self._circular_topology
        elif topology == 'ring':
            return self._ring_topology
        else:
            raise ValueError(f"Invalid topology: {topology}")

    def _rectangular_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _hexagonal_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        ax, az = self._to_cube_coordinates(x1, y1)
        bx, bz = self._to_cube_coordinates(x2, y2)
        return np.sqrt((ax - bx) ** 2 + (az - bz) ** 2)

    def _circular_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        angle1 = np.arctan2(y1 - self.y_size // 2, x1 - self.x_size // 2)
        angle2 = np.arctan2(y2 - self.y_size // 2, x2 - self.x_size // 2)
        return np.abs(angle1 - angle2)

    def _ring_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        r1, theta1 = self._to_polar_coordinates(x1, y1)
        r2, theta2 = self._to_polar_coordinates(x2, y2)
        return np.sqrt((r1 - r2) ** 2 + (theta1 - theta2) ** 2)

    def _to_cube_coordinates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x3 = x - (y + (y % 2)) // 2
        z3 = y
        # y3 = -x3 - z3  # y3 is not used in the current implementation, but this equation holds for cube coordinates.
        return x3, z3

    def _to_polar_coordinates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = x - self.x_size // 2
        dy = y - self.y_size // 2
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)
        return r, theta
