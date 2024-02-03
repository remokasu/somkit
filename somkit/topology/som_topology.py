from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


def _to_cube_coordinates(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x3 = x - (y + (y % 2)) // 2
    z3 = y
    return x3, z3


class Topology(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def topology_function(
        self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int
    ) -> np.ndarray:
        raise NotImplementedError


class HexaglnalTopology(Topology):
    def topology_function(
        self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int
    ) -> np.ndarray:
        ax, az = _to_cube_coordinates(x1, y1)
        bx, bz = _to_cube_coordinates(x2, y2)
        return np.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


class RectangularTopology(Topology):
    def topology_function(
        self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int
    ) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
