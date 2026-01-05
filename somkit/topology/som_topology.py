from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from matplotlib.patches import Rectangle, RegularPolygon


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

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the topology."""
        raise NotImplementedError

    @abstractmethod
    def get_visualization_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Get the visualization coordinates for a node at grid position (x, y)."""
        raise NotImplementedError

    @abstractmethod
    def create_patch(self, x: float, y: float, size: float, **kwargs):
        """Create a matplotlib patch for visualization at position (x, y)."""
        raise NotImplementedError

    @abstractmethod
    def get_map_dimensions(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Get the actual dimensions of the map for visualization."""
        raise NotImplementedError


class HexagonalTopology(Topology):
    def topology_function(
        self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int
    ) -> np.ndarray:
        ax, az = _to_cube_coordinates(x1, y1)
        bx, bz = _to_cube_coordinates(x2, y2)
        return np.sqrt((ax - bx) ** 2 + (az - bz) ** 2)

    def get_name(self) -> str:
        return "hexagonal"

    def get_visualization_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Get hexagonal grid coordinates with offset for odd rows."""
        hex_height_coeff = np.sqrt(3) / 2
        vis_x = x + 0.5 * (y % 2)
        vis_y = y * hex_height_coeff
        return vis_x, vis_y

    def create_patch(self, x: float, y: float, size: float = 0.58, **kwargs):
        """Create a hexagonal patch."""
        return RegularPolygon(
            (x, y),
            numVertices=6,
            radius=size,
            orientation=np.radians(0),
            **kwargs
        )

    def get_map_dimensions(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Get dimensions for hexagonal map."""
        hex_height_coeff = np.sqrt(3) / 2
        width = grid_x + 0.5 * (grid_y % 2)
        height = grid_y * hex_height_coeff
        return width, height


class RectangularTopology(Topology):
    def topology_function(
        self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int
    ) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_name(self) -> str:
        return "rectangular"

    def get_visualization_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Get rectangular grid coordinates (no offset needed)."""
        return float(x), float(y)

    def create_patch(self, x: float, y: float, size: float = 0.9, **kwargs):
        """Create a rectangular patch."""
        # Center the rectangle at (x, y)
        return Rectangle(
            (x - size/2, y - size/2),
            width=size,
            height=size,
            **kwargs
        )

    def get_map_dimensions(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Get dimensions for rectangular map."""
        return float(grid_x), float(grid_y)
