from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from matplotlib.patches import Rectangle, RegularPolygon


# SOM_PAK hexa_dist constants (som_rout.c:434-455).
# x-offset applied between rows of differing parity.
_HEX_ROW_OFFSET = 0.5
# vertical scaling (sqrt(3)/2)**2 for the hexagonal row spacing.
_HEX_VERTICAL_SQ = 0.75


class Topology(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def topology_function(
        self,
        x1: int | np.ndarray,
        y1: int | np.ndarray,
        x2: int | np.ndarray,
        y2: int | np.ndarray,
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

    @abstractmethod
    def get_visualization_coords_array(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`get_visualization_coords` for index arrays.

        Args:
            x: Grid x-indices (any shape).
            y: Grid y-indices (same shape as ``x``).

        Returns:
            ``(vis_x, vis_y)`` arrays with the same shape as the inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def patch_vertices(self) -> np.ndarray:
        """Cell-outline vertices relative to the cell center.

        Used to draw all cells of a map as one
        :class:`~matplotlib.collections.PolyCollection` instead of one patch
        object per cell. Must describe the same shape as :meth:`create_patch`
        with its default size.

        Returns:
            A ``(n_vertices, 2)`` float array.
        """
        raise NotImplementedError


class HexagonalTopology(Topology):
    def topology_function(
        self,
        x1: int | np.ndarray,
        y1: int | np.ndarray,
        x2: int | np.ndarray,
        y2: int | np.ndarray,
    ) -> np.ndarray:
        """Grid distance matching SOM_PAK ``hexa_dist`` (som_rout.c:434-455).

        ``(x2, y2)`` is the BMU and ``(x1, y1)`` the target unit(s). When the
        rows differ in parity, the x-difference is shifted by ±0.5 depending on
        the BMU row's parity, then ``dist = sqrt(diff^2 + 0.75 * dy^2)``.

        Args:
            x1: Target x-coordinate(s) (scalar or array).
            y1: Target y-coordinate(s) (scalar or array).
            x2: BMU x-coordinate.
            y2: BMU y-coordinate.

        Returns:
            The hexagonal grid distance(s).
        """
        # SOM_PAK hexa_dist(bx=x2, by=y2, tx=x1, ty=y1). The offset sign depends
        # on the BMU row parity (y2); ``np.where`` keeps this vectorized so y2
        # may be a scalar (som_step) or an array of BMUs (batch update).
        diff = np.asarray(x2 - x1, dtype=np.float64)
        row_diff = np.asarray(y2 - y1)
        offset = np.where(np.asarray(y2) % 2 == 0, -_HEX_ROW_OFFSET, _HEX_ROW_OFFSET)
        diff = np.where(row_diff % 2 != 0, diff + offset, diff)
        return np.sqrt(diff ** 2 + _HEX_VERTICAL_SQ * row_diff.astype(np.float64) ** 2)

    def get_name(self) -> str:
        return "hexagonal"

    def get_visualization_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Get hexagonal grid coordinates with offset for odd rows."""
        hex_height_coeff = np.sqrt(3) / 2
        vis_x = x + 0.5 * (y % 2)
        vis_y = y * hex_height_coeff
        return vis_x, vis_y

    def get_visualization_coords_array(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized variant of :meth:`get_visualization_coords`."""
        hex_height_coeff = np.sqrt(3) / 2
        return x + 0.5 * (y % 2), y * hex_height_coeff

    def create_patch(self, x: float, y: float, size: float = 0.58, **kwargs):
        """Create a hexagonal patch."""
        return RegularPolygon(
            (x, y),
            numVertices=6,
            radius=size,
            orientation=np.radians(0),
            **kwargs
        )

    def patch_vertices(self) -> np.ndarray:
        """Pointy-top hexagon vertices in data coordinates (circumradius 0.58).

        Renders the same shape as ``create_patch`` with its default size; note
        that ``RegularPolygon.get_path()`` itself returns unit-circle vertices
        (matplotlib applies the radius via a transform), so the radius is baked
        in here instead.
        """
        angles = np.pi / 2 + np.arange(6) * (np.pi / 3)
        return 0.58 * np.column_stack([np.cos(angles), np.sin(angles)])

    def get_map_dimensions(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Get dimensions for hexagonal map."""
        hex_height_coeff = np.sqrt(3) / 2
        width = grid_x + 0.5 * (grid_y % 2)
        height = grid_y * hex_height_coeff
        return width, height


class RectangularTopology(Topology):
    def topology_function(
        self,
        x1: int | np.ndarray,
        y1: int | np.ndarray,
        x2: int | np.ndarray,
        y2: int | np.ndarray,
    ) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_name(self) -> str:
        return "rectangular"

    def get_visualization_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Get rectangular grid coordinates (no offset needed)."""
        return float(x), float(y)

    def get_visualization_coords_array(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized variant of :meth:`get_visualization_coords`."""
        return x.astype(float), y.astype(float)

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

    def patch_vertices(self) -> np.ndarray:
        """Axis-aligned square vertices (Rectangle with default size=0.9)."""
        h = 0.9 / 2
        return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
