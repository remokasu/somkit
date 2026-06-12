from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

from somkit.exceptions import SomkitError
from somkit.functions.learning import find_bmu_pak_batch

# The vectorized variant is numerically identical to the loop variant but
# ~70x faster at a few hundred points, which plotting routinely hits.
from somkit.projection.sammon_mapping import sammon_mapping_batch as sammon_mapping
from somkit.trainer.som_trainer import SOMTrainer


def _class_colors(
    n_classes: int, colormap: str | None = None
) -> list[tuple[float, float, float, float]]:
    """Return ``n_classes`` mutually distinct RGBA colors for class rendering.

    Sampling a 10-color qualitative palette (tab10) at ``i / n_classes`` makes
    classes collide as soon as ``n_classes > 10``, so qualitative palettes are
    indexed by integer and switched by size instead.

    Args:
        n_classes: Number of classes to color.
        colormap: Optional matplotlib colormap name. Qualitative (listed)
            colormaps are indexed directly and fall back to the automatic
            palette choice when they have fewer colors than ``n_classes``;
            continuous colormaps are sampled evenly over ``[0, 1]``.

    Returns:
        A list of ``n_classes`` distinct RGBA tuples.
    """
    if colormap is not None:
        cmap = plt.get_cmap(colormap)
        if isinstance(cmap, ListedColormap):
            if n_classes <= cmap.N:
                return [cmap(i) for i in range(n_classes)]
            # Requested qualitative palette is too small: fall through to the
            # automatic choice below to keep the colors distinguishable.
        else:
            return [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    if n_classes <= 10:
        return [plt.cm.tab10(i) for i in range(n_classes)]
    if n_classes <= 20:
        return [plt.cm.tab20(i) for i in range(n_classes)]
    return [plt.cm.hsv(i / n_classes) for i in range(n_classes)]


def _capped_figsize(
    width: float, height: float, max_edge: float = 16.0
) -> tuple[float, float]:
    """Scale ``(width, height)`` so the longer edge is at most ``max_edge`` inches.

    Figure sizes proportional to the map dimensions explode for large maps —
    a 100x100 map would request a 100-inch canvas and a >100-megapixel PNG.
    Capping preserves the aspect ratio and leaves small maps unchanged.

    Args:
        width: Requested width in inches.
        height: Requested height in inches.
        max_edge: Maximum allowed edge length in inches.

    Returns:
        The possibly scaled ``(width, height)``.
    """
    scale = min(1.0, max_edge / max(width, height, 1e-9))
    return (width * scale, height * scale)


def distance_map(weights: np.ndarray) -> np.ndarray:
    """Calculate the legacy per-unit distance map of the SOM.

    Each unit's value is the mean Euclidean distance to its 8 grid neighbors
    (Moore neighborhood). Kept for the ``style="legacy"`` U-matrix; the
    SOM_PAK-conformant interpolated U-matrix is :func:`compute_umatrix_pak`.

    Args:
        weights: SOM codebook of shape ``(x_size, y_size, input_dim)``.

    Returns:
        A ``(x_size, y_size)`` array of per-unit mean neighbor distances.
    """
    size_x, size_y = weights.shape[0], weights.shape[1]
    um = np.zeros((size_x, size_y, 8))

    # Left neighbor
    um[1:, :, 0] = np.linalg.norm(weights[1:, :] - weights[:-1, :], axis=2)
    # Right neighbor
    um[:-1, :, 1] = np.linalg.norm(weights[:-1, :] - weights[1:, :], axis=2)
    # Top neighbor
    um[:, 1:, 2] = np.linalg.norm(weights[:, 1:] - weights[:, :-1], axis=2)
    # Bottom neighbor
    um[:, :-1, 3] = np.linalg.norm(weights[:, :-1] - weights[:, 1:], axis=2)
    # Top-left neighbor
    um[1:, 1:, 4] = np.linalg.norm(weights[1:, 1:] - weights[:-1, :-1], axis=2)
    # Bottom-right neighbor
    um[:-1, :-1, 5] = np.linalg.norm(weights[:-1, :-1] - weights[1:, 1:], axis=2)
    # Top-right neighbor
    um[:-1, 1:, 6] = np.linalg.norm(weights[:-1, 1:] - weights[1:, :-1], axis=2)
    # Bottom-left neighbor
    um[1:, :-1, 7] = np.linalg.norm(weights[1:, :-1] - weights[:-1, 1:], axis=2)

    return um.mean(axis=2)


def _pak_median(values: list[float]) -> float:
    """Return SOM_PAK's "median" of a neighbor table (``map.c`` ``compar``+qsort).

    SOM_PAK sorts the table ascending and, for an even count, takes the mean of
    the two central values; for an odd count it takes the single central value
    (``map.c:309-475``). The neighbor count varies by position: 2/3/4/5/6, so
    this handles any table length (not just two values).

    Args:
        values: The unsorted neighbor U-values (length 2-6).

    Returns:
        The SOM_PAK median value.
    """
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def compute_umatrix_pak(weights: np.ndarray, topology_name: str = "hexagonal") -> np.ndarray:
    """Compute the SOM_PAK ``umat`` U-matrix on the interpolated grid (SPEC-0003).

    Ports ``SOM_PAK/map.c:calc_umatrix`` (L154-476) one-to-one. The result is a
    ``(2*x_size-1, 2*y_size-1)`` array where:

    - **wall cells** (between adjacent units) hold the raw Euclidean distance
      between those units' reference vectors,
    - **unit cells** (even/even indices) hold the SOM_PAK *median* of the
      surrounding wall cells (number of neighbors 2/3/4/6 depending on
      edge/corner/middle position and, for hexa, row parity).

    Unlike :func:`distance_map` (the 10x10, 8-neighbor *mean* map kept for the
    legacy view / training diagnostics), this is the SOM_PAK-conformant
    interpolated U-matrix. Values are returned **raw** (no [0,1] normalization;
    that is a rendering concern — see ``map.c:517-521``).

    Args:
        weights: SOM codebook of shape ``(x_size, y_size, input_dim)`` where the
            ``(i, j)`` indexing matches SOM_PAK's ``mvalue[i][j]``.
        topology_name: ``"hexagonal"`` (SOM_PAK ``hexa``) or ``"rectangular"``
            (``rect``).

    Returns:
        Raw U-matrix of shape ``(2*x_size-1, 2*y_size-1)``.

    Raises:
        SomkitError: If ``topology_name`` is not a known SOM_PAK topology, or if
            the map is smaller than 2x2 in either dimension (SOM_PAK ``umat``
            requires at least one inter-unit wall per dimension).
    """
    if topology_name not in ("hexagonal", "rectangular"):
        raise SomkitError(
            f"unknown topology {topology_name!r}; expected 'hexagonal' or 'rectangular'."
        )

    mxdim, mydim = weights.shape[0], weights.shape[1]
    if mxdim < 2 or mydim < 2:
        raise SomkitError(
            f"map must be at least 2x2 (got {mxdim}x{mydim}); SOM_PAK umat "
            "requires at least one inter-unit wall in each dimension."
        )
    uxdim, uydim = 2 * mxdim - 1, 2 * mydim - 1
    uvalue = np.zeros((uxdim, uydim), dtype=np.float64)
    w = weights.astype(np.float64)
    sqrt2 = np.sqrt(2.0)

    if topology_name == "rectangular":
        # map.c:162-202 -- rectangular wall cells
        for j in range(mydim):
            for i in range(mxdim):
                bx = by = bz = False
                dx = dy = dz1 = dz2 = 0.0
                if i < mxdim - 1:
                    dx = float(np.sum((w[i, j] - w[i + 1, j]) ** 2))
                    bx = True
                if j < mydim - 1:
                    dy = float(np.sum((w[i, j] - w[i, j + 1]) ** 2))
                    by = True
                if i < mxdim - 1 and j < mydim - 1:
                    dz1 = float(np.sum((w[i, j] - w[i + 1, j + 1]) ** 2))
                    dz2 = float(np.sum((w[i, j + 1] - w[i + 1, j]) ** 2))
                    bz = True
                # map.c:194
                dz = (np.sqrt(dz1) / sqrt2 + np.sqrt(dz2) / sqrt2) / 2
                if bx:
                    uvalue[2 * i + 1][2 * j] = np.sqrt(dx)
                if by:
                    uvalue[2 * i][2 * j + 1] = np.sqrt(dy)
                if bz:
                    uvalue[2 * i + 1][2 * j + 1] = dz
    else:
        # map.c:207-294 -- hexagonal wall cells (neighbor depends on row parity)
        for j in range(mydim):
            for i in range(mxdim):
                bx = by = bz = False
                dx = dy = dz = 0.0
                if i < mxdim - 1:  # map.c:214-222
                    dx = float(np.sum((w[i, j] - w[i + 1, j]) ** 2))
                    bx = True
                if j < mydim - 1:  # map.c:224-248 (dy)
                    if j % 2:
                        dy = float(np.sum((w[i, j] - w[i, j + 1]) ** 2))
                        by = True
                    elif i > 0:
                        dy = float(np.sum((w[i, j] - w[i - 1, j + 1]) ** 2))
                        by = True
                if j < mydim - 1:  # map.c:250-272 (dz)
                    if not (j % 2):
                        dz = float(np.sum((w[i, j] - w[i, j + 1]) ** 2))
                        bz = True
                    elif i < mxdim - 1:
                        dz = float(np.sum((w[i, j] - w[i + 1, j + 1]) ** 2))
                        bz = True
                if bx:  # map.c:276-277
                    uvalue[2 * i + 1][2 * j] = np.sqrt(dx)
                if by:  # map.c:279-285
                    if j % 2:
                        uvalue[2 * i][2 * j + 1] = np.sqrt(dy)
                    else:
                        uvalue[2 * i - 1][2 * j + 1] = np.sqrt(dy)
                if bz:  # map.c:287-293
                    if j % 2:
                        uvalue[2 * i + 1][2 * j + 1] = np.sqrt(dz)
                    else:
                        uvalue[2 * i][2 * j + 1] = np.sqrt(dz)

    _fill_unit_cell_medians(uvalue, topology_name, uxdim, uydim)
    return uvalue


def _fill_unit_cell_medians(
    uvalue: np.ndarray, topology_name: str, uxdim: int, uydim: int
) -> None:
    """Fill unit (even/even) cells with the median of surrounding walls.

    Ports ``SOM_PAK/map.c:297-476`` (the "model vector" cells). Mutates
    ``uvalue`` in place. The neighbor set differs by position (middle / edge /
    corner) and, for hexa, by row parity (``j%4``).

    Args:
        uvalue: U-matrix buffer of shape ``(uxdim, uydim)``, mutated in place.
        topology_name: ``"hexagonal"`` or ``"rectangular"``.
        uxdim: First dimension of ``uvalue`` (``= 2*x_size - 1``).
        uydim: Second dimension of ``uvalue`` (``= 2*y_size - 1``).
    """
    U = uvalue  # local alias matching map.c uvalue[i][j]
    if topology_name == "rectangular":
        # map.c:299-357 -- medians of the 4-neighborhood
        for j in range(0, uydim, 2):
            for i in range(0, uxdim, 2):
                if 0 < i < uxdim - 1 and 0 < j < uydim - 1:
                    U[i][j] = _pak_median(
                        [U[i - 1][j], U[i + 1][j], U[i][j - 1], U[i][j + 1]]
                    )
                elif j == 0 and 0 < i < uxdim - 1:
                    U[i][j] = _pak_median([U[i - 1][j], U[i + 1][j], U[i][j + 1]])
                elif j == uydim - 1 and 0 < i < uxdim - 1:
                    U[i][j] = _pak_median([U[i - 1][j], U[i + 1][j], U[i][j - 1]])
                elif i == 0 and 0 < j < uydim - 1:
                    U[i][j] = _pak_median([U[i + 1][j], U[i][j - 1], U[i][j + 1]])
                elif i == uxdim - 1 and 0 < j < uydim - 1:
                    U[i][j] = _pak_median([U[i - 1][j], U[i][j - 1], U[i][j + 1]])
                elif i == 0 and j == 0:
                    U[i][j] = (U[i + 1][j] + U[i][j + 1]) / 2.0
                elif i == uxdim - 1 and j == 0:
                    U[i][j] = (U[i - 1][j] + U[i][j + 1]) / 2.0
                elif i == 0 and j == uydim - 1:
                    U[i][j] = (U[i + 1][j] + U[i][j - 1]) / 2.0
                elif i == uxdim - 1 and j == uydim - 1:
                    U[i][j] = (U[i - 1][j] + U[i][j - 1]) / 2.0
        return

    # map.c:358-476 -- HEXA medians (neighbor set varies with j%4)
    for j in range(0, uydim, 2):
        for i in range(0, uxdim, 2):
            if 0 < i < uxdim - 1 and 0 < j < uydim - 1:  # middle (map.c:361-379)
                tbl = [U[i - 1][j], U[i + 1][j]]
                if not (j % 4):
                    tbl += [U[i - 1][j - 1], U[i][j - 1], U[i - 1][j + 1], U[i][j + 1]]
                else:
                    tbl += [U[i][j - 1], U[i + 1][j - 1], U[i][j + 1], U[i + 1][j + 1]]
                U[i][j] = _pak_median(tbl)
            elif j == 0 and 0 < i < uxdim - 1:  # upper edge (map.c:380-389)
                U[i][j] = _pak_median(
                    [U[i - 1][j], U[i + 1][j], U[i][j + 1], U[i - 1][j + 1]]
                )
            elif j == uydim - 1 and 0 < i < uxdim - 1:  # lower edge (map.c:390-404)
                tbl = [U[i - 1][j], U[i + 1][j]]
                if not (j % 4):
                    tbl += [U[i - 1][j - 1], U[i][j - 1]]
                else:
                    tbl += [U[i][j - 1], U[i + 1][j - 1]]
                U[i][j] = _pak_median(tbl)
            elif i == 0 and 0 < j < uydim - 1:  # left edge (map.c:405-422)
                if not (j % 4):
                    U[i][j] = _pak_median([U[i + 1][j], U[i][j - 1], U[i][j + 1]])
                else:
                    U[i][j] = _pak_median(
                        [U[i + 1][j], U[i][j - 1], U[i + 1][j - 1],
                         U[i][j + 1], U[i + 1][j + 1]]
                    )
            elif i == uxdim - 1 and 0 < j < uydim - 1:  # right edge (map.c:423-440)
                if j % 4:
                    U[i][j] = _pak_median([U[i - 1][j], U[i][j - 1], U[i][j + 1]])
                else:
                    U[i][j] = _pak_median(
                        [U[i - 1][j], U[i][j - 1], U[i - 1][j - 1],
                         U[i][j + 1], U[i - 1][j + 1]]
                    )
            elif i == 0 and j == 0:  # upper-left corner (map.c:441-443)
                U[i][j] = (U[i + 1][j] + U[i][j + 1]) / 2.0
            elif i == uxdim - 1 and j == 0:  # upper-right corner (map.c:444-451)
                U[i][j] = _pak_median([U[i - 1][j], U[i - 1][j + 1], U[i][j + 1]])
            elif i == 0 and j == uydim - 1:  # lower-left corner (map.c:452-463)
                if not (j % 4):
                    U[i][j] = (U[i + 1][j] + U[i][j - 1]) / 2.0
                else:
                    U[i][j] = _pak_median([U[i + 1][j], U[i][j - 1], U[i + 1][j - 1]])
            elif i == uxdim - 1 and j == uydim - 1:  # lower-right corner (map.c:464-475)
                if j % 4:
                    U[i][j] = (U[i - 1][j] + U[i][j - 1]) / 2.0
                else:
                    U[i][j] = _pak_median([U[i - 1][j], U[i][j - 1], U[i - 1][j - 1]])


class SOMVisualizer:
    def __init__(
        self,
        som: SOMTrainer,
        font_path: str | None = None,
        font_size: int | None = None,
    ):
        self.som: SOMTrainer = som
        self.font_path: str | None = font_path

        self.data: np.ndarray = som.data
        self.target: np.ndarray = som.target
        self.target_names: list = som.target_names

        if self.font_path is not None:
            self.font_prop = FontProperties(fname=font_path)
        else:
            self.font_prop = (
                FontProperties()
            )  # This will use the default font properties.

        if font_size is None:
            self.font_size = self.font_prop.get_size()
        else:
            self.font_size = font_size
        self.font_prop.set_size(self.font_size)
        self.point_size = 200  # size of ○ on hex.

    def _bmu_coords(self) -> np.ndarray:
        """Per-sample BMU coordinates, computed batched instead of per sample.

        Uses :func:`find_bmu_pak_batch` (chunked internally), which both avoids
        one full-map distance pass per sample and resolves distance ties in
        SOM_PAK scan order — consistent with ``calibrate_labels``, unlike the
        legacy :meth:`SOMTrainer.winner` whose flatten order differs on ties.

        Returns:
            An ``(n_samples, 2)`` int array of BMU grid coordinates.
        """
        return find_bmu_pak_batch(self.som.weights, self.data)

    def add_some_coloured_hexagons(self, umatrix: np.ndarray, colormap: str, ax):
        """Add colored patches (hexagons or rectangles based on topology) to the axis.

        Legacy helper kept for the ``style="legacy"`` U-matrix path. New plots use
        :meth:`_render_hex_field`, which adds the consistent orientation/limits.
        """
        linewidth = 0.1
        patches = []
        topology = self.som.topology

        for y in range(umatrix.shape[0]):
            for x in range(umatrix.shape[1]):
                vis_x, vis_y = topology.get_visualization_coords(x, y)
                patch = topology.create_patch(
                    vis_x, vis_y,
                    edgecolor="k",
                    linewidth=linewidth,
                )
                patches.append(patch)
        pc = PatchCollection(patches, array=np.ravel(umatrix), cmap=colormap)
        ax.add_collection(pc)
        return ax

    def _render_hex_field(
        self,
        ax: Axes,
        values_ij: np.ndarray,
        colormap: str,
        *,
        linewidth: float = 0.1,
        edgecolor: str = "k",
        colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_shrink: float = 1.0,
    ) -> PolyCollection:
        """Render a per-cell scalar field as a hex/rect grid (canonical orientation).

        Single source of truth for somkit's grid rendering: cell ``(i, j)`` is
        drawn at ``get_visualization_coords(i, j)`` and the y-axis is inverted so
        grid row 0 is at the **top** — matching the SOM_PAK ``umat`` orientation
        and keeping every somkit map (U-matrix, component planes, hit map)
        consistent so the same unit appears at the same place across figures.

        Args:
            ax: Target matplotlib Axes.
            values_ij: 2D array; ``values_ij[i, j]`` is the value at grid cell
                ``(i, j)`` (``i`` -> x axis, ``j`` -> y axis). A per-unit map
                ``(x_size, y_size)`` or the U-matrix interpolated grid
                ``(2*x_size-1, 2*y_size-1)`` are both accepted (for the latter,
                even/even cells are units and odd-index cells are walls).
            colormap: Matplotlib colormap name.
            linewidth: Cell edge line width.
            edgecolor: Cell edge color.
            colorbar: Whether to attach a colorbar.
            colorbar_label: Optional colorbar label.
            colorbar_shrink: Colorbar shrink factor.

        Returns:
            The drawn :class:`~matplotlib.collections.PolyCollection`.
        """
        topology = self.som.topology
        # One PolyCollection for the whole grid: per-cell patch objects are
        # ~10x slower to build and render at large map sizes (e.g. 100x100).
        ii, jj = np.meshgrid(
            np.arange(values_ij.shape[0]),
            np.arange(values_ij.shape[1]),
            indexing="ij",
        )
        vis_x, vis_y = topology.get_visualization_coords_array(ii, jj)
        centers = np.stack([vis_x.ravel(), vis_y.ravel()], axis=1)
        # (n_cells, n_vertices, 2); cell order (i outer, j inner) matches
        # values_ij.ravel() row-major order.
        verts = centers[:, np.newaxis, :] + topology.patch_vertices()[np.newaxis, :, :]
        pc = PolyCollection(
            verts,
            array=values_ij.ravel(),
            cmap=colormap,
            edgecolors=edgecolor,
            linewidths=linewidth,
        )
        ax.add_collection(pc)

        map_width, map_height = topology.get_map_dimensions(
            values_ij.shape[0], values_ij.shape[1]
        )
        ax.set_xlim(-1, map_width + 1)
        ax.set_ylim(-1, map_height + 1)
        ax.set_aspect("equal")
        # row 0 at top, consistent with SOM_PAK umat and all somkit maps
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            fig = ax.figure
            assert fig is not None  # always set for Axes from plt.subplots
            cb = fig.colorbar(pc, ax=ax, shrink=colorbar_shrink)
            if colorbar_label:
                cb.set_label(colorbar_label)
        return pc

    def add_data_points(self):
        """Overlay each input sample at its BMU on the legacy U-matrix axes.

        Used by the ``style="legacy"`` U-matrix path. The winner is
        ``(wx, wy)`` (``winner()`` returns weight indices ``i=x, j=y``) and is
        placed at ``get_visualization_coords(wx, wy)`` — the same mapping the
        legacy grid uses (no x/y swap).
        """
        topology = self.som.topology
        class_colors = (
            _class_colors(len(self.target_names))
            if self.target_names is not None and len(self.target_names) > 0
            else []
        )
        winners = self._bmu_coords()
        for i in range(len(self.data)):
            winner_node = tuple(winners[i])
            x, y = topology.get_visualization_coords(winner_node[0], winner_node[1])
            if (
                self.target is not None
                and self.target_names is not None
                and len(self.target_names) > 0
                and self.target[i] < len(self.target_names)
            ):
                color = class_colors[self.target[i]]
                plt.scatter(
                    x, y, color=color, s=self.point_size, marker="o", edgecolors="k"
                )
                plt.annotate(
                    self.target_names[self.target[i]],
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontproperties=self.font_prop,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="white",
                        boxstyle="round,pad=0.1",
                        alpha=0.8,
                    ),
                )
            else:
                plt.scatter(
                    x, y, color="black", s=self.point_size, marker="o", edgecolors="k"
                )

    def add_legend(self):
        if self.target_names is None or len(self.target_names) == 0:
            return
        class_colors = _class_colors(len(self.target_names))
        legend_elements = [
            Patch(
                facecolor=class_colors[i],
                edgecolor="k",
                label=self.target_names[i],
            )
            for i in range(len(self.target_names))
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            prop=self.font_prop,
        )

    def plot_umatrix(
        self,
        style: str = "pak",
        colormap: str | None = None,
        show_labels: bool = True,
        numlabs: int = 1,
        label_fontsize: float = 18.0,
        show_nodes: bool = True,
        node_size: float = 20.0,
        show_data_points: bool = False,
        show_legend: bool = False,
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """Plot the U-Matrix of the trained SOM (SPEC-0003).

        Two rendering styles are available:

        - ``"pak"`` (default): SOM_PAK ``umat``-conformant. Uses the
          ``(2*x-1, 2*y-1)`` interpolated grid (:func:`compute_umatrix_pak`)
          with explicit inter-unit "wall" cells, rendered grayscale (larger
          inter-unit distance = darker). This is what matches the SOM_PAK
          reference figures.
        - ``"legacy"``: the historical somkit view (one cell per unit,
          8-neighbor mean, ``bone_r``). Kept for backward compatibility.

        Labeling: by default (``show_labels=True``) each map unit gets one
        SOM_PAK ``vcal``-style majority label (:meth:`SOMTrainer.calibrate_labels`),
        drawn once per unit — no marker overlap. The older per-sample marker
        overlay is still available via ``show_data_points=True`` but is not
        recommended for the U-matrix (markers pile up when several samples share
        a BMU; use :meth:`plot_hit_map` for density instead).

        Args:
            style: ``"pak"`` (default) or ``"legacy"``.
            colormap: Matplotlib colormap. Defaults to ``"gray_r"`` for ``pak``
                and ``"bone_r"`` for ``legacy`` when ``None``.
            show_labels: Draw one ``vcal`` majority label per unit (pak style).
                Independent of ``show_nodes``. If the dataset has no targets,
                no labels exist, so every unit falls back to a node dot.
            numlabs: Max labels per unit when ``show_labels`` (SOM_PAK
                ``-numlabs``); ``0`` means all.
            label_fontsize: Font size (points) for the per-unit labels.
            show_nodes: Draw a small dot on units without a drawn label,
                marking map nodes no data mapped to (SOM_PAK ``LN`` marker).
                Works independently of ``show_labels`` (with ``show_labels=False``
                every unit gets a dot).
            node_size: Marker area (points^2) for the node dots.
            show_data_points: Overlay one marker per input sample at its BMU
                (legacy overlay; off by default).
            show_legend: Whether to draw the class legend.
            title: Optional plot title (``None`` = no title).
            file_name: Optional path to save the figure.
            show: Whether to display the figure.

        Raises:
            SomkitError: If ``style`` is not ``"pak"`` or ``"legacy"``.
        """
        if style == "legacy":
            self._plot_umatrix_legacy(
                colormap or "bone_r", show_data_points, show_legend, title, file_name, show
            )
            return
        if style != "pak":
            raise SomkitError(f"unknown style {style!r}; expected 'pak' or 'legacy'.")

        cmap = colormap or "gray_r"
        topology_name = self.som.topology.get_name()
        # uvalue[i][j] on the (2*x-1, 2*y-1) interpolated grid.
        uvalue = compute_umatrix_pak(self.som.get_weights(), topology_name)

        width_padding = 10 if show_legend else 0
        # Size the figure by the interpolated-grid extent (2N-1), not the unit
        # count: the U-matrix draws ~2x more cells per axis than the per-unit
        # plots, so this keeps each hex cell ~1 inch (same physical size as hit
        # map / component planes) and the labels legible.
        fig, ax = plt.subplots(
            figsize=_capped_figsize(uvalue.shape[0] + width_padding, uvalue.shape[1])
        )

        # Shared renderer: draws the grid and sets the canonical orientation
        # (row 0 at top), so the U-matrix lines up with the other maps.
        self._render_hex_field(ax, uvalue, cmap)
        if show_labels or show_nodes:
            self._add_unit_labels_pak(
                ax,
                show_labels=show_labels,
                numlabs=numlabs,
                fontsize=label_fontsize,
                show_nodes=show_nodes,
                node_size=node_size,
            )
        if show_data_points:
            self._add_data_points_pak(ax)
        if show_legend and self.target is not None and self.target_names is not None:
            self.add_legend()

        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        plt.xticks([])
        plt.yticks([])
        # SOM_PAK's umat output has no surrounding frame; hide the axes box.
        for spine in ax.spines.values():
            spine.set_visible(False)
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def _add_unit_labels_pak(
        self,
        ax: Axes,
        show_labels: bool = True,
        numlabs: int = 1,
        fontsize: float = 18.0,
        show_nodes: bool = True,
        node_size: float = 20.0,
    ) -> None:
        """Draw SOM_PAK ``vcal``-style per-unit labels and/or empty-node markers.

        Mirrors the SOM_PAK ``umat`` label loop (``umat.c:609-651``): for every
        map unit ``(i, j)``, if it has a calibrated label
        (:meth:`SOMTrainer.calibrate_labels`) the label *text* is drawn at the
        unit's doubled-grid position ``(2*i, 2*j)``; otherwise a small filled
        circle (the ``LN`` marker) marks the **empty node** (a unit no data
        sample mapped to). ``show_labels`` and ``show_nodes`` are independent:

        - both on: labeled units -> text, empty units -> dot (SOM_PAK default);
        - labels only: labeled -> text, empty -> nothing;
        - nodes only (``show_labels=False``): a dot on every unit;
        - dataset with no targets: every unit is treated as empty (dots only).

        Args:
            ax: The matplotlib Axes to draw on (avoids relying on the current
                Axes, which may differ in multi-subplot contexts).
            show_labels: Draw the ``vcal`` label text on labeled units.
            numlabs: Max labels per unit (SOM_PAK ``-numlabs``); ``0`` = all.
            fontsize: Font size (points) for the label text.
            show_nodes: Draw the ``LN`` dot on units without a drawn label.
            node_size: Marker area (points^2) for the node dots.
        """
        has_targets = (
            self.target is not None
            and self.target_names is not None
            and len(self.target_names) > 0
        )
        labels = (
            self.som.calibrate_labels(numlabs=numlabs)
            if (show_labels and has_targets)
            else None
        )
        topology = self.som.topology
        node_xs: list[float] = []
        node_ys: list[float] = []
        for i in range(self.som.x_size):
            for j in range(self.som.y_size):
                x, y = topology.get_visualization_coords(2 * i, 2 * j)
                cell = labels[i, j] if labels is not None else None
                if cell is not None and len(cell) > 0:
                    # labeled unit -> SOM_PAK LAB/ML (label text)
                    ax.annotate(
                        ", ".join(str(c) for c in cell),
                        (x, y),
                        ha="center",
                        va="center",
                        fontproperties=self.font_prop,
                        fontsize=fontsize,
                        color="black",
                        bbox=dict(
                            facecolor="white",
                            edgecolor="0.6",
                            boxstyle="round,pad=0.2",
                            alpha=0.9,
                        ),
                    )
                elif show_nodes:
                    # empty node (or labels off) -> SOM_PAK LN (filled dot)
                    node_xs.append(x)
                    node_ys.append(y)
        if node_xs:
            ax.scatter(node_xs, node_ys, s=node_size, c="black", marker="o", zorder=3)

    def _add_data_points_pak(self, ax: Axes) -> None:
        """Overlay data points at their BMUs on the SOM_PAK interpolated grid.

        Like :meth:`add_data_points`, but maps a winner unit ``(wx, wy)`` to its
        position on the doubled grid (``2*wx, 2*wy``), keeping data points
        aligned with the SOM_PAK-style U-matrix without the legacy transpose.

        Args:
            ax: The matplotlib Axes to draw on.
        """
        topology = self.som.topology
        class_colors = (
            _class_colors(len(self.target_names))
            if self.target_names is not None and len(self.target_names) > 0
            else []
        )
        winners = self._bmu_coords()
        for i in range(len(self.data)):
            winner_node = tuple(winners[i])
            x, y = topology.get_visualization_coords(2 * winner_node[0], 2 * winner_node[1])
            if (
                self.target is not None
                and self.target_names is not None
                and len(self.target_names) > 0
                and self.target[i] < len(self.target_names)
            ):
                color = class_colors[self.target[i]]
                ax.scatter(x, y, color=color, s=self.point_size, marker="o", edgecolors="k")
                ax.annotate(
                    self.target_names[self.target[i]],
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontproperties=self.font_prop,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="white",
                        boxstyle="round,pad=0.1",
                        alpha=0.8,
                    ),
                )
            else:
                ax.scatter(x, y, color="black", s=self.point_size, marker="o", edgecolors="k")

    def _plot_umatrix_legacy(
        self,
        colormap: str,
        show_data_points: bool,
        show_legend: bool,
        title: str | None,
        file_name: str | None,
        show: bool,
    ):
        """Legacy U-Matrix view (one cell per unit, 8-neighbor mean).

        Preserved unchanged for backward compatibility; reachable via
        ``plot_umatrix(style="legacy")``. Prefer the SOM_PAK-conformant default.
        """
        umatrix: np.ndarray = distance_map(self.som.get_weights()).T
        width_padding = 10 if show_legend else 0  # Add extra space for the legend
        fig, ax = plt.subplots(
            figsize=_capped_figsize(
                self.som.weights.shape[1] + width_padding,
                self.som.weights.shape[0],
            )
        )

        ax = self.add_some_coloured_hexagons(umatrix, colormap, ax)
        if show_data_points:
            self.add_data_points()
        if show_legend and self.target is not None and self.target_names is not None:
            self.add_legend()

        xlim_padding = 1
        ylim_padding = 1
        map_width, map_height = self.som.topology.get_map_dimensions(
            umatrix.shape[1], umatrix.shape[0]
        )
        ax.set_xlim(-xlim_padding, map_width + xlim_padding)
        ax.set_ylim(-ylim_padding, map_height + ylim_padding)
        ax.set_aspect("equal")

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        plt.xticks([])
        plt.yticks([])
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_component_planes(
        self,
        colormap: str = "viridis",
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """Plot component planes: the distribution of each input feature.

        One subplot per input dimension, sharing the canonical orientation with
        the U-matrix (see :meth:`_render_hex_field`).

        Args:
            colormap: Matplotlib colormap for the feature values.
            title: Optional figure title (``None`` = no title).
            file_name: Optional path to save the figure.
            show: Whether to display the figure.
        """
        weights = self.som.get_weights()
        n_features = weights.shape[2]

        # Calculate grid size for subplots
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            # weights[:, :, k] is indexed [i][j]; the shared renderer handles
            # orientation/limits/colorbar consistently with the U-matrix.
            self._render_hex_field(
                ax, weights[:, :, feature_idx], colormap, colorbar=True
            )
            ax.set_title(f"Feature {feature_idx + 1}", fontproperties=self.font_prop)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis("off")

        # Set title only if provided
        if title is not None:
            fig.suptitle(title, fontproperties=self.font_prop, fontsize=self.font_size + 6)

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_hit_map(
        self,
        colormap: str = "YlOrRd",
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """Plot a hit map: the number of samples mapped to each unit (BMU).

        The natural place to see data density (the per-sample overlay was
        removed from the U-matrix in favor of this). Shares the canonical
        orientation with the other maps.

        Args:
            colormap: Matplotlib colormap for the hit counts.
            title: Optional plot title (``None`` = no title).
            file_name: Optional path to save the figure.
            show: Whether to display the figure.
        """
        weights = self.som.get_weights()
        # hit_map[i, j] = number of samples whose BMU is unit (i, j).
        hit_map = np.zeros((weights.shape[0], weights.shape[1]))
        winners = self._bmu_coords()
        np.add.at(hit_map, (winners[:, 0], winners[:, 1]), 1)

        fig, ax = plt.subplots(
            figsize=_capped_figsize(weights.shape[0], weights.shape[1])
        )
        # Shared renderer -> same orientation as the U-matrix / component planes.
        self._render_hex_field(
            ax,
            hit_map,
            colormap,
            colorbar=True,
            colorbar_label="Number of hits",
            colorbar_shrink=0.7,
        )

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_class_distribution(
        self,
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """Plot a class-distribution map: a per-unit pie chart of class counts.

        Each unit shows the mix of classes of the samples mapped to it. Shares
        the canonical orientation with the other maps. Requires labeled data
        (``target``/``target_names``); otherwise it warns and returns.

        Args:
            title: Optional plot title (``None`` = no title).
            file_name: Optional path to save the figure.
            show: Whether to display the figure.
        """
        if self.target is None or self.target_names is None or len(self.target_names) == 0:
            print("Warning: No target data available for class distribution map.")
            return

        weights = self.som.get_weights()
        # Store class counts for each node
        class_distribution = {}

        # Count class occurrences for each node
        winners = self._bmu_coords()
        for i in range(len(self.data)):
            winner_node = tuple(winners[i])
            if winner_node not in class_distribution:
                class_distribution[winner_node] = {}
            class_idx = self.target[i]
            if class_idx < len(self.target_names):
                class_name = self.target_names[class_idx]
                class_distribution[winner_node][class_name] = (
                    class_distribution[winner_node].get(class_name, 0) + 1
                )

        fig, ax = plt.subplots(
            figsize=_capped_figsize(
                self.som.weights.shape[0] + 2, self.som.weights.shape[1]
            )
        )

        topology = self.som.topology
        linewidth = 0.5
        class_colors = _class_colors(len(self.target_names))

        # Draw patches (hexagons or rectangles). Unit (i, j) is drawn at
        # get_visualization_coords(i, j) -- the canonical mapping shared with the
        # U-matrix / component planes (no x/y swap).
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                vis_x, vis_y = topology.get_visualization_coords(i, j)
                patch = topology.create_patch(
                    vis_x, vis_y,
                    edgecolor="gray",
                    facecolor="white",
                    linewidth=linewidth,
                )
                ax.add_patch(patch)

                # Add pie chart if node has data
                node = (i, j)
                if node in class_distribution:
                    classes = list(class_distribution[node].keys())
                    counts = list(class_distribution[node].values())
                    # Convert target_names to list if it's a numpy array
                    target_names_list = list(self.target_names) if isinstance(self.target_names, np.ndarray) else self.target_names
                    colors = [
                        class_colors[target_names_list.index(c)] for c in classes
                    ]

                    # Draw mini pie chart at the visualization coordinates
                    pie_radius = 0.4
                    ax.pie(
                        counts,
                        colors=colors,
                        center=(vis_x, vis_y),
                        radius=pie_radius,
                        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
                    )

        xlim_padding = 1
        ylim_padding = 1
        map_width, map_height = topology.get_map_dimensions(
            weights.shape[0], weights.shape[1]
        )
        ax.set_xlim(-xlim_padding, map_width + xlim_padding)
        ax.set_ylim(-ylim_padding, map_height + ylim_padding)
        ax.set_aspect("equal")
        # row 0 at top, consistent with the U-matrix and the other maps
        ax.invert_yaxis()

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = [
            Patch(
                facecolor=class_colors[i],
                edgecolor="k",
                label=self.target_names[i],
            )
            for i in range(len(self.target_names))
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            prop=self.font_prop,
        )

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_sammon_projection(
        self,
        show_nodes: bool = True,
        show_data_points: bool = True,
        show_connections: bool = False,
        show_legend: bool = True,
        show_labels: bool = False,
        node_size: int = 150,
        data_point_size: int = 80,
        connection_style: str = "line",  # "line" or "spring"
        colormap: str = "tab10",
        max_iter: int = 500,
        learning_rate: float = 0.2,
        random_state: int | None = None,
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """Plot a Sammon's mapping projection of SOM weights and/or data points.

        Sammon's mapping projects high-dimensional data to 2D while preserving
        inter-point distances, giving a view of the data structure that is
        independent of the SOM grid topology.

        Args:
            show_nodes: Whether to show SOM nodes in the projection.
            show_data_points: Whether to show data points in the projection.
            show_connections: Whether to show connections between adjacent nodes.
            show_legend: Whether to show the legend for data points.
            show_labels: Whether to show labels on data points.
            node_size: Size of SOM node markers.
            data_point_size: Size of data point markers.
            connection_style: Node connection style (``"line"`` or ``"spring"``).
            colormap: Colormap to use for class colors. When it names a
                qualitative (listed) palette with fewer colors than the number
                of classes (e.g. the default ``"tab10"`` with >10 classes),
                :func:`_class_colors` falls back to its automatic palette
                choice — the same colors the other plots use — so class colors
                stay consistent across figures.
            max_iter: Maximum iterations for Sammon's mapping optimization.
            learning_rate: Learning rate for gradient descent.
            random_state: Random seed for reproducibility.
            title: Optional plot title (``None`` = no title).
            file_name: Optional path to save the figure.
            show: Whether to display the figure.
        """
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')

        # Combine weights and data for projection if both are shown
        if show_nodes and show_data_points:
            # Flatten SOM weights to get node vectors
            weights_flat = self.som.weights.reshape(-1, self.som.weights.shape[2])
            combined_data = np.vstack([weights_flat, self.data])

            # Project combined data
            projected = sammon_mapping(
                combined_data,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )

            # Split projected data back
            n_nodes = weights_flat.shape[0]
            nodes_projected = projected[:n_nodes]
            data_projected = projected[n_nodes:]

        elif show_nodes:
            # Project only SOM nodes
            weights_flat = self.som.weights.reshape(-1, self.som.weights.shape[2])
            nodes_projected = sammon_mapping(
                weights_flat,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            data_projected = None

        elif show_data_points:
            # Project only data points
            data_projected = sammon_mapping(
                self.data,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            nodes_projected = None

        else:
            raise ValueError("At least one of show_nodes or show_data_points must be True")

        # Plot SOM nodes
        if show_nodes and nodes_projected is not None:
            # Plot connections between adjacent nodes if requested
            if show_connections:
                # Determine connection style
                if connection_style == "spring":
                    # Spring-like connections with varying thickness based on distance
                    for i in range(self.som.x_size):
                        for j in range(self.som.y_size):
                            idx = i * self.som.y_size + j
                            # Connect to right neighbor
                            if j < self.som.y_size - 1:
                                idx_right = i * self.som.y_size + (j + 1)
                                dist = np.linalg.norm(
                                    nodes_projected[idx] - nodes_projected[idx_right]
                                )
                                # Thicker lines for shorter distances (stronger connections)
                                linewidth = max(0.3, 2.0 / (1 + dist))
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_right, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_right, 1]],
                                    color='#CCCCCC',
                                    alpha=0.4,
                                    linewidth=linewidth,
                                    zorder=1
                                )
                            # Connect to bottom neighbor
                            if i < self.som.x_size - 1:
                                idx_bottom = (i + 1) * self.som.y_size + j
                                dist = np.linalg.norm(
                                    nodes_projected[idx] - nodes_projected[idx_bottom]
                                )
                                linewidth = max(0.3, 2.0 / (1 + dist))
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_bottom, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_bottom, 1]],
                                    color='#CCCCCC',
                                    alpha=0.4,
                                    linewidth=linewidth,
                                    zorder=1
                                )
                else:  # "line" style
                    for i in range(self.som.x_size):
                        for j in range(self.som.y_size):
                            idx = i * self.som.y_size + j
                            # Connect to right neighbor
                            if j < self.som.y_size - 1:
                                idx_right = i * self.som.y_size + (j + 1)
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_right, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_right, 1]],
                                    color='#AAAAAA',
                                    alpha=0.35,
                                    linewidth=0.8,
                                    zorder=1
                                )
                            # Connect to bottom neighbor
                            if i < self.som.x_size - 1:
                                idx_bottom = (i + 1) * self.som.y_size + j
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_bottom, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_bottom, 1]],
                                    color='#AAAAAA',
                                    alpha=0.35,
                                    linewidth=0.8,
                                    zorder=1
                                )

            # Plot nodes with improved styling
            ax.scatter(
                nodes_projected[:, 0],
                nodes_projected[:, 1],
                c='#E8E8E8',
                s=node_size,
                marker='o',
                edgecolors='#555555',
                linewidths=1.5,
                alpha=0.7,
                label='SOM Nodes',
                zorder=2
            )

        # Plot data points
        if show_data_points and data_projected is not None:
            if (
                self.target is not None
                and self.target_names is not None
                and len(self.target_names) > 0
            ):
                # Distinct per-class colors honoring the requested colormap
                class_colors = _class_colors(len(self.target_names), colormap)

                # Plot with class colors
                for class_idx in range(len(self.target_names)):
                    mask = self.target == class_idx
                    if np.any(mask):
                        color = class_colors[class_idx]
                        ax.scatter(
                            data_projected[mask, 0],
                            data_projected[mask, 1],
                            c=[color],
                            s=data_point_size,
                            marker='o',
                            edgecolors='white',
                            linewidths=1.5,
                            alpha=0.85,
                            label=self.target_names[class_idx],
                            zorder=3
                        )

                        # Add labels if requested
                        if show_labels:
                            for i, idx in enumerate(np.where(mask)[0]):
                                ax.annotate(
                                    self.target_names[class_idx],
                                    (data_projected[idx, 0], data_projected[idx, 1]),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.7,
                                    fontproperties=self.font_prop
                                )
            else:
                # Plot without class information
                ax.scatter(
                    data_projected[:, 0],
                    data_projected[:, 1],
                    c='#3498db',
                    s=data_point_size,
                    marker='o',
                    edgecolors='white',
                    linewidths=1.5,
                    alpha=0.85,
                    label='Data Points',
                    zorder=3
                )

        # Styling
        ax.set_xlabel('Sammon Dimension 1', fontproperties=self.font_prop, fontsize=self.font_size + 2)
        ax.set_ylabel('Sammon Dimension 2', fontproperties=self.font_prop, fontsize=self.font_size + 2)

        # Set title only if provided
        if title is not None:
            ax.set_title(
                title,
                fontproperties=self.font_prop,
                fontsize=self.font_size + 6,
                fontweight='bold',
                pad=20
            )

        ax.set_aspect('equal')

        # Improved grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Legend styling
        if show_legend:
            legend = ax.legend(
                prop=self.font_prop,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                frameon=True,
                fancybox=True,
                shadow=True,
                ncol=1,
                fontsize=self.font_size
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('#CCCCCC')

        # Tight layout
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.2, dpi=300, facecolor='white')
        if show:
            plt.show()
