"""SOM_PAK ``.vis`` output (SPEC-0004 FR-2).

A ``.vis`` file holds, for each input sample, the coordinates of its best
matching unit and the per-sample quantization error — the output of the
SOM_PAK ``visual`` tool.

`.vis` format:
    line 1: ``3 topol xdim ydim neigh`` (the "dimension" is always 3:
    ``x y qerror``)
    remaining lines: ``x y qerror [label ...]`` per sample, in input order;
    a sample with no valid components (fully masked) is ``-1 -1 -1``.

References:
    - visual.c:47-155 (compute_visual_data)
    - datafile.c:433-480 (write_header / write_entry, ``%g`` floats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class VisualResult:
    """Per-sample BMU coordinates and quantization errors.

    Attributes:
        coords: ``(n_samples, 2)`` int array of BMU grid coordinates ``(x, y)``;
            a fully masked sample has ``(-1, -1)``.
        qerrors: ``(n_samples,)`` float array of per-sample L2 quantization
            errors (``sqrt(sum(diff^2))`` over unmasked components); a fully
            masked sample has ``-1.0``.
        labels: Per-sample label lists taken from the BMU **unit** (SOM_PAK
            ``visual`` copies the winner's calibration labels, visual.c:128),
            or ``None`` when no unit labels are available.
    """

    coords: np.ndarray
    qerrors: np.ndarray
    labels: Optional[List[List[str]]]


def write_vis(
    path: str,
    result: VisualResult,
    *,
    topol: str,
    xdim: int,
    ydim: int,
    neigh: str,
) -> None:
    """Write a :class:`VisualResult` as a SOM_PAK ``.vis`` file.

    Args:
        path: Output path.
        result: The per-sample BMU/qerror data to write.
        topol: SOM_PAK topology string (``"hexa"`` / ``"rect"``).
        xdim: Map width recorded in the header.
        ydim: Map height recorded in the header.
        neigh: SOM_PAK neighborhood string (``"bubble"`` / ``"gaussian"``).

    References:
        visual.c:99 (header with dimension=3), visual.c:135 (one line per
        sample), datafile.c:468 (``%g`` float format).
    """
    with open(path, "w") as f:
        f.write(f"3 {topol} {xdim} {ydim} {neigh}\n")
        for i in range(len(result.coords)):
            x, y = result.coords[i]
            parts = [f"{x:g}", f"{y:g}", f"{result.qerrors[i]:g}"]
            if result.labels is not None:
                parts.extend(str(lab) for lab in result.labels[i])
            # Trailing space matches SOM_PAK's write_entry output format.
            f.write(" ".join(parts) + " \n")
