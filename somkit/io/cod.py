"""Public SOM_PAK ``.cod`` codebook I/O (SPEC-0002 FR-4 / ADR-0006).

Read and write SOM_PAK 3.1 codebook files so somkit can interoperate with the
SOM_PAK C tools and existing ``.cod`` assets.

`.cod` format:
    line 1: ``dim topol xdim ydim neigh``
    ``#``-prefixed lines: comments (skipped)
    remaining lines: one reference vector per unit in ``index = y*xdim + x``
    order (y outer, x inner); the first ``dim`` tokens are the components and any
    trailing tokens are calibration labels.

The ``index = y*xdim + x`` convention is encoded in two mutually-inverse helpers
(:func:`flat_index` for grid->index, :func:`_unravel` for index->grid) used by
both read and write, so an ``[x,y]``/``[y,x]`` mix-up cannot arise.

References:
    - datafile.c:433-480 (write_header / write_entry)
    - som_rout.c:643-644 (index = y*xdim + x)
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, TypedDict

import numpy as np

from somkit.exceptions import CodFormatError

#: Number of header fields: ``dim topol xdim ydim neigh``.
_HEADER_FIELDS = 5


class CodHeader(TypedDict):
    """Parsed ``.cod`` header fields."""

    dim: int
    topol: str
    xdim: int
    ydim: int
    neigh: str


class CodResult(NamedTuple):
    """Parsed ``.cod`` contents.

    Attributes:
        header: the :class:`CodHeader` fields.
        weights: ``(xdim, ydim, dim)`` reference vectors (``weights[x, y]``).
    """

    header: CodHeader
    weights: np.ndarray


def flat_index(x: int, y: int, x_size: int) -> int:
    """Return the SOM_PAK flat unit index ``y*x_size + x`` (y outer, x inner).

    The single source of truth for the grid->file-order mapping; also used by
    vcal label placement (ADR-0006).
    """
    return y * x_size + x


def _unravel(k: int, x_size: int) -> Tuple[int, int]:
    """Inverse of :func:`flat_index`: flat index ``k`` -> grid ``(x, y)``."""
    return k % x_size, k // x_size


def read_cod(path: str) -> CodResult:
    """Read a SOM_PAK ``.cod`` codebook file.

    Args:
        path: Path to the ``.cod`` file.

    Returns:
        A :class:`CodResult` with the parsed header and
        ``weights[xdim, ydim, dim]``.

    Raises:
        CodFormatError: On a missing/short header, a row with fewer than ``dim``
            components, or a vector count that does not match ``xdim * ydim``.
    """
    header: Optional[CodHeader] = None
    rows: List[List[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                parts = line.split()
                if len(parts) < _HEADER_FIELDS:
                    raise CodFormatError(
                        f"{path}: malformed header (expected {_HEADER_FIELDS} "
                        f"fields, got {len(parts)})."
                    )
                header = CodHeader(
                    dim=int(parts[0]),
                    topol=parts[1],
                    xdim=int(parts[2]),
                    ydim=int(parts[3]),
                    neigh=parts[4],
                )
                continue
            tokens = line.split()
            # First `dim` tokens are components; trailing tokens are labels.
            row = [float(v) for v in tokens[: header["dim"]]]
            if len(row) < header["dim"]:
                raise CodFormatError(
                    f"{path}: a vector row has {len(row)} components, "
                    f"expected {header['dim']}."
                )
            rows.append(row)

    if header is None:
        raise CodFormatError(f"{path}: missing .cod header line.")

    xdim, ydim, dim = header["xdim"], header["ydim"], header["dim"]
    if len(rows) != xdim * ydim:
        raise CodFormatError(
            f"{path}: expected {xdim * ydim} vectors ({xdim}x{ydim}), "
            f"found {len(rows)}."
        )

    weights = np.empty((xdim, ydim, dim), dtype=np.float64)
    for k, row in enumerate(rows):
        x, y = _unravel(k, xdim)
        weights[x, y] = row
    return CodResult(header, weights)


def write_cod(
    path: str,
    weights: np.ndarray,
    *,
    topol: str,
    neigh: str,
    labels: Optional[np.ndarray] = None,
    comments: Optional[List[str]] = None,
) -> None:
    """Write a codebook to a SOM_PAK ``.cod`` file.

    Args:
        path: Output path.
        weights: ``(xdim, ydim, dim)`` reference vectors (``weights[x, y]``).
        topol: SOM_PAK topology string (e.g. ``"hexa"`` / ``"rect"``).
        neigh: SOM_PAK neighborhood string (e.g. ``"bubble"`` / ``"gaussian"``).
        labels: Optional ``(xdim, ydim)`` object array; each cell is a list of
            label strings appended after the components (SOM_PAK vcal output).
        comments: Optional comment lines written after the header, each
            prefixed with ``#`` (SOM_PAK snapshots record ``#iterations: ...``,
            lvq_pak.c:530-531). :func:`read_cod` skips them.

    Raises:
        CodFormatError: If ``weights`` is not a 3-D array.

    References:
        datafile.c:433-480 (header ``dim topol xdim ydim neigh``; each unit's
        components as ``%g`` then trailing labels, ``index = y*xdim + x`` order).
    """
    if weights.ndim != 3:
        raise CodFormatError(
            f"weights must be (xdim, ydim, dim), got shape {weights.shape}."
        )
    xdim, ydim, dim = weights.shape

    with open(path, "w") as f:
        f.write(f"{dim} {topol} {xdim} {ydim} {neigh}\n")
        for comment in comments or []:
            f.write(f"#{comment}\n")
        for k in range(xdim * ydim):
            x, y = _unravel(k, xdim)
            parts = [f"{v:g}" for v in weights[x, y]]
            if labels is not None:
                parts.extend(str(lab) for lab in labels[x, y])
            f.write(" ".join(parts) + " \n")
