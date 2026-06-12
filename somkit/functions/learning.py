"""SOM_PAK-conformant sequential learning core (SPEC-0001 FR-2/FR-5).

``weighted_alpha`` (per-sample learning-rate weighting) belongs to SPEC-0002 FR-2.

Pure, vectorized building blocks for ``train_pak``: data presentation order,
BMU search with SOM_PAK's tie-break, and a single sequential update step. These
are kept side-effect-light (``som_step`` updates weights in place) and free of
trainer state so they can be unit-tested at the ADR-0001 layer-A level.

The inner per-unit work is numpy-vectorized while the outer step loop stays
sequential (ADR-0003 Option B): each update is applied immediately, preserving
the order-dependence of SOM_PAK's ``som_training``.

References:
    - som_rout.c:602-648 (som_training loop)
    - lvq_pak.c:38-91 (find_winner_euc, strict-< tie-break)
    - lvq_pak.c:202-216 (adapt_vector)
    - datafile.c:1144-1172 (randomize_entry_order)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from somkit.functions.neighborhood import PakNeighborhood, bubble_neighborhood
from somkit.functions.rng import OrandRNG
from somkit.topology.som_topology import Topology


def presentation_order(
    n_samples: int, rlen: int, rng: OrandRNG, random_order: bool
) -> np.ndarray:
    """Build the data presentation order for ``rlen`` steps.

    When ``random_order`` is True, the sample list is shuffled *once* using
    SOM_PAK's ``randomize_entry_order`` (repeatedly removing the
    ``orand() % remaining``-th element), then cycled modulo ``n_samples``. When
    False, the natural order ``0..n_samples-1`` is cycled.

    Args:
        n_samples: Number of data samples (``> 0``).
        rlen: Total number of training steps.
        rng: A seeded :class:`OrandRNG` (used only when ``random_order``).
        random_order: Whether to shuffle the presentation order.

    Returns:
        Integer index array of length ``rlen``.

    Raises:
        ValueError: If ``n_samples <= 0``.

    References:
        datafile.c:1144-1172 (randomize_entry_order); som_rout.c:602-612 (cycling).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}.")

    if random_order:
        remaining = list(range(n_samples))
        base = []
        for nol in range(n_samples, 0, -1):
            base.append(remaining.pop(rng.next_int() % nol))
        base = np.array(base, dtype=np.intp)
    else:
        base = np.arange(n_samples, dtype=np.intp)

    return base[np.arange(rlen) % n_samples]


def weighted_alpha(alpha: float, weight: float) -> float:
    """Apply a per-sample weight to the learning rate (SOM_PAK).

    ``talp = 1 - (1 - alpha)^weight`` when ``weight > 0`` (the effect of
    presenting the sample ``weight`` times); otherwise ``alpha`` unchanged.

    Args:
        alpha: Base learning rate at this step.
        weight: Per-sample weight; ``<= 0`` leaves ``alpha`` unchanged.

    Returns:
        The weight-adjusted learning rate.

    References:
        som_rout.c:624-626 (talp formula); vsom.c:96 (the ``-weights`` opt-in —
        here, supplying ``SOMData.weights`` is the opt-in).
    """
    if weight > 0:
        return 1.0 - (1.0 - alpha) ** weight
    return alpha


@lru_cache(maxsize=None)
def _unit_grid(x_size: int, y_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached ``(grid_x, grid_y)`` unit-coordinate meshgrid for a map size.

    The grid only depends on the map dimensions, so it is computed once per
    ``(x_size, y_size)`` and reused across all training steps. Callers must
    treat the returned arrays as read-only (they are shared).
    """
    return np.meshgrid(np.arange(x_size), np.arange(y_size), indexing="ij")


def find_bmu_pak(
    weights: np.ndarray, sample: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """Find the Best Matching Unit using SOM_PAK's tie-break.

    The winner is the unit minimizing the Euclidean distance in weight space;
    ties are broken by the smallest SOM_PAK index ``y*x_size + x`` (SOM_PAK
    updates the winner only on a strict ``<``, so the first unit in scan order
    wins — see lvq_pak.c:76).

    Args:
        weights: Codebook of shape ``(x_size, y_size, dim)``.
        sample: Input vector of shape ``(dim,)``.
        mask: Optional boolean ``(dim,)``; ``True`` components are excluded from
            the distance (SOM_PAK ``mask[i] != 0``). The distance is **not**
            normalized by the number of active components (matches SOM_PAK).
            When ``None`` the original fast path is used (bit-identical).

    Returns:
        The BMU grid coordinates ``(x, y)``.

    Note:
        A fully-masked sample (all components ignored) has no well-defined BMU
        and returns ``(0, 0)`` here; callers must exclude such samples first
        (``train_pak`` skips them, per som_rout.c:637-642).

    References:
        lvq_pak.c:38-91 (find_winner_euc; masked components skipped, no normalize).
    """
    # Compare *squared* distances, exactly like find_winner_euc
    # (lvq_pak.c:67 accumulates diff*diff and never takes a square root).
    # This is also faster and avoids sqrt-induced float ties.
    if mask is None:
        diff = weights - sample
    else:
        diff = np.where(mask, 0.0, weights - sample)
    distances = np.einsum("xyd,xyd->xy", diff, diff)  # (x_size, y_size)
    x_size = distances.shape[0]
    # Flatten in SOM_PAK index order (y outer, x inner); argmin returns the
    # smallest flat index on ties == smallest SOM_PAK index.
    flat_index = int(np.argmin(distances.T))
    y, x = divmod(flat_index, x_size)
    return (x, y)


def find_bmu_pak_batch(
    weights: np.ndarray,
    samples: np.ndarray,
    mask: Optional[np.ndarray] = None,
    chunk_size: int = 512,
) -> np.ndarray:
    """Vectorized :func:`find_bmu_pak` over many samples (same tie-break).

    Computes the same squared distances and resolves ties by the same SOM_PAK
    scan order (``y`` outer, ``x`` inner) as the single-sample version, but in
    chunks, avoiding one full-map distance pass per sample.

    Args:
        weights: Codebook of shape ``(x_size, y_size, dim)``.
        samples: Input vectors of shape ``(n_samples, dim)``.
        mask: Optional boolean ``(n_samples, dim)`` per-sample component masks.
        chunk_size: Samples per chunk; bounds the ``(chunk, x*y)`` distance
            buffer instead of materializing all ``n_samples * x * y`` distances.

    Returns:
        An ``(n_samples, 2)`` int array of BMU grid coordinates ``(x, y)``.
    """
    x_size = weights.shape[0]
    # Flatten the codebook in SOM_PAK scan order (y outer, x inner) so that
    # np.argmin's first-minimum rule reproduces find_bmu_pak's tie-break.
    weights_flat = weights.transpose(1, 0, 2).reshape(-1, weights.shape[2])
    bmus = np.empty((len(samples), 2), dtype=np.intp)
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        diff = weights_flat[np.newaxis, :, :] - chunk[:, np.newaxis, :]
        if mask is not None:
            chunk_mask = mask[start : start + chunk_size]
            diff = np.where(chunk_mask[:, np.newaxis, :], 0.0, diff)
        distances = np.einsum("nkd,nkd->nk", diff, diff)
        flat_index = np.argmin(distances, axis=1)
        y, x = np.divmod(flat_index, x_size)
        bmus[start : start + chunk_size, 0] = x
        bmus[start : start + chunk_size, 1] = y
    return bmus


def som_step(
    weights: np.ndarray,
    sample: np.ndarray,
    bmu: Tuple[int, int],
    radius: float,
    alpha: float,
    neighborhood_fn: PakNeighborhood,
    topology: Topology,
    mask: Optional[np.ndarray] = None,
) -> None:
    """Apply one SOM_PAK update step in place.

    For every unit ``i``: ``w_i += alpha * h(d_i, radius) * (sample - w_i)``,
    where ``d_i`` is the grid distance to the BMU and ``h`` is the neighborhood
    function (gaussian = all units, bubble = within radius). This matches
    ``bubble_adapt``/``gaussian_adapt`` followed by ``adapt_vector``.

    Args:
        weights: Codebook of shape ``(x_size, y_size, dim)``; modified in place.
        sample: Input vector of shape ``(dim,)``.
        bmu: BMU grid coordinates ``(x, y)``.
        radius: Current neighborhood radius.
        alpha: Current learning rate.
        neighborhood_fn: Neighborhood function with signature ``(distance, radius)``.
        topology: Topology providing ``topology_function`` (grid distances).
        mask: Optional boolean ``(dim,)``; ``True`` components are not moved
            (SOM_PAK ``adapt_vector`` skips masked components). ``None`` uses the
            original path (bit-identical).

    References:
        som_rout.c:472-551, lvq_pak.c:202-216.
    """
    x_size, y_size = weights.shape[0], weights.shape[1]
    grid_x, grid_y = _unit_grid(x_size, y_size)

    # Bubble fast path: h(d) is exactly 0 outside the radius, so restricting
    # the update to a bounding box that covers {d <= radius} changes nothing
    # (every skipped unit would receive += 0). The per-unit arithmetic inside
    # the window is element-wise and therefore bit-identical to the full-map
    # path. Gaussian has no cutoff (som_rout.c:543-546) and must stay full-map.
    if neighborhood_fn is bubble_neighborhood:
        # Window bounds covering {d <= radius} for both topologies:
        # |dx|: hexa_dist uses diff = dx -+ 0.5 (parity shift), and d <= radius
        #   requires |diff| <= radius, so |dx| <= radius + 0.5; the integer dx
        #   range is covered by rx = int(radius) + 1 >= floor(radius + 0.5).
        # |dy|: hexa_dist d^2 = diff^2 + 0.75*dy^2, worst case diff = 0 gives
        #   |dy| <= radius / sqrt(0.75); ry = int(radius / 0.866...) + 1 covers it.
        # The rectangular Euclidean distance needs only |dx|, |dy| <= radius,
        # which both bounds also cover.
        rx = int(radius) + 1
        ry = int(radius / 0.8660254037844386) + 1
        x0, x1 = max(0, bmu[0] - rx), min(x_size, bmu[0] + rx + 1)
        y0, y1 = max(0, bmu[1] - ry), min(y_size, bmu[1] + ry + 1)
        grid_x = grid_x[x0:x1, y0:y1]
        grid_y = grid_y[x0:x1, y0:y1]
        weights = weights[x0:x1, y0:y1]  # view: updates the original in place

    distance = topology.topology_function(grid_x, grid_y, bmu[0], bmu[1])
    influence = neighborhood_fn(distance, radius)
    delta = sample - weights
    if mask is not None:
        delta = np.where(mask, 0.0, delta)
    weights += alpha * influence[:, :, np.newaxis] * delta
