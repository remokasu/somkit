"""SOM_PAK-conformant codebook initialization (SPEC-0001 FR-7).

Reproduces SOM_PAK's ``randinit_codes``: each weight component is drawn
uniformly from the data's per-component ``[min, max]`` range using the ported
``orand`` generator (:class:`somkit.functions.rng.OrandRNG`). Matching the
generator *and the draw order* is what makes the initial codebook bit-compatible
with SOM_PAK (ADR-0001 layer B).

References:
    - som_rout.c:34-162 — ``randinit_codes``
    - som_rout.c:146-147 — ``points[i] = mival[i] + (maval[i]-mival[i]) * orand()/32768.0``
"""

from __future__ import annotations

import numpy as np

from somkit.functions.rng import OrandRNG


def random_init(
    data: np.ndarray, x_size: int, y_size: int, rng: OrandRNG
) -> np.ndarray:
    """Initialize a codebook the SOM_PAK ``randinit_codes`` way.

    Each component of each unit is drawn as
    ``min_i + (max_i - min_i) * rng.uniform()`` where ``min_i``/``max_i`` are the
    per-component minimum/maximum of ``data``.

    The ``orand`` draws are consumed in SOM_PAK's scan order: codebook entries
    ``index = y*x_size + x`` (``y`` outer, ``x`` inner) and, within each entry,
    components ``i = 0..dim-1``. This ordering is required for bit-compatibility
    with SOM_PAK; the returned array is laid out as ``weights[x, y, i]``.

    Args:
        data: Input data of shape ``(n_samples, dim)``.
        x_size: Number of units along the x axis.
        y_size: Number of units along the y axis.
        rng: A seeded :class:`OrandRNG`.

    Returns:
        Weights of shape ``(x_size, y_size, dim)``.

    Raises:
        ValueError: If ``data`` has no samples.

    Note:
        Components whose values are all identical (``max == min``) are
        initialized to that constant value (zero-width range). SOM_PAK's
        FLT_MIN/FLT_MAX min/max seeding quirk is intentionally not reproduced;
        the real per-component ``min``/``max`` is used (see SPEC-0001 FR-7).

    References:
        som_rout.c:142-153.
    """
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one sample for random_init.")

    mival = data.min(axis=0)
    maval = data.max(axis=0)
    dim = data.shape[1]

    # Consume orand draws in SOM_PAK order (entry y-outer/x-inner, then component)
    # as a flat scalar sequence, then reshape to that order and transpose to the
    # somkit (x, y, dim) layout. Only the draw generation is scalar-sequential
    # (initialization runs once).
    n = y_size * x_size * dim
    draws = np.fromiter(
        (rng.uniform() for _ in range(n)), dtype=np.float64, count=n
    ).reshape(y_size, x_size, dim)
    draws = np.transpose(draws, (1, 0, 2))  # (x_size, y_size, dim)

    span = maval - mival
    return mival + span * draws
