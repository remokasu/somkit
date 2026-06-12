from typing import Callable, Dict

import numpy as np


def gaussian(radius: float, distance: np.ndarray, n_radius: float) -> float:
    return np.exp(-(distance ** 2) / (2 * n_radius ** 2))


def mexican_hat(radius: float, distance: float, n_radius: float) -> float:
    return (1 - (distance ** 2) / (n_radius ** 2)) * np.exp(
        -(distance ** 2) / (2 * n_radius ** 2)
    )


def bubble(radius: np.float64, distance: np.ndarray, n_radius: float) -> float:
    return np.where(distance <= radius, 1, 0)


def cone(radius: float, distance: np.ndarray, n_radius: float) -> float:
    return np.where(distance <= radius, 1 - (distance / radius), 0)


# ----------------------------------------------------------------------
# SOM_PAK-conformant neighborhood functions (SPEC-0001 FR-5)
#
# These use the simplified ``(distance, radius)`` signature (ADR-0003), in
# contrast to the legacy ``(radius, distance, n_radius)`` functions above which
# are kept for the non-conformant ``train``/``train_batch`` paths.
# ----------------------------------------------------------------------


def gaussian_neighborhood(distance: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian neighborhood without cutoff (SOM_PAK ``gaussian_adapt``).

    ``h(d) = exp(-d^2 / (2 * radius^2))`` for *all* units (no hard cutoff at the
    radius), matching som_rout.c:543-546.

    Args:
        distance: Grid distance(s) from the BMU.
        radius: Current neighborhood radius (``> 0``).

    Returns:
        Neighborhood coefficients, same shape as ``distance``.

    References:
        som_rout.c:543-546.
    """
    return np.exp(-(distance ** 2) / (2.0 * radius ** 2))


def bubble_neighborhood(distance: np.ndarray, radius: float) -> np.ndarray:
    """Bubble (step) neighborhood (SOM_PAK ``bubble_adapt``).

    ``h(d) = 1`` for ``d <= radius`` else ``0`` (the cutoff is intrinsic to the
    bubble definition), matching som_rout.c:497.

    Args:
        distance: Grid distance(s) from the BMU.
        radius: Current neighborhood radius.

    Returns:
        Neighborhood coefficients (0.0/1.0), same shape as ``distance``.

    References:
        som_rout.c:497.
    """
    return (distance <= radius).astype(np.float64)


PakNeighborhood = Callable[[np.ndarray, float], np.ndarray]

#: Name -> SOM_PAK neighborhood function (simplified ``(distance, radius)`` I/F).
PAK_NEIGHBORHOODS: Dict[str, PakNeighborhood] = {
    "bubble": bubble_neighborhood,
    "gaussian": gaussian_neighborhood,
}


def get_pak_neighborhood(name: str) -> PakNeighborhood:
    """Resolve a SOM_PAK neighborhood function by name.

    Args:
        name: One of the keys in :data:`PAK_NEIGHBORHOODS`.

    Returns:
        The matching neighborhood callable.

    Raises:
        ValueError: If ``name`` is not a known neighborhood.
    """
    try:
        return PAK_NEIGHBORHOODS[name]
    except KeyError:
        valid = ", ".join(sorted(PAK_NEIGHBORHOODS))
        raise ValueError(
            f"Unknown neighborhood {name!r}. Valid options: {valid}."
        ) from None
