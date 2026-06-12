"""Decay schedulers for the learning rate (alpha) and neighborhood radius.

These pure functions reproduce the SOM_PAK 3.1 decay schedules so that somkit
can be made numerically conformant with the reference C implementation
(SPEC-0001 FR-1). Each scheduler maps the current step ``t`` and the total
training length ``length`` to a decayed value, given the initial value.

The ``linear`` and ``inverse_t`` schedules are the SOM_PAK-conformant options;
the ``exponential`` schedule is the legacy somkit behavior, preserved as a
non-conformant opt-in (it is *not* used by SOM_PAK).

Notes:
    The schedulers are stateless and frame the time axis in *steps*, matching
    SOM_PAK's ``rlen`` total-step model. Wiring them into a step-based training
    loop (``train_pak``) is handled separately in SPEC-0001 FR-2/FR-3; this
    module only provides the interchangeable scheduler strategies.

References:
    - lvq_pak.c:620-634 — ``linear_alpha`` / ``inverse_t_alpha`` / ``INV_ALPHA_CONSTANT``
    - som_rout.c:617 — radius: ``trad = 1.0 + (radius - 1.0) * (length - le) / length``
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


# SOM_PAK ``INV_ALPHA_CONSTANT`` (lvq_pak.c:625): the inverse-t schedule decays
# with time constant ``c = length / INV_ALPHA_CONSTANT``.
INV_ALPHA_CONSTANT: float = 100.0


def _check_length(length: int) -> None:
    """Validate the total training length.

    Args:
        length: The total number of training steps.

    Raises:
        ValueError: If ``length`` is not strictly positive (would divide by zero).
    """
    if length <= 0:
        raise ValueError(f"length must be a positive integer, got {length!r}.")


# ----------------------------------------------------------------------
# Learning rate (alpha) schedulers
# ----------------------------------------------------------------------
def linear_alpha(t: int, length: int, alpha: float) -> float:
    """Linearly decaying learning rate (SOM_PAK default).

    ``alpha(t) = alpha * (length - t) / length``, reaching exactly ``0`` at
    ``t == length``.

    Args:
        t: Current step (``0 <= t <= length``).
        length: Total number of training steps.
        alpha: Initial learning rate.

    Returns:
        The decayed learning rate at step ``t``.

    Raises:
        ValueError: If ``length <= 0``.

    References:
        lvq_pak.c:620-623 (``linear_alpha``).
    """
    _check_length(length)
    return alpha * (length - t) / length


def inverse_t_alpha(t: int, length: int, alpha: float) -> float:
    """Inverse-time decaying learning rate.

    ``alpha(t) = alpha * c / (c + t)`` with ``c = length / INV_ALPHA_CONSTANT``.
    Equals ``alpha`` at ``t == 0`` and decays monotonically.

    Args:
        t: Current step (``t >= 0``).
        length: Total number of training steps.
        alpha: Initial learning rate.

    Returns:
        The decayed learning rate at step ``t``.

    Raises:
        ValueError: If ``length <= 0``.

    References:
        lvq_pak.c:627-634 (``inverse_t_alpha``).
    """
    _check_length(length)
    c = length / INV_ALPHA_CONSTANT
    return alpha * c / (c + t)


def exponential_alpha(
    t: int, length: int, alpha: float, tau: Optional[float] = None
) -> float:
    """Exponentially decaying learning rate (legacy, non-SOM_PAK option).

    ``alpha(t) = alpha * exp(-t / tau)``. This is the historical somkit behavior
    and is *not* part of SOM_PAK; it is preserved as an opt-in schedule.

    Args:
        t: Current step (``t >= 0``).
        length: Total number of training steps. Used as the default time
            constant when ``tau`` is not given (matches the legacy behavior
            where ``tau`` defaulted to the number of epochs).
        alpha: Initial learning rate.
        tau: Time constant for the exponential decay. Defaults to ``length``.

    Returns:
        The decayed learning rate at step ``t``.

    Raises:
        ValueError: If ``length <= 0``.
    """
    _check_length(length)
    if tau is None:
        tau = length
    return float(alpha * np.exp(-t / tau))


# ----------------------------------------------------------------------
# Neighborhood radius schedulers
# ----------------------------------------------------------------------
def linear_radius(t: int, length: int, initial: float) -> float:
    """Linearly decaying neighborhood radius (SOM_PAK default).

    ``radius(t) = 1.0 + (initial - 1.0) * (length - t) / length``.

    For the normal case ``initial >= 1`` the radius decreases linearly and
    reaches exactly ``1`` at ``t == length`` (the "floor at 1" behavior). The
    formula is applied faithfully to SOM_PAK with **no clamp**: for
    ``initial < 1`` the value rises from ``initial`` toward ``1`` and stays
    below ``1`` in between.

    Args:
        t: Current step (``0 <= t <= length``).
        length: Total number of training steps.
        initial: Initial neighborhood radius.

    Returns:
        The decayed neighborhood radius at step ``t``.

    Raises:
        ValueError: If ``length <= 0``.

    References:
        som_rout.c:617 (``trad = 1.0 + (radius - 1.0) * (length - le) / length``).
    """
    _check_length(length)
    return 1.0 + (initial - 1.0) * (length - t) / length


def exponential_radius(
    t: int, length: int, initial: float, tau: Optional[float] = None
) -> float:
    """Exponentially decaying neighborhood radius (legacy, non-SOM_PAK option).

    ``radius(t) = initial * exp(-t / tau)``. This is the historical somkit
    behavior and is *not* part of SOM_PAK; it is preserved as an opt-in schedule.

    Args:
        t: Current step (``t >= 0``).
        length: Total number of training steps. Used as the default time
            constant when ``tau`` is not given.
        initial: Initial neighborhood radius.
        tau: Time constant for the exponential decay. Defaults to ``length``.

    Returns:
        The decayed neighborhood radius at step ``t``.

    Raises:
        ValueError: If ``length <= 0``.
    """
    _check_length(length)
    if tau is None:
        tau = length
    return float(initial * np.exp(-t / tau))


# ----------------------------------------------------------------------
# Registries / resolvers for string-based selection (e.g. ``alpha_type=``)
# ----------------------------------------------------------------------
# Uniform scheduler call signature ``(t, length, initial) -> float``. The
# ``exponential_*`` schedulers additionally accept an optional ``tau`` keyword
# (default ``length``); it is compatible with this signature when called
# positionally and is only needed for direct use, not via the registries.
AlphaScheduler = Callable[[int, int, float], float]
RadiusScheduler = Callable[[int, int, float], float]

#: Name -> alpha scheduler. Names mirror SOM_PAK's ``alpha_type`` values.
ALPHA_SCHEDULERS: Dict[str, AlphaScheduler] = {
    "linear": linear_alpha,
    "inverse_t": inverse_t_alpha,
    "exponential": exponential_alpha,
}

#: Name -> radius scheduler. SOM_PAK has no ``inverse_t`` schedule for the radius.
RADIUS_SCHEDULERS: Dict[str, RadiusScheduler] = {
    "linear": linear_radius,
    "exponential": exponential_radius,
}


def get_alpha_scheduler(name: str) -> AlphaScheduler:
    """Resolve a learning-rate scheduler by name.

    Args:
        name: One of the keys in :data:`ALPHA_SCHEDULERS`.

    Returns:
        The matching scheduler callable.

    Raises:
        ValueError: If ``name`` is not a known alpha scheduler.
    """
    try:
        return ALPHA_SCHEDULERS[name]
    except KeyError:
        valid = ", ".join(sorted(ALPHA_SCHEDULERS))
        raise ValueError(
            f"Unknown alpha scheduler {name!r}. Valid options: {valid}."
        ) from None


def get_radius_scheduler(name: str) -> RadiusScheduler:
    """Resolve a neighborhood-radius scheduler by name.

    Args:
        name: One of the keys in :data:`RADIUS_SCHEDULERS`.

    Returns:
        The matching scheduler callable.

    Raises:
        ValueError: If ``name`` is not a known radius scheduler.
    """
    try:
        return RADIUS_SCHEDULERS[name]
    except KeyError:
        valid = ", ".join(sorted(RADIUS_SCHEDULERS))
        raise ValueError(
            f"Unknown radius scheduler {name!r}. Valid options: {valid}."
        ) from None
