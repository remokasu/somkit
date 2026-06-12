"""Port of the SOM_PAK ``orand`` random number generator (SPEC-0001 FR-6).

SOM_PAK uses its own linear congruential generator (LCG) for codebook
initialization and data presentation order. To make somkit numerically
conformant (ADR-0001 layer B), the same integer stream must be reproducible in
Python; numpy's RNG cannot match it. :class:`OrandRNG` reproduces the SOM_PAK
recurrence exactly using Python's arbitrary-precision integers.

The trainer-side injection (``rng=``) and a numpy adapter are intentionally out
of scope here and handled in SPEC-0001 FR-7 (randinit) / FR-2 (train_pak); this
module only provides the RNG primitive.

Note:
    SOM_PAK's ``init_random(0)`` seeds from ``time(NULL)`` (non-deterministic).
    That behavior is deliberately *not* ported (to keep CI deterministic):
    :class:`OrandRNG` requires a strictly positive seed. This is the only point
    where it diverges from SOM_PAK.

References:
    - lvq_pak.c:311-323 — ``next`` / ``osrand`` / ``orand`` / ``RND_MAX``
    - lvq_pak.c:328-334 — ``init_random``
    - som_rout.c:147 — randinit scaling ``orand()/32768.0``
    - som_rout.c:290 — lininit scaling ``orand()/16384.0 - 1.0``
"""

from __future__ import annotations


class OrandRNG:
    """SOM_PAK ``orand`` linear congruential generator.

    The recurrence is ``state = (state * 23) % 100000001`` and each draw returns
    ``state % 32767`` (an integer in ``[0, 32766]``). State is held per instance
    to avoid the global mutable state of the C original.

    Args:
        seed: Strictly positive integer seed (SOM_PAK ``osrand`` value).
            Defaults to ``1`` (the C ``static next = 1`` initial value).

    Raises:
        ValueError: If ``seed`` is not a strictly positive integer.

    References:
        lvq_pak.c:311-323.
    """

    # LCG parameters (lvq_pak.c:312, 322).
    _MULTIPLIER: int = 23
    _MODULUS: int = 100000001
    #: Upper bound of the returned integer range; draws are ``state % RND_MAX``.
    RND_MAX: int = 32767

    # Caller-side scaling divisors used by SOM_PAK.
    #: randinit_codes divides ``orand()`` by this (som_rout.c:147).
    UNIFORM_DIVISOR: float = 32768.0
    #: find_eigenvectors (lininit) uses ``orand()/16384.0 - 1.0`` (som_rout.c:290).
    SYMMETRIC_DIVISOR: float = 16384.0

    def __init__(self, seed: int = 1) -> None:
        # ``_state`` is bound (to a valid, positive value) by ``seed``; we avoid
        # a placeholder so the only observable state is a fully-seeded one.
        self._state: int
        self.seed(seed)

    def seed(self, value: int) -> None:
        """Reset the generator state (SOM_PAK ``osrand``).

        Args:
            value: Strictly positive integer seed.

        Raises:
            ValueError: If ``value`` is not a strictly positive integer.

        References:
            lvq_pak.c:315-318. SOM_PAK's ``seed == 0`` (time-based) path is not
            ported; pass an explicit positive seed instead.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"seed must be an int, got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(
                f"seed must be a strictly positive integer (SOM_PAK's seed=0 "
                f"time-based path is not ported), got {value!r}."
            )
        self._state = value

    def next_int(self) -> int:
        """Return the next integer draw in ``[0, RND_MAX - 1]`` (i.e. 0..32766).

        Returns:
            The next ``orand()`` value.

        References:
            lvq_pak.c:320-323 (``(next = (next*23) % 100000001) % 32767``).
        """
        self._state = (self._state * self._MULTIPLIER) % self._MODULUS
        return self._state % self.RND_MAX

    def uniform(self) -> float:
        """Return the next draw scaled by 1/32768 (SOM_PAK randinit).

        Returns:
            ``next_int() / 32768.0`` — a float in ``[0.0, 32766/32768.0]``
            (i.e. ``[0.0, 0.999939...]``).

        References:
            som_rout.c:147.
        """
        return self.next_int() / self.UNIFORM_DIVISOR

    def symmetric(self) -> float:
        """Return the next draw scaled to a near-symmetric range (SOM_PAK lininit).

        Returns:
            ``next_int() / 16384.0 - 1.0`` — a float in
            ``[-1.0, 32766/16384.0 - 1.0]`` (i.e. ``[-1.0, 0.999878...]``).

        References:
            som_rout.c:290.
        """
        return self.next_int() / self.SYMMETRIC_DIVISOR - 1.0
