"""SOMData container for data plus per-sample metadata (SPEC-0002 / ADR-0004).

``SOMData`` bundles the training data with optional per-sample metadata used by
SOM_PAK-conformant features:

- ``mask``    — boolean ``(n, dim)``; ``True`` marks a component to ignore in BMU
  search and updates (FR-1, SOM_PAK ``mask[i] != 0``).
- ``weights`` — float ``(n,)`` per-sample learning weight (FR-2).
- ``fixed`` / ``fixed_valid`` — int ``(n, 2)`` forced BMU coords + bool ``(n,)``
  validity (FR-3).
- ``labels``  — ``(n,)`` per-sample labels (vcal input).

The container is frozen (immutable) so it can act as the single source of truth
for per-sample metadata during training. All metadata is optional (``None``);
when every field is ``None`` the trainer falls back to the SPEC-0001
bit-identical fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from somkit.exceptions import SOMDataError


@dataclass(frozen=True)
class SOMData:
    """Training data with optional per-sample metadata.

    Args:
        data: ``(n_samples, dim)`` float array.
        mask: Optional boolean ``(n_samples, dim)``; ``True`` = ignore component.
        weights: Optional float ``(n_samples,)`` per-sample learning weight.
        fixed: Optional int ``(n_samples, 2)`` forced BMU ``(x, y)`` coordinates.
        fixed_valid: Optional boolean ``(n_samples,)`` marking which rows of
            ``fixed`` are active.
        labels: Optional ``(n_samples,)`` per-sample labels.

    Raises:
        SOMDataError: If any field's shape/dtype is inconsistent with ``data``.
    """

    data: np.ndarray
    mask: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    fixed: Optional[np.ndarray] = None
    fixed_valid: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise SOMDataError(
                f"data must be 2-D (n_samples, dim), got shape {self.data.shape}."
            )
        n = self.data.shape[0]

        if self.mask is not None:
            if self.mask.shape != self.data.shape:
                raise SOMDataError(
                    f"mask shape {self.mask.shape} != data shape {self.data.shape}."
                )
            if self.mask.dtype != np.bool_:
                raise SOMDataError(
                    f"mask must be boolean, got dtype {self.mask.dtype}."
                )

        if self.weights is not None and self.weights.shape != (n,):
            raise SOMDataError(
                f"weights shape {self.weights.shape} != ({n},)."
            )

        if self.fixed is not None and self.fixed.shape != (n, 2):
            raise SOMDataError(
                f"fixed shape {self.fixed.shape} != ({n}, 2)."
            )

        if self.fixed_valid is not None:
            if self.fixed is None:
                raise SOMDataError("fixed_valid requires fixed to be set.")
            if self.fixed_valid.shape != (n,):
                raise SOMDataError(
                    f"fixed_valid shape {self.fixed_valid.shape} != ({n},)."
                )

        if self.labels is not None and self.labels.shape != (n,):
            raise SOMDataError(
                f"labels shape {self.labels.shape} != ({n},)."
            )
