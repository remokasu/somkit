"""Test helpers for SOM_PAK golden files (SPEC-0001 FR-8 / SPEC-0002 FR-4).

``read_cod`` now delegates to the public ``somkit.io.cod.read_cod`` (the read
logic was promoted to a public API in FR-4); this module re-exports it under the
legacy name for existing tests and keeps the ``read_qerror`` fixture (which
parses SOM_PAK's ``qerror`` stdout, not a ``.cod`` file).
"""

from __future__ import annotations

import re

from somkit.exceptions import SomkitError
from somkit.io.cod import read_cod  # noqa: F401  (re-exported for test imports)

__all__ = ["read_cod", "read_qerror"]


def read_qerror(path: str) -> float:
    """Read the per-sample quantization error from SOM_PAK ``qerror`` output.

    Args:
        path: Path to the captured ``qerror`` stdout (``... is X per sample ...``).

    Returns:
        The per-sample quantization error ``X``.

    Raises:
        SomkitError: If no quantization-error value is found.
    """
    with open(path) as f:
        text = f.read()
    match = re.search(r"is\s+([0-9.eE+-]+)\s+per sample", text)
    if match is None:
        raise SomkitError(f"{path}: could not parse quantization error.")
    return float(match.group(1))
