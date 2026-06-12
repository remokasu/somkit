"""Domain-specific exceptions for somkit (SPEC-0002 / ADR-0004).

All somkit errors derive from :class:`SomkitError`, which subclasses the
built-in :class:`ValueError` for backward compatibility (existing
``except ValueError`` handlers keep working — e.g. the SOM_PAK loader that
previously raised plain ``ValueError``; see bug-report 2026-06-05).
"""

from __future__ import annotations


class SomkitError(ValueError):
    """Base class for somkit domain errors (a ``ValueError`` for back-compat)."""


class CodFormatError(SomkitError):
    """A SOM_PAK ``.cod`` file is malformed.

    Raised on a missing/short header, a vector count that does not match
    ``xdim * ydim``, or a dimension mismatch.
    """


class SOMDataError(SomkitError):
    """Invalid :class:`somkit.data_loader.SOMData` contents.

    Raised when per-sample metadata (mask / weights / fixed / labels) does not
    match the data's shape or expected dtype.
    """
