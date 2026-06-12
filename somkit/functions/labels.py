"""SOM_PAK vcal label calibration (SPEC-0002 FR-5).

After training, each codebook unit is labelled by majority vote of the data
samples that map to it (their BMU). This reproduces SOM_PAK's ``vcal``.

The hitlist order is the subtle part: SOM_PAK keeps each unit's labels in
**frequency-descending** order, and ties are broken by **arrival order at that
frequency** (``add_hit`` swaps a label toward the front only while the previous
label has a strictly smaller frequency — ``<``, never ``==``). A plain
``Counter`` would not reproduce this tie order, so :func:`_add_hit` simulates the
C list maintenance exactly.

References:
    - vcal.c:105-161 (find_labels, numlabs)
    - labels.c:365-402 (add_hit: frequency-descending, tie = arrival order)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from somkit.io.cod import flat_index

#: SOM_PAK's empty label (loader assigns "" to unlabelled samples).
_EMPTY_LABEL = ""


#: A hitlist entry is a ``[label: str, freq: int]`` pair.
HitEntry = List

def _add_hit(hitlist: List[HitEntry], label: str) -> None:
    """Add one hit for ``label`` to ``hitlist`` (SOM_PAK ``add_hit``).

    ``hitlist`` is a list of ``[label, freq]`` pairs kept in frequency-descending
    order. A new label is appended at the tail (freq 1); an existing label's freq
    is incremented and then bubbled toward the front while the previous entry has
    a *strictly* smaller frequency (ties keep their arrival order).

    References:
        labels.c:365-402.
    """
    for i, entry in enumerate(hitlist):
        if entry[0] == label:
            entry[1] += 1
            while i > 0 and hitlist[i - 1][1] < hitlist[i][1]:
                hitlist[i - 1], hitlist[i] = hitlist[i], hitlist[i - 1]
                i -= 1
            return
    hitlist.append([label, 1])


def calibrate_labels(
    bmus: Sequence[Tuple[int, int]],
    labels: Sequence[str],
    x_size: int,
    y_size: int,
    numlabs: int = 1,
) -> np.ndarray:
    """Assign majority-vote labels to each unit (SOM_PAK ``vcal``).

    Args:
        bmus: Per-sample BMU grid coordinates ``(x, y)`` (same order as ``labels``).
        labels: Per-sample label strings; ``""`` means no label (ignored).
        x_size: Number of units along x.
        y_size: Number of units along y.
        numlabs: Max labels per unit; ``0`` means all. Default ``1``.

    Returns:
        An ``(x_size, y_size)`` object array; each cell is a list of label
        strings in frequency-descending order (empty list for units with no hit).

    References:
        vcal.c:60 (``numlabs < 0`` normalized to 0), vcal.c:105-161, labels.c:365-402.
    """
    if numlabs < 0:  # SOM_PAK vcal.c:60 normalizes negative numlabs to "all".
        numlabs = 0
    hitlists: List[List[HitEntry]] = [[] for _ in range(x_size * y_size)]
    for (x, y), label in zip(bmus, labels):
        if label == _EMPTY_LABEL:
            continue
        _add_hit(hitlists[flat_index(x, y, x_size)], label)

    result = np.empty((x_size, y_size), dtype=object)
    for x in range(x_size):
        for y in range(y_size):
            hl = hitlists[flat_index(x, y, x_size)]
            n = len(hl) if numlabs == 0 else min(len(hl), numlabs)
            result[x, y] = [entry[0] for entry in hl[:n]]
    return result
