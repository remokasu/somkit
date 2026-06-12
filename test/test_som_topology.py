"""Tests for topology distance functions (SPEC-0001 FR-4).

HexagonalTopology.topology_function must match SOM_PAK's ``hexa_dist`` and
RectangularTopology.topology_function its ``rect_dist``.

Convention: ``topology_function(x1, y1, x2, y2)`` computes the grid distance
where ``(x2, y2)`` is the BMU and ``(x1, y1)`` is the target unit (matching the
``som_step`` call ``topology_function(grid_x, grid_y, bmu_x, bmu_y)``). SOM_PAK
uses ``hexa_dist(bx, by, tx, ty)`` with the BMU first, so the row-parity offset
depends on the BMU's row parity (``by == y2``).

References:
    - som_rout.c:434-455 (hexa_dist)
    - som_rout.c:457-468 (rect_dist)
"""

import math

import numpy as np

from somkit.topology import HexagonalTopology, RectangularTopology


def _ref_hexa_dist(tx, ty, bx, by):
    """Independent port of SOM_PAK hexa_dist, called target-first.

    Args are ``(target_x, target_y, bmu_x, bmu_y)`` to match somkit's
    ``topology_function(x1, y1, x2, y2)`` argument order; internally it computes
    SOM_PAK's ``hexa_dist(bx, by, tx, ty)``.
    """
    diff = float(bx - tx)
    if (by - ty) % 2 != 0:
        if by % 2 == 0:
            diff -= 0.5
        else:
            diff += 0.5
    return math.sqrt(diff * diff + 0.75 * (by - ty) ** 2)


def test_som_topology_init():
    topo = HexagonalTopology()
    assert callable(topo.topology_function)


class TestHexaDist:
    def test_matches_som_pak_scalar(self):
        # target=(2,3), bmu=(5,7): diff=3, (7-3)%2==0 -> no offset,
        # dist = sqrt(9 + 0.75*16) = sqrt(21).
        d = HexagonalTopology().topology_function(2, 3, 5, 7)
        assert np.isclose(d, math.sqrt(21.0))
        assert np.isclose(d, _ref_hexa_dist(2, 3, 5, 7))

    def test_same_point_zero(self):
        assert HexagonalTopology().topology_function(4, 4, 4, 4) == 0.0

    def test_row_offset_even_bmu_row(self):
        # bmu row even (by=0), target one row away -> diff -= 0.5
        # target=(0,1), bmu=(0,0): diff=0-0.5=-0.5, dist=sqrt(0.25+0.75)=1.0
        d = HexagonalTopology().topology_function(0, 1, 0, 0)
        assert np.isclose(d, 1.0)
        assert np.isclose(d, _ref_hexa_dist(0, 1, 0, 0))

    def test_row_offset_odd_bmu_row(self):
        # bmu row odd (by=1) -> diff += 0.5
        # target=(0,0), bmu=(0,1): diff=0+0.5=0.5, dist=sqrt(0.25+0.75)=1.0
        d = HexagonalTopology().topology_function(0, 0, 0, 1)
        assert np.isclose(d, 1.0)
        assert np.isclose(d, _ref_hexa_dist(0, 0, 0, 1))

    def test_offset_sign_depends_on_bmu_parity(self):
        # Same target/bmu rows differing by 1 but different bmu parity should
        # produce different x-offsets (asymmetry of hexa_dist).
        topo = HexagonalTopology()
        d_even = topo.topology_function(1, 1, 0, 0)  # by=0 even -> diff -=0.5
        d_odd = topo.topology_function(1, 0, 0, 1)   # by=1 odd  -> diff +=0.5
        assert np.isclose(d_even, _ref_hexa_dist(1, 1, 0, 0))
        assert np.isclose(d_odd, _ref_hexa_dist(1, 0, 0, 1))

    def test_vectorized_grid_matches_reference(self):
        topo = HexagonalTopology()
        x_size, y_size = 5, 4
        gx, gy = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing="ij")
        for bmu in [(0, 0), (2, 1), (4, 3), (1, 2)]:
            produced = topo.topology_function(gx, gy, bmu[0], bmu[1])
            expected = np.array(
                [
                    [_ref_hexa_dist(x, y, bmu[0], bmu[1]) for y in range(y_size)]
                    for x in range(x_size)
                ]
            )
            np.testing.assert_allclose(produced, expected, rtol=0, atol=0)

    def test_array_bmu_broadcast(self):
        # _batch_update passes an array of BMUs; offset must stay vectorized.
        topo = HexagonalTopology()
        node_x = np.arange(3)[:, None, None]  # (3,1,1)
        node_y = np.arange(4)[None, :, None]  # (1,4,1)
        bmu_x = np.array([0, 2])[None, None, :]  # (1,1,2)
        bmu_y = np.array([0, 1])[None, None, :]  # mixed parity BMUs
        produced = topo.topology_function(node_x, node_y, bmu_x, bmu_y)
        assert produced.shape == (3, 4, 2)
        for xi in range(3):
            for yi in range(4):
                for s in range(2):
                    assert np.isclose(
                        produced[xi, yi, s],
                        _ref_hexa_dist(xi, yi, int(bmu_x[0, 0, s]), int(bmu_y[0, 0, s])),
                    )


class TestRectDist:
    def test_matches_euclidean_scalar(self):
        d = RectangularTopology().topology_function(2, 3, 5, 7)
        assert np.isclose(d, math.sqrt((5 - 2) ** 2 + (7 - 3) ** 2))

    def test_vectorized(self):
        topo = RectangularTopology()
        gx, gy = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        produced = topo.topology_function(gx, gy, 1, 2)
        expected = np.sqrt((gx - 1) ** 2 + (gy - 2) ** 2)
        np.testing.assert_allclose(produced, expected, rtol=0, atol=0)
