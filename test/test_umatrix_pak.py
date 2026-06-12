"""SOM_PAK ``umat`` conformance tests for the U-matrix (SPEC-0003).

The primary acceptance criterion (SPEC-0003 Q5) is **numerical agreement of the
(2x-1)x(2y-1) value array** with SOM_PAK ``umat``. Golden int matrices were
extracted from SOM_PAK's PS output (``(int)(100*uvalue)``, ``umat.c:596``) and
committed under ``test/golden/`` so CI compares without a C toolchain
(ADR-0001 golden-file strategy). Regenerate with ``test/golden/regenerate.sh``.
"""

from pathlib import Path

import numpy as np
import pytest

from somkit.exceptions import SomkitError
from somkit.trainer.som_trainer import SOMTrainer
from somkit.visualizer import compute_umatrix_pak, distance_map

GOLDEN = Path(__file__).parent / "golden"


def _load_golden(name: str) -> np.ndarray:
    """Load a committed golden umat matrix (rows=y, cols=x) as int ``[y][x]``."""
    rows = [
        [int(v) for v in line.split()]
        for line in (GOLDEN / name).read_text().splitlines()
        if line and not line.startswith("#")
    ]
    return np.array(rows)


def _to_som_pak_ints(uvalue: np.ndarray) -> np.ndarray:
    """Apply SOM_PAK's display transform: normalize to [0,1] then ``int(100*v)``.

    Mirrors ``map.c:517-521`` (``1 - (v-min)/bw``) and ``umat.c:596``
    (``(int)(100*uvalue)``).
    """
    mn, mx = uvalue.min(), uvalue.max()
    norm = 1.0 - (uvalue - mn) / (mx - mn)
    return (100 * norm).astype(int)


@pytest.mark.parametrize(
    "cod_name, golden_name, expected_shape",
    [
        ("animal.cod", "animal_umat_pak.txt", (9, 9)),       # 5x5 -> 9x9
        ("ex_trained.cod", "ex_trained_umat_pak.txt", (19, 19)),  # 10x10 -> 19x19
    ],
)
def test_matches_som_pak_umat_golden(cod_name, golden_name, expected_shape):
    """compute_umatrix_pak reproduces SOM_PAK umat values exactly (int level)."""
    som = SOMTrainer.load_cod(str(GOLDEN / cod_name))
    uvalue = compute_umatrix_pak(som.get_weights(), "hexagonal")
    assert uvalue.shape == expected_shape

    mine = _to_som_pak_ints(uvalue)          # [x][y]
    golden = _load_golden(golden_name).T     # txt is [y][x] -> [x][y]
    assert mine.shape == golden.shape
    np.testing.assert_array_equal(mine, golden)


def test_shape_hexa_and_rect():
    """Output grid is (2*x-1, 2*y-1) for both topologies."""
    w = np.random.RandomState(0).rand(4, 6, 3)
    assert compute_umatrix_pak(w, "hexagonal").shape == (7, 11)
    assert compute_umatrix_pak(w, "rectangular").shape == (7, 11)


def test_wall_cell_is_euclidean_distance_rect():
    """A rectangular horizontal wall cell equals the inter-unit Euclidean distance."""
    w = np.zeros((2, 2, 3))  # 2x2 map -> 3x3 grid (avoids degenerate 1-row case)
    w[0, 0] = [0.0, 0.0, 0.0]
    w[1, 0] = [3.0, 4.0, 0.0]  # distance 5 from (0,0)
    uvalue = compute_umatrix_pak(w, "rectangular")  # shape (3, 3)
    # wall between unit (0,0) and (1,0) sits at uvalue[1][0] (map.c:197)
    assert uvalue[1, 0] == pytest.approx(5.0)


def test_center_cell_is_median_of_walls_rect():
    """A middle rect unit cell is the mean-of-two-medians of its 4 wall neighbors."""
    # 3x3 map -> 5x5 grid; center unit is uvalue[2][2], walls at the 4-neighborhood.
    rng = np.random.RandomState(1)
    w = rng.rand(3, 3, 4)
    uvalue = compute_umatrix_pak(w, "rectangular")
    walls = sorted(
        [uvalue[1, 2], uvalue[3, 2], uvalue[2, 1], uvalue[2, 3]]
    )
    expected = (walls[1] + walls[2]) / 2.0  # map.c:312 mean of two medians
    assert uvalue[2, 2] == pytest.approx(expected)


def test_hexa_row_parity_branches_differ():
    """Hexa wall placement depends on row parity; the grid is fully populated."""
    rng = np.random.RandomState(2)
    w = rng.rand(4, 4, 5)
    uvalue = compute_umatrix_pak(w, "hexagonal")
    # All unit cells (even/even) must be filled with a finite median value.
    units = uvalue[0::2, 0::2]
    assert np.all(np.isfinite(units))
    assert np.all(units >= 0)


def test_unknown_topology_raises():
    w = np.random.RandomState(0).rand(3, 3, 2)
    with pytest.raises(SomkitError):
        compute_umatrix_pak(w, "triangular")


@pytest.mark.parametrize("shape", [(1, 5, 3), (5, 1, 3), (1, 1, 3)])
def test_degenerate_map_raises(shape):
    """Maps smaller than 2x2 raise SomkitError instead of an IndexError."""
    w = np.random.RandomState(0).rand(*shape)
    with pytest.raises(SomkitError):
        compute_umatrix_pak(w, "hexagonal")


def test_identical_weights_give_zero_umatrix():
    """All-identical reference vectors -> every inter-unit distance is zero."""
    w = np.full((4, 4, 3), 0.7)
    uvalue = compute_umatrix_pak(w, "hexagonal")
    assert np.allclose(uvalue, 0.0)


def test_plot_umatrix_pak_with_labels_smoke(tmp_path):
    """plot_umatrix(style='pak', show_labels=True) renders without error.

    Smoke test: builds a tiny labeled trainer, draws the per-unit vcal labels
    (no per-sample markers) and saves the figure. Asserts the file is produced.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; pyplot is imported via somkit.visualizer
    import somkit

    rng = np.random.RandomState(3)
    x = np.vstack([rng.rand(8, 4), rng.rand(8, 4) + 3.0])  # two separable blobs
    som = somkit.create_trainer(
        data=x, size=(4, 4), learning_rate=0.3, initial_radius=4,
        random_seed=1, topology="hexagonal", checkpoint_interval=10**9,
    )
    som.target = np.array([0] * 8 + [1] * 8)
    som.target_names = np.array(["A", "B"])
    som.initialize_weights_randomly()
    som.train_pak(rlen=200, alpha=0.3, radius=4, neighborhood="bubble", seed=1)

    viz = somkit.SOMVisualizer(som)
    out = tmp_path / "umat.png"
    viz.plot_umatrix(show_labels=True, file_name=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0

    # show_nodes works independently of show_labels (nodes-only render)
    out2 = tmp_path / "umat_nodes.png"
    viz.plot_umatrix(
        show_labels=False, show_nodes=True, file_name=str(out2), show=False
    )
    assert out2.exists() and out2.stat().st_size > 0


def test_legacy_distance_map_api_unchanged():
    """The legacy per-unit distance_map is untouched by SPEC-0003.

    The SOM_PAK path is a separate function (compute_umatrix_pak); distance_map
    must keep its one-cell-per-unit contract for the legacy view / diagnostics.
    """
    w = np.random.RandomState(0).rand(5, 7, 3)
    dm = distance_map(w)
    assert dm.shape == (5, 7)  # one value per unit, not the (2n-1) interpolated grid
    assert np.all(np.isfinite(dm))
    assert np.all(dm >= 0)
