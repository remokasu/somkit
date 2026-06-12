"""Tests for vcal label calibration (SPEC-0002 FR-5).

References:
    - vcal.c:105-161 (find_labels / numlabs)
    - labels.c:365-402 (add_hit: frequency-descending, tie = arrival order)
"""

import numpy as np
import pytest

from somkit import SOMTrainer
from somkit.exceptions import SomkitError
from somkit.functions.labels import calibrate_labels
from somkit.io.cod import read_cod


def _single_unit(labels, numlabs=0):
    """Calibrate a 1x1 map: all samples hit unit (0,0). Returns that cell's list."""
    bmus = [(0, 0)] * len(labels)
    grid = calibrate_labels(bmus, labels, 1, 1, numlabs=numlabs)
    return grid[0, 0]


class TestHitlistOrder:
    def test_tie_arrival_order_abab(self):
        # A,B,A,B: both reach freq 2; A reaches it first -> [A, B]
        assert _single_unit(["A", "B", "A", "B"]) == ["A", "B"]

    def test_tie_arrival_order_baba(self):
        assert _single_unit(["B", "A", "B", "A"]) == ["B", "A"]

    def test_majority_moves_to_front(self):
        # A,B,B: B has freq 2 -> front
        assert _single_unit(["A", "B", "B"]) == ["B", "A"]

    def test_three_way_tie_keeps_arrival_order(self):
        assert _single_unit(["X", "Y", "Z"]) == ["X", "Y", "Z"]

    def test_descending_frequency(self):
        # C x3, B x2, A x1
        labels = ["A", "B", "B", "C", "C", "C"]
        assert _single_unit(labels) == ["C", "B", "A"]


class TestNumlabs:
    def test_default_numlabs_1_keeps_top(self):
        grid = calibrate_labels([(0, 0)] * 3, ["A", "B", "B"], 1, 1, numlabs=1)
        assert grid[0, 0] == ["B"]

    def test_numlabs_0_keeps_all(self):
        assert _single_unit(["A", "B", "B"], numlabs=0) == ["B", "A"]

    def test_numlabs_2(self):
        grid = calibrate_labels(
            [(0, 0)] * 6, ["A", "B", "B", "C", "C", "C"], 1, 1, numlabs=2
        )
        assert grid[0, 0] == ["C", "B"]


class TestEmptyAndMissing:
    def test_empty_label_ignored(self):
        # "" samples are not counted (LABEL_EMPTY)
        assert _single_unit(["A", "", "", "A"]) == ["A"]

    def test_all_empty_gives_empty_unit(self):
        assert _single_unit(["", "", ""]) == []

    def test_unhit_unit_is_empty_list(self):
        grid = calibrate_labels([(0, 0)], ["A"], 2, 2, numlabs=1)
        assert grid[1, 1] == []
        assert grid[0, 0] == ["A"]


class TestShapeAndIndex:
    def test_shape(self):
        grid = calibrate_labels([(0, 0)], ["A"], 3, 4, numlabs=1)
        assert grid.shape == (3, 4)

    def test_index_placement(self):
        # A sample whose BMU is (x=2, y=1) must place its label at grid[2, 1].
        grid = calibrate_labels([(2, 1)], ["L"], 3, 3, numlabs=1)
        assert grid[2, 1] == ["L"]
        assert grid[0, 0] == []


class TestTrainerCalibrate:
    def _trained(self, seed=1):
        rng = np.random.RandomState(0)
        data = rng.rand(30, 4)
        target = np.array([i % 3 for i in range(30)])
        target_names = np.array(["a", "b", "c"])
        bunch = type("B", (), {"data": data, "target": target, "target_names": target_names})()
        som = SOMTrainer(data=bunch, size=(5, 5), input_dim=4, learning_rate=0.1,
                         topology="hexagonal", random_seed=seed)
        som.initialize_weights_randomly()
        som.train_pak(rlen=300, alpha=0.3, radius=3.0)
        return som

    def test_returns_grid(self):
        som = self._trained()
        grid = som.calibrate_labels(numlabs=1)
        assert grid.shape == (5, 5)
        # every cell is a list
        assert all(isinstance(grid[x, y], list) for x in range(5) for y in range(5))

    def test_without_weights_raises(self):
        data = np.random.RandomState(0).rand(5, 4)
        som = SOMTrainer(data=data, size=(5, 5), input_dim=4, learning_rate=0.1)
        with pytest.raises(SomkitError):
            som.calibrate_labels()

    def test_calibrate_then_save_cod_round_trip(self, tmp_path):
        som = self._trained()
        grid = som.calibrate_labels(numlabs=1)
        p = tmp_path / "cal.cod"
        som.save_cod(str(p), labels=grid)
        # labels appear in the file and weights still round-trip
        text = (tmp_path / "cal.cod").read_text()
        assert any(lab in text for lab in ["a", "b", "c"])
        _, weights = read_cod(str(p))
        np.testing.assert_allclose(weights, som.weights, atol=1e-5)
