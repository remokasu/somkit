"""Tests for the SOM_PAK ``visual`` port (SPEC-0004 FR-2).

References:
    - visual.c:47-155 (compute_visual_data: BMU coords, sqrt(diff) qerror,
      empty samples as (-1, -1, -1), winner-unit labels)
"""

import os

import numpy as np
import pytest

from somkit import SOMTrainer, create_trainer, load_som_pak_data
from somkit.data_loader import SOMData
from somkit.exceptions import SomkitError
from somkit.functions.learning import find_bmu_pak

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def _make_trainer(n_samples: int = 12, dim: int = 4, size=(5, 4)):
    rng = np.random.RandomState(0)
    data = rng.rand(n_samples, dim)
    som = create_trainer(data=data, size=size, learning_rate=0.05, random_seed=1)
    som.initialize_weights_randomly()
    return som, data


class TestComputeVisual:
    def test_shapes_and_bmu_match_find_bmu_pak(self):
        som, data = _make_trainer()
        res = som.compute_visual()

        assert res.coords.shape == (len(data), 2)
        assert res.qerrors.shape == (len(data),)
        expected = np.array([find_bmu_pak(som.weights, s) for s in data])
        np.testing.assert_array_equal(res.coords, expected)

    def test_qerror_is_l2_distance_to_bmu(self):
        som, data = _make_trainer()
        res = som.compute_visual()

        for i, (x, y) in enumerate(res.coords):
            expected = np.linalg.norm(data[i] - som.weights[x, y])
            assert res.qerrors[i] == pytest.approx(expected)

    def test_explicit_data_overrides_training_data(self):
        som, _ = _make_trainer()
        other = np.random.RandomState(7).rand(3, 4)
        res = som.compute_visual(other)
        assert res.coords.shape == (3, 2)

    def test_fully_masked_sample_yields_sentinel(self):
        rng = np.random.RandomState(0)
        data = rng.rand(6, 4)
        mask = np.zeros((6, 4), dtype=bool)
        mask[2, :] = True  # sample 2: every component masked -> no BMU
        som = create_trainer(
            data=SOMData(data=data, mask=mask), size=(3, 3),
            learning_rate=0.05, random_seed=1,
        )
        som.initialize_weights_randomly()

        res = som.compute_visual()
        np.testing.assert_array_equal(res.coords[2], [-1, -1])
        assert res.qerrors[2] == -1.0
        # other samples are unaffected
        assert (res.coords[0] >= 0).all()

    def test_dimension_mismatch_raises(self):
        som, _ = _make_trainer(dim=4)
        with pytest.raises(SomkitError):
            som.compute_visual(np.zeros((3, 7)))

    def test_labels_come_from_bmu_unit(self):
        som, data = _make_trainer()
        som.target = np.arange(len(data)) % 3
        som.target_names = ["a", "b", "c"]
        unit_labels = som.calibrate_labels(numlabs=1)

        res = som.compute_visual(unit_labels=unit_labels)
        assert res.labels is not None
        for i, (x, y) in enumerate(res.coords):
            assert res.labels[i] == list(unit_labels[x, y])


class TestWriteVis:
    def test_vis_file_format(self, tmp_path):
        som, data = _make_trainer(size=(5, 4))
        path = tmp_path / "out.vis"
        som.write_vis(str(path))

        lines = path.read_text().splitlines()
        assert lines[0].split() == ["3", "hexa", "5", "4", "bubble"]
        assert len(lines) == 1 + len(data)
        res = som.compute_visual()
        for i, line in enumerate(lines[1:]):
            tokens = line.split()
            assert int(tokens[0]) == res.coords[i, 0]
            assert int(tokens[1]) == res.coords[i, 1]
            assert float(tokens[2]) == pytest.approx(res.qerrors[i], rel=1e-5)

    def test_vis_empty_sample_row(self, tmp_path):
        rng = np.random.RandomState(0)
        data = rng.rand(4, 3)
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, :] = True
        som = create_trainer(
            data=SOMData(data=data, mask=mask), size=(3, 3),
            learning_rate=0.05, random_seed=1,
        )
        som.initialize_weights_randomly()

        path = tmp_path / "out.vis"
        som.write_vis(str(path))
        lines = path.read_text().splitlines()
        assert lines[2].split() == ["-1", "-1", "-1"]


def _read_vis(path):
    """Parse a .vis file into (header_tokens, rows of token lists)."""
    lines = open(path).read().splitlines()
    return lines[0].split(), [line.split() for line in lines[1:] if line.strip()]


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN, "ex_trained.vis")),
    reason="SOM_PAK golden .vis files not generated",
)
class TestVisGolden:
    """Layer-B conformance: write_vis output vs SOM_PAK visual output."""

    def test_ex_trained_vis_matches_som_pak(self, tmp_path):
        som = SOMTrainer.load_cod(os.path.join(GOLDEN, "ex_trained.cod"))
        data = np.loadtxt(
            os.path.join(GOLDEN, "ex.dat"), skiprows=1,
            usecols=range(som.input_dim), dtype=np.float64,
        )
        som.set_data(data)
        ours_path = tmp_path / "ours.vis"
        som.write_vis(str(ours_path))

        golden_header, golden_rows = _read_vis(os.path.join(GOLDEN, "ex_trained.vis"))
        header, rows = _read_vis(str(ours_path))
        assert header == golden_header
        assert len(rows) == len(golden_rows)
        for ours, gold in zip(rows, golden_rows):
            assert ours[0] == gold[0] and ours[1] == gold[1]  # BMU coords exact
            assert float(ours[2]) == pytest.approx(float(gold[2]), rel=1e-4)

    def test_animal_cal_vis_matches_som_pak_including_labels(self, tmp_path):
        som = SOMTrainer.load_cod(os.path.join(GOLDEN, "animal_cal.cod"))
        bunch = load_som_pak_data(os.path.join(GOLDEN, "animal.dat"))
        som.set_data(bunch)
        unit_labels = som.calibrate_labels(numlabs=1)
        ours_path = tmp_path / "ours.vis"
        som.write_vis(str(ours_path), unit_labels=unit_labels)

        golden_header, golden_rows = _read_vis(os.path.join(GOLDEN, "animal_cal.vis"))
        header, rows = _read_vis(str(ours_path))
        assert header == golden_header
        assert len(rows) == len(golden_rows)
        for ours, gold in zip(rows, golden_rows):
            assert ours[0] == gold[0] and ours[1] == gold[1]
            assert float(ours[2]) == pytest.approx(float(gold[2]), rel=1e-4)
            assert ours[3:] == gold[3:]  # BMU unit labels
