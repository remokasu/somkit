"""Tests for the public SOM_PAK .cod I/O API (SPEC-0002 FR-4).

References:
    - datafile.c:433-480 (write_header / write_entry)
    - som_rout.c:643-644 (index = y*xdim + x)
"""

import os

import numpy as np
import pytest

from somkit import SOMTrainer
from somkit.exceptions import CodFormatError, SomkitError
from somkit.io.cod import CodResult, flat_index, read_cod, write_cod

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
golden_available = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN, "ex_init.cod")),
    reason="golden files missing; run `bash test/golden/regenerate.sh`",
)


class TestFlatIndex:
    def test_value(self):
        # index = y*xdim + x
        assert flat_index(0, 0, 10) == 0
        assert flat_index(2, 0, 10) == 2
        assert flat_index(0, 1, 10) == 10
        assert flat_index(3, 4, 10) == 43


class TestReadCod:
    @golden_available
    def test_reads_som_pak_golden(self):
        result = read_cod(os.path.join(GOLDEN, "ex_init.cod"))
        assert isinstance(result, CodResult)
        assert result.header == {
            "dim": 5, "topol": "hexa", "xdim": 10, "ydim": 10, "neigh": "bubble",
        }
        assert result.weights.shape == (10, 10, 5)
        # namedtuple unpacking stays compatible
        header, weights = read_cod(os.path.join(GOLDEN, "ex_init.cod"))
        assert header["dim"] == 5 and weights.shape == (10, 10, 5)

    def test_ignores_trailing_label_tokens(self, tmp_path):
        p = tmp_path / "labeled.cod"
        p.write_text("2 hexa 1 2 bubble\n0.5 0.6 catA\n1.0 2.0 catB\n")
        _, weights = read_cod(str(p))
        assert weights.shape == (1, 2, 2)
        np.testing.assert_array_equal(weights[0, 0], [0.5, 0.6])
        np.testing.assert_array_equal(weights[0, 1], [1.0, 2.0])

    def test_skips_comment_lines(self, tmp_path):
        p = tmp_path / "c.cod"
        p.write_text("2 rect 2 1 bubble\n# a comment\n0.1 0.2 \n0.3 0.4 \n")
        _, weights = read_cod(str(p))
        assert weights.shape == (2, 1, 2)


class TestReadCodErrors:
    def test_missing_header_fields(self, tmp_path):
        p = tmp_path / "bad.cod"
        p.write_text("2 hexa 2\n0.1 0.2\n")
        with pytest.raises(CodFormatError):
            read_cod(str(p))

    def test_vector_count_mismatch(self, tmp_path):
        p = tmp_path / "short.cod"
        p.write_text("2 hexa 2 2 bubble\n0.1 0.2\n0.3 0.4\n")  # 2 vectors, need 4
        with pytest.raises(CodFormatError):
            read_cod(str(p))

    def test_codformat_is_somkit_error_and_value_error(self):
        assert issubclass(CodFormatError, SomkitError)
        assert issubclass(SomkitError, ValueError)


class TestWriteRoundTrip:
    def test_round_trip_numeric(self, tmp_path):
        rng = np.random.RandomState(0)
        weights = rng.rand(4, 3, 5)
        p = tmp_path / "rt.cod"
        write_cod(str(p), weights, topol="hexa", neigh="bubble")
        result = read_cod(str(p))
        assert result.header == {
            "dim": 5, "topol": "hexa", "xdim": 4, "ydim": 3, "neigh": "bubble",
        }
        # %g keeps ~6 significant digits
        np.testing.assert_allclose(result.weights, weights, atol=1e-5)

    def test_round_trip_preserves_index_order(self, tmp_path):
        # Distinct per-unit values so a [x,y]/[y,x] swap would be caught.
        weights = np.zeros((3, 2, 1))
        for x in range(3):
            for y in range(2):
                weights[x, y, 0] = x * 10 + y
        p = tmp_path / "ord.cod"
        write_cod(str(p), weights, topol="rect", neigh="bubble")
        _, back = read_cod(str(p))
        np.testing.assert_allclose(back, weights, atol=1e-9)

    def test_round_trip_with_labels(self, tmp_path):
        weights = np.zeros((2, 2, 1))
        labels = np.empty((2, 2), dtype=object)
        for x in range(2):
            for y in range(2):
                labels[x, y] = [f"u{x}{y}"]
        p = tmp_path / "lab.cod"
        write_cod(str(p), weights, topol="hexa", neigh="bubble", labels=labels)
        # labels are trailing tokens; read still recovers the numeric weights
        _, back = read_cod(str(p))
        assert back.shape == (2, 2, 1)
        # the label tokens are present in the file
        assert "u01" in (tmp_path / "lab.cod").read_text()


class TestTrainerCodMethods:
    def _trained(self):
        data = np.random.RandomState(1).rand(20, 4)
        som = SOMTrainer(data=data, size=(5, 5), input_dim=4, learning_rate=0.1,
                         topology="hexagonal")
        som.initialize_weights_randomly()
        return som

    def test_save_then_load(self, tmp_path):
        som = self._trained()
        p = tmp_path / "model.cod"
        som.save_cod(str(p))
        loaded = SOMTrainer.load_cod(str(p))
        np.testing.assert_allclose(loaded.weights, som.weights, atol=1e-5)
        assert loaded.weights.shape == som.weights.shape

    def test_save_writes_topology_string(self, tmp_path):
        som = self._trained()
        p = tmp_path / "m.cod"
        som.save_cod(str(p))
        header, _ = read_cod(str(p))
        assert header["topol"] == "hexa"

    def test_save_without_weights_raises(self):
        data = np.random.RandomState(1).rand(5, 4)
        som = SOMTrainer(data=data, size=(5, 5), input_dim=4, learning_rate=0.1)
        # weights not initialized
        with pytest.raises(SomkitError):
            som.save_cod("/tmp/should_not_write.cod")
