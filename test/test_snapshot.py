"""Tests for train_pak training snapshots (SPEC-0004 FR-3).

References:
    - som_rout.c:652 (save when ``(le % interval == 0) && (le > 0)``)
    - lvq_pak.c:515-536 (save_snapshot: filename from template, #iterations
      comment)
"""

import os

import numpy as np
import pytest

from somkit import SOMTrainer, create_trainer, read_cod
from somkit.exceptions import SomkitError

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def _make_trainer(seed: int = 1):
    rng = np.random.RandomState(0)
    data = rng.rand(20, 4)
    som = create_trainer(data=data, size=(4, 4), learning_rate=0.05, random_seed=seed)
    som.initialize_weights_randomly()
    return som


class TestTrainPakSnapshot:
    def test_snapshot_files_created_at_interval_steps(self, tmp_path):
        som = _make_trainer()
        som.train_pak(
            rlen=100, alpha=0.05, radius=3.0, seed=1,
            snapshot_interval=25, snapshot_path=str(tmp_path / "map.cod"),
        )
        # le==0 and the final step are not saved (som_rout.c:652)
        created = sorted(p.name for p in tmp_path.iterdir())
        assert created == ["map_00025.cod", "map_00050.cod", "map_00075.cod"]

    def test_snapshot_readable_and_consistent(self, tmp_path):
        som = _make_trainer()
        som.train_pak(
            rlen=50, alpha=0.05, radius=3.0, seed=1,
            snapshot_interval=25, snapshot_path=str(tmp_path / "map.cod"),
        )
        result = read_cod(str(tmp_path / "map_00025.cod"))
        assert result.header["topol"] == "hexa"
        assert result.header["neigh"] == "bubble"
        assert result.weights.shape == som.weights.shape

    def test_no_snapshot_when_interval_exceeds_rlen(self, tmp_path):
        som = _make_trainer()
        som.train_pak(
            rlen=10, alpha=0.05, radius=3.0, seed=1,
            snapshot_interval=50, snapshot_path=str(tmp_path / "map.cod"),
        )
        assert list(tmp_path.iterdir()) == []

    def test_snapshot_does_not_change_training_result(self, tmp_path):
        plain = _make_trainer()
        plain.train_pak(rlen=100, alpha=0.05, radius=3.0, seed=1)

        snapped = _make_trainer()
        snapped.train_pak(
            rlen=100, alpha=0.05, radius=3.0, seed=1,
            snapshot_interval=25, snapshot_path=str(tmp_path / "map.cod"),
        )
        np.testing.assert_array_equal(plain.weights, snapped.weights)

    def test_interval_without_path_raises(self):
        som = _make_trainer()
        with pytest.raises(SomkitError):
            som.train_pak(rlen=10, alpha=0.05, radius=3.0, snapshot_interval=5)

    def test_path_without_interval_raises(self):
        som = _make_trainer()
        with pytest.raises(SomkitError):
            som.train_pak(
                rlen=10, alpha=0.05, radius=3.0, snapshot_path="map.cod"
            )

    def test_missing_parent_directory_raises(self, tmp_path):
        som = _make_trainer()
        with pytest.raises(SomkitError):
            som.train_pak(
                rlen=10, alpha=0.05, radius=3.0, seed=1,
                snapshot_interval=5,
                snapshot_path=str(tmp_path / "nope" / "map.cod"),
            )

    def test_snapshot_comment_records_iterations(self, tmp_path):
        som = _make_trainer()
        som.train_pak(
            rlen=50, alpha=0.05, radius=3.0, seed=1,
            snapshot_interval=25, snapshot_path=str(tmp_path / "map.cod"),
        )
        text = (tmp_path / "map_00025.cod").read_text()
        assert "#iterations: 25 (50 total)" in text


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN, "ex_snap_00500.cod")),
    reason="SOM_PAK snapshot golden not generated",
)
def test_snapshot_matches_som_pak_golden(tmp_path):
    """Layer-B conformance: somkit's step-500 snapshot equals SOM_PAK's."""
    som = SOMTrainer.load_cod(os.path.join(GOLDEN, "ex_init.cod"))
    data = np.loadtxt(
        os.path.join(GOLDEN, "ex.dat"), skiprows=1,
        usecols=range(som.input_dim), dtype=np.float64,
    )
    som.set_data(data)
    som.train_pak(
        rlen=1000, alpha=0.05, radius=10.0, seed=1,
        snapshot_interval=500, snapshot_path=str(tmp_path / "ex.cod"),
    )
    _, golden_weights = read_cod(os.path.join(GOLDEN, "ex_snap_00500.cod"))
    _, ours = read_cod(str(tmp_path / "ex_00500.cod"))
    np.testing.assert_allclose(ours, golden_weights, atol=1e-2)


def test_interval_zero_disables_snapshots(tmp_path):
    """interval=0 means disabled (no ZeroDivisionError, no files)."""
    som = _make_trainer()
    som.train_pak(rlen=10, alpha=0.05, radius=3.0, seed=1, snapshot_interval=0)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_file_comment_matches_som_pak(tmp_path):
    """Snapshot comments mirror SOM_PAK save_snapshot (lvq_pak.c:529-531)."""
    som = _make_trainer()
    som.train_pak(
        rlen=50, alpha=0.05, radius=3.0, seed=1,
        snapshot_interval=25, snapshot_path=str(tmp_path / "map.cod"),
    )
    lines = (tmp_path / "map_00025.cod").read_text().splitlines()
    assert lines[1] == "#SNAPSHOT FILE"
    assert lines[2] == "#iterations: 25 (50 total)"
