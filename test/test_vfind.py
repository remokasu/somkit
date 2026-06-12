"""Tests for the SOM_PAK ``vfind`` port (SPEC-0004 FR-1).

References:
    - vfind.c:245-304 (trial loop: init_random(not) -> two-phase training ->
      qerror comparison with strict ``<``)
"""

import logging

import numpy as np
import pytest

from somkit import SOMTrainer
from somkit.exceptions import SomkitError

PHASE1 = dict(rlen=100, alpha=0.05, radius=4.0)
PHASE2 = dict(rlen=200, alpha=0.02, radius=2.0)


def _data(n=24, dim=4, seed=0):
    return np.random.RandomState(seed).rand(n, dim)


class TestVfind:
    def test_returns_trained_trainer_with_metadata(self):
        data = _data()
        som = SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, phase2=PHASE2, n_trials=3
        )
        assert som.weights is not None
        assert som.vfind_best_seed in (1, 2, 3)
        assert set(som.vfind_qerrors) == {1, 2, 3}
        assert som.vfind_best_qerror == min(som.vfind_qerrors.values())
        assert som.vfind_best_qerror == som.vfind_qerrors[som.vfind_best_seed]

    def test_deterministic_across_runs(self):
        data = _data()
        a = SOMTrainer.vfind(data, (4, 4), phase1=PHASE1, n_trials=3)
        b = SOMTrainer.vfind(data, (4, 4), phase1=PHASE1, n_trials=3)
        assert a.vfind_best_seed == b.vfind_best_seed
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_best_matches_single_trial_rerun(self):
        data = _data()
        som = SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, phase2=PHASE2, n_trials=3
        )
        # Re-running vfind with only the winning seed must reproduce the
        # winning codebook bit-identically.
        solo = SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, phase2=PHASE2,
            n_trials=1, seeds=[som.vfind_best_seed],
        )
        np.testing.assert_array_equal(som.weights, solo.weights)

    def test_test_data_none_equals_training_data(self):
        data = _data()
        a = SOMTrainer.vfind(data, (4, 4), phase1=PHASE1, n_trials=3)
        b = SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, n_trials=3, test_data=data
        )
        assert a.vfind_best_seed == b.vfind_best_seed
        assert a.vfind_qerrors == b.vfind_qerrors

    def test_explicit_seeds(self):
        data = _data()
        som = SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, n_trials=2, seeds=[10, 20]
        )
        assert set(som.vfind_qerrors) == {10, 20}

    def test_tie_break_keeps_first_seed(self, monkeypatch):
        """Strict ``<`` comparison: equal qerrors keep the earlier trial
        (vfind.c:290)."""
        from somkit.io.vis import VisualResult

        data = _data()

        def constant_qerror(self, data=None, unit_labels=None):
            return VisualResult(
                coords=np.zeros((2, 2), dtype=int),
                qerrors=np.array([0.5, 0.5]),
                labels=None,
            )

        monkeypatch.setattr(SOMTrainer, "compute_visual", constant_qerror)
        som = SOMTrainer.vfind(data, (4, 4), phase1=PHASE1, n_trials=3)
        assert som.vfind_best_seed == 1

    def test_invalid_n_trials_raises(self):
        with pytest.raises(SomkitError):
            SOMTrainer.vfind(_data(), (4, 4), phase1=PHASE1, n_trials=0)

    def test_seeds_length_mismatch_raises(self):
        with pytest.raises(SomkitError):
            SOMTrainer.vfind(
                _data(), (4, 4), phase1=PHASE1, n_trials=3, seeds=[1, 2]
            )

    def test_phase_dict_with_seed_key_raises(self):
        with pytest.raises(SomkitError):
            SOMTrainer.vfind(
                _data(), (4, 4),
                phase1=dict(rlen=10, alpha=0.05, radius=2.0, seed=7),
                n_trials=1,
            )

    def test_logs_trial_qerrors(self, caplog):
        data = _data()
        with caplog.at_level(logging.INFO, logger="somkit.trainer.som_trainer"):
            SOMTrainer.vfind(data, (4, 4), phase1=PHASE1, n_trials=2)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "seed=1" in messages and "seed=2" in messages
        assert "qerror" in messages


def test_all_empty_test_data_raises():
    """Fully masked test data has no quantization error (explicit failure)."""
    from somkit.data_loader import SOMData

    data = _data(n=8)
    masked = SOMData(
        data=np.zeros((2, 4)), mask=np.ones((2, 4), dtype=bool)
    )
    with pytest.raises(SomkitError):
        SOMTrainer.vfind(
            data, (4, 4), phase1=PHASE1, n_trials=1, test_data=masked
        )
