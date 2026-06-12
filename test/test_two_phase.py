"""Tests for two-phase (coarse -> fine) training (SPEC-0001 FR-3).

``train_two_phase`` must be a thin wrapper that calls ``train_pak`` twice with
no duplicated logic, matching SOM_PAK's command.sh Step2 (two vsom runs).

References:
    - command.sh Step2 (rlen=1000,alpha=0.05,r=10) -> (rlen=10000,alpha=0.02,r=3)
"""

import numpy as np
import pytest

from somkit import SOMTrainer


def _make(seed=1):
    # data is always fixed; seed controls SOMTrainer's weight-init RNG.
    data = np.random.RandomState(0).rand(20, 3)
    return SOMTrainer(
        data=data, size=(6, 6), input_dim=3, learning_rate=0.1, random_seed=seed
    )


class TestTrainTwoPhase:
    def test_equivalent_to_two_train_pak_calls(self):
        # The wrapper must produce bit-identical weights to two manual calls.
        p1 = dict(rlen=200, alpha=0.1, radius=3.0, seed=1)
        p2 = dict(rlen=300, alpha=0.05, radius=1.5, seed=2)

        a = _make(seed=42)
        a.train_two_phase(phase1=p1, phase2=p2)

        b = _make(seed=42)
        b.train_pak(**p1)
        b.train_pak(**p2)

        np.testing.assert_array_equal(a.weights, b.weights)

    def test_reproducible(self):
        p1 = dict(rlen=200, alpha=0.1, radius=3.0, seed=1)
        p2 = dict(rlen=300, alpha=0.05, radius=1.5, seed=2)
        a = _make(seed=7)
        a.train_two_phase(phase1=p1, phase2=p2)
        b = _make(seed=7)
        b.train_two_phase(phase1=p1, phase2=p2)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_shape_and_finite(self):
        som = _make()
        som.train_two_phase(
            phase1=dict(rlen=100, alpha=0.1, radius=3.0),
            phase2=dict(rlen=200, alpha=0.05, radius=1.5),
        )
        assert som.weights.shape == (6, 6, 3)
        assert np.all(np.isfinite(som.weights))

    def test_coarse_to_fine_reduces_qerror(self):
        from somkit.functions.learning import find_bmu_pak

        som = _make()
        som.initialize_weights_randomly()
        bmu0 = np.array(
            [som.weights[tuple(find_bmu_pak(som.weights, x))] for x in som.data]
        )
        qe_before = np.mean(np.linalg.norm(som.data - bmu0, axis=1))

        som.train_two_phase(
            phase1=dict(rlen=1000, alpha=0.3, radius=4.0),
            phase2=dict(rlen=2000, alpha=0.1, radius=1.5),
        )
        bmu1 = np.array(
            [som.weights[tuple(find_bmu_pak(som.weights, x))] for x in som.data]
        )
        qe_after = np.mean(np.linalg.norm(som.data - bmu1, axis=1))
        assert qe_after < qe_before

    def test_phase2_continues_from_phase1(self):
        # After phase1 the weights are already set; phase2 must continue from
        # them (not re-initialize), i.e. two_phase != phase2-only.
        som_two = _make(seed=3)
        som_two.train_two_phase(
            phase1=dict(rlen=300, alpha=0.3, radius=4.0, seed=1),
            phase2=dict(rlen=300, alpha=0.1, radius=1.5, seed=2),
        )
        som_one = _make(seed=3)
        som_one.train_pak(rlen=300, alpha=0.1, radius=1.5, seed=2)
        assert not np.array_equal(som_two.weights, som_one.weights)

    def test_invalid_phase_dict_raises(self):
        som = _make()
        with pytest.raises(TypeError):
            # missing required positional arg 'radius' -> train_pak TypeError
            som.train_two_phase(
                phase1=dict(rlen=100, alpha=0.1),
                phase2=dict(rlen=100, alpha=0.1, radius=1.5),
            )
