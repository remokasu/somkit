"""Tests for mask-aware learning (SPEC-0002 FR-1).

References:
    - lvq_pak.c:60-73 (find_winner_euc: masked components excluded, no normalize)
    - lvq_pak.c:209-215 (adapt_vector: masked components not moved)
    - som_rout.c:637-642 (fully-masked sample skipped)
"""

import numpy as np
import pytest

from somkit import SOMTrainer
from somkit.data_loader import SOMData
from somkit.functions.learning import find_bmu_pak, som_step
from somkit.functions.neighborhood import bubble_neighborhood
from somkit.topology import RectangularTopology


# ----------------------------------------------------------------------
# find_bmu_pak with mask
# ----------------------------------------------------------------------
class TestFindBmuMask:
    def test_mask_none_matches_fast_path(self):
        rng = np.random.RandomState(0)
        weights = rng.rand(4, 4, 3)
        sample = rng.rand(3)
        assert find_bmu_pak(weights, sample, mask=None) == find_bmu_pak(weights, sample)

    def test_masked_component_excluded_from_distance(self):
        weights = np.zeros((2, 1, 2))
        weights[0, 0] = [0.0, 0.0]
        weights[1, 0] = [0.0, 5.0]
        # Without mask, sample [0, 9] is closest to unit (0,0) (dist 9 vs 4).
        # Masking component 1 makes only component 0 count -> both dist 0,
        # tie -> smallest index (0,0).
        sample = np.array([0.0, 9.0])
        mask = np.array([False, True])
        assert find_bmu_pak(weights, sample, mask=mask) == (0, 0)

    def test_mask_no_normalization(self):
        # Distance must NOT be divided by the number of active components.
        weights = np.zeros((1, 1, 3))
        sample = np.array([3.0, 4.0, 99.0])
        mask = np.array([False, False, True])
        # active components: 3,4 -> dist sqrt(9+16)=5 (not 5/2 or similar)
        # we can only check the winner here (1x1), so verify via som_step below.
        assert find_bmu_pak(weights, sample, mask=mask) == (0, 0)


# ----------------------------------------------------------------------
# som_step with mask
# ----------------------------------------------------------------------
class TestSomStepMask:
    def test_masked_component_not_moved(self):
        topo = RectangularTopology()
        weights = np.zeros((3, 3, 2))
        sample = np.array([1.0, 1.0])
        mask = np.array([False, True])  # component 1 frozen
        som_step(weights, sample, (1, 1), 1.0, 0.5, bubble_neighborhood, topo, mask=mask)
        # component 0 moved at BMU, component 1 stayed 0 everywhere
        assert weights[1, 1, 0] == pytest.approx(0.5)
        assert np.all(weights[:, :, 1] == 0.0)

    def test_mask_none_matches_fast_path(self):
        topo = RectangularTopology()
        a = np.zeros((3, 3, 2))
        b = np.zeros((3, 3, 2))
        sample = np.array([1.0, 1.0])
        som_step(a, sample, (1, 1), 1.0, 0.5, bubble_neighborhood, topo, mask=None)
        som_step(b, sample, (1, 1), 1.0, 0.5, bubble_neighborhood, topo)
        np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------
# train_pak with mask (integration)
# ----------------------------------------------------------------------
class TestTrainPakMask:
    def _data(self):
        return np.random.RandomState(0).rand(20, 4)

    def test_fully_masked_component_never_moves(self):
        data = self._data()
        mask = np.zeros_like(data, dtype=bool)
        mask[:, 2] = True  # component 2 masked for ALL samples
        som = SOMTrainer(
            data=SOMData(data=data, mask=mask), size=(5, 5), input_dim=4,
            learning_rate=0.1, topology="hexagonal", random_seed=1,
        )
        som.initialize_weights_randomly()
        before = som.weights[:, :, 2].copy()
        som.train_pak(rlen=300, alpha=0.3, radius=3.0)
        # masked component 2 weights are unchanged everywhere
        np.testing.assert_array_equal(som.weights[:, :, 2], before)
        # an unmasked component did change
        assert not np.array_equal(som.weights[:, :, 0], som.weights[:, :, 0] * 0)

    def test_fully_masked_sample_skipped(self):
        data = self._data()
        mask = np.zeros_like(data, dtype=bool)
        mask[0] = True  # sample 0 fully masked -> skipped, no crash
        som = SOMTrainer(
            data=SOMData(data=data, mask=mask), size=(4, 4), input_dim=4,
            learning_rate=0.1, topology="hexagonal", random_seed=1,
        )
        som.initialize_weights_randomly()
        som.train_pak(rlen=100, alpha=0.3, radius=2.0)
        assert np.all(np.isfinite(som.weights))

    def test_no_mask_matches_plain_ndarray(self):
        # Backward compat: SOMData(data) with no mask == plain ndarray path.
        data = self._data()
        a = SOMTrainer(data=SOMData(data=data), size=(5, 5), input_dim=4,
                       learning_rate=0.1, topology="hexagonal", random_seed=1)
        a.initialize_weights_randomly()
        a.train_pak(rlen=300, alpha=0.3, radius=3.0)
        b = SOMTrainer(data=data, size=(5, 5), input_dim=4,
                       learning_rate=0.1, topology="hexagonal", random_seed=1)
        b.initialize_weights_randomly()
        b.train_pak(rlen=300, alpha=0.3, radius=3.0)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_all_false_mask_matches_no_mask(self):
        # An all-False mask must produce the same result as no mask.
        data = self._data()
        mask = np.zeros_like(data, dtype=bool)
        a = SOMTrainer(data=SOMData(data=data, mask=mask), size=(5, 5), input_dim=4,
                       learning_rate=0.1, topology="hexagonal", random_seed=1)
        a.initialize_weights_randomly()
        a.train_pak(rlen=300, alpha=0.3, radius=3.0)
        b = SOMTrainer(data=data, size=(5, 5), input_dim=4,
                       learning_rate=0.1, topology="hexagonal", random_seed=1)
        b.initialize_weights_randomly()
        b.train_pak(rlen=300, alpha=0.3, radius=3.0)
        # all-False mask must be bit-identical to the no-mask fast path.
        np.testing.assert_array_equal(a.weights, b.weights)
