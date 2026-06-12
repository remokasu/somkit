"""Tests for sample weights (FR-2) and fixed points (FR-3) in train_pak.

References:
    - som_rout.c:614, 624-626 (weight: talp = 1 - (1-talp)^weight)
    - som_rout.c:630-634 (fixed: BMU forced to (xfix, yfix), winner skipped)
"""

import numpy as np
import pytest

from somkit import SOMTrainer
from somkit.data_loader import SOMData
from somkit.exceptions import SOMDataError
from somkit.functions.learning import weighted_alpha


# ----------------------------------------------------------------------
# FR-2: weighted_alpha (pure function)
# ----------------------------------------------------------------------
class TestWeightedAlpha:
    def test_weight_one_is_identity(self):
        assert weighted_alpha(0.5, 1.0) == pytest.approx(0.5)

    def test_weight_two(self):
        # 1 - (1-0.5)^2 = 0.75
        assert weighted_alpha(0.5, 2.0) == pytest.approx(0.75)

    def test_weight_zero_no_change(self):
        assert weighted_alpha(0.5, 0.0) == 0.5

    def test_negative_weight_no_change(self):
        assert weighted_alpha(0.5, -3.0) == 0.5

    def test_large_weight_saturates_toward_one(self):
        # 1 - (1-alpha)^w -> 1 as w grows; in float64 it reaches exactly 1.0.
        v = weighted_alpha(0.5, 100.0)
        assert 0.99 < v <= 1.0
        # a moderate weight stays strictly below 1
        assert weighted_alpha(0.5, 10.0) < 1.0


# ----------------------------------------------------------------------
# FR-3: fixed points
# ----------------------------------------------------------------------
class TestFixedPoints:
    def _data(self):
        return np.random.RandomState(0).rand(20, 3)

    def test_fixed_bmu_is_used(self):
        # All samples fixed to unit (1,1); radius<1 (rect) updates only the BMU.
        data = self._data()
        fixed = np.tile([1, 1], (20, 1))
        som = SOMTrainer(
            data=SOMData(data=data, fixed=fixed), size=(4, 4), input_dim=3,
            learning_rate=0.5, topology="rectangular", random_seed=1,
        )
        som.initialize_weights_randomly()
        before = som.weights.copy()
        # radius 0 -> only the (fixed) BMU is updated; avoid depending on the
        # neighborhood cutoff for far units.
        som.train_pak(rlen=100, alpha=0.5, radius=0.0)
        assert not np.array_equal(som.weights[1, 1], before[1, 1])

    def test_fixed_out_of_range_raises(self):
        data = self._data()
        fixed = np.tile([4, 0], (20, 1))  # x=4 >= x_size 4
        som = SOMTrainer(
            data=SOMData(data=data, fixed=fixed), size=(4, 4), input_dim=3,
            learning_rate=0.5,
        )
        som.initialize_weights_randomly()
        with pytest.raises(SOMDataError):
            som.train_pak(rlen=10, alpha=0.5, radius=1.0)

    def test_fixed_valid_partial(self):
        # Only row 0 is a valid fixed point; others use normal BMU search.
        data = self._data()
        fixed = np.zeros((20, 2), dtype=int)
        fixed[0] = [2, 2]
        fixed_valid = np.zeros(20, dtype=bool)
        fixed_valid[0] = True
        som = SOMTrainer(
            data=SOMData(data=data, fixed=fixed, fixed_valid=fixed_valid),
            size=(4, 4), input_dim=3, learning_rate=0.5,
            topology="rectangular", random_seed=1,
        )
        som.initialize_weights_randomly()
        som.train_pak(rlen=50, alpha=0.5, radius=1.0)
        assert np.all(np.isfinite(som.weights))


# ----------------------------------------------------------------------
# Backward compatibility
# ----------------------------------------------------------------------
class TestBackwardCompat:
    def _data(self):
        return np.random.RandomState(0).rand(20, 3)

    def _train(self, somdata_or_array):
        som = SOMTrainer(
            data=somdata_or_array, size=(5, 5), input_dim=3, learning_rate=0.3,
            topology="hexagonal", random_seed=1,
        )
        som.initialize_weights_randomly()
        som.train_pak(rlen=300, alpha=0.3, radius=3.0)
        return som.weights

    def test_no_weights_no_fixed_matches_plain(self):
        data = self._data()
        a = self._train(SOMData(data=data))
        b = self._train(data)
        np.testing.assert_array_equal(a, b)

    def test_weight_all_ones_approximates_no_weight(self):
        # weight=1 takes the pow path (talp = 1-(1-talp)^1), which is talp up to
        # float rounding (1-(1-x) != x bit-exactly), so this is NOT bit-identical
        # to the no-weight fast path — only numerically close. SOM_PAK behaves the
        # same (weights, when present, always go through pow).
        data = self._data()
        a = self._train(SOMData(data=data, weights=np.ones(20)))
        b = self._train(data)
        # measured max diff ~2.2e-16 (machine epsilon from 1-(1-x) != x).
        np.testing.assert_allclose(a, b, atol=1e-12)
