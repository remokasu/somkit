"""Unit tests for SOM_PAK-conformant random initialization (SPEC-0001 FR-7).

ADR-0001 "層A" (strict-conformance) tests: with a fixed dataset and a seeded
``OrandRNG``, ``random_init`` must reproduce the SOM_PAK ``randinit_codes``
scan order and values exactly.

References:
    - som_rout.c:34-162 (``randinit_codes``)
    - som_rout.c:146-147 (``points[i] = mival[i] + (maval[i]-mival[i]) * orand()/32768.0``)
"""

import numpy as np
import pytest

from somkit.functions.initialization import random_init
from somkit.functions.rng import OrandRNG


def _reference_random_init(data, x_size, y_size, seed):
    """Reimplementation of SOM_PAK's randinit scan order.

    Entry order is ``index = y*xdim + x`` (y outer, x inner); within each entry
    the components ``i=0..dim-1`` consume one ``orand()`` draw each.

    Note:
        ``OrandRNG`` is tested independently in FR-6 (test_rng.py). The
        independence here is in the *scan-order logic* (this explicit triple
        loop vs. ``random_init``'s reshape/transpose), not in the RNG itself.
    """
    mival = data.min(axis=0)
    maval = data.max(axis=0)
    dim = data.shape[1]
    rng = OrandRNG(seed)
    weights = np.zeros((x_size, y_size, dim))
    for y in range(y_size):
        for x in range(x_size):
            for i in range(dim):
                weights[x, y, i] = mival[i] + (maval[i] - mival[i]) * rng.uniform()
    return weights


@pytest.fixture
def data():
    return np.array(
        [
            [0.0, 10.0, -5.0],
            [2.0, 14.0, -1.0],
            [4.0, 12.0, -3.0],
        ]
    )


class TestScanOrderAndValues:
    def test_matches_reference_exactly(self, data):
        produced = random_init(data, 3, 4, OrandRNG(1))
        expected = _reference_random_init(data, 3, 4, seed=1)
        np.testing.assert_array_equal(produced, expected)

    @pytest.mark.parametrize("seed", [1, 7, 9999])
    @pytest.mark.parametrize("size", [(2, 2), (3, 4), (5, 1), (1, 5)])
    def test_matches_reference_various(self, data, seed, size):
        x_size, y_size = size
        produced = random_init(data, x_size, y_size, OrandRNG(seed))
        expected = _reference_random_init(data, x_size, y_size, seed)
        np.testing.assert_array_equal(produced, expected)


class TestShapeAndRange:
    def test_shape(self, data):
        w = random_init(data, 3, 4, OrandRNG(1))
        assert w.shape == (3, 4, data.shape[1])

    def test_within_component_min_max(self, data):
        # Upper bound is strict: uniform() max = 32766/32768 < 1.0, so weights
        # approach but never reach maval.
        w = random_init(data, 6, 6, OrandRNG(3))
        mival = data.min(axis=0)
        maval = data.max(axis=0)
        for i in range(data.shape[1]):
            assert np.all(w[:, :, i] >= mival[i])
            assert np.all(w[:, :, i] <= maval[i])


class TestConstantComponent:
    def test_constant_component_stays_constant(self):
        # Component 1 is identical across all rows -> max == min -> constant.
        data = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
        w = random_init(data, 4, 4, OrandRNG(1))
        assert np.all(w[:, :, 1] == 5.0)


class TestReproducibility:
    def test_same_seed_identical(self, data):
        a = random_init(data, 4, 4, OrandRNG(2024))
        b = random_init(data, 4, 4, OrandRNG(2024))
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self, data):
        a = random_init(data, 4, 4, OrandRNG(1))
        b = random_init(data, 4, 4, OrandRNG(2))
        assert not np.array_equal(a, b)


class TestEmptyData:
    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            random_init(np.empty((0, 3)), 2, 2, OrandRNG(1))


class TestTrainerIntegration:
    def _make(self, seed=42):
        from somkit import SOMTrainer

        data = np.array(
            [[0.0, 10.0, -5.0], [2.0, 14.0, -1.0], [4.0, 12.0, -3.0], [1.0, 11.0, -2.0]]
        )
        return SOMTrainer(
            data=data, size=(5, 5), input_dim=3, learning_rate=0.1, random_seed=seed
        )

    def test_randomly_shape_and_range(self):
        som = self._make()
        som.initialize_weights_randomly()
        assert som.weights.shape == (5, 5, 3)
        mival = som.data.min(axis=0)
        maval = som.data.max(axis=0)
        for i in range(3):
            assert np.all(som.weights[:, :, i] >= mival[i])
            assert np.all(som.weights[:, :, i] <= maval[i])

    def test_randomly_reproducible_with_seed(self):
        a = self._make(seed=42)
        a.initialize_weights_randomly()
        b = self._make(seed=42)
        b.initialize_weights_randomly()
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_randomly_accepts_injected_rng(self):
        som = self._make()
        som.initialize_weights_randomly(rng=OrandRNG(7))
        expected = random_init(som.data, 5, 5, OrandRNG(7))
        np.testing.assert_array_equal(som.weights, expected)

    def test_uniform_preserves_legacy_unit_interval(self):
        som = self._make()
        som.initialize_weights_uniform()
        assert som.weights.shape == (5, 5, 3)
        assert np.all(som.weights >= 0.0)
        assert np.all(som.weights < 1.0)
