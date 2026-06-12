"""Unit tests for the SOM_PAK-conformant sequential learning core (SPEC-0001 FR-2/FR-5).

ADR-0001 "層A" (strict-conformance) tests for the deterministic learning parts:
presentation order, BMU tie-break, single-step update, and the new neighborhood
I/F (gaussian without cutoff / bubble).

References:
    - som_rout.c:602-648 (som_training loop)
    - lvq_pak.c:76 (BMU strict-< tie-break), lvq_pak.c:213-214 (adapt_vector)
    - som_rout.c:497 (bubble cutoff), som_rout.c:543-546 (gaussian, no cutoff)
    - datafile.c:1144-1172 (randomize_entry_order)
"""

import numpy as np
import pytest

from somkit.functions.learning import (
    find_bmu_pak,
    presentation_order,
    som_step,
)
from somkit.functions.neighborhood import (
    bubble_neighborhood,
    gaussian_neighborhood,
    get_pak_neighborhood,
)
from somkit.functions.rng import OrandRNG
from somkit.topology import RectangularTopology


# ----------------------------------------------------------------------
# presentation_order — datafile.c:1144-1172
# ----------------------------------------------------------------------
def _reference_order(n, rlen, seed, random_order):
    rng = OrandRNG(seed)
    if random_order:
        rem = list(range(n))
        base = []
        for nol in range(n, 0, -1):
            base.append(rem.pop(rng.next_int() % nol))
    else:
        base = list(range(n))
    return np.array([base[t % n] for t in range(rlen)])


class TestPresentationOrder:
    @pytest.mark.parametrize("seed", [1, 42])
    @pytest.mark.parametrize("n,rlen", [(5, 5), (5, 12), (3, 10), (7, 3)])
    def test_random_order_matches_reference(self, seed, n, rlen):
        produced = presentation_order(n, rlen, OrandRNG(seed), random_order=True)
        expected = _reference_order(n, rlen, seed, random_order=True)
        np.testing.assert_array_equal(produced, expected)

    def test_random_order_is_permutation_per_cycle(self):
        order = presentation_order(6, 6, OrandRNG(1), random_order=True)
        assert sorted(order.tolist()) == list(range(6))

    def test_sequential_order(self):
        order = presentation_order(4, 10, OrandRNG(1), random_order=False)
        np.testing.assert_array_equal(order, np.array([0, 1, 2, 3] * 2 + [0, 1]))

    def test_cycles_when_rlen_exceeds_n(self):
        order = presentation_order(3, 7, OrandRNG(1), random_order=True)
        assert len(order) == 7
        # second cycle repeats the same (fixed) shuffled list
        assert order[0] == order[3] == order[6]
        assert order[1] == order[4]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            presentation_order(0, 5, OrandRNG(1), random_order=True)


# ----------------------------------------------------------------------
# find_bmu_pak — SOM_PAK tie-break (index = y*xdim + x, strict <)
# ----------------------------------------------------------------------
class TestFindBmuPak:
    def test_normal_winner(self):
        weights = np.zeros((3, 3, 2))
        weights[2, 1] = [5.0, 5.0]
        bmu = find_bmu_pak(weights, np.array([5.0, 5.0]))
        assert bmu == (2, 1)

    def test_tie_break_smallest_index(self):
        # Two units equidistant; SOM_PAK keeps the smallest index = y*xdim + x.
        # x_size=3. Candidates (x=2,y=0)->index 2 and (x=0,y=1)->index 3.
        # The winner must be index 2 = (x=2, y=0).
        weights = np.zeros((3, 2, 2))
        sample = np.array([1.0, 1.0])
        weights[2, 0] = [1.0, 1.0]
        weights[0, 1] = [1.0, 1.0]
        assert find_bmu_pak(weights, sample) == (2, 0)

    def test_tie_break_all_equal_returns_index_zero(self):
        # All units identical -> smallest index 0 = (x=0, y=0).
        weights = np.full((4, 4, 2), 2.0)
        assert find_bmu_pak(weights, np.array([2.0, 2.0])) == (0, 0)


# ----------------------------------------------------------------------
# neighborhood new I/F (FR-5)
# ----------------------------------------------------------------------
class TestNeighborhood:
    def test_gaussian_formula(self):
        d = np.array([0.0, 1.0, 2.0, 5.0])
        radius = 2.0
        expected = np.exp(-(d**2) / (2.0 * radius**2))
        np.testing.assert_allclose(gaussian_neighborhood(d, radius), expected, rtol=0, atol=0)

    def test_gaussian_no_cutoff_beyond_radius(self):
        # Distances well beyond the radius must still be strictly positive.
        d = np.array([3.0, 10.0])
        assert np.all(gaussian_neighborhood(d, 1.0) > 0.0)

    def test_bubble_step(self):
        d = np.array([0.0, 1.0, 2.0, 2.5])
        np.testing.assert_array_equal(
            bubble_neighborhood(d, 2.0), np.array([1.0, 1.0, 1.0, 0.0])
        )

    def test_get_pak_neighborhood_resolves(self):
        assert get_pak_neighborhood("bubble") is bubble_neighborhood
        assert get_pak_neighborhood("gaussian") is gaussian_neighborhood

    def test_get_pak_neighborhood_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown neighborhood 'nope'"):
            get_pak_neighborhood("nope")


# ----------------------------------------------------------------------
# som_step — single-step update (adapt_vector + neighborhood)
# ----------------------------------------------------------------------
class TestSomStep:
    def test_bubble_update_matches_manual(self):
        topo = RectangularTopology()
        weights = np.zeros((3, 3, 2))
        sample = np.array([1.0, 1.0])
        bmu = (1, 1)
        trad, talp = 1.0, 0.5
        before = weights.copy()
        som_step(weights, sample, bmu, trad, talp, bubble_neighborhood, topo)

        gx, gy = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        d = topo.topology_function(gx, gy, 1, 1)
        infl = bubble_neighborhood(d, trad)
        expected = before + talp * infl[:, :, None] * (sample - before)
        np.testing.assert_allclose(weights, expected, rtol=0, atol=0)

    def test_bubble_leaves_far_units_unchanged(self):
        topo = RectangularTopology()
        weights = np.zeros((5, 5, 2))
        som_step(weights, np.array([9.0, 9.0]), (0, 0), 1.0, 0.5, bubble_neighborhood, topo)
        # A unit far outside radius 1 from (0,0) stays at its initial value.
        assert np.all(weights[4, 4] == 0.0)

    def test_gaussian_updates_all_units(self):
        topo = RectangularTopology()
        weights = np.zeros((5, 5, 2))
        som_step(weights, np.array([9.0, 9.0]), (0, 0), 1.0, 0.5, gaussian_neighborhood, topo)
        # gaussian has no cutoff: even the far corner moves.
        assert np.all(weights[4, 4] != 0.0)

    def test_bmu_moves_toward_sample(self):
        topo = RectangularTopology()
        weights = np.zeros((3, 3, 2))
        sample = np.array([1.0, 1.0])
        som_step(weights, sample, (1, 1), 1.0, 0.5, bubble_neighborhood, topo)
        np.testing.assert_allclose(weights[1, 1], [0.5, 0.5], rtol=0, atol=0)


# ----------------------------------------------------------------------
# train_pak integration
# ----------------------------------------------------------------------
class TestTrainPakIntegration:
    def _make(self, seed=1):
        from somkit import SOMTrainer

        rng_data = np.random.RandomState(0)
        data = rng_data.rand(20, 3)
        return SOMTrainer(
            data=data, size=(6, 6), input_dim=3, learning_rate=0.1, random_seed=seed
        )

    def test_runs_and_preserves_shape(self):
        som = self._make()
        som.train_pak(rlen=200, alpha=0.1, radius=3.0)
        assert som.weights.shape == (6, 6, 3)
        assert np.all(np.isfinite(som.weights))

    def test_reproducible_with_seed(self):
        a = self._make(seed=1)
        a.train_pak(rlen=200, alpha=0.1, radius=3.0)
        b = self._make(seed=1)
        b.train_pak(rlen=200, alpha=0.1, radius=3.0)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_reduces_quantization_error(self):
        som = self._make()
        som.initialize_weights_randomly()
        bmu0 = np.array([som.weights[tuple(find_bmu_pak(som.weights, x))] for x in som.data])
        qe_before = np.mean(np.linalg.norm(som.data - bmu0, axis=1))
        som.train_pak(rlen=2000, alpha=0.3, radius=3.0)
        bmu1 = np.array([som.weights[tuple(find_bmu_pak(som.weights, x))] for x in som.data])
        qe_after = np.mean(np.linalg.norm(som.data - bmu1, axis=1))
        assert qe_after < qe_before

    def test_unknown_alpha_type_raises(self):
        som = self._make()
        with pytest.raises(ValueError):
            som.train_pak(rlen=10, alpha=0.1, radius=3.0, alpha_type="bogus")

    def test_unknown_neighborhood_raises(self):
        som = self._make()
        with pytest.raises(ValueError):
            som.train_pak(rlen=10, alpha=0.1, radius=3.0, neighborhood="nope")

    def test_non_positive_rlen_raises(self):
        som = self._make()
        with pytest.raises(ValueError):
            som.train_pak(rlen=0, alpha=0.1, radius=3.0)


class TestBubbleWindowEquivalence:
    """The bubble bounding-box fast path must be bit-identical to the full-map
    update (skipped units receive influence 0, so omitting them changes nothing).
    """

    @pytest.mark.parametrize("topology_name", ["hexagonal", "rectangular"])
    @pytest.mark.parametrize("radius", [1.0, 1.0002, 2.5, 7.0, 50.0])
    @pytest.mark.parametrize("bmu", [(0, 0), (5, 3), (9, 9)])
    def test_window_path_matches_full_map_path(self, topology_name, radius, bmu):
        if topology_name == "hexagonal":
            from somkit.topology import HexagonalTopology

            topology = HexagonalTopology()
        else:
            from somkit.topology import RectangularTopology

            topology = RectangularTopology()

        rng = np.random.RandomState(42)
        weights_fast = rng.rand(10, 10, 4)
        weights_full = weights_fast.copy()
        sample = rng.rand(4)

        # Real bubble function -> identity check enables the window fast path.
        som_step(
            weights_fast, sample, bmu, radius, 0.3, bubble_neighborhood, topology
        )
        # Wrapped bubble -> identity check fails, forcing the full-map path.
        som_step(
            weights_full, sample, bmu, radius, 0.3,
            lambda d, r: bubble_neighborhood(d, r), topology,
        )

        np.testing.assert_array_equal(weights_fast, weights_full)

    def test_window_path_with_mask_matches_full_map_path(self):
        from somkit.topology import HexagonalTopology

        rng = np.random.RandomState(7)
        weights_fast = rng.rand(8, 8, 5)
        weights_full = weights_fast.copy()
        sample = rng.rand(5)
        mask = np.array([False, True, False, False, True])

        som_step(
            weights_fast, sample, (3, 4), 2.0, 0.2,
            bubble_neighborhood, HexagonalTopology(), mask=mask,
        )
        som_step(
            weights_full, sample, (3, 4), 2.0, 0.2,
            lambda d, r: bubble_neighborhood(d, r), HexagonalTopology(), mask=mask,
        )

        np.testing.assert_array_equal(weights_fast, weights_full)


class TestFindBmuPakBatch:
    """find_bmu_pak_batch must reproduce find_bmu_pak exactly, incl. ties."""

    def test_matches_single_sample_version(self):
        from somkit.functions.learning import find_bmu_pak_batch

        rng = np.random.RandomState(3)
        weights = rng.rand(7, 9, 5)
        samples = rng.rand(100, 5)

        batch = find_bmu_pak_batch(weights, samples, chunk_size=16)
        single = np.array([find_bmu_pak(weights, s) for s in samples])
        np.testing.assert_array_equal(batch, single)

    def test_tie_break_matches_som_pak_scan_order(self):
        from somkit.functions.learning import find_bmu_pak_batch

        # All units identical -> every unit ties; SOM_PAK keeps the first in
        # scan order (y outer, x inner), i.e. unit (0, 0).
        weights = np.ones((4, 3, 2))
        samples = np.zeros((5, 2))
        batch = find_bmu_pak_batch(weights, samples)
        np.testing.assert_array_equal(batch, np.zeros((5, 2), dtype=int))

    def test_masked_batch_matches_single(self):
        from somkit.functions.learning import find_bmu_pak_batch

        rng = np.random.RandomState(11)
        weights = rng.rand(6, 6, 4)
        samples = rng.rand(30, 4)
        mask = rng.rand(30, 4) < 0.3

        batch = find_bmu_pak_batch(weights, samples, mask=mask, chunk_size=7)
        single = np.array(
            [find_bmu_pak(weights, s, mask=m) for s, m in zip(samples, mask)]
        )
        np.testing.assert_array_equal(batch, single)
