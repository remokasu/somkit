"""Unit tests for the SOM_PAK ``orand`` port (SPEC-0001 FR-6).

ADR-0001 "層A" (strict-conformance) tests: ``OrandRNG`` is a deterministic
integer LCG, so its output is compared bit-exactly against the SOM_PAK
recurrence recomputed in Python.

References:
    - lvq_pak.c:311-323 (``next``, ``osrand``, ``orand``, ``RND_MAX=32767``)
    - som_rout.c:147 (randinit: ``orand()/32768.0``)
    - som_rout.c:290 (lininit: ``orand()/16384.0 - 1.0``)
"""

import pytest

from somkit.functions.rng import OrandRNG


# SOM_PAK orand recurrence, reimplemented independently for the reference.
_MULTIPLIER = 23
_MODULUS = 100000001
_RND_MAX = 32767


def _reference_sequence(seed: int, n: int) -> list[int]:
    """Recompute the SOM_PAK ``orand`` sequence with plain Python integers."""
    state = seed
    out = []
    for _ in range(n):
        state = (state * _MULTIPLIER) % _MODULUS
        out.append(state % _RND_MAX)
    return out


class TestKnownSequence:
    @pytest.mark.parametrize("seed", [1, 2, 7, 12345])
    def test_next_int_matches_reference(self, seed):
        rng = OrandRNG(seed)
        produced = [rng.next_int() for _ in range(20)]
        assert produced == _reference_sequence(seed, 20)

    def test_first_values_seed_1(self):
        # Hand-computed: 1->23, 23->529, 529->12167, 12167->279841%32767=17705
        rng = OrandRNG(1)
        assert [rng.next_int() for _ in range(4)] == [23, 529, 12167, 17705]


class TestRanges:
    def test_next_int_within_rnd_max(self):
        rng = OrandRNG(1)
        for _ in range(1000):
            v = rng.next_int()
            assert 0 <= v <= _RND_MAX - 1  # 0..32766

    def test_uniform_within_unit_interval(self):
        rng = OrandRNG(3)
        for _ in range(1000):
            u = rng.uniform()
            assert 0.0 <= u < 1.0

    def test_symmetric_within_pm_one(self):
        rng = OrandRNG(3)
        for _ in range(1000):
            s = rng.symmetric()
            assert -1.0 <= s < 1.0


class TestScalingRelationship:
    def test_uniform_is_next_int_over_32768(self):
        # Two RNGs in the same state must scale the same draw consistently.
        rng_int = OrandRNG(99)
        rng_uni = OrandRNG(99)
        for _ in range(50):
            expected = rng_int.next_int() / 32768.0
            assert rng_uni.uniform() == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_symmetric_is_next_int_over_16384_minus_one(self):
        rng_int = OrandRNG(99)
        rng_sym = OrandRNG(99)
        for _ in range(50):
            expected = rng_int.next_int() / 16384.0 - 1.0
            assert rng_sym.symmetric() == pytest.approx(expected, abs=0.0, rel=0.0)


class TestReproducibility:
    def test_same_seed_same_sequence(self):
        a = OrandRNG(2024)
        b = OrandRNG(2024)
        assert [a.next_int() for _ in range(100)] == [b.next_int() for _ in range(100)]

    def test_different_seed_different_sequence(self):
        a = [OrandRNG(1).next_int() for _ in range(10)]
        b = [OrandRNG(2).next_int() for _ in range(10)]
        assert a != b


class TestSeedReset:
    def test_seed_restarts_sequence(self):
        rng = OrandRNG(5)
        first = [rng.next_int() for _ in range(10)]
        rng.next_int()  # advance further
        rng.seed(5)
        assert [rng.next_int() for _ in range(10)] == first

    def test_seed_changes_stream(self):
        rng = OrandRNG(5)
        rng.seed(7)
        assert [rng.next_int() for _ in range(10)] == _reference_sequence(7, 10)


class TestSeedValidation:
    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_init_rejects_non_positive_seed(self, bad):
        with pytest.raises(ValueError):
            OrandRNG(bad)

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_seed_method_rejects_non_positive(self, bad):
        rng = OrandRNG(1)
        with pytest.raises(ValueError):
            rng.seed(bad)

    # bool is an int subclass, so it is rejected explicitly (not a valid seed).
    @pytest.mark.parametrize("bad", [1.0, "1", None, True, False])
    def test_init_rejects_non_int_seed(self, bad):
        with pytest.raises(ValueError):
            OrandRNG(bad)

    @pytest.mark.parametrize("bad", [1.0, "1", None, True, False])
    def test_seed_method_rejects_non_int(self, bad):
        rng = OrandRNG(1)
        with pytest.raises(ValueError):
            rng.seed(bad)
