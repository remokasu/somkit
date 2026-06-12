"""Unit tests for SOM_PAK-conformant decay schedulers (SPEC-0001 FR-1).

These are ADR-0001 "層A" (strict-conformance) tests: each scheduler is a
deterministic pure function, so we compare against the SOM_PAK formula
recomputed in Python with exact (atol=rtol=0) agreement.

References:
    - lvq_pak.c:620-634 (linear_alpha / inverse_t_alpha, INV_ALPHA_CONSTANT)
    - som_rout.c:617 (radius: trad = 1 + (radius-1)*(length-le)/length)
"""

import math

import pytest

from somkit.functions.decay import (
    INV_ALPHA_CONSTANT,
    ALPHA_SCHEDULERS,
    RADIUS_SCHEDULERS,
    exponential_alpha,
    exponential_radius,
    get_alpha_scheduler,
    get_radius_scheduler,
    inverse_t_alpha,
    linear_alpha,
    linear_radius,
)


# ----------------------------------------------------------------------
# linear_alpha — lvq_pak.c:620-623
# ----------------------------------------------------------------------
class TestLinearAlpha:
    @pytest.mark.parametrize(
        "t, length, alpha",
        [
            (0, 1000, 0.05),
            (250, 1000, 0.05),
            (500, 1000, 0.5),
            (999, 1000, 0.05),
            (37, 10000, 0.02),
        ],
    )
    def test_matches_som_pak_formula(self, t, length, alpha):
        expected = alpha * (length - t) / length
        assert linear_alpha(t, length, alpha) == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_start_equals_initial(self):
        assert linear_alpha(0, 1000, 0.05) == 0.05

    def test_end_reaches_zero(self):
        # At t == length the linear schedule reaches exactly 0.
        assert linear_alpha(1000, 1000, 0.05) == 0.0

    def test_monotonic_decreasing(self):
        vals = [linear_alpha(t, 100, 0.5) for t in range(0, 101, 10)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------------
# inverse_t_alpha — lvq_pak.c:627-634
# ----------------------------------------------------------------------
class TestInverseTAlpha:
    def test_constant_value(self):
        assert INV_ALPHA_CONSTANT == 100.0

    @pytest.mark.parametrize(
        "t, length, alpha",
        [
            (0, 1000, 0.05),
            (250, 1000, 0.05),
            (1000, 1000, 0.05),
            (5000, 10000, 0.02),
        ],
    )
    def test_matches_som_pak_formula(self, t, length, alpha):
        c = length / INV_ALPHA_CONSTANT
        expected = alpha * c / (c + t)
        assert inverse_t_alpha(t, length, alpha) == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_start_equals_initial(self):
        # At t=0: alpha * c / c == alpha.
        assert inverse_t_alpha(0, 1000, 0.05) == pytest.approx(0.05, abs=0.0, rel=0.0)

    def test_monotonic_decreasing(self):
        vals = [inverse_t_alpha(t, 1000, 0.5) for t in range(0, 1001, 100)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------------
# linear_radius — som_rout.c:617 (no clamp)
# ----------------------------------------------------------------------
class TestLinearRadius:
    @pytest.mark.parametrize(
        "t, length, initial",
        [
            (0, 1000, 10.0),
            (500, 1000, 10.0),
            (1000, 1000, 10.0),
            (123, 10000, 3.0),
        ],
    )
    def test_matches_som_pak_formula(self, t, length, initial):
        expected = 1.0 + (initial - 1.0) * (length - t) / length
        assert linear_radius(t, length, initial) == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_start_equals_initial(self):
        assert linear_radius(0, 1000, 10.0) == pytest.approx(10.0, abs=0.0, rel=0.0)

    def test_floor_at_one_for_normal_radius(self):
        # r0 >= 1: radius decreases linearly down to exactly 1 at t == length.
        assert linear_radius(1000, 1000, 10.0) == pytest.approx(1.0, abs=0.0, rel=0.0)

    def test_no_clamp_below_one(self):
        # r0 < 1: faithful to SOM_PAK (no clamp) -> rises from r0 toward 1.
        assert linear_radius(0, 1000, 0.5) == pytest.approx(0.5, abs=0.0, rel=0.0)
        assert linear_radius(1000, 1000, 0.5) == pytest.approx(1.0, abs=0.0, rel=0.0)
        # midpoint stays strictly below 1
        assert linear_radius(500, 1000, 0.5) < 1.0


# ----------------------------------------------------------------------
# exponential_* — legacy (non-SOM_PAK) preserved option
# ----------------------------------------------------------------------
class TestExponential:
    @pytest.mark.parametrize("t", [0, 100, 500, 999])
    def test_alpha_default_tau_is_length(self, t):
        length, alpha = 1000, 0.5
        expected = alpha * math.exp(-t / length)
        assert exponential_alpha(t, length, alpha) == pytest.approx(expected, abs=0.0, rel=0.0)

    @pytest.mark.parametrize("t", [0, 100, 500, 999])
    def test_radius_default_tau_is_length(self, t):
        length, initial = 1000, 10.0
        expected = initial * math.exp(-t / length)
        assert exponential_radius(t, length, initial) == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_explicit_tau(self):
        expected = 0.5 * math.exp(-100 / 250)
        assert exponential_alpha(100, 1000, 0.5, tau=250) == pytest.approx(expected, abs=0.0, rel=0.0)

    def test_alpha_start_equals_initial(self):
        assert exponential_alpha(0, 1000, 0.5) == pytest.approx(0.5, abs=0.0, rel=0.0)


# ----------------------------------------------------------------------
# Registries / resolvers
# ----------------------------------------------------------------------
class TestRegistries:
    def test_alpha_registry_keys(self):
        assert set(ALPHA_SCHEDULERS) == {"linear", "inverse_t", "exponential"}

    def test_radius_registry_keys(self):
        # SOM_PAK has no inverse_t schedule for the radius.
        assert set(RADIUS_SCHEDULERS) == {"linear", "exponential"}

    def test_get_alpha_scheduler_resolves(self):
        assert get_alpha_scheduler("linear") is linear_alpha
        assert get_alpha_scheduler("inverse_t") is inverse_t_alpha
        assert get_alpha_scheduler("exponential") is exponential_alpha

    def test_get_radius_scheduler_resolves(self):
        assert get_radius_scheduler("linear") is linear_radius
        assert get_radius_scheduler("exponential") is exponential_radius

    def test_unknown_alpha_name_raises_value_error(self):
        with pytest.raises(ValueError, match=r"Unknown alpha scheduler 'nope'"):
            get_alpha_scheduler("nope")

    def test_unknown_radius_name_raises_value_error(self):
        # "inverse_t" is valid for alpha but not for the radius.
        with pytest.raises(ValueError, match=r"Unknown radius scheduler 'inverse_t'"):
            get_radius_scheduler("inverse_t")


# ----------------------------------------------------------------------
# Domain validation: length must be > 0 (avoid division by zero)
# ----------------------------------------------------------------------
class TestLengthValidation:
    @pytest.mark.parametrize("bad_length", [0, -1, -1000])
    @pytest.mark.parametrize(
        "func, args",
        [
            (linear_alpha, (0, 0.05)),
            (inverse_t_alpha, (0, 0.05)),
            (linear_radius, (0, 10.0)),
            (exponential_alpha, (0, 0.5)),
            (exponential_radius, (0, 10.0)),
        ],
    )
    def test_non_positive_length_raises(self, func, args, bad_length):
        t, initial = args
        with pytest.raises(ValueError):
            func(t, bad_length, initial)
