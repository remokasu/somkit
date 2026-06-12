"""Tests for the SOMData container (SPEC-0002 / ADR-0004)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from somkit.data_loader import SOMData
from somkit.exceptions import SOMDataError


class TestSOMDataConstruction:
    def test_minimal_data_only(self):
        d = SOMData(data=np.zeros((5, 3)))
        assert d.data.shape == (5, 3)
        assert d.mask is None
        assert d.weights is None
        assert d.fixed is None
        assert d.labels is None

    def test_mask_ok(self):
        d = SOMData(data=np.zeros((5, 3)), mask=np.zeros((5, 3), dtype=bool))
        assert d.mask.shape == (5, 3)
        assert d.mask.dtype == bool

    def test_weights_ok(self):
        d = SOMData(data=np.zeros((5, 3)), weights=np.ones(5))
        assert d.weights.shape == (5,)

    def test_fixed_ok(self):
        d = SOMData(
            data=np.zeros((5, 3)),
            fixed=np.zeros((5, 2), dtype=int),
            fixed_valid=np.ones(5, dtype=bool),
        )
        assert d.fixed.shape == (5, 2)
        assert d.fixed_valid.shape == (5,)

    def test_labels_ok(self):
        d = SOMData(data=np.zeros((3, 2)), labels=np.array(["a", "b", "c"]))
        assert d.labels.shape == (3,)


class TestSOMDataValidation:
    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), mask=np.zeros((5, 2), dtype=bool))

    def test_mask_not_bool_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), mask=np.zeros((5, 3)))  # float, not bool

    def test_weights_row_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), weights=np.ones(4))

    def test_fixed_shape_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), fixed=np.zeros((5, 3), dtype=int))

    def test_fixed_row_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), fixed=np.zeros((4, 2), dtype=int))

    def test_fixed_valid_row_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(
                data=np.zeros((5, 3)),
                fixed=np.zeros((5, 2), dtype=int),
                fixed_valid=np.ones(4, dtype=bool),
            )

    def test_labels_row_mismatch_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), labels=np.array(["a", "b"]))

    def test_data_not_2d_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5,)))

    def test_fixed_valid_without_fixed_raises(self):
        with pytest.raises(SOMDataError):
            SOMData(data=np.zeros((5, 3)), fixed_valid=np.ones(5, dtype=bool))

    def test_frozen(self):
        d = SOMData(data=np.zeros((5, 3)))
        with pytest.raises(FrozenInstanceError):
            d.data = np.zeros((2, 2))  # frozen dataclass
