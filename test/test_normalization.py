import unittest

import numpy as np

from somkit import create_trainer


class TestNormalization(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create simple test data with known properties
        np.random.seed(42)
        self.data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0],
        ])

    def test_standard_normalization(self):
        """Test standard (Z-score) normalization."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        som.normalize_data(method='standard')

        # Check mean is close to 0
        mean = np.mean(som.data, axis=0)
        np.testing.assert_array_almost_equal(mean, [0, 0, 0], decimal=10)

        # Check std is close to 1
        std = np.std(som.data, axis=0)
        np.testing.assert_array_almost_equal(std, [1, 1, 1], decimal=10)

    def test_minmax_normalization(self):
        """Test Min-Max normalization."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        som.normalize_data(method='minmax')

        # Check min is 0
        min_vals = np.min(som.data, axis=0)
        np.testing.assert_array_almost_equal(min_vals, [0, 0, 0], decimal=10)

        # Check max is 1
        max_vals = np.max(som.data, axis=0)
        np.testing.assert_array_almost_equal(max_vals, [1, 1, 1], decimal=10)

    def test_variance_normalization(self):
        """Test variance normalization."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        # Store original mean
        original_mean = np.mean(self.data, axis=0)

        som.normalize_data(method='variance')

        # Check std is 1
        std = np.std(som.data, axis=0)
        np.testing.assert_array_almost_equal(std, [1, 1, 1], decimal=10)

        # Check mean is preserved (scaled version of original)
        # After variance normalization: data / std
        # Mean should be: original_mean / std
        original_std = np.std(self.data, axis=0)
        expected_mean = original_mean / original_std
        actual_mean = np.mean(som.data, axis=0)
        np.testing.assert_array_almost_equal(actual_mean, expected_mean, decimal=10)

    def test_standardize_data_backward_compatibility(self):
        """Test that standardize_data still works (backward compatibility)."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        som.standardize_data()

        # Should behave same as standard normalization
        mean = np.mean(som.data, axis=0)
        std = np.std(som.data, axis=0)
        np.testing.assert_array_almost_equal(mean, [0, 0, 0], decimal=10)
        np.testing.assert_array_almost_equal(std, [1, 1, 1], decimal=10)

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        with self.assertRaises(ValueError):
            som.normalize_data(method='invalid_method')

    def test_zero_std_handling(self):
        """Test handling of features with zero standard deviation."""
        # Create data with one constant feature
        data_with_const = np.array([
            [1.0, 5.0, 100.0],
            [2.0, 5.0, 200.0],
            [3.0, 5.0, 300.0],
        ])

        som = create_trainer(
            data=data_with_const,
            size=(3, 3),
            learning_rate=0.1,
            random_seed=42
        )

        # Should not raise error
        som.normalize_data(method='variance')

        # Constant feature should remain constant
        self.assertTrue(np.all(som.data[:, 1] == 5.0))


if __name__ == "__main__":
    unittest.main()
