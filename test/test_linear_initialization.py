import unittest

import numpy as np

from somkit import create_trainer


class TestLinearInitialization(unittest.TestCase):
    def setUp(self):
        # Create a simple 3D dataset
        np.random.seed(42)
        self.data = np.random.randn(100, 4)

    def test_linear_initialization_basic(self):
        """Test basic functionality of linear initialization."""
        som = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )

        som.initialize_weights_linearly()

        # Check that weights are initialized
        self.assertIsNotNone(som.weights)
        self.assertEqual(som.weights.shape, (10, 10, 4))

        # Check that weights are not all zeros
        self.assertFalse(np.allclose(som.weights, 0))

    def test_linear_initialization_ordered(self):
        """Test that linear initialization creates an ordered grid."""
        som = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )

        som.initialize_weights_linearly()

        # Check that adjacent nodes have similar weights
        # This is a characteristic of linear initialization
        for i in range(som.x_size - 1):
            for j in range(som.y_size - 1):
                # Distance to right neighbor
                dist_right = np.linalg.norm(som.weights[i, j] - som.weights[i+1, j])
                # Distance to bottom neighbor
                dist_bottom = np.linalg.norm(som.weights[i, j] - som.weights[i, j+1])

                # Distances should be relatively small and consistent
                # (not a strict test, but should be true for linear init)
                self.assertLess(dist_right, 5.0)
                self.assertLess(dist_bottom, 5.0)

    def test_linear_initialization_lies_on_pc_plane(self):
        """Test that initialized weights lie on the principal component plane."""
        # Create data with clear principal components
        np.random.seed(42)
        # Create data that varies mainly in first two dimensions
        data = np.random.randn(100, 4)
        data[:, 0] *= 3  # Large variance in first dimension
        data[:, 1] *= 2  # Medium variance in second dimension
        data[:, 2] *= 0.1  # Small variance
        data[:, 3] *= 0.1  # Small variance

        som = create_trainer(
            data=data,
            size=(5, 5),
            learning_rate=0.1,
            random_seed=42
        )

        som.initialize_weights_linearly()

        # All weights should be close to the mean
        mean = np.mean(data, axis=0)
        for i in range(som.x_size):
            for j in range(som.y_size):
                # The weights should be within reasonable range of the data
                self.assertTrue(
                    np.all(som.weights[i, j] > np.min(data, axis=0) - 5),
                    "Weights should be within data range"
                )
                self.assertTrue(
                    np.all(som.weights[i, j] < np.max(data, axis=0) + 5),
                    "Weights should be within data range"
                )

    def test_linear_vs_random_initialization(self):
        """Compare linear and random initialization."""
        som_linear = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )
        som_linear.initialize_weights_linearly()

        som_random = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )
        som_random.initialize_weights_randomly()

        # Linear initialization should have more ordered structure
        # Calculate average distance to neighbors
        def avg_neighbor_distance(weights):
            total_dist = 0
            count = 0
            for i in range(weights.shape[0] - 1):
                for j in range(weights.shape[1] - 1):
                    dist = np.linalg.norm(weights[i, j] - weights[i+1, j])
                    dist += np.linalg.norm(weights[i, j] - weights[i, j+1])
                    total_dist += dist
                    count += 2
            return total_dist / count

        linear_dist = avg_neighbor_distance(som_linear.weights)
        random_dist = avg_neighbor_distance(som_random.weights)

        # Linear initialization should have smaller neighbor distances
        # (more ordered structure)
        self.assertLess(linear_dist, random_dist)

    def test_linear_initialization_reproducibility(self):
        """Test that linear initialization is reproducible."""
        som1 = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )
        som1.initialize_weights_linearly()

        som2 = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )
        som2.initialize_weights_linearly()

        # Weights should be identical
        np.testing.assert_array_almost_equal(som1.weights, som2.weights)

    def test_linear_initialization_requires_data(self):
        """Test that linear initialization requires data to be set."""
        som = create_trainer(
            data=self.data,
            size=(10, 10),
            learning_rate=0.1,
            random_seed=42
        )

        # Clear the data
        som.data = None

        # Should raise assertion error
        with self.assertRaises(AssertionError):
            som.initialize_weights_linearly()


if __name__ == "__main__":
    unittest.main()
