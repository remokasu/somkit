import unittest

import numpy as np

from somkit import create_trainer


class TestBatchSOM(unittest.TestCase):
    def setUp(self):
        # Create a simple 2D dataset with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 3) + [0, 0, 0]
        cluster2 = np.random.randn(50, 3) + [5, 5, 5]
        self.data = np.vstack([cluster1, cluster2])

    def test_batch_som_basic(self):
        """Test basic functionality of batch SOM."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.1,
            initial_radius=2.0,
            random_seed=42
        )

        som.initialize_weights_randomly()
        som.train_batch(n_epochs=10)

        # Check that weights are updated
        self.assertIsNotNone(som.weights)
        self.assertEqual(som.weights.shape, (5, 5, 3))

    def test_batch_vs_sequential(self):
        """Compare batch and sequential training."""
        # Batch SOM
        som_batch = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            random_seed=42
        )
        som_batch.standardize_data()
        som_batch.initialize_weights_linearly()

        # Sequential SOM
        som_seq = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            random_seed=42
        )
        som_seq.standardize_data()
        som_seq.initialize_weights_linearly()

        # Train both
        som_batch.train_batch(n_epochs=20)
        som_seq.train(n_epochs=20, batch_size=1)

        # Both should converge (weights should not be NaN or Inf)
        self.assertFalse(np.any(np.isnan(som_batch.weights)))
        self.assertFalse(np.any(np.isinf(som_batch.weights)))
        self.assertFalse(np.any(np.isnan(som_seq.weights)))
        self.assertFalse(np.any(np.isinf(som_seq.weights)))

        # Weights should be different (different algorithms)
        self.assertFalse(np.allclose(som_batch.weights, som_seq.weights))

    def test_batch_som_convergence(self):
        """Test that batch SOM completes training without errors."""
        from somkit.evaluator import SOMEvaluator

        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=3.0,
            dynamic_radius=True,
            random_seed=42
        )
        som.standardize_data()
        som.initialize_weights_randomly()

        # Train
        som.train_batch(n_epochs=100)

        # Verify that training completed and weights are valid
        self.assertIsNotNone(som.weights)
        self.assertFalse(np.any(np.isnan(som.weights)))
        self.assertFalse(np.any(np.isinf(som.weights)))

        # Calculate quantization error to ensure evaluation works
        evaluator = SOMEvaluator(som)
        qe = evaluator.calculate_quantization_error()
        self.assertGreater(qe, 0)  # QE should be positive

    def test_batch_som_with_different_topologies(self):
        """Test batch SOM with different topologies."""
        # Hexagonal topology
        som_hex = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            topology="hexagonal",
            random_seed=42
        )
        som_hex.initialize_weights_randomly()
        som_hex.train_batch(n_epochs=10)

        # Rectangular topology
        som_rect = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            topology="rectangular",
            random_seed=42
        )
        som_rect.initialize_weights_randomly()
        som_rect.train_batch(n_epochs=10)

        # Both should work without errors
        self.assertIsNotNone(som_hex.weights)
        self.assertIsNotNone(som_rect.weights)

    def test_batch_som_reproducibility(self):
        """Test that batch SOM is reproducible with same random seed."""
        som1 = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            random_seed=42
        )
        som1.initialize_weights_randomly()
        som1.train_batch(n_epochs=20, shuffle_each_epoch=False)

        som2 = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            random_seed=42
        )
        som2.initialize_weights_randomly()
        som2.train_batch(n_epochs=20, shuffle_each_epoch=False)

        # Weights should be identical
        np.testing.assert_array_almost_equal(som1.weights, som2.weights)

    def test_batch_som_dynamic_radius(self):
        """Test batch SOM with dynamic radius."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=3.0,
            dynamic_radius=True,
            random_seed=42
        )
        som.initialize_weights_randomly()

        # Get initial radius
        initial_radius = som.get_radius()

        # Train
        som.train_batch(n_epochs=20)

        # Radius should have decreased
        final_radius = som.get_radius()
        self.assertLess(final_radius, initial_radius)

    def test_batch_som_requires_data(self):
        """Test that batch SOM requires data to be set."""
        som = create_trainer(
            data=self.data,
            size=(5, 5),
            learning_rate=0.5,
            initial_radius=2.0,
            random_seed=42
        )

        # Clear the data
        som.data = None

        # Should raise assertion error
        with self.assertRaises(AssertionError):
            som.train_batch(n_epochs=10)


if __name__ == "__main__":
    unittest.main()
