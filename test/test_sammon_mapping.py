import unittest
import warnings

import numpy as np

from somkit.exceptions import SomkitError
from somkit.projection.sammon_mapping import sammon_mapping, sammon_mapping_batch


class TestSammonMapping(unittest.TestCase):
    def setUp(self):
        # Create a simple 3D dataset with known structure
        np.random.seed(42)
        self.data_3d = np.random.randn(50, 3)

        # Create a higher dimensional dataset
        self.data_5d = np.random.randn(30, 5)

    def test_sammon_mapping_basic(self):
        """Test basic functionality of Sammon's mapping."""
        projected = sammon_mapping(
            self.data_3d,
            n_components=2,
            max_iter=100,
            random_state=42
        )

        # Check output shape
        self.assertEqual(projected.shape, (50, 2))

        # Check that projection is not all zeros
        self.assertFalse(np.allclose(projected, 0))

    def test_sammon_mapping_with_random_init(self):
        """Test Sammon's mapping with random initialization."""
        projected = sammon_mapping(
            self.data_3d,
            n_components=2,
            init='random',
            max_iter=100,
            random_state=42
        )

        # Check output shape
        self.assertEqual(projected.shape, (50, 2))

        # Check that projection is not all zeros
        self.assertFalse(np.allclose(projected, 0))

    def test_sammon_distance_preservation(self):
        """Test that Sammon's mapping preserves relative distances."""
        # Create a simple dataset with known structure
        data = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        projected = sammon_mapping(
            data,
            n_components=2,
            max_iter=500,
            random_state=42
        )

        # Calculate distances in original space
        orig_dist_01 = np.linalg.norm(data[0] - data[1])
        orig_dist_02 = np.linalg.norm(data[0] - data[2])

        # Calculate distances in projected space
        proj_dist_01 = np.linalg.norm(projected[0] - projected[1])
        proj_dist_02 = np.linalg.norm(projected[0] - projected[2])

        # The ratio of distances should be similar
        orig_ratio = orig_dist_01 / orig_dist_02
        proj_ratio = proj_dist_01 / proj_dist_02

        # Allow some tolerance due to optimization
        self.assertAlmostEqual(orig_ratio, proj_ratio, delta=0.5)

    def test_invalid_n_components(self):
        """Test that invalid n_components raises an error."""
        with self.assertRaises(ValueError):
            sammon_mapping(
                self.data_3d,
                n_components=5,  # More than input dimensions
                random_state=42
            )

    def test_initialization_methods(self):
        """Test different initialization methods."""
        # PCA initialization
        projected_pca = sammon_mapping(
            self.data_5d,
            n_components=2,
            init='pca',
            max_iter=50,
            random_state=42
        )
        self.assertEqual(projected_pca.shape, (30, 2))

        # Random initialization
        projected_random = sammon_mapping(
            self.data_5d,
            n_components=2,
            init='random',
            max_iter=50,
            random_state=42
        )
        self.assertEqual(projected_random.shape, (30, 2))

        # Results should be different for different initializations
        self.assertFalse(np.allclose(projected_pca, projected_random))

    def test_invalid_initialization(self):
        """Test that invalid initialization method raises an error."""
        with self.assertRaises(ValueError):
            sammon_mapping(
                self.data_3d,
                n_components=2,
                init='invalid_method',
                random_state=42
            )

    def test_convergence(self):
        """Test that the algorithm converges."""
        # Run with very small tolerance
        projected = sammon_mapping(
            self.data_3d[:10],  # Use smaller dataset for speed
            n_components=2,
            max_iter=1000,
            tol=1e-6,
            random_state=42
        )

        # Should produce valid output
        self.assertEqual(projected.shape, (10, 2))
        self.assertFalse(np.any(np.isnan(projected)))
        self.assertFalse(np.any(np.isinf(projected)))

    def test_sammon_mapping_finite_with_duplicate_rows(self):
        """Duplicate input rows must not cause numerical divergence (NaN/inf).

        Regression test for the overflow observed when projecting SOM node
        weights that contain identical vectors (cf. SOM_PAK sammon.c
        remove_identicals).
        """
        rng = np.random.RandomState(0)
        base = rng.rand(10, 5)
        data = np.vstack([base, base[:3]])  # rows 10-12 duplicate rows 0-2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            projected = sammon_mapping(
                data, n_components=2, max_iter=100, random_state=42
            )

        self.assertEqual(projected.shape, (13, 2))
        self.assertFalse(np.any(np.isnan(projected)))
        self.assertFalse(np.any(np.isinf(projected)))
        # Duplicate rows must map to identical coordinates
        np.testing.assert_array_almost_equal(projected[10:13], projected[:3])

    def test_sammon_mapping_batch_finite_with_duplicate_rows(self):
        """Batch variant must also stay finite with duplicate input rows."""
        rng = np.random.RandomState(0)
        base = rng.rand(10, 5)
        data = np.vstack([base, base[:3]])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            projected = sammon_mapping_batch(
                data, n_components=2, max_iter=100, random_state=42
            )

        self.assertEqual(projected.shape, (13, 2))
        self.assertFalse(np.any(np.isnan(projected)))
        self.assertFalse(np.any(np.isinf(projected)))
        np.testing.assert_array_almost_equal(projected[10:13], projected[:3])

    def test_too_few_unique_rows_raises(self):
        """All-identical input leaves fewer unique rows than n_components."""
        data = np.tile([[1.0, 2.0, 3.0]], (5, 1))  # 5 rows, all identical

        with self.assertRaises(SomkitError):
            sammon_mapping(data, n_components=2, random_state=42)

        with self.assertRaises(SomkitError):
            sammon_mapping_batch(data, n_components=2, random_state=42)

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        projected1 = sammon_mapping(
            self.data_3d,
            n_components=2,
            max_iter=100,
            random_state=42
        )

        projected2 = sammon_mapping(
            self.data_3d,
            n_components=2,
            max_iter=100,
            random_state=42
        )

        np.testing.assert_array_almost_equal(projected1, projected2)


if __name__ == "__main__":
    unittest.main()
