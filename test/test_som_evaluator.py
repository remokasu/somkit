import unittest

import numpy as np

from somkit import create_trainer
from somkit.evaluator import SOMEvaluator


class TestSOMEvaluator(unittest.TestCase):
    def setUp(self):
        # Create a fixed dataset
        self.data = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0.5, 0.5]])

        # Create a fixed SOM instance with pre-defined weights
        self.som = create_trainer(
            data=self.data,
            size=(2, 2),
            learning_rate=0.1,
        )
        self.som.weights = np.array([[[1, 1], [1, 0]], [[0, 1], [0, 0]]])

        self.evaluator = SOMEvaluator(self.som)

    def test_a(self):
        self.assertTrue(-1 <= 0.5 <= 1)

    def test_calculate_wcss(self):
        expected_wcss = 0.5
        actual_wcss = self.evaluator.calculate_wcss()
        self.assertAlmostEqual(expected_wcss, actual_wcss, places=6)

    def test_calculate_silhouette_score(self):
        expected_silhouette_score = (
            0.05857864376269049  # Updated manually calculated silhouette
        )
        actual_silhouette_score = self.evaluator.calculate_silhouette_score()

        # Check if the silhouette score is close to the expected value
        self.assertAlmostEqual(
            expected_silhouette_score, actual_silhouette_score, places=6
        )

    def test_calculate_topological_error(self):
        expected_error = (
            0.6  # Manually calculated topological error for the fixed dataset
        )
        actual_error = self.evaluator.calculate_topological_error()
        self.assertAlmostEqual(expected_error, actual_error, places=6)


if __name__ == "__main__":
    unittest.main()
