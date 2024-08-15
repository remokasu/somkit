from __future__ import annotations

import unittest

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

from somkit import SOMTrainer


class TestSOMTrainer(unittest.TestCase):
    def setUp(self):
        self.iris_data = datasets.load_iris()
        self.som = SOMTrainer(
            self.iris_data,
            size=(10, 10),
            input_dim=4,
            learning_rate=0.5,
        )

    def test_initialize_weights_randomly(self):
        self.som.initialize_weights_randomly()
        self.assertIsNotNone(self.som.weights)
        self.assertEqual(self.som.weights.shape, (10, 10, 4))

    def test_initialize_weights_with_pca(self):
        self.som.standardize_data()
        self.som.initialize_weights_with_pca()
        self.assertIsNotNone(self.som.weights)
        self.assertEqual(self.som.weights.shape, (10, 10, 4))

    def test_train(self):
        self.som.standardize_data()
        self.som.initialize_weights_randomly()
        self.som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)

    def test_winner(self):
        self.som.standardize_data()
        self.som.initialize_weights_randomly()
        self.som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)
        test_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.5, 3.0, 5.2, 2.0]])
        winner_coords = self.som.winner(test_data)
        self.assertEqual(winner_coords.shape, (2, 2))

    # def test_distance_map(self):
    #     self.som.standardize_data()
    #     self.som.initialize_weights_randomly()
    #     self.som.train(n_epochs=100, batch_size=1, shuffle_each_epoch=True)
    #     distance_map = self.som.distance_map()
    #     self.assertEqual(distance_map.shape, (10, 10))

    def test_shuffle_each_epoch_enabled(self):
        data = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0.5, 0.5]])

        som = SOMTrainer(
            data=data,
            size=(10, 10),
            input_dim=2,
            learning_rate=0.5,
        )
        original_data = data.copy()

        som.train(n_epochs=1, batch_size=1, shuffle_each_epoch=True)
        assert not np.array_equal(
            som.data, original_data
        ), "Data should be shuffled when shuffle_each_epoch is enabled."

    def test_shuffle_each_epoch_disabled(self):
        data = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [0.5, 0.5]])

        som = SOMTrainer(
            data=data,
            size=(10, 10),
            input_dim=2,
            learning_rate=0.5,
        )
        original_data = data.copy()

        som.train(n_epochs=1, batch_size=1, shuffle_each_epoch=False)
        assert np.array_equal(
            som.data, original_data
        ), "Data should not be shuffled when shuffle_each_epoch is disabled."

    def test_shuffle_preserves_data_target_combination(self):
        iris = load_iris()
        som = SOMTrainer(
            data=iris,
            size=(10, 10),
            input_dim=4,
            learning_rate=0.5,
        )

        original_data = som.data.copy()
        original_target = som.target.copy()

        som.train(n_epochs=1, batch_size=1, shuffle_each_epoch=True)

        shuffled_data = som.data
        shuffled_target = som.target

        # for i in range(len(original_data)):
        #     shuffled_index = np.where((shuffled_data == original_data[i]).all(axis=1))[0][0]
        #     assert original_target[i] == shuffled_target[shuffled_index], "Data and target combination should be preserved during shuffling."

        for i in range(len(original_data)):
            original_index = np.where((original_data == shuffled_data[i]).all(axis=1))[
                0
            ][0]
            assert (
                original_target[original_index] == shuffled_target[i]
            ), "Data and target combination should be preserved during shuffling."

    # def test_save_and_load_checkpoint(self):
    #     iris = load_iris()
    #     # Create an SOM instance
    #     som_original = SOMTrainer(
    #         data=iris,
    #         size=(10, 10),
    #         input_dim=4,
    #         learning_rate=0.5,
    #     )

    #     # Set the attributes for the instance
    #     som_original.weights = np.random.rand(5, 5, 3)

    #     epoch = 10
    #     # Save the SOM instance to a temporary file
    #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    #         som_original._save_checkpoint(tmp_file.name)

    #         # Load the SOM instance from the temporary file
    #         som_loaded = SOMTrainer(
    #             data=iris,
    #             size=(10, 10),
    #             input_dim=4,
    #             learning_rate=0.5,
    #         )
    #         som_loaded.load_checkpoint(tmp_file.name)

    #     # Check if the attributes are equal
    #     assert som_original.x_size == som_loaded.x_size
    #     assert som_original.y_size == som_loaded.y_size
    #     assert som_original.input_dim == som_loaded.input_dim
    #     assert np.allclose(som_original.weights, som_loaded.weights)
    #     assert (
    #         som_original.n_func.__name__
    #         == som_loaded.n_func.__name__
    #     )
    #     assert som_original.n_radius == som_loaded.n_radius

    #     # Delete the temporary file
    #     os.remove(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
