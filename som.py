from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from som_topology import SOMTopology
from son_neighborhood_functions import create_neighborhood_function


class SOM:
    def __init__(
        self, data, x_size: int, y_size: int, input_dim: int,
        epochs: int, learning_rate: float, initial_radius: float, final_radius: float,
        topology: str | callable = 'rectangular',
        neighborhood_function: str = "gaussian",
        neighborhood_width=1.0
    ) -> None:
        """
        Initialize the Self-Organizing Map (SOM) with the given parameters.

        :param x_size: The number of nodes in the x dimension.
        :param y_size: The number of nodes in the y dimension.
        :param input_dim: The dimensionality of the input data.
        :param epochs: The number of epochs for training.
        :param learning_rate: The initial learning rate for weight updates.
        """
        self.data = data.data
        self.target = data.target
        self.target_names = data.target_names

        self.x_size = x_size
        self.y_size = y_size
        self.input_dim = input_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.weights = None
        self._topology = topology
        self.topology = SOMTopology(topology)
        self.neighborhood_function = create_neighborhood_function(neighborhood_function)
        self.neighborhood_width = neighborhood_width

    def get_data(self):
        return self.data

    def standardize_data(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def initialize_weights_randomly(self) -> None:
        """_summary_
        """
        self.weights = np.random.rand(self.x_size, self.y_size, self.input_dim)

    def initialize_weights_with_pca(self) -> None:
        """
        Initialize the weight matrix using the first two principal components
        of the input data. This method can provide a better starting point for
        the SOM training, potentially leading to faster convergence and a more
        accurate representation of the input data.

        Note: This method should be called after setting the input data using the `set_data` method.
        """
        assert self.data is not None, "Data must be set using 'set_data' before initializing weights with PCA."

        # Calculate the first two principal components of the data using PCA
        pca = PCA(n_components=2)
        pca.fit(self.data)

        # Initialize the weight matrix using the first two principal components
        two_principal_components = pca.components_[:2]
        ranges = [np.linspace(0, 1, num) for num in (self.x_size, self.y_size)]
        grid = np.meshgrid(*ranges, indexing='ij')
        grid = np.stack(grid, axis=-1)

        # Initialize the weight matrix using the first two principal components
        self.weights = np.tensordot(grid, two_principal_components, axes=1) + pca.mean_

    def train(self) -> None:
        """
        Train the SOM using the given input data.

        :param data: A 2D numpy array containing the input data.
        """
        assert self.data is not None, "Data must be set using 'set_data' before training."

        if self.weights is None:
            self.initialize_weights_randomly()

        for epoch in tqdm(range(self.epochs)):
            current_radius = self._decay_function(epoch)  # Update this line
            for sample in self.data:
                bmu, bmu_idx = self._find_bmu(sample)
                self._update_weights(sample, bmu_idx, epoch, current_radius)

        self._compute_performance_metrics(self.data)

    def _find_bmu(self, sample: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        x_indices, y_indices = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size), indexing='ij')
        sample_x, sample_y = np.unravel_index(np.argmin(np.linalg.norm(self.weights - sample, axis=2)), (self.x_size, self.y_size))
        euclidean_distances = self.topology.topology_function(x_indices, y_indices, sample_x, sample_y)
        bmu_idx = np.unravel_index(np.argmin(euclidean_distances), (self.x_size, self.y_size))
        bmu = self.weights[bmu_idx]
        return bmu, bmu_idx

    def _update_weights(self, sample: np.ndarray, bmu_idx: Tuple[int, int], epoch: int, current_radius: float) -> None:
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        grid = np.concatenate((x, y), axis=1)

        distance = np.linalg.norm(grid - np.array(bmu_idx), axis=1)
        influence = self.neighborhood_function(current_radius, distance, self.neighborhood_width)

        """_update_weightsでは、distance = np.linalg.norm(grid - np.array(bmu_idx), axis=1)としています。このデータ型に合わせ、近傍関数をすべて変更する必要があります。
        """

        mask = distance <= current_radius
        influence = influence[mask].reshape(-1, 1)

        affected_nodes = grid[mask]

        affected_weights = self.weights[affected_nodes[:, 0], affected_nodes[:, 1], :]
        new_weights = affected_weights + self.learning_rate * influence * (sample - affected_weights)
        self.weights[affected_nodes[:, 0], affected_nodes[:, 1], :] = new_weights

    def _decay_function(self, epoch: int) -> float:
        """
        Calculate the decay function value for the given epoch.

        :param epoch: The current epoch of training.
        :return: The decay function value for the given epoch.
        """
        return np.exp(-epoch / self.epochs) * max(self.x_size, self.y_size) / 2

    def _compute_performance_metrics(self, data: ndarray) -> None:
        """
        Compute performance metrics (quantization error, topological error, and silhouette coefficient) for the SOM.

        :param data: A 2D numpy array containing the input data.
        """
        bmus_idx = self._get_bmus(data)
        self.quantization_error = self._compute_quantization_error(data, bmus_idx)
        self.topological_error = self._compute_topological_error(bmus_idx)

        cluster_labels = [y * self.x_size + x for x, y in bmus_idx]

        n_labels = len(set(cluster_labels))
        if 1 < n_labels < len(data):
            self.silhouette_coefficient = silhouette_score(data, cluster_labels)
        else:
            self.silhouette_coefficient = None

    def _get_bmus(self, data: ndarray) -> List[Tuple[int, int]]:
        """
        Get the Best Matching Units (BMUs) for each input sample in the given data.

        :param data: A 2D numpy array containing the input data.
        :return: A list of BMU indices in the SOM grid for each input sample.
        """
        data_expanded = data[:, np.newaxis, np.newaxis, :]
        distance_map = np.linalg.norm(self.weights - data_expanded, axis=3)
        bmu_indices = np.unravel_index(np.argmin(distance_map.reshape(data.shape[0], -1), axis=1), (self.x_size, self.y_size))
        return list(zip(bmu_indices[0], bmu_indices[1]))

    def _compute_topological_error(self, bmus_idx: List[Tuple[int, int]]) -> float:
        """
        Compute the topological error for the SOM based on the given BMUs.

        :param bmus_idx: A list of BMU indices in the SOM grid.
        :return: The topological error for the SOM.
        """
        bmus_idx_array = np.array(bmus_idx)
        distances = np.linalg.norm(bmus_idx_array[:-1] - bmus_idx_array[1:], axis=1)
        errors = np.sum(distances > 1)
        return errors / len(bmus_idx)

    def _compute_quantization_error(self, data: ndarray, bmus_idx: List[Tuple[int, int]]) -> float:
        """
        Compute the quantization error for the SOM based on the given data and BMUs.

        :param data: A 2D numpy array containing the input data.
        :param bmus_idx: A list of BMU indices in the SOM grid.
        :return: The quantization error for the SOM.
        """
        bmus = np.array([self.weights[x, y, :] for x, y in bmus_idx])
        errors = np.linalg.norm(data - bmus, axis=1)
        return np.mean(errors)

    def get_cluster_count(self) -> int:
        """
        Get the number of clusters in the trained SOM.

        :return: The number of clusters in the trained SOM.
        """
        unique_weights = np.unique(self.weights.reshape(-1, self.input_dim), axis=0)
        return len(unique_weights)

    def winner(self, data_points: Union[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Find the winning node(s) in the SOM for the given data point(s).

        :param data_points: A single data point or an array of data points with shape (n_points, n_features).
        :return: The coordinates of the winning node(s) as an array of shape (n_points, 2).
                 If only one data point is provided, a 1D array of shape (2,) is returned.
        """
        data_points = np.asarray(data_points)
        if data_points.ndim == 1:
            data_points = data_points[np.newaxis, :]

        # Calculate Euclidean distances between each data point and the nodes in the weights matrix
        distances = np.linalg.norm(self.weights - data_points[:, np.newaxis, np.newaxis, :], axis=-1)

        # Find the indices of the minimum distance(s) in the flattened distances array
        winner_indices = np.argmin(distances.reshape(data_points.shape[0], -1), axis=1)

        # Convert the indices of the minimum distance(s) to coordinates of the winning node(s)
        winner_coordinates = np.column_stack(np.unravel_index(winner_indices, self.weights.shape[:2]))

        if winner_coordinates.shape[0] == 1:
            return winner_coordinates[0]
        else:
            return winner_coordinates

    def distance_map(self) -> np.ndarray:
        """
        Calculate the distance map of the SOM.

        Returns:
            np.ndarray: A 2D numpy array representing the distance map.
        """
        size_x, size_y = self.weights.shape[0], self.weights.shape[1]
        um = np.zeros((size_x, size_y, 8))

        # Left neighbor
        um[1:, :, 0] = np.linalg.norm(self.weights[1:, :] - self.weights[:-1, :], axis=2)
        # Right neighbor
        um[:-1, :, 1] = np.linalg.norm(self.weights[:-1, :] - self.weights[1:, :], axis=2)
        # Top neighbor
        um[:, 1:, 2] = np.linalg.norm(self.weights[:, 1:] - self.weights[:, :-1], axis=2)
        # Bottom neighbor
        um[:, :-1, 3] = np.linalg.norm(self.weights[:, :-1] - self.weights[:, 1:], axis=2)
        # Top-left neighbor
        um[1:, 1:, 4] = np.linalg.norm(self.weights[1:, 1:] - self.weights[:-1, :-1], axis=2)
        # Bottom-right neighbor
        um[:-1, :-1, 5] = np.linalg.norm(self.weights[:-1, :-1] - self.weights[1:, 1:], axis=2)
        # Top-right neighbor
        um[:-1, 1:, 6] = np.linalg.norm(self.weights[:-1, 1:] - self.weights[1:, :-1], axis=2)
        # Bottom-left neighbor
        um[1:, :-1, 7] = np.linalg.norm(self.weights[1:, :-1] - self.weights[:-1, 1:], axis=2)

        return um.mean(axis=2)
