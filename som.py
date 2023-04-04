from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class SOM:
    def __init__(self, x_size: int, y_size: int, input_dim: int, epochs: int, learning_rate: float, topology: str | callable = 'rectangular') -> None:
        """
        Initialize the Self-Organizing Map (SOM) with the given parameters.

        :param x_size: The number of nodes in the x dimension.
        :param y_size: The number of nodes in the y dimension.
        :param input_dim: The dimensionality of the input data.
        :param epochs: The number of epochs for training.
        :param learning_rate: The initial learning rate for weight updates.
        """
        self.x_size = x_size
        self.y_size = y_size
        self.input_dim = input_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.data = None
        self.topology_function = self._get_topology_function(topology)

    def set_data(self, data) -> None:
        self.data = data

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

        # for epoch in range(self.epochs):
        for epoch in tqdm(range(self.epochs)):
            for sample in self.data:
                bmu, bmu_idx = self._find_bmu(sample)
                self._update_weights(sample, bmu_idx, epoch)

        self._compute_performance_metrics(self.data)

    # def _find_bmu(self, sample: ndarray) -> Tuple[ndarray, Tuple[int, int]]:
    #     """
    #     Find the Best Matching Unit (BMU) for a given input sample.

    #     :param sample: A 1D numpy array representing the input sample.
    #     :return: A tuple containing the BMU and its index in the SOM grid.
    #     """
    #     distance_map = np.linalg.norm(self.weights - sample, axis=2)
    #     bmu_idx = np.unravel_index(np.argmin(distance_map), (self.x_size, self.y_size))
    #     bmu = self.weights[bmu_idx]
    #     return bmu, bmu_idx

    def _find_bmu(self, sample: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        x_indices, y_indices = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size), indexing='ij')
        sample_x, sample_y = np.unravel_index(np.argmin(np.linalg.norm(self.weights - sample, axis=2)), (self.x_size, self.y_size))
        distances = self.topology_function(x_indices, y_indices, sample_x, sample_y)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.x_size, self.y_size))
        bmu = self.weights[bmu_idx]
        return bmu, bmu_idx

    def _update_weights(self, sample: ndarray, bmu_idx: Tuple[int, int], epoch: int) -> None:
        """
        Update the weights of the SOM nodes based on the given input sample and BMU.

        :param sample: A 1D numpy array representing the input sample.
        :param bmu_idx: A tuple containing the index of the BMU in the SOM grid.
        :param epoch: The current epoch of training.
        """
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        grid = np.concatenate((x, y), axis=1)

        distance = np.linalg.norm(grid - np.array(bmu_idx), axis=1)
        sigma = self._decay_function(epoch)
        influence = np.exp(-distance ** 2 / (2 * sigma ** 2))

        mask = distance <= sigma
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

    def _get_topology_function(self, topology: str | callable) -> callable:
        if callable(topology):
            return topology
        elif topology == 'rectangular':
            return self._rectangular_topology
        elif topology == 'hexagonal':
            return self._hexagonal_topology
        elif topology == 'circular':
            return self._circular_topology
        elif topology == 'ring':
            return self._ring_topology
        else:
            raise ValueError(f"Invalid topology: {topology}")

    def _rectangular_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _hexagonal_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        ax, ay = self._to_cube_coordinates(x1, y1)
        bx, by = self._to_cube_coordinates(x2, y2)
        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def _circular_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        angle1 = np.arctan2(y1 - self.y_size // 2, x1 - self.x_size // 2)
        angle2 = np.arctan2(y2 - self.y_size // 2, x2 - self.x_size // 2)
        return np.abs(angle1 - angle2)

    def _ring_topology(self, x1: np.ndarray, y1: np.ndarray, x2: int, y2: int) -> np.ndarray:
        r1, theta1 = self._to_polar_coordinates(x1, y1)
        r2, theta2 = self._to_polar_coordinates(x2, y2)
        return np.sqrt((r1 - r2) ** 2 + (theta1 - theta2) ** 2)

    def _to_cube_coordinates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x3 = x - (y + (y % 2)) // 2
        z3 = y
        # y3 = -x3 - z3  # y3 is not used in the current implementation, but this equation holds for cube coordinates.
        return x3, z3

    def _to_polar_coordinates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = x - self.x_size // 2
        dy = y - self.y_size // 2
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)
        return r, theta
