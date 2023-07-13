from __future__ import annotations

import os
from typing import List, Tuple, Union

import h5py
import numpy as np
from numpy import ndarray

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils._bunch import Bunch
from tqdm import tqdm

from pysom.data_loader import DatasetWrapper
from pysom.topology import SOMTopology
from pysom.functions import create_neighborhood_function

class SOM:
    def __init__(
        self,
        data: Bunch | DatasetWrapper | np.ndarray,
        x_size: int,
        y_size: int,
        input_dim: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        initial_radius: float,
        final_radius: float,
        topology: str | callable = 'rectangular',
        neighborhood_function: str = "gaussian",
        neighborhood_width=1.0,
        shuffle_each_epoch: bool = True,
        checkpoint_interval: int = 1
    ) -> None:
        """
        Initialize the Self-Organizing Map (SOM) with the given parameters.

        :param x_size: The number of nodes in the x dimension.
        :param y_size: The number of nodes in the y dimension.
        :param input_dim: The dimensionality of the input data.
        :param epochs: The number of epochs for training.
        :param learning_rate: The initial learning rate for weight updates.
        """
        self._org_data = data
        self.data = data.data if hasattr(data, 'data') and not isinstance(data, np.ndarray) else data
        self.target = getattr(data, 'target', None)
        self.target_names = getattr(data, 'target_names', None)

        self.x_size = x_size
        self.y_size = y_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.weights = None
        self._topology = topology
        self.topology = SOMTopology(topology)
        self.neighborhood_function = create_neighborhood_function(neighborhood_function)
        self.neighborhood_width = neighborhood_width
        self.shuffle_each_epoch = shuffle_each_epoch

        # Initialize performance metrics
        self.quantization_error = None
        self.topological_error = None
        self.silhouette_coefficient = None

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir: str = 'checkpoints'

    def get_data(self):
        return self.data

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

    def shuffle_data(self):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = self.data[indices]
        if self.target is not None:
            self.target = self.target[indices]

    def standardize_data(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def train(self) -> None:
        """
        Train the SOM using the given input data.

        :param data: A 2D numpy array containing the input data.
        :param batch_size: The batch size for training. If None, online learning will be used.
        """
        assert self.data is not None, "Data must be set using 'set_data' before training."

        if self.weights is None:
            self.initialize_weights_randomly()

        if self.batch_size == 1:
            # Online learning
            for epoch in tqdm(range(self.epochs)):
                if self.shuffle_each_epoch:
                    self.shuffle_data()
                current_radius = self._decay_function(epoch)
                for sample in self.data:
                    bmu, bmu_idx = self._find_bmu(sample)
                    self._update_weights(sample, bmu_idx, epoch, current_radius)

                # Save checkpoint at specified intervals
                if epoch % self.checkpoint_interval == 0:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.h5')
                    self.save_checkpoint(checkpoint_path, epoch)

        else:
            # Batch learning
            for epoch in tqdm(range(self.epochs)):
                if self.shuffle_each_epoch:
                    self.shuffle_data()
                current_radius = self._decay_function(epoch)
                batch_indices = np.arange(0, self.data.shape[0], self.batch_size)
                for batch_index in batch_indices:
                    batch = self.data[batch_index:batch_index + self.batch_size]
                    bmu_indices = [self._find_bmu(sample)[1] for sample in batch]
                    self._update_weights_batch(batch, bmu_indices, epoch, current_radius)

                # Save checkpoint at specified intervals
                if epoch % self.checkpoint_interval == 0:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.h5')
                    self.save_checkpoint(checkpoint_path, epoch)

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

    def _update_weights_batch(self, batch: np.ndarray, bmu_indices: List[Tuple[int, int]], epoch: int, current_radius: float) -> None:
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        grid = np.concatenate((x, y), axis=1)

        for sample, bmu_idx in zip(batch, bmu_indices):
            distance = np.linalg.norm(grid - np.array(bmu_idx), axis=1)
            influence = self.neighborhood_function(current_radius, distance, self.neighborhood_width)

            mask = distance <= current_radius
            influence = influence[mask].reshape(-1, 1)

            affected_nodes = grid[mask]

            affected_weights = self.weights[affected_nodes[:, 0], affected_nodes[:, 1], :]
            new_weights = affected_weights + (self.learning_rate / len(batch)) * influence * (sample - affected_weights)
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

    # save

    def save_checkpoint(self, file_path: str, epoch: int) -> None:
        """
        Save the model state to a checkpoint file.

        :param file_path: The path to the checkpoint file.
        :param epoch: The current epoch of training.
        """
        with h5py.File(file_path, 'w') as f:
            f.attrs['x_size'] = self.x_size
            f.attrs['y_size'] = self.y_size
            f.attrs['input_dim'] = self.input_dim
            f.create_dataset('weights', data=self.weights)
            f.attrs['topology'] = self._topology
            f.attrs['neighborhood_function'] = self.neighborhood_function.__name__
            f.attrs['neighborhood_width'] = self.neighborhood_width
            # f.attrs['quantization_error'] = self.quantization_error
            # f.attrs['topological_error'] = self.topological_error
            # f.attrs['silhouette_coefficient'] = self.silhouette_coefficient
            f.create_dataset('quantization_error', data=self.quantization_error if self.quantization_error is not None else np.nan)
            f.create_dataset('topological_error', data=self.topological_error if self.topological_error is not None else np.nan)
            f.create_dataset('silhouette_coefficient', data=self.silhouette_coefficient if self.silhouette_coefficient is not None else np.nan)
            f.attrs['epoch'] = epoch

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load the model state from a checkpoint file.

        :param file_path: The path to the checkpoint file.
        """
        with h5py.File(file_path, 'r') as f:
            self.x_size = f.attrs['x_size']
            self.y_size = f.attrs['y_size']
            self.input_dim = f.attrs['input_dim']
            self.weights = f['weights'][:]
            self._topology = f.attrs['topology']
            self.topology = SOMTopology(self._topology)
            self.neighborhood_function = create_neighborhood_function(f.attrs['neighborhood_function'])
            self.neighborhood_width = f.attrs['neighborhood_width']
            self.quantization_error = f['quantization_error'][()] if not np.isnan(f['quantization_error'][()]) else None
            self.topological_error = f['topological_error'][()] if not np.isnan(f['topological_error'][()]) else None
            self.silhouette_coefficient = f['silhouette_coefficient'][()] if not np.isnan(f['silhouette_coefficient'][()]) else None
            #  loading the epochs here, but not using them specifically. Please use them as needed."
            loaded_epoch = f.attrs['epoch'] 
