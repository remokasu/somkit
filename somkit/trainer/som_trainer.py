from __future__ import annotations

import os
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import copy
import types


from somkit.data_loader import Bunch, DatasetWrapper
from somkit.decomposition import PCA
from somkit.functions import gaussian
from somkit.metrics import silhouette_score
from somkit.preprocessing import fit_transform
from somkit.topology import HexaglnalTopology, RectangularTopology


__n_radius__ = 1.0
__checkpoint_interval__ = 1


class SOMTrainer:
    def __init__(
        self,
        data: Bunch | DatasetWrapper | np.ndarray,
        size: Tuple[int, int],
        input_dim: int,
        learning_rate: float,
        n_func: Callable = gaussian,
        n_radius: float = __n_radius__,
        checkpoint_interval: int = __checkpoint_interval__,
        random_seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the Self-Organizing Map (SOM) with the given parameters.

        :param size: The number of nodes in the x, y dimension.
        :param input_dim: The dimensionality of the input data.
        :param epochs: The number of epochs for training.
        :param learning_rate: The initial learning rate for weight updates.
        :param n_func: The neighborhood function to use for updating weights.
        :param n_radius: The initial radius of the neighborhood function.
        :param checkpoint_interval: The interval at which to save checkpoints during training.
        :param random_seed: The random seed to use for reproducible results.
        """
        self._org_data = data
        self.data = (
            data.data
            if hasattr(data, "data") and not isinstance(data, np.ndarray)
            else data
        )
        self.target = getattr(data, "target", np.array([]))
        self.target_names = getattr(data, "target_names", np.array([]))
        self.x_size = size[0]
        self.y_size = size[1]
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = None
        self.topology = HexaglnalTopology()
        self.n_func = n_func
        self.n_radius = n_radius
        # Initialize performance metrics
        self.quantization_error = None
        self.topological_error = None
        self.silhouette_coefficient = None

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir: str = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.random_seed = random_seed
        self.rng = rng
        if self.rng is None:
            if self.random_seed is not None:
                self.rng = np.random.RandomState(self.random_seed)
            else:
                self.rng = np.random.RandomState()

    # ====================
    # training
    # ====================

    def initialize_weights_randomly(self) -> None:
        """_summary_"""
        self.weights = self.rng.rand(self.x_size, self.y_size, self.input_dim)

    def initialize_weights_with_pca(self) -> None:
        """
        Initialize the weight matrix using the first two principal components
        of the input data. This method can provide a better starting point for
        the SOM training, potentially leading to faster convergence and a more
        accurate representation of the input data.

        Note: This method should be called after setting the input data using the `set_data` method.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before initializing weights with PCA."

        # Calculate the first two principal components of the data using PCA
        pca = PCA(n_components=2)
        pca.fit(self.data)

        # Initialize the weight matrix using the first two principal components
        two_principal_components = pca.components_[:2]
        ranges = [np.linspace(0, 1, num) for num in (self.x_size, self.y_size)]
        grid = np.meshgrid(*ranges, indexing="ij")
        grid = np.stack(grid, axis=-1)

        # Initialize the weight matrix using the first two principal components
        self.weights = np.tensordot(grid, two_principal_components, axes=1) + pca.mean_

    def shuffle_data(self):
        """
        Shuffle the input data and target labels (if available) in unison.
        """
        indices = np.arange(len(self.data))
        self.rng.shuffle(indices)
        self.data = self.data[indices]
        # if self.target is not None:
        if len(self.target) > 0:
            self.target = self.target[indices]

    def standardize_data(self):
        """
        Standardize the input data to have a mean of 0 and a standard deviation of 1.
        """
        self.data = fit_transform(self.data)

    def train(
        self, n_epochs: int, batch_size: int = 1, shuffle_each_epoch: bool = True
    ) -> None:
        """
        Train the SOM using the given input data.

        :param n_epochs: The number of epochs for training.
        :param batch_size: The batch size for training. If None, online learning will be used.
        :param shuffle_each_epoch: Whether to shuffle the input data before each epoch.
        """
        assert (
            self.data is not None
        ), "Data must be set using 'set_data' before training."

        if self.weights is None:
            self.initialize_weights_randomly()

        for epoch in tqdm(range(n_epochs)):
            if shuffle_each_epoch:
                self.shuffle_data()
            current_radius = self._decay_function(n_epochs, epoch)
            batch_indices = np.arange(0, self.data.shape[0], batch_size)
            for batch_index in batch_indices:
                batch = self.data[batch_index : batch_index + batch_size]
                # bmu = [self._find_bmu(sample)[0] for sample in batch]
                bmu_indices = [self._find_bmu(sample)[1] for sample in batch]
                self._update_weights_batch(batch, bmu_indices, current_radius)

            # Save checkpoint at specified intervals
            if epoch % self.checkpoint_interval == 0:
                checkpoint_path = self._get_checkpoint_file_path(epoch)
                self._save_checkpoint(checkpoint_path)

        self._compute_performance_metrics(self.data)

    def _find_bmu(self, sample: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Find the Best Matching Unit (BMU) in the SOM for the given input sample.

        :param sample: The input sample for which to find the BMU.
        :return bmu: The weights of the BMU.
        :return bmu_idx: The indices of the BMU in the SOM grid.
        """
        x_indices, y_indices = np.meshgrid(
            np.arange(self.x_size), np.arange(self.y_size), indexing="ij"
        )
        sample_x, sample_y = np.unravel_index(
            np.argmin(np.linalg.norm(self.weights - sample, axis=2)),
            (self.x_size, self.y_size),
        )
        euclidean_distances = self.topology.topology_function(
            x_indices, y_indices, sample_x, sample_y
        )
        bmu_idx = np.unravel_index(
            np.argmin(euclidean_distances), (self.x_size, self.y_size)
        )
        bmu = self.weights[bmu_idx]
        return bmu, bmu_idx

    def _update_weights_batch(
        self,
        batch: np.ndarray,
        bmu_indices: List[Tuple[int, int]],
        current_radius: float,
    ) -> None:
        """
        Update the weights of the SOM using the given batch of input samples.

        :param batch: A batch of input samples.
        :param bmu_indices: The BMU indices in the SOM grid for each input sample.
        :param current_radius: The current radius of the neighborhood function.
        """
        x, y = np.meshgrid(np.arange(self.x_size), np.arange(self.y_size))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        grid = np.concatenate((x, y), axis=1)

        for sample, bmu_idx in zip(batch, bmu_indices):
            distance = np.linalg.norm(grid - np.array(bmu_idx), axis=1)
            influence = self.n_func(current_radius, distance, self.n_radius)

            mask = distance <= current_radius
            influence = influence[mask].reshape(-1, 1)

            affected_nodes = grid[mask]

            affected_weights = self.weights[
                affected_nodes[:, 0], affected_nodes[:, 1], :
            ]
            new_weights = affected_weights + (
                self.learning_rate / len(batch)
            ) * influence * (sample - affected_weights)
            self.weights[affected_nodes[:, 0], affected_nodes[:, 1], :] = new_weights

    def _decay_function(self, n_epochs: int, epoch: int) -> float:
        """
        Calculate the decay function value for the given epoch.

        :param n_epochs: The total number of epochs for training.
        :param epoch: The current epoch of training.
        :return: The decay function value for the given epoch.
        """
        return np.exp(-epoch / n_epochs) * max(self.x_size, self.y_size) / 2.0

    # ====================
    # evaluation
    # ====================

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
        bmu_indices = np.unravel_index(
            np.argmin(distance_map.reshape(data.shape[0], -1), axis=1),
            (self.x_size, self.y_size),
        )
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

    def _compute_quantization_error(
        self, data: ndarray, bmus_idx: List[Tuple[int, int]]
    ) -> float:
        """
        Compute the quantization error for the SOM based on the given data and BMUs.

        :param data: A 2D numpy array containing the input data.
        :param bmus_idx: A list of BMU indices in the SOM grid.
        :return: The quantization error for the SOM.
        """
        bmus = np.array([self.weights[x, y, :] for x, y in bmus_idx])
        errors = np.linalg.norm(data - bmus, axis=1)
        return np.mean(errors)

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
        distances = np.linalg.norm(
            self.weights - data_points[:, np.newaxis, np.newaxis, :], axis=-1
        )

        # Find the indices of the minimum distance(s) in the flattened distances array
        winner_indices = np.argmin(distances.reshape(data_points.shape[0], -1), axis=1)

        # Convert the indices of the minimum distance(s) to coordinates of the winning node(s)
        winner_coordinates = np.column_stack(
            np.unravel_index(winner_indices, self.weights.shape[:2])
        )

        if winner_coordinates.shape[0] == 1:
            return winner_coordinates[0]
        else:
            return winner_coordinates

    # ====================
    # visualization
    # ====================

    def distance_map(self) -> np.ndarray:
        """
        Calculate the distance map of the SOM.

        :return: A 2D numpy array containing the distance map of the SOM.
        """
        size_x, size_y = self.weights.shape[0], self.weights.shape[1]
        um = np.zeros((size_x, size_y, 8))

        # Left neighbor
        um[1:, :, 0] = np.linalg.norm(
            self.weights[1:, :] - self.weights[:-1, :], axis=2
        )
        # Right neighbor
        um[:-1, :, 1] = np.linalg.norm(
            self.weights[:-1, :] - self.weights[1:, :], axis=2
        )
        # Top neighbor
        um[:, 1:, 2] = np.linalg.norm(
            self.weights[:, 1:] - self.weights[:, :-1], axis=2
        )
        # Bottom neighbor
        um[:, :-1, 3] = np.linalg.norm(
            self.weights[:, :-1] - self.weights[:, 1:], axis=2
        )
        # Top-left neighbor
        um[1:, 1:, 4] = np.linalg.norm(
            self.weights[1:, 1:] - self.weights[:-1, :-1], axis=2
        )
        # Bottom-right neighbor
        um[:-1, :-1, 5] = np.linalg.norm(
            self.weights[:-1, :-1] - self.weights[1:, 1:], axis=2
        )
        # Top-right neighbor
        um[:-1, 1:, 6] = np.linalg.norm(
            self.weights[:-1, 1:] - self.weights[1:, :-1], axis=2
        )
        # Bottom-left neighbor
        um[1:, :-1, 7] = np.linalg.norm(
            self.weights[1:, :-1] - self.weights[:-1, 1:], axis=2
        )

        return um.mean(axis=2)

    # ====================
    # save
    # ====================

    def _get_checkpoint_file_path(self, epoch: int) -> str:
        """
        Get the file path for the checkpoint file for the given epoch.

        :param epoch: The epoch for which to get the checkpoint file path.
        :return: The file path for the checkpoint file for the given epoch.
        """
        return os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.h5")

    def _save_checkpoint(self, file_path: str) -> None:
        """
        Save the trained SOM model to a file.

        :param file_path: The path to the file where the model will be saved.
        """
        random_state = self.rng.get_state()

        target_names = np.array([])
        if len(self.target_names) > 0:
            target_names = np.array(
                [name.encode("utf-8") for name in self.target_names]
            )

        with h5py.File(file_path, "w") as f:
            f.attrs["x_size"] = self.x_size
            f.attrs["y_size"] = self.y_size
            f.attrs["input_dim"] = self.input_dim
            f.attrs["n_radius"] = self.n_radius
            f.attrs["learning_rate"] = self.learning_rate
            f.create_dataset("data", data=self.data)
            f.create_dataset("target", data=self.target)
            f.create_dataset("target_names", data=target_names)
            f.create_dataset("weights", data=self.weights)
            f.create_dataset(
                "quantization_error",
                data=self.quantization_error
                if self.quantization_error is not None
                else np.nan,
            )
            f.create_dataset(
                "topological_error",
                data=self.topological_error
                if self.topological_error is not None
                else np.nan,
            )
            f.create_dataset(
                "silhouette_coefficient",
                data=self.silhouette_coefficient
                if self.silhouette_coefficient is not None
                else np.nan,
            )
            grp = f.create_group("random_state")
            grp["0"] = random_state[0]
            grp["1"] = f.create_dataset("stete", data=random_state[1])
            grp["2"] = random_state[2]
            grp["3"] = random_state[3]
            grp["4"] = random_state[4]

    def save_model(self, file_path: str) -> None:
        """
        Save the trained SOM model to a file.

        :param file_path: The path to the file where the model will be saved.
        """
        self._save_checkpoint(file_path)

    # ====================
    # getter and setter
    # ====================

    def get_data(self):
        return self.data

    def set_data(self, data: Bunch | DatasetWrapper | np.ndarray) -> None:
        """
        Set the input data for the SOM.

        :param data: The input data for the SOM.
        """
        self.data = (
            data.data
            if hasattr(data, "data") and not isinstance(data, np.ndarray)
            else data
        )
        self.target = getattr(data, "target", None)
        self.target_names = getattr(data, "target_names", None)

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the weights of the SOM.

        :param weights: The weights of the SOM.
        """
        self.weights = weights

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the SOM.

        :return: The weights of the SOM.
        """
        return self.weights

    def set_function(self, n_func: Callable) -> None:
        """
        Set the neighborhood function for the SOM.

        :param n_func: The neighborhood function to use for updating weights.
        """
        self.n_func = n_func

    def set_radius(self, n_radius: float) -> None:
        """
        Set the radius of the neighborhood function for the SOM.

        :param n_radius: The radius of the neighborhood function.
        """
        self.n_radius = n_radius

    def set_neighbor_func(self, func, radius: float = __n_radius__):
        self.n_func = func
        self.n_radius = radius


def create_trainer(
    data: Bunch | DatasetWrapper | np.ndarray,
    size: Tuple[int, int],
    learning_rate: float,
    n_func: Callable = gaussian,
    n_radius: float = __n_radius__,
    checkpoint_interval: int = __checkpoint_interval__,
    random_seed: int | None = None,
):
    if isinstance(data, np.ndarray):
        input_dim = data.shape[1]
    elif isinstance(data, Bunch) or isinstance(data, DatasetWrapper):
        input_dim = data.data.shape[1]
    else:
        raise ValueError(
            "Invalid input data type. The input data must be a numpy array or a Bunch object."
        )

    return SOMTrainer(
        data,
        size,
        input_dim,
        learning_rate,
        n_func=n_func,
        n_radius=n_radius,
        checkpoint_interval=checkpoint_interval,
        random_seed=random_seed,
    )


def load_trainer(
    checkpoint_file_path: str,
    learning_rate: float,
    n_func: Callable,
    n_radius: float | None = None,
) -> SOMTrainer:
    with h5py.File(checkpoint_file_path, "r") as f:
        _x_size = f.attrs["x_size"]
        _y_size = f.attrs["y_size"]
        _input_dim = f.attrs["input_dim"]
        _weights = f["weights"][:]
        _n_radius = f.attrs["n_radius"]
        _learning_rate = f.attrs["learning_rate"]
        _data = f["data"][:]
        _target = f["target"][:]
        _target_names = f["target_names"][:]
        _state = (
            f["random_state"]["0"][()].decode("utf-8"),
            f["random_state"]["1"][:],
            f["random_state"]["2"][()],
            f["random_state"]["3"][()],
            f["random_state"]["4"][()],
        )

    if len(_target_names) > 0:
        target_names = [name.decode("utf-8") for name in _target_names]

    rng = np.random.RandomState()
    rng.set_state(_state)

    if n_radius is None:
        n_radius = _n_radius

    som = SOMTrainer(
        data=_data,
        size=(_x_size, _y_size),
        input_dim=_input_dim,
        learning_rate=_learning_rate,
        n_func=n_func,
        n_radius=n_radius,
        rng=rng,
    )
    som.weights = _weights
    som.target = _target
    som.target_names = target_names
    return som
