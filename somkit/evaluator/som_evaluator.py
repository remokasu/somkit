from typing import Tuple

import numpy as np

from somkit.metrics import silhouette_score
from somkit.trainer.som_trainer import SOMTrainer


class SOMEvaluator:
    def __init__(self, som: SOMTrainer):
        self.som = som
        self.data = self.som.data
        self.weights = self.som.weights

    def calculate_wcss(self) -> float:
        """
        Calculate the Within-Cluster Sum of Squares (WCSS) for the SOM.

        :return: The WCSS for the SOM.
        """
        bmus_idx = self.som.get_bmus(self.data)
        bmus = np.array([self.weights[x, y, :] for x, y in bmus_idx])
        _data = self.data.astype(np.float64)  # Ensure data type consistency
        errors = np.linalg.norm(_data - bmus, axis=1)
        wcss = np.sum(errors ** 2)
        return wcss

    def calculate_silhouette_score(self) -> float:
        """
        Calculate the silhouette score for the SOM.

        The silhouette score measures how similar a data point is to its own cluster compared to other clusters.
        It ranges from -1 to 1; a higher score indicates better clustering.

        Returns:
            silhouette_score (float): The calculated silhouette score for the SOM.
        """
        # Get BMUs for all data points at once (more efficient than looping)
        bmus_idx = self.som.get_bmus(self.data)

        # Convert BMU indices to single integer labels
        labels = np.array([
            np.ravel_multi_index(bmu, self.weights.shape[:2])
            for bmu in bmus_idx
        ])

        # Calculate and return the silhouette score using the data points and their assigned labels
        try:
            return silhouette_score(self.data, labels)
        except ValueError as e:
            # If an error occurs, return a default value
            print(f"Warning: {e}")

            # Note: The default value should be chosen appropriately. Since the silhouette score is evaluated
            # in the range of -1 (worst) to 1 (optimal), -1.0 can be considered as an appropriate value when
            # clustering fails. Make sure the chosen default value is suitable for the context of the problem
            # to ensure the optimization process functions properly.
            return -1.0

    def calculate_topological_error(self) -> float:
        """
        Calculate the topological error for the SOM.

        Topological error is the proportion of all data points for which the first and
        second BMUs are not adjacent in the grid. This measures how well the SOM preserves
        the topology of the input space.

        :return: The topological error (0 = perfect topology preservation, 1 = worst).
        """
        num_incorrect_topology = 0
        num_data_points = len(self.data)

        for data_point in self.data:
            # Calculate distances to all nodes
            distances = np.linalg.norm(self.weights - data_point, axis=2).flatten()

            # Find indices of two closest nodes (BMU and second BMU)
            sorted_indices = np.argsort(distances)
            bmu1_flat = sorted_indices[0]
            bmu2_flat = sorted_indices[1]

            # Convert flat indices to 2D coordinates
            bmu1 = np.unravel_index(bmu1_flat, (self.weights.shape[0], self.weights.shape[1]))
            bmu2 = np.unravel_index(bmu2_flat, (self.weights.shape[0], self.weights.shape[1]))

            # Check if BMU and second BMU are adjacent (Manhattan distance <= 1)
            manhattan_distance = abs(bmu1[0] - bmu2[0]) + abs(bmu1[1] - bmu2[1])

            if manhattan_distance > 1:
                num_incorrect_topology += 1

        # Calculate the topological error
        topological_error = num_incorrect_topology / num_data_points

        return topological_error

    def calculate_quantization_error(self) -> float:
        """
        Calculate the quantization error for the SOM.

        The quantization error measures the average distance between each data point and its corresponding
        winning node (i.e., the node with the smallest distance to the data point).

        Returns:
            quantization_error (float): The calculated quantization error for the SOM.
        """
        # Calculate the BMUs for each data point
        bmus = self.som.get_bmus(self.data)

        # Calculate the Euclidean distance between each data point and its corresponding BMU
        distances = np.linalg.norm(self.data - self.weights[bmus], axis=1)

        # Calculate the average distance
        quantization_error = np.mean(distances)

        return quantization_error

    def get_neighbors(self, node: Tuple[int, int], radius: int = 1) -> np.ndarray:
        x, y = np.meshgrid(
            np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)
        )
        x, y = x.flatten(), y.flatten()

        # Remove the center (0, 0) since it's not a neighbor
        center_mask = (x != 0) | (y != 0)
        x, y = x[center_mask], y[center_mask]

        neighbor_coords = np.array([x + node[0], y + node[1]]).T
        neighbor_coords = neighbor_coords[
            (0 <= neighbor_coords[:, 0])
            & (neighbor_coords[:, 0] < self.weights.shape[0])
            & (0 <= neighbor_coords[:, 1])
            & (neighbor_coords[:, 1] < self.weights.shape[1])
        ]

        return neighbor_coords
