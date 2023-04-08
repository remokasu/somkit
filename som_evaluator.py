from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import silhouette_score

from som import SOM


class SOMEvaluator:
    def __init__(self, som: SOM):
        self.som = som
        self.data = self.som.data
        self.weights = self.som.weights

    def calculate_wcss(self) -> float:
        """
        Calculate the Within-Cluster Sum of Squares (WCSS) for the SOM.

        :param data: A 2D numpy array containing the input data. If not provided, the data used during training is used.
        :return: The WCSS for the SOM.
        """
        bmus_idx = self.som._get_bmus(self.data)
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
        # Create an array to store the labels assigned to each data point by the SOM
        labels: np.ndarray = np.zeros(len(self.data))

        # Assign a label to each data point based on the index of its winning node
        for i, data_point in enumerate(self.data):
            # Find the winning node for the current data point
            winner_node: Tuple[int, int] = self.som.winner(data_point)

            # Convert the winner_node's index to a single integer label and assign it to the data point
            labels[i] = np.ravel_multi_index(winner_node, self.weights.shape[:2])

        # Calculate and return the silhouette score using the data points and their assigned labels
        # return silhouette_score(self.data, labels)
        try:
            return silhouette_score(self.data, labels)
        except ValueError as e:
            # エラーが発生した場合、デフォルト値を返す
            print(f"Warning: {e}")

            # 注意: デフォルト値は適切に選択する必要があります。シルエットスコアは -1（最悪）から
            # 1（最適）までの範囲で評価されるため、-1.0 はクラスタリングが失敗した場合に適切な値と
            # 考えられます。最適化プロセスが適切に機能するためには、選択したデフォルト値が問題の
            # コンテキストに適していることを確認してください。
            return -1.0



    def calculate_topological_error(self) -> float:
        num_incorrect_topology = 0
        num_data_points = len(self.data)

        for data_point in self.data:
            # Find the winner node and its neighbors
            winner_node = self.som.winner(data_point)
            neighbors = self.get_neighbors(winner_node)

            # Calculate distances between the data_point and the winner node & its neighbors
            winner_distance = np.linalg.norm(data_point - self.weights[winner_node])
            neighbor_distances = [np.linalg.norm(data_point - self.weights[neighbor]) for neighbor in neighbors]

            # Check if the winner_distance is not the smallest distance
            if any(winner_distance >= neighbor_distance for neighbor_distance in neighbor_distances):
                num_incorrect_topology += 1

        # Calculate the topological error
        topological_error = num_incorrect_topology / num_data_points

        return topological_error

    def get_neighbors(self, node: Tuple[int, int], radius: int = 1) -> np.ndarray:
        x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        x, y = x.flatten(), y.flatten()

        # Remove the center (0, 0) since it's not a neighbor
        center_mask = (x != 0) | (y != 0)
        x, y = x[center_mask], y[center_mask]

        neighbor_coords = np.array([x + node[0], y + node[1]]).T
        neighbor_coords = neighbor_coords[
            (0 <= neighbor_coords[:, 0]) & (neighbor_coords[:, 0] < self.weights.shape[0]) &
            (0 <= neighbor_coords[:, 1]) & (neighbor_coords[:, 1] < self.weights.shape[1])
        ]

        return neighbor_coords
