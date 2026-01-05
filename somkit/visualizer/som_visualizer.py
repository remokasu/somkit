from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, RegularPolygon

from somkit.trainer.som_trainer import SOMTrainer


def distance_map(weights: np.ndarray) -> np.ndarray:
    """
    Calculate the distance map of the SOM.

    :return: A 2D numpy array containing the distance map of the SOM.
    """
    size_x, size_y = weights.shape[0], weights.shape[1]
    um = np.zeros((size_x, size_y, 8))

    # Left neighbor
    um[1:, :, 0] = np.linalg.norm(weights[1:, :] - weights[:-1, :], axis=2)
    # Right neighbor
    um[:-1, :, 1] = np.linalg.norm(weights[:-1, :] - weights[1:, :], axis=2)
    # Top neighbor
    um[:, 1:, 2] = np.linalg.norm(weights[:, 1:] - weights[:, :-1], axis=2)
    # Bottom neighbor
    um[:, :-1, 3] = np.linalg.norm(weights[:, :-1] - weights[:, 1:], axis=2)
    # Top-left neighbor
    um[1:, 1:, 4] = np.linalg.norm(weights[1:, 1:] - weights[:-1, :-1], axis=2)
    # Bottom-right neighbor
    um[:-1, :-1, 5] = np.linalg.norm(weights[:-1, :-1] - weights[1:, 1:], axis=2)
    # Top-right neighbor
    um[:-1, 1:, 6] = np.linalg.norm(weights[:-1, 1:] - weights[1:, :-1], axis=2)
    # Bottom-left neighbor
    um[1:, :-1, 7] = np.linalg.norm(weights[1:, :-1] - weights[:-1, 1:], axis=2)

    return um.mean(axis=2)


class SOMVisualizer:
    def __init__(
        self,
        som: SOMTrainer,
        font_path: str | None = None,
        font_size: int | None = None,
    ):
        self.som: SOMTrainer = som
        self.font_path: str | None = font_path

        self.data: np.ndarray = som.data
        self.target: np.ndarray = som.target
        self.target_names: list = som.target_names

        if self.font_path is not None:
            self.font_prop = FontProperties(fname=font_path)
        else:
            self.font_prop = (
                FontProperties()
            )  # This will use the default font properties.

        if font_size is None:
            self.font_size = self.font_prop.get_size()
        else:
            self.font_size = font_size
        self.font_prop.set_size(self.font_size)
        self.point_size = 200  # size of ○ on hex.

    def add_some_coloured_hexagons(self, umatrix: np.ndarray, colormap: str, ax):
        """Add colored patches (hexagons or rectangles based on topology) to the axis."""
        linewidth = 0.1
        patches = []
        topology = self.som.topology

        for y in range(umatrix.shape[0]):
            for x in range(umatrix.shape[1]):
                vis_x, vis_y = topology.get_visualization_coords(x, y)
                patch = topology.create_patch(
                    vis_x, vis_y,
                    edgecolor="k",
                    linewidth=linewidth,
                )
                patches.append(patch)
        pc = PatchCollection(patches, array=np.ravel(umatrix), cmap=colormap)
        ax.add_collection(pc)
        return ax

    def add_data_points(self):
        topology = self.som.topology
        for i, data_point in enumerate(self.data):
            winner_node = tuple(self.som.winner(data_point))
            x, y = topology.get_visualization_coords(winner_node[1], winner_node[0])
            if (
                self.target is not None
                and self.target_names is not None
                and len(self.target_names) > 0
                and self.target[i] < len(self.target_names)
            ):
                color = plt.cm.tab10(float(self.target[i]) / len(self.target_names))
                plt.scatter(
                    x, y, color=color, s=self.point_size, marker="o", edgecolors="k"
                )
                plt.annotate(
                    self.target_names[self.target[i]],
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontproperties=self.font_prop,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="white",
                        boxstyle="round,pad=0.1",
                        alpha=0.8,
                    ),
                )
            else:
                plt.scatter(
                    x, y, color="black", s=self.point_size, marker="o", edgecolors="k"
                )

    def add_legend(self):
        if self.target_names is None or len(self.target_names) == 0:
            return
        legend_elements = [
            Patch(
                facecolor=plt.cm.tab10(float(i) / len(self.target_names)),
                edgecolor="k",
                label=self.target_names[i],
            )
            for i in range(len(self.target_names))
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            prop=self.font_prop,
        )

    def plot_umatrix(
        self,
        colormap: str = "bone_r",
        show_data_points: bool = False,
        show_legend: bool = True,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot the U-Matrix of the trained SOM.

        :param colormap: A string representing the colormap to be used for the U-Matrix visualization.
        :param show_data_points: A boolean indicating whether to show the data points on the U-Matrix.
        :param show_legend: A boolean indicating whether to show the legend for the data points.
        """

        # umatrix: np.ndarray = self.som.distance_map().T
        umatrix: np.ndarray = distance_map(self.som.get_weights()).T
        width_padding = 10 if show_legend else 0  # Add extra space for the legend
        fig, ax = plt.subplots(
            figsize=(
                self.som.weights.shape[1] + width_padding,
                self.som.weights.shape[0],
            )
        )

        ax = self.add_some_coloured_hexagons(umatrix, colormap, ax)
        if show_data_points:
            self.add_data_points()
        if show_legend and self.target is not None and self.target_names is not None:
            self.add_legend()

        xlim_padding = 1
        ylim_padding = 1
        map_width, map_height = self.som.topology.get_map_dimensions(
            umatrix.shape[1], umatrix.shape[0]
        )
        ax.set_xlim(-xlim_padding, map_width + xlim_padding)
        ax.set_ylim(-ylim_padding, map_height + ylim_padding)
        ax.set_aspect("equal")
        plt.xticks([])
        plt.yticks([])
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_component_planes(
        self,
        colormap: str = "viridis",
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot component planes showing the distribution of each input feature across the SOM.

        :param colormap: A string representing the colormap to be used for the visualization.
        :param file_name: Optional filename to save the plot.
        :param show: Whether to display the plot.
        """
        weights = self.som.get_weights()
        n_features = weights.shape[2]

        # Calculate grid size for subplots
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        topology = self.som.topology
        linewidth = 0.1

        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            feature_map = weights[:, :, feature_idx].T

            patches = []
            for y in range(feature_map.shape[0]):
                for x in range(feature_map.shape[1]):
                    vis_x, vis_y = topology.get_visualization_coords(x, y)
                    patch = topology.create_patch(
                        vis_x, vis_y,
                        edgecolor="k",
                        linewidth=linewidth,
                    )
                    patches.append(patch)

            pc = PatchCollection(patches, array=np.ravel(feature_map), cmap=colormap)
            ax.add_collection(pc)

            xlim_padding = 1
            ylim_padding = 1
            map_width, map_height = topology.get_map_dimensions(
                feature_map.shape[1], feature_map.shape[0]
            )
            ax.set_xlim(-xlim_padding, map_width + xlim_padding)
            ax.set_ylim(-ylim_padding, map_height + ylim_padding)
            ax.set_aspect("equal")
            ax.set_title(f"Feature {feature_idx + 1}", fontproperties=self.font_prop)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(pc, ax=ax)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_hit_map(
        self,
        colormap: str = "YlOrRd",
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot a hit map showing the number of data points mapped to each node.

        :param colormap: A string representing the colormap to be used for the visualization.
        :param file_name: Optional filename to save the plot.
        :param show: Whether to display the plot.
        """
        weights = self.som.get_weights()
        hit_map = np.zeros((weights.shape[0], weights.shape[1]))

        # Count hits for each node
        for data_point in self.data:
            winner_node = tuple(self.som.winner(data_point))
            hit_map[winner_node] += 1

        hit_map = hit_map.T

        fig, ax = plt.subplots(figsize=(self.som.weights.shape[1], self.som.weights.shape[0]))

        topology = self.som.topology
        linewidth = 0.1
        patches = []

        for y in range(hit_map.shape[0]):
            for x in range(hit_map.shape[1]):
                vis_x, vis_y = topology.get_visualization_coords(x, y)
                patch = topology.create_patch(
                    vis_x, vis_y,
                    edgecolor="k",
                    linewidth=linewidth,
                )
                patches.append(patch)

        pc = PatchCollection(patches, array=np.ravel(hit_map), cmap=colormap)
        ax.add_collection(pc)

        xlim_padding = 1
        ylim_padding = 1
        map_width, map_height = topology.get_map_dimensions(
            hit_map.shape[1], hit_map.shape[0]
        )
        ax.set_xlim(-xlim_padding, map_width + xlim_padding)
        ax.set_ylim(-ylim_padding, map_height + ylim_padding)
        ax.set_aspect("equal")
        ax.set_title("Hit Map", fontproperties=self.font_prop, fontsize=self.font_size + 4)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(pc, ax=ax, label="Number of hits")

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_class_distribution(
        self,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot class distribution map showing the distribution of classes at each node with pie charts.

        :param file_name: Optional filename to save the plot.
        :param show: Whether to display the plot.
        """
        if self.target is None or self.target_names is None or len(self.target_names) == 0:
            print("Warning: No target data available for class distribution map.")
            return

        weights = self.som.get_weights()
        # Store class counts for each node
        class_distribution = {}

        # Count class occurrences for each node
        for i, data_point in enumerate(self.data):
            winner_node = tuple(self.som.winner(data_point))
            if winner_node not in class_distribution:
                class_distribution[winner_node] = {}
            class_idx = self.target[i]
            if class_idx < len(self.target_names):
                class_name = self.target_names[class_idx]
                class_distribution[winner_node][class_name] = (
                    class_distribution[winner_node].get(class_name, 0) + 1
                )

        fig, ax = plt.subplots(
            figsize=(self.som.weights.shape[1] + 2, self.som.weights.shape[0])
        )

        topology = self.som.topology
        linewidth = 0.5

        # Draw patches (hexagons or rectangles)
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                vis_x, vis_y = topology.get_visualization_coords(x, y)
                patch = topology.create_patch(
                    vis_x, vis_y,
                    edgecolor="gray",
                    facecolor="white",
                    linewidth=linewidth,
                )
                ax.add_patch(patch)

                # Add pie chart if node has data
                node = (y, x)
                if node in class_distribution:
                    classes = list(class_distribution[node].keys())
                    counts = list(class_distribution[node].values())
                    # Convert target_names to list if it's a numpy array
                    target_names_list = list(self.target_names) if isinstance(self.target_names, np.ndarray) else self.target_names
                    colors = [
                        plt.cm.tab10(float(target_names_list.index(c)) / len(self.target_names))
                        for c in classes
                    ]

                    # Draw mini pie chart at the visualization coordinates
                    pie_radius = 0.4
                    ax.pie(
                        counts,
                        colors=colors,
                        center=(vis_x, vis_y),
                        radius=pie_radius,
                        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
                    )

        xlim_padding = 1
        ylim_padding = 1
        map_width, map_height = topology.get_map_dimensions(
            weights.shape[1], weights.shape[0]
        )
        ax.set_xlim(-xlim_padding, map_width + xlim_padding)
        ax.set_ylim(-ylim_padding, map_height + ylim_padding)
        ax.set_aspect("equal")
        ax.set_title(
            "Class Distribution Map", fontproperties=self.font_prop, fontsize=self.font_size + 4
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = [
            Patch(
                facecolor=plt.cm.tab10(float(i) / len(self.target_names)),
                edgecolor="k",
                label=self.target_names[i],
            )
            for i in range(len(self.target_names))
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            prop=self.font_prop,
        )

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()
