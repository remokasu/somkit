from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, RegularPolygon

from somkit.projection import sammon_mapping
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
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot the U-Matrix of the trained SOM.

        :param colormap: A string representing the colormap to be used for the U-Matrix visualization.
        :param show_data_points: A boolean indicating whether to show the data points on the U-Matrix.
        :param show_legend: A boolean indicating whether to show the legend for the data points.
        :param title: Optional title for the plot. If None, no title is displayed.
        :param file_name: Optional filename to save the plot.
        :param show: Whether to display the plot.
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

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        plt.xticks([])
        plt.yticks([])
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_component_planes(
        self,
        colormap: str = "viridis",
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot component planes showing the distribution of each input feature across the SOM.

        :param colormap: A string representing the colormap to be used for the visualization.
        :param title: Optional title for the plot. If None, no title is displayed.
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

        # Set title only if provided
        if title is not None:
            fig.suptitle(title, fontproperties=self.font_prop, fontsize=self.font_size + 6)

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_hit_map(
        self,
        colormap: str = "YlOrRd",
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot a hit map showing the number of data points mapped to each node.

        :param colormap: A string representing the colormap to be used for the visualization.
        :param title: Optional title for the plot. If None, no title is displayed.
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

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(pc, ax=ax, label="Number of hits", shrink=0.7)

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        if show:
            plt.show()

    def plot_class_distribution(
        self,
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot class distribution map showing the distribution of classes at each node with pie charts.

        :param title: Optional title for the plot. If None, no title is displayed.
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

        # Set title only if provided
        if title is not None:
            ax.set_title(title, fontproperties=self.font_prop, fontsize=self.font_size + 4)

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

    def plot_sammon_projection(
        self,
        show_nodes: bool = True,
        show_data_points: bool = True,
        show_connections: bool = False,
        show_legend: bool = True,
        show_labels: bool = False,
        node_size: int = 150,
        data_point_size: int = 80,
        connection_style: str = "line",  # "line" or "spring"
        colormap: str = "tab10",
        max_iter: int = 500,
        learning_rate: float = 0.2,
        random_state: int | None = None,
        title: str | None = None,
        file_name: str | None = None,
        show: bool = True,
    ):
        """
        Plot Sammon's mapping projection of SOM weights and/or data points.

        Sammon's mapping projects high-dimensional data to 2D while preserving
        inter-point distances, providing an alternative view of the data structure
        that is independent of the SOM grid topology.

        :param show_nodes: Whether to show SOM nodes in the projection.
        :param show_data_points: Whether to show data points in the projection.
        :param show_connections: Whether to show connections between adjacent SOM nodes.
        :param show_legend: Whether to show the legend for data points.
        :param show_labels: Whether to show labels on data points.
        :param node_size: Size of SOM node markers.
        :param data_point_size: Size of data point markers.
        :param connection_style: Style for node connections ("line" or "spring").
        :param colormap: Colormap to use for class colors.
        :param max_iter: Maximum iterations for Sammon's mapping optimization.
        :param learning_rate: Learning rate for gradient descent.
        :param random_state: Random seed for reproducibility.
        :param title: Optional title for the plot. If None, no title is displayed.
        :param file_name: Optional filename to save the plot.
        :param show: Whether to display the plot.
        """
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        ax.set_facecolor('white')

        # Combine weights and data for projection if both are shown
        if show_nodes and show_data_points:
            # Flatten SOM weights to get node vectors
            weights_flat = self.som.weights.reshape(-1, self.som.weights.shape[2])
            combined_data = np.vstack([weights_flat, self.data])

            # Project combined data
            projected = sammon_mapping(
                combined_data,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )

            # Split projected data back
            n_nodes = weights_flat.shape[0]
            nodes_projected = projected[:n_nodes]
            data_projected = projected[n_nodes:]

        elif show_nodes:
            # Project only SOM nodes
            weights_flat = self.som.weights.reshape(-1, self.som.weights.shape[2])
            nodes_projected = sammon_mapping(
                weights_flat,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            data_projected = None

        elif show_data_points:
            # Project only data points
            data_projected = sammon_mapping(
                self.data,
                n_components=2,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            nodes_projected = None

        else:
            raise ValueError("At least one of show_nodes or show_data_points must be True")

        # Plot SOM nodes
        if show_nodes and nodes_projected is not None:
            # Plot connections between adjacent nodes if requested
            if show_connections:
                # Determine connection style
                if connection_style == "spring":
                    # Spring-like connections with varying thickness based on distance
                    for i in range(self.som.x_size):
                        for j in range(self.som.y_size):
                            idx = i * self.som.y_size + j
                            # Connect to right neighbor
                            if j < self.som.y_size - 1:
                                idx_right = i * self.som.y_size + (j + 1)
                                dist = np.linalg.norm(
                                    nodes_projected[idx] - nodes_projected[idx_right]
                                )
                                # Thicker lines for shorter distances (stronger connections)
                                linewidth = max(0.3, 2.0 / (1 + dist))
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_right, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_right, 1]],
                                    color='#CCCCCC',
                                    alpha=0.4,
                                    linewidth=linewidth,
                                    zorder=1
                                )
                            # Connect to bottom neighbor
                            if i < self.som.x_size - 1:
                                idx_bottom = (i + 1) * self.som.y_size + j
                                dist = np.linalg.norm(
                                    nodes_projected[idx] - nodes_projected[idx_bottom]
                                )
                                linewidth = max(0.3, 2.0 / (1 + dist))
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_bottom, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_bottom, 1]],
                                    color='#CCCCCC',
                                    alpha=0.4,
                                    linewidth=linewidth,
                                    zorder=1
                                )
                else:  # "line" style
                    for i in range(self.som.x_size):
                        for j in range(self.som.y_size):
                            idx = i * self.som.y_size + j
                            # Connect to right neighbor
                            if j < self.som.y_size - 1:
                                idx_right = i * self.som.y_size + (j + 1)
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_right, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_right, 1]],
                                    color='#AAAAAA',
                                    alpha=0.35,
                                    linewidth=0.8,
                                    zorder=1
                                )
                            # Connect to bottom neighbor
                            if i < self.som.x_size - 1:
                                idx_bottom = (i + 1) * self.som.y_size + j
                                ax.plot(
                                    [nodes_projected[idx, 0], nodes_projected[idx_bottom, 0]],
                                    [nodes_projected[idx, 1], nodes_projected[idx_bottom, 1]],
                                    color='#AAAAAA',
                                    alpha=0.35,
                                    linewidth=0.8,
                                    zorder=1
                                )

            # Plot nodes with improved styling
            ax.scatter(
                nodes_projected[:, 0],
                nodes_projected[:, 1],
                c='#E8E8E8',
                s=node_size,
                marker='o',
                edgecolors='#555555',
                linewidths=1.5,
                alpha=0.7,
                label='SOM Nodes',
                zorder=2
            )

        # Plot data points
        if show_data_points and data_projected is not None:
            if (
                self.target is not None
                and self.target_names is not None
                and len(self.target_names) > 0
            ):
                # Get colormap
                cmap = plt.get_cmap(colormap)

                # Plot with class colors
                for class_idx in range(len(self.target_names)):
                    mask = self.target == class_idx
                    if np.any(mask):
                        color = cmap(float(class_idx) / len(self.target_names))
                        ax.scatter(
                            data_projected[mask, 0],
                            data_projected[mask, 1],
                            c=[color],
                            s=data_point_size,
                            marker='o',
                            edgecolors='white',
                            linewidths=1.5,
                            alpha=0.85,
                            label=self.target_names[class_idx],
                            zorder=3
                        )

                        # Add labels if requested
                        if show_labels:
                            for i, idx in enumerate(np.where(mask)[0]):
                                ax.annotate(
                                    self.target_names[class_idx],
                                    (data_projected[idx, 0], data_projected[idx, 1]),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.7,
                                    fontproperties=self.font_prop
                                )
            else:
                # Plot without class information
                ax.scatter(
                    data_projected[:, 0],
                    data_projected[:, 1],
                    c='#3498db',
                    s=data_point_size,
                    marker='o',
                    edgecolors='white',
                    linewidths=1.5,
                    alpha=0.85,
                    label='Data Points',
                    zorder=3
                )

        # Styling
        ax.set_xlabel('Sammon Dimension 1', fontproperties=self.font_prop, fontsize=self.font_size + 2)
        ax.set_ylabel('Sammon Dimension 2', fontproperties=self.font_prop, fontsize=self.font_size + 2)

        # Set title only if provided
        if title is not None:
            ax.set_title(
                title,
                fontproperties=self.font_prop,
                fontsize=self.font_size + 6,
                fontweight='bold',
                pad=20
            )

        ax.set_aspect('equal')

        # Improved grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Legend styling
        if show_legend:
            legend = ax.legend(
                prop=self.font_prop,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                frameon=True,
                fancybox=True,
                shadow=True,
                ncol=1,
                fontsize=self.font_size
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('#CCCCCC')

        # Tight layout
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name, bbox_inches="tight", pad_inches=0.2, dpi=300, facecolor='white')
        if show:
            plt.show()
