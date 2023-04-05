from collections import Counter, defaultdict
from math import cos, pi, sin
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Patch, Polygon, RegularPolygon
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# class SOMVisualizer:
#     def __init__(self, som, data, target, target_names, font_path: str = None):
#         self.som = som
#         self.data = data
#         self.target = target
#         self.target_names = target_names
#         self.font_path = font_path
#         if self.font_path is not None:
#             self.font_prop = FontProperties(fname=font_path)
#         else:
#             self.font_prop = FontProperties()  # This will use the default font properties.

#     def plot(self, label_type='cluster', show_legend: bool = False):
#         # Determine the grid type based on the SOM's topology property
#         if isinstance(self.som.topology, str):
#             grid_type = self.som.topology
#         elif callable(self.som.topology):
#             grid_type = self.som.topology.__name__
#         else:
#             raise ValueError("Invalid topology. The SOM's topology must be either a string or a callable object.")

#         assert grid_type in ('square', 'hexagonal'), "Invalid grid_type. Choose either 'square' or 'hexagonal'."
#         assert label_type in ('cluster', 'block'), "Invalid label_type. Choose either 'cluster' or 'block'."

#         if grid_type == 'square':
#             if label_type == 'cluster':
#                 self.plot_som_cluster_labels(show_legend)
#             elif label_type == 'block':
#                 self.plot_som_block_labels(show_legend)
#         elif grid_type == 'hexagonal':
#             if label_type == 'cluster':
#                 self.plot_hexagonal_som_cluster_labels(show_legend)
#             elif label_type == 'block':
#                 self.plot_hexagonal_som_block_labels(show_legend)

#     def find_cluster_centers(self):
#         node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)

#         # Compute the class label for each node
#         for sample, label in zip(self.data, self.target):
#             bmu_idx = self.som._find_bmu(sample)[1]
#             node_labels[bmu_idx] = label

#         cluster_centers = {}
#         for idx, label in enumerate(np.unique(self.target)):
#             label_mask = (node_labels == label)
#             coords = np.column_stack(np.where(label_mask))
#             cluster_centers[label] = coords.mean(axis=0)

#         return cluster_centers

#     def hexagon_coordinates(self, x_center, y_center, size):
#         angle = np.arange(0, 2 * np.pi, np.pi / 3)
#         x = x_center + size * np.cos(angle)
#         y = y_center + size * np.sin(angle)
#         return np.column_stack((x, y))

#     def plot_som_cluster_labels(self, show_legend: bool):
#         plt.figure(figsize=(10, 10))

#         # Plot the SOM map with average weights
#         plt.imshow(self.som.weights.mean(axis=2), cmap="viridis")
#         plt.colorbar()

#         # Find cluster centers and add labels
#         cluster_centers = self.find_cluster_centers()
#         for label, center in cluster_centers.items():
#             if not np.isfinite(center[0]) or not np.isfinite(center[1]):
#                 print(f"Warning: Invalid coordinates for cluster center {label}. Please check your data and SOM settings.")
#                 continue
#             plt.text(center[1], center[0], self.target_names[label], ha='center', va='center', color='white', fontsize=12, fontweight='bold', fontproperties=self.font_prop)

#         # Add a legend for the class labels
#         if show_legend is True:
#             legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
#             leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
#             for text in leg.get_texts():
#                 text.set_fontproperties(self.font_prop)

#         plt.show()

#     def plot_som_block_labels(self, show_legend: bool):
#         plt.figure(figsize=(10, 10))

#         # Compute the class label for each node
#         node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)
#         for sample, label in zip(self.data, self.target):
#             bmu_idx = self.som._find_bmu(sample)[1]
#             node_labels[bmu_idx] = label

#         # Plot the SOM map with average weights
#         plt.imshow(self.som.weights.mean(axis=2), cmap="viridis")
#         plt.colorbar()

#         # Add class labels to each node
#         for x in range(self.som.x_size):
#             for y in range(self.som.y_size):
#                 if not np.isfinite(x) or not np.isfinite(y):
#                     print(f"Warning: Invalid coordinates for data point with label {self.target_names[node_labels[x, y]]} at position ({x}, {y}). Please check your data and SOM settings.")
#                     continue
#                 plt.text(y, x, self.target_names[node_labels[x, y]], ha='center', va='center', color='white', fontsize=10, fontproperties=self.font_prop)

#         # Add a legend for the class labels
#         if show_legend is True:
#             legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
#             leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
#             for text in leg.get_texts():
#                 text.set_fontproperties(self.font_prop)

#         plt.show()

#     def plot_hexagonal_som_cluster_labels(self, show_legend: bool):
#         plt.figure(figsize=(10, 10))

#         # Plot hexagonal grid
#         hex_size = 0.5
#         for x in range(self.som.x_size):
#             for y in range(self.som.y_size):
#                 hex_center = (y + 0.5 * (x % 2), x * np.sqrt(3) / 2)
#                 hex_coords = self.hexagon_coordinates(hex_center[0], hex_center[1], hex_size)
#                 hex_color = self.som.weights[x, y].mean()
#                 hex_patch = Polygon(hex_coords, facecolor=plt.cm.viridis(hex_color), edgecolor='k', lw=1)
#                 plt.gca().add_patch(hex_patch)

#         # Find cluster centers and add labels
#         cluster_centers = self.find_cluster_centers()
#         for label, center in cluster_centers.items():
#             hex_center = (center[1] + 0.5 * (center[0] % 2), center[0] * np.sqrt(3) / 2)
#             if not np.isfinite(center[0]) or not np.isfinite(center[1]):
#                 print(f"Warning: Invalid coordinates for cluster center {label}. Please check your data and SOM settings.")
#                 continue
#             plt.text(hex_center[0], hex_center[1], self.target_names[label], ha='center', va='center', color='white', fontsize=12, fontweight='bold', fontproperties=self.font_prop)

#         plt.xlim(-0.5, self.som.y_size + (self.som.x_size % 2) * 0.5 - 0.5)
#         plt.ylim(-0.5, self.som.x_size * np.sqrt(3) / 2)

#         # Add a legend for the class labels
#         if show_legend is True:
#             legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
#             leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
#             for text in leg.get_texts():
#                 text.set_fontproperties(self.font_prop)

#         plt.axis('off')
#         plt.show()

#     def plot_hexagonal_som_block_labels(self, show_legend: bool):
#         plt.figure(figsize=(10, 10))

#         # Compute the class label for each node
#         node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)
#         for sample, label in zip(self.data, self.target):
#             bmu_idx = self.som._find_bmu(sample)[1]
#             node_labels[bmu_idx] = label

#         # Plot hexagonal grid
#         hex_size = 0.5
#         for x in range(self.som.x_size):
#             for y in range(self.som.y_size):
#                 hex_center = (y + 0.5 * (x % 2), x * np.sqrt(3) / 2)
#                 if not np.isfinite(hex_center[0]) or not np.isfinite(hex_center[1]):
#                     print(f"Warning: Invalid coordinates for data point with label {self.target_names[node_labels[x, y]]} at position {hex_center}. Please check your data and SOM settings.")
#                     continue
#                 hex_center = (y + 0.5 * (x % 2), x * np.sqrt(3) / 2)
#                 hex_coords = self.hexagon_coordinates(hex_center[0], hex_center[1], hex_size)
#                 hex_color = self.som.weights[x, y].mean()
#                 hex_patch = Polygon(hex_coords, facecolor=plt.cm.viridis(hex_color), edgecolor='k', lw=1)
#                 plt.gca().add_patch(hex_patch)
#                 plt.text(hex_center[0], hex_center[1], self.target_names[node_labels[x, y]], ha='center', va='center', color='white', fontsize=10, fontproperties=self.font_prop)

#         plt.xlim(-0.5, self.som.y_size + (self.som.x_size % 2) * 0.5 - 0.5)
#         plt.ylim(-0.5, self.som.x_size * np.sqrt(3) / 2)

#         # Add a legend for the class labels
#         if show_legend is True:
#             legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
#             leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
#             for text in leg.get_texts():
#                 text.set_fontproperties(self.font_prop)

#         plt.axis('off')
#         plt.show()


class SOMVisualizer:
    def __init__(self, som, data, target, target_names, font_path: str = None):
        self.som = som
        self.data = data
        self.target = target
        self.target_names = target_names
        self.font_path = font_path
        if self.font_path is not None:
            self.font_prop = FontProperties(fname=font_path)
        else:
            self.font_prop = FontProperties()  # This will use the default font properties.

    def plot(self, grid_type: str, label_type: str, show_legend: bool = True, custom_plot_function=None):
        if grid_type == "rectangular":
            self.plot_rectangular_som_labels(label_type, show_legend)
        elif grid_type == "hexagonal":
            self.plot_hexagonal_som_labels(label_type, show_legend)
        elif grid_type == "circular":
            self.plot_circular_som_labels(label_type, show_legend)
        elif grid_type == "ring":
            self.plot_ring_som_labels(label_type, show_legend)
        elif custom_plot_function is not None:
            custom_plot_function(self, label_type, show_legend)
        else:
            raise ValueError("Invalid grid_type. Supported grid types are 'rectangular', 'hexagonal', 'circular', 'ring', or provide a custom_plot_function.")

    def create_winner_map(self):
        winner_map = defaultdict(list)
        for i, data_point in enumerate(self.data):
            winner_node = tuple(self.som.winner(self.data[i]))
            winner_map[winner_node].append(i)
        return winner_map

    def plot_rectangular_som_labels(self, label_type: str, show_legend: bool = True):
        plt.figure(figsize=(self.som.weights.shape[0], self.som.weights.shape[1]))
        plt.pcolor(self.som.distance_map().T, cmap='bone_r', edgecolors='k', linewidths=1)

        font_size = min(self.som.weights.shape[0], self.som.weights.shape[1]) * 2
        self.font_prop = FontProperties(size=font_size)

        winner_map = self.create_winner_map()

        if label_type == "cluster":
            for i, (x, y) in enumerate(winner_map.keys()):
                plt.text(x + 0.5, y + 0.5, self.target_names[self.target[i]],
                        color=plt.cm.tab10(float(self.target[i]) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        elif label_type == "block":
            for (x, y), data_indices in winner_map.items():
                labels, counts = np.unique(self.target[data_indices], return_counts=True)
                most_frequent_label = labels[np.argmax(counts)]
                plt.text(x + 0.5, y + 0.5, self.target_names[most_frequent_label],
                        color=plt.cm.tab10(float(most_frequent_label) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        else:
            raise ValueError("Invalid label_type. Supported label types are 'cluster' and 'block'.")

        if show_legend:
            legend_elements = [Patch(facecolor=plt.cm.tab10(float(i) / len(self.target_names)), edgecolor='k',
                                    label=self.target_names[i]) for i in range(len(self.target_names))]
            plt.legend(handles=legend_elements, loc='upper right', prop=self.font_prop)

        plt.xticks(np.arange(self.som.weights.shape[0]) + 0.5, [])
        plt.yticks(np.arange(self.som.weights.shape[1]) + 0.5, [])
        plt.show()

    def plot_hexagonal_som_labels(self, label_type: str, show_legend: bool = True):
        plt.figure(figsize=(self.som.weights.shape[0], self.som.weights.shape[1]))

        font_size = min(self.som.weights.shape[0], self.som.weights.shape[1]) * 2
        self.font_prop = FontProperties(size=font_size)

        distance_map = self.som.distance_map()
        for i in range(distance_map.shape[0]):
            for j in range(distance_map.shape[1]):
                hexagon = RegularPolygon((i * 1.5 + 0.5 * (j % 2), j * np.sqrt(3) / 2), numVertices=6, radius=0.5, edgecolor='k', linewidth=1, facecolor=plt.cm.bone_r(distance_map[i, j]))
                plt.gca().add_patch(hexagon)

        winner_map = self.create_winner_map()

        if label_type == "cluster":
            for i, (x, y) in enumerate(winner_map.keys()):
                plt.text(x * 1.5 + 0.5 * (y % 2), y * np.sqrt(3) / 2, self.target_names[self.target[i]],
                        color=plt.cm.tab10(float(self.target[i]) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')
        elif label_type == "block":
            for (x, y), data_indices in winner_map.items():
                labels, counts = np.unique(self.target[data_indices], return_counts=True)
                most_frequent_label = labels[np.argmax(counts)]
                plt.text(x * 1.5 + 0.5 * (y % 2), y * np.sqrt(3) / 2, self.target_names[most_frequent_label],
                        color=plt.cm.tab10(float(most_frequent_label) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')
        else:
            raise ValueError("Invalid label_type. Supported label types are 'cluster' and 'block'.")

        if show_legend:
            legend_elements = [Patch(facecolor=plt.cm.tab10(float(i) / len(self.target_names)), edgecolor='k',
                                    label=self.target_names[i]) for i in range(len(self.target_names))]
            plt.legend(handles=legend_elements, loc='upper right', prop=self.font_prop)

        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_circular_som_labels(self, label_type: str, show_legend: bool = True):
        n_neurons = self.som.weights.shape[0]

        # Polar coordinate system
        angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        winner_map = self.create_winner_map()

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_xticks(angles)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        if label_type == "cluster":
            for i, angle in enumerate(angles):
                winner_node = tuple(self.som.winner(self.data[i]))
                ax.text(angle, 1, self.target_names[self.target[i]],
                        color=plt.cm.tab10(float(self.target[i]) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        elif label_type == "block":
            for angle, (x, y) in zip(angles, winner_map.keys()):
                data_indices = winner_map[(x, y)]
                labels, counts = np.unique(self.target[data_indices], return_counts=True)
                most_frequent_label = labels[np.argmax(counts)]
                ax.text(angle, 1, self.target_names[most_frequent_label],
                        color=plt.cm.tab10(float(most_frequent_label) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        else:
            raise ValueError("Invalid label_type. Supported label types are 'cluster' and 'block'.")

        if show_legend:
            legend_elements = [Circle((0, 0), 0.1, facecolor=plt.cm.tab10(float(i) / len(self.target_names)), edgecolor='k',
                                    label=self.target_names[i]) for i in range(len(self.target_names))]
            ax.legend(handles=legend_elements, loc='upper right', prop=self.font_prop)

        plt.show()

    # def plot_ring_som_labels(self, label_type: str, show_legend: bool = True):
    #     pass
