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

        # font_size = min(self.som.weights.shape[0], self.som.weights.shape[1]) * 2
        # self.font_prop = FontProperties(size=font_size)

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
            # plt.legend(handles=legend_elements, loc='upper right', prop=self.font_prop)
            plt.legend(
                handles=legend_elements, loc='upper left',
                bbox_to_anchor=(1.05, 1), prop=self.font_prop
            )

        plt.xticks(np.arange(self.som.weights.shape[0]) + 0.5, [])
        plt.yticks(np.arange(self.som.weights.shape[1]) + 0.5, [])
        plt.show()

    def plot_hexagonal_som_labels(self, label_type: str, show_legend: bool = True):
        plt.figure(figsize=(self.som.weights.shape[0], self.som.weights.shape[1]))

        # font_size = 10 * (1 / np.sqrt(self.som.weights.shape[0] * self.som.weights.shape[1]))
        # self.font_prop.set_size(font_size)

        distance_map = self.som.distance_map().T

        for i in range(self.som.weights.shape[1]):
            for j in range(self.som.weights.shape[0]):
                color = plt.cm.bone_r(distance_map[i, j])
                y_shift = np.sqrt(3) / 2 * (j % 2)
                hexagon = RegularPolygon((j * 1.5, i * np.sqrt(3) / 2 + y_shift), numVertices=6, radius=0.5, edgecolor='k', linewidth=1, facecolor=color)
                plt.gca().add_patch(hexagon)

        winner_map = self.create_winner_map()

        if label_type == "cluster":
            for i, (x, y) in enumerate(winner_map.keys()):
                y_shift = np.sqrt(3) / 2 * (y % 2)
                plt.text(y * 1.5, x * np.sqrt(3) / 2 + y_shift, self.target_names[self.target[i]],
                        color=plt.cm.tab10(float(self.target[i]) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        elif label_type == "block":
            for (x, y), data_indices in winner_map.items():
                labels, counts = np.unique(self.target[data_indices], return_counts=True)
                most_frequent_label = labels[np.argmax(counts)]
                y_shift = np.sqrt(3) / 2 * (y % 2)
                plt.text(y * 1.5, x * np.sqrt(3) / 2 + y_shift, self.target_names[most_frequent_label],
                        color=plt.cm.tab10(float(most_frequent_label) / len(self.target_names)),
                        fontproperties=self.font_prop,
                        ha='center', va='center')

        else:
            raise ValueError("Invalid label_type. Supported label types are 'cluster' and 'block'.")

        if show_legend:
            legend_elements = [Patch(facecolor=plt.cm.tab10(float(i) / len(self.target_names)), edgecolor='k',
                                    label=self.target_names[i]) for i in range(len(self.target_names))]
            # plt.legend(handles=legend_elements, loc='upper right', prop=self.font_prop)
            plt.legend(
                handles=legend_elements, loc='upper left',
                bbox_to_anchor=(1.05, 1), prop=self.font_prop
            )

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
