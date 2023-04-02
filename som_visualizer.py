import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
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

    def plot(self, grid_type='square', label_type='cluster', show_legend:bool=False):
        assert grid_type in ('square', 'hexagonal'), "Invalid grid_type. Choose either 'square' or 'hexagonal'."
        assert label_type in ('cluster', 'block'), "Invalid label_type. Choose either 'cluster' or 'block'."

        if grid_type == 'square':
            if label_type == 'cluster':
                self.plot_som_cluster_labels(show_legend)
            elif label_type == 'block':
                self.plot_som_block_labels(show_legend)
        elif grid_type == 'hexagonal':
            if label_type == 'cluster':
                self.plot_hexagonal_som_cluster_labels(show_legend)
            elif label_type == 'block':
                self.plot_hexagonal_som_block_labels(show_legend)

    def find_cluster_centers(self):
        node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)

        # Compute the class label for each node
        for sample, label in zip(self.data, self.target):
            bmu_idx = self.som._find_bmu(sample)[1]
            node_labels[bmu_idx] = label

        cluster_centers = {}
        for idx, label in enumerate(np.unique(self.target)):
            label_mask = (node_labels == label)
            coords = np.column_stack(np.where(label_mask))
            cluster_centers[label] = coords.mean(axis=0)

        return cluster_centers

    def hexagon_coordinates(self, x_center, y_center, size):
        angle = np.arange(0, 2 * np.pi, np.pi / 3)
        x = x_center + size * np.cos(angle)
        y = y_center + size * np.sin(angle)
        return np.column_stack((x, y))

    def plot_som_cluster_labels(self, show_legend: bool):
        plt.figure(figsize=(10, 10))

        # Plot the SOM map with average weights
        plt.imshow(self.som.weights.mean(axis=2), cmap="viridis")
        plt.colorbar()

        # Find cluster centers and add labels
        cluster_centers = self.find_cluster_centers()
        for label, center in cluster_centers.items():
            plt.text(center[1], center[0], self.target_names[label], ha='center', va='center', color='white', fontsize=12, fontweight='bold', fontproperties=self.font_prop)

        # Add a legend for the class labels
        if show_legend is True:
            legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
            leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
            for text in leg.get_texts():
                text.set_fontproperties(self.font_prop)

        plt.show()

    def plot_som_block_labels(self, show_legend: bool):
        plt.figure(figsize=(10, 10))

        # Compute the class label for each node
        node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)
        for sample, label in zip(self.data, self.target):
            bmu_idx = self.som._find_bmu(sample)[1]
            node_labels[bmu_idx] = label

        # Plot the SOM map with average weights
        plt.imshow(self.som.weights.mean(axis=2), cmap="viridis")
        plt.colorbar()

        # Add class labels to each node
        for x in range(self.som.x_size):
            for y in range(self.som.y_size):
                plt.text(y, x, self.target_names[node_labels[x, y]], ha='center', va='center', color='white', fontsize=10, fontproperties=self.font_prop)

        # Add a legend for the class labels
        if show_legend is True:
            legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
            leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
            for text in leg.get_texts():
                text.set_fontproperties(self.font_prop)

        plt.show()

    def plot_hexagonal_som_cluster_labels(self, show_legend: bool):
        plt.figure(figsize=(10, 10))

        # Plot hexagonal grid
        hex_size = 0.5
        for x in range(self.som.x_size):
            for y in range(self.som.y_size):
                hex_center = (y + 0.5 * (x % 2), x * np.sqrt(3) / 2)
                hex_coords = self.hexagon_coordinates(hex_center[0], hex_center[1], hex_size)
                hex_color = self.som.weights[x, y].mean()
                hex_patch = Polygon(hex_coords, facecolor=plt.cm.viridis(hex_color), edgecolor='k', lw=1)
                plt.gca().add_patch(hex_patch)

        # Find cluster centers and add labels
        cluster_centers = self.find_cluster_centers()
        for label, center in cluster_centers.items():
            hex_center = (center[1] + 0.5 * (center[0] % 2), center[0] * np.sqrt(3) / 2)
            plt.text(hex_center[0], hex_center[1], self.target_names[label], ha='center', va='center', color='white', fontsize=12, fontweight='bold', fontproperties=self.font_prop)

        plt.xlim(-0.5, self.som.y_size + (self.som.x_size % 2) * 0.5 - 0.5)
        plt.ylim(-0.5, self.som.x_size * np.sqrt(3) / 2)

        # Add a legend for the class labels
        if show_legend is True:
            legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
            leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
            for text in leg.get_texts():
                text.set_fontproperties(self.font_prop)

        plt.axis('off')
        plt.show()

    def plot_hexagonal_som_block_labels(self, show_legend: bool):
        plt.figure(figsize=(10, 10))

        # Compute the class label for each node
        node_labels = np.zeros((self.som.x_size, self.som.y_size), dtype=int)
        for sample, label in zip(self.data, self.target):
            bmu_idx = self.som._find_bmu(sample)[1]
            node_labels[bmu_idx] = label

        # Plot hexagonal grid
        hex_size = 0.5
        for x in range(self.som.x_size):
            for y in range(self.som.y_size):
                hex_center = (y + 0.5 * (x % 2), x * np.sqrt(3) / 2)
                hex_coords = self.hexagon_coordinates(hex_center[0], hex_center[1], hex_size)
                hex_color = self.som.weights[x, y].mean()
                hex_patch = Polygon(hex_coords, facecolor=plt.cm.viridis(hex_color), edgecolor='k', lw=1)
                plt.gca().add_patch(hex_patch)
                plt.text(hex_center[0], hex_center[1], self.target_names[node_labels[x, y]], ha='center', va='center', color='white', fontsize=10, fontproperties=self.font_prop)

        plt.xlim(-0.5, self.som.y_size + (self.som.x_size % 2) * 0.5 - 0.5)
        plt.ylim(-0.5, self.som.x_size * np.sqrt(3) / 2)

        # Add a legend for the class labels
        if show_legend is True:
            legend_elements = [mpatches.Patch(facecolor='white', edgecolor='black', label=name) for name in self.target_names]
            leg = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
            for text in leg.get_texts():
                text.set_fontproperties(self.font_prop)

        plt.axis('off')
        plt.show()
