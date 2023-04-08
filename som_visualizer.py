from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, RegularPolygon

from som import SOM


class SOMVisualizer:
    def __init__(self, som: SOM, font_path: str = None):
        self.som = som
        self.data = som.data
        self.target = som.target
        self.target_names = som.target_names
        self.font_path = font_path
        if self.font_path is not None:
            self.font_prop = FontProperties(fname=font_path)
        else:
            self.font_prop = FontProperties()  # This will use the default font properties.

        current_font_size = self.font_prop.get_size()
        self.font_prop.set_size(current_font_size * 2)


    def plot_umatrix(self, colormap: str = 'bone_r', show_data_points: bool = False, show_legend: bool = True):
        """
        Plot the U-Matrix of the trained SOM.

        :param colormap: A string representing the colormap to be used for the U-Matrix visualization.
        :param show_data_points: A boolean indicating whether to show the data points on the U-Matrix.
        :param show_legend: A boolean indicating whether to show the legend for the data points.
        """

        _point_size = 200

        umatrix = self.som.distance_map().T
        width_padding = 10 if show_legend else 0  # Add extra space for the legend
        fig, ax = plt.subplots(figsize=(self.som.weights.shape[0] + width_padding, self.som.weights.shape[1]))

        patches = []
        hex_radius = 0.5  # Adjust this value to control the gap between hexagons
        linewidth = 0.1
        for y in range(umatrix.shape[0]):
            for x in range(umatrix.shape[1]):
                hexagon = RegularPolygon((x + 0.5 * (y % 2), y * 0.75), numVertices=6, radius=hex_radius, edgecolor='k', linewidth=linewidth)
                patches.append(hexagon)

        pc = PatchCollection(patches, array=np.ravel(umatrix), cmap=colormap)
        ax.add_collection(pc)

        fontsize = 11
        if show_data_points:
            for i, data_point in enumerate(self.data):
                winner_node = tuple(self.som.winner(data_point))
                x = winner_node[1] + 0.5 * (winner_node[0] % 2)
                y = winner_node[0] * 0.75
                color = plt.cm.tab10(float(self.target[i]) / len(self.target_names))
                plt.scatter(x, y, color=color, s=_point_size, marker='o', edgecolors='k')
                plt.annotate(self.target_names[self.target[i]], (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=fontsize, color='black', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.1', alpha=0.8))


            if show_legend:
                legend_elements = [Patch(facecolor=plt.cm.tab10(float(i) / len(self.target_names)), edgecolor='k',
                                         label=self.target_names[i]) for i in range(len(self.target_names))]
                plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), prop=self.font_prop)

        xlim_padding = 1
        ylim_padding = 1
        ax.set_xlim(-xlim_padding, umatrix.shape[1] + 0.5 * (umatrix.shape[0] % 2) + xlim_padding)
        ax.set_ylim(-ylim_padding, umatrix.shape[0] * 0.75 + ylim_padding)
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.show()