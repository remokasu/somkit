import unittest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / test runs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from somkit import create_trainer
from somkit.visualizer import SOMVisualizer


def _make_visualizer(n_classes: int) -> SOMVisualizer:
    """Build a tiny trained SOM with one sample per class."""
    rng = np.random.RandomState(0)
    data = rng.rand(n_classes, 5)
    som = create_trainer(data=data, size=(4, 4), learning_rate=0.1, random_seed=1)
    som.initialize_weights_randomly()
    som.target = np.arange(n_classes)
    som.target_names = [f"class_{i}" for i in range(n_classes)]
    return SOMVisualizer(som)


class TestClassColors(unittest.TestCase):
    """Class color assignment must stay distinguishable beyond 10 classes."""

    def tearDown(self):
        plt.close("all")

    def test_class_colors_unique_for_16_classes(self):
        """Regression: tab10(i/16) collapsed 16 classes onto 10 colors."""
        from somkit.visualizer.som_visualizer import _class_colors

        colors = _class_colors(16)
        self.assertEqual(len(colors), 16)
        self.assertEqual(len(set(map(tuple, colors))), 16)

    def test_class_colors_unique_up_to_20_and_beyond(self):
        from somkit.visualizer.som_visualizer import _class_colors

        for n in (2, 10, 20, 25):
            colors = _class_colors(n)
            self.assertEqual(len(set(map(tuple, colors))), n, f"n_classes={n}")

    def test_umatrix_pak_data_point_colors_unique_for_16_classes(self):
        """The pak U-matrix per-sample overlay must also avoid color collisions.

        Covers _add_data_points_pak, which had its own tab10 color lookup.
        """
        visualizer = _make_visualizer(16)
        visualizer.plot_umatrix(
            show_data_points=True, show_labels=False, show_nodes=False, show=False
        )

        ax = plt.gca()
        # Each per-sample scatter call adds a single-offset PathCollection
        # (the hex grid itself is a PatchCollection, excluded by type).
        point_colors = [
            tuple(np.asarray(c.get_facecolor()).ravel())
            for c in ax.collections
            if isinstance(c, PathCollection) and len(c.get_offsets()) == 1
        ]
        self.assertEqual(len(point_colors), 16)
        self.assertEqual(len(set(point_colors)), 16)

    def test_umatrix_pak_has_no_axes_frame(self):
        """SOM_PAK's umat output has no surrounding frame; ours must not either."""
        visualizer = _make_visualizer(16)
        visualizer.plot_umatrix(show_labels=False, show_nodes=False, show=False)

        ax = plt.gca()
        self.assertFalse(any(s.get_visible() for s in ax.spines.values()))

    def test_class_distribution_legend_colors_unique_for_16_classes(self):
        """plot_class_distribution must give 16 classes 16 distinct colors."""
        visualizer = _make_visualizer(16)
        visualizer.plot_class_distribution(show=False)

        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        colors = [tuple(np.asarray(h.get_facecolor()).ravel()) for h in handles]
        self.assertEqual(len(colors), 16)
        self.assertEqual(len(set(colors)), 16)


if __name__ == "__main__":
    unittest.main()
