import numpy as np

from somkit.topology import HexagonalTopology


def test_som_topology_init():
    # Test initialization with string topology
    som_topology_rectangular = HexagonalTopology()
    assert callable(som_topology_rectangular.topology_function)


def test_som_topology_hexagonal():
    som_topology = HexagonalTopology()
    distance = som_topology.topology_function(2, 3, 5, 7)
    assert np.isclose(distance, 4.123105625617661)  # Update the expected value
