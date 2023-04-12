import numpy as np
import pytest

from som_topology import SOMTopology


def test_som_topology_init():
    # Test initialization with string topology
    som_topology_rectangular = SOMTopology("rectangular")
    assert callable(som_topology_rectangular.topology_function)

    som_topology_hexagonal = SOMTopology("hexagonal")
    assert callable(som_topology_hexagonal.topology_function)

    # Test initialization with custom callable
    def custom_topology(x1, y1, x2, y2):
        return 0

    som_topology_custom = SOMTopology(custom_topology)
    assert som_topology_custom.topology_function == custom_topology

    # Test initialization with invalid topology
    with pytest.raises(ValueError, match="Invalid topology: invalid"):
        SOMTopology("invalid")


def test_som_topology_rectangular():
    som_topology = SOMTopology("rectangular")
    distance = som_topology._rectangular_topology(2, 3, 5, 7)
    assert np.isclose(distance, 5)


def test_som_topology_hexagonal():
    som_topology = SOMTopology("hexagonal")
    distance = som_topology._hexagonal_topology(2, 3, 5, 7)
    assert np.isclose(distance, 5.0990195135927845)


def test_som_topology_to_cube_coordinates():
    som_topology = SOMTopology("rectangular")
    x3, z3 = som_topology._to_cube_coordinates(5, 7)
    assert x3 == -1
    assert z3 == 7


def test_som_topology_to_polar_coordinates():
    som_topology = SOMTopology("rectangular")
    r, theta = som_topology._to_polar_coordinates(5, 7)
    assert np.isclose(r, 7.211102550927978)
    assert np.isclose(theta, 0.5404195002705842)
