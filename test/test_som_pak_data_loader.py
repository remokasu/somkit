import tempfile

import numpy as np

from somkit.data_loader.som_pak_data_loader import SOMPakDataLoader


def test_load_data_coverage():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        # Write test data to the temporary file
        temp_file.write("3\n")
        temp_file.write("0.5 0.5 0.5 A\n")
        temp_file.write("0.0 1.0 0.0 B\n")
        temp_file.write("1.0 0.0 0.0 C\n")
        temp_file.flush()

        # Load the data using SOMPakDataLoader
        loader = SOMPakDataLoader(temp_file.name)
        dataset = loader.load_data()

        # Check the data
        np.testing.assert_array_equal(
            dataset.data, np.array([[0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )
        np.testing.assert_array_equal(dataset.target, np.array([0, 1, 2]))
        np.testing.assert_array_equal(dataset.target_names, np.array(["A", "B", "C"]))


def test_load_data_boundary():
    # Test with empty file
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        # Write an empty file
        temp_file.write("")
        temp_file.flush()

        try:
            loader = SOMPakDataLoader(temp_file.name)
            dataset = loader.load_data()
            assert False, "ValueError should be raised for empty file"
        except IndexError:
            pass


def test_load_data_functionality():
    # Test if the loader handles different number of dimensions and different labels properly
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        # Write test data to the temporary file
        temp_file.write("2\n")
        temp_file.write("0.5 0.5 X\n")
        temp_file.write("0.0 1.0 Y\n")
        temp_file.write("1.0 0.0 X\n")
        temp_file.flush()

        # Load the data using SOMPakDataLoader
        loader = SOMPakDataLoader(temp_file.name)
        dataset = loader.load_data()

        # Check the data
        np.testing.assert_array_equal(
            dataset.data, np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        )
        np.testing.assert_array_equal(dataset.target, np.array([0, 1, 0]))
        np.testing.assert_array_equal(dataset.target_names, np.array(["X", "Y"]))


if __name__ == "__main__":
    test_load_data_coverage()
    test_load_data_boundary()
    test_load_data_functionality()
