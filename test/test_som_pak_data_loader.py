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
            loader.load_data()
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


def test_load_data_without_labels_keeps_all_columns():
    # SOM_PAK labels (per-row case names) are optional. A row with exactly `dim`
    # numeric tokens and no label must keep all `dim` columns. Regression: the
    # loader used to drop the last column unconditionally, yielding dim-1.
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        temp_file.write("3\n")
        temp_file.write("0.5 0.6 0.7\n")
        temp_file.write("1.0 2.0 3.0\n")
        temp_file.flush()

        loader = SOMPakDataLoader(temp_file.name)
        dataset = loader.load_data()

        assert dataset.data.shape == (2, 3), "shape must be (N, dim), not (N, dim-1)"
        np.testing.assert_array_equal(
            dataset.data, np.array([[0.5, 0.6, 0.7], [1.0, 2.0, 3.0]])
        )
        # No labels -> all rows share a single (empty) label class.
        np.testing.assert_array_equal(dataset.target, np.array([0, 0]))
        np.testing.assert_array_equal(dataset.target_names, np.array([""]))


def test_load_data_rejects_row_with_too_few_tokens():
    # A row with fewer than `dim` numeric tokens is malformed; loading must fail
    # loudly instead of silently producing a ragged / dim-1 array.
    with tempfile.NamedTemporaryFile(mode="w+t") as temp_file:
        temp_file.write("3\n")
        temp_file.write("0.5 0.6 0.7\n")
        temp_file.write("1.0 2.0\n")  # only 2 tokens, dim=3
        temp_file.flush()

        loader = SOMPakDataLoader(temp_file.name)
        try:
            loader.load_data()
            assert False, "ValueError should be raised for a too-short row"
        except ValueError:
            pass


if __name__ == "__main__":
    test_load_data_coverage()
    test_load_data_boundary()
    test_load_data_functionality()
    test_load_data_without_labels_keeps_all_columns()
