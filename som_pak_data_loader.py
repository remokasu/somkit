from typing import List

import numpy as np
from sklearn.utils import Bunch


class DatasetWrapper:
    def __init__(self, dataset):
        if not hasattr(dataset, 'data') or not hasattr(dataset, 'target') or not hasattr(dataset, 'target_names'):
            raise ValueError("Input dataset must have attributes: 'data', 'target', and 'target_names'.")

        self.data = dataset.data
        self.target = dataset.target
        self.target_names = dataset.target_names


class SOMPakDataLoader:
    def __init__(self, filepath: str) -> None:
        """
        A class for converting SOM_PAK format data to the format of sklearn.datasets.

        :param filepath : The file path of the SOM_PAK format data.
        :param encoding : The encoding of the SOM_PAK format data
        """
        self.filepath = filepath

    def load_data(self) -> Bunch:
        """
        Convert SOM_PAK format data to the format of sklearn.datasets.

        SOM_PAK format example.

            16
            1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0.5 dove
            0 1 0 0.5 0 1 1 0 0 0 0 1 0 0 0 0 fox
            1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0.5 hen
            0 0 1 0 0 1 1 0 1 0 0 1 1 0 0 0 lion
            1 0 0 0 1 0 0 0 0 1 0 0 0 1 1 0.5 goose
            0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 eagle
            0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 dog
            0 1 0 1 0 1 1 0 1 0 0 1 1 0 0 0 wolf
            0 0 1 0 0 1 1 1 1 0 1 0 1 0 0 1 zebra
            1 0 0 0 1 0 0 0 0 1 0.3 0 0 1 1 0.5 duck
            1 0 0 0.5 0 1 1 0 0 0 0 1 0 0 0 0 cat
            1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 owl
            0 0 1 0.5 0 1 1 0 0 0 1 1 1 0 0 0 tiger
            0 0 1 0 0 1 1 1 1 0 0 0 1 0 0 1 horse
            1 0 0 0 1 0 0 0 0 1 0 1 0 1 0 0 hawk
            0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 1 cow

        Returns
        -------
        data : Bunch
            Data in the format of sklearn.datasets.
            It has four keys: 'data', 'target', 'feature_names', 'target_names'.
        """
        # Load SOM_PAK format data
        with open(self.filepath, 'r') as f:
            data = f.readlines()

        # Get the number of dimensions
        num_features = int(data[0].strip())

        # Separate the data and labels
        X = []
        y = []
        for line in data[1:]:
            line = line.strip().split()
            X.append(list(map(float, line[:-1])))
            y.append(line[-1])

        # Convert labels to numbers
        label_map = {}
        for label in y:
            if label not in label_map:
                label_map[label] = len(label_map)
        y_numeric = np.array([label_map[label] for label in y])

        # Create a dictionary to convert numbers back to the original labels
        label_map_inv = {i: label for label, i in label_map.items()}

        # Convert to the original labels
        y = np.array([label_map_inv[i] for i in y_numeric])

        # Convert the data to the format of sklearn.datasets
        data = Bunch(data=np.array(X), target=y_numeric, feature_names=[f'feature_{i+1}' for i in range(num_features)],
                     target_names=list(label_map_inv.values()))
        return DatasetWrapper(data)
