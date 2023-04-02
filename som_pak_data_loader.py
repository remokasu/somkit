from typing import List

import numpy as np
from sklearn.utils import Bunch


class SOMPakDataLoader:
    def __init__(self, filepath: str):
        """
        A class for converting SOM_PAK format data to the format of sklearn.datasets.

        Parameters
        ----------
        filepath : str
            The file path of the SOM_PAK format data.
        """
        self.filepath = filepath

    def load_data(self) -> Bunch:
        """
        Convert SOM_PAK format data to the format of sklearn.datasets.

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
        return data
