import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def fit_transform(data, method='standard'):
    """
    Normalize the input data using the specified method.

    Args:
    - data : array-like of shape (n_samples, n_features)
        Input data to be normalized.
    - method : str, default='standard'
        Normalization method. Options:
        - 'standard': Z-score normalization (mean=0, std=1)
        - 'minmax': Min-Max normalization to [0, 1]
        - 'variance': Variance normalization (only divide by std, keep mean)

    Returns:
    - normalized_data : array-like of shape (n_samples, n_features)
        Normalized data.
    """
    if method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    elif method == 'variance':
        # Variance normalization: divide by standard deviation only
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return data / std
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Choose from 'standard', 'minmax', or 'variance'.")
