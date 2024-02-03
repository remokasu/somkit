from sklearn.preprocessing import StandardScaler


def fit_transform(data):
    """
    Standardize the input data using the StandardScaler.

    Args:
    - data : array-like of shape (n_samples, n_features)
        Input data to be standardized.

    Returns:
    - standardized_data : array-like of shape (n_samples, n_features)
        Standardized data.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)
