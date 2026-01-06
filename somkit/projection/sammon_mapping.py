from __future__ import annotations

import numpy as np

from somkit.decomposition import PCA


def sammon_mapping(
    data: np.ndarray,
    n_components: int = 2,
    max_iter: int = 500,
    tol: float = 1e-4,
    learning_rate: float = 0.2,
    init: str = "pca",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Perform Sammon's mapping to project high-dimensional data to lower dimensions
    while preserving inter-point distances.

    Sammon's mapping is a non-linear dimensionality reduction technique that
    minimizes the stress function:

    E = (1 / Σ d*ij) × Σ [(d*ij - dij)² / d*ij]

    where d*ij is the distance in the original space and dij is the distance
    in the projected space.

    :param data: Input data of shape (n_samples, n_features).
    :param n_components: Number of dimensions in the projected space (default: 2).
    :param max_iter: Maximum number of iterations for optimization (default: 500).
    :param tol: Convergence tolerance for the stress function (default: 1e-4).
    :param learning_rate: Learning rate for gradient descent (default: 0.2).
    :param init: Initialization method ('pca' or 'random') (default: 'pca').
    :param random_state: Random seed for reproducibility (default: None).
    :return: Projected data of shape (n_samples, n_components).
    """
    n_samples, n_features = data.shape

    if n_components >= n_features:
        raise ValueError(
            f"n_components ({n_components}) must be less than n_features ({n_features})"
        )

    # Set random seed
    rng = np.random.RandomState(random_state)

    # Calculate pairwise distances in the original space
    # d*ij: distance matrix in high-dimensional space
    dist_original = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(data[i] - data[j])
            dist_original[i, j] = dist
            dist_original[j, i] = dist

    # Avoid division by zero
    dist_original[dist_original == 0] = 1e-10

    # Compute the normalization constant
    c = 1.0 / np.sum(dist_original)

    # Initialize the projection
    if init == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        Y = pca.fit_transform(data)
    elif init == "random":
        Y = rng.randn(n_samples, n_components) * 0.1
    else:
        raise ValueError(f"Unknown initialization method: {init}")

    # Optimization loop
    prev_stress = float("inf")

    for iteration in range(max_iter):
        # Calculate pairwise distances in the projected space
        # dij: distance matrix in low-dimensional space
        dist_projected = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(Y[i] - Y[j])
                dist_projected[i, j] = dist
                dist_projected[j, i] = dist

        # Avoid division by zero
        dist_projected[dist_projected == 0] = 1e-10

        # Calculate stress (error function)
        diff = dist_original - dist_projected
        stress = c * np.sum((diff ** 2) / dist_original)

        # Check for convergence
        if abs(prev_stress - stress) < tol:
            break

        prev_stress = stress

        # Calculate gradient for each point
        for i in range(n_samples):
            delta = np.zeros(n_components)

            for j in range(n_samples):
                if i == j:
                    continue

                # Distance difference
                dist_diff = dist_original[i, j] - dist_projected[i, j]

                # Gradient contribution from point j
                if dist_projected[i, j] > 1e-10:
                    factor = -2 * c * dist_diff / (dist_original[i, j] * dist_projected[i, j])
                    delta += factor * (Y[i] - Y[j])

            # Update position with gradient descent
            Y[i] -= learning_rate * delta

    return Y


def sammon_mapping_batch(
    data: np.ndarray,
    n_components: int = 2,
    max_iter: int = 500,
    tol: float = 1e-4,
    learning_rate: float = 0.2,
    init: str = "pca",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Vectorized version of Sammon's mapping for better performance.

    This implementation uses numpy broadcasting to compute gradients
    more efficiently than the point-by-point approach.

    :param data: Input data of shape (n_samples, n_features).
    :param n_components: Number of dimensions in the projected space (default: 2).
    :param max_iter: Maximum number of iterations for optimization (default: 500).
    :param tol: Convergence tolerance for the stress function (default: 1e-4).
    :param learning_rate: Learning rate for gradient descent (default: 0.2).
    :param init: Initialization method ('pca' or 'random') (default: 'pca').
    :param random_state: Random seed for reproducibility (default: None).
    :return: Projected data of shape (n_samples, n_components).
    """
    n_samples, n_features = data.shape

    if n_components >= n_features:
        raise ValueError(
            f"n_components ({n_components}) must be less than n_features ({n_features})"
        )

    # Set random seed
    rng = np.random.RandomState(random_state)

    # Calculate pairwise distances in the original space (vectorized)
    data_expanded_1 = data[:, np.newaxis, :]
    data_expanded_2 = data[np.newaxis, :, :]
    dist_original = np.linalg.norm(data_expanded_1 - data_expanded_2, axis=2)

    # Avoid division by zero
    np.fill_diagonal(dist_original, 1e-10)
    dist_original[dist_original == 0] = 1e-10

    # Compute the normalization constant
    c = 1.0 / np.sum(dist_original)

    # Initialize the projection
    if init == "pca":
        pca = PCA(n_components=n_components, random_state=random_state)
        Y = pca.fit_transform(data)
    elif init == "random":
        Y = rng.randn(n_samples, n_components) * 0.1
    else:
        raise ValueError(f"Unknown initialization method: {init}")

    # Optimization loop
    prev_stress = float("inf")

    for iteration in range(max_iter):
        # Calculate pairwise distances in the projected space (vectorized)
        Y_expanded_1 = Y[:, np.newaxis, :]
        Y_expanded_2 = Y[np.newaxis, :, :]
        dist_projected = np.linalg.norm(Y_expanded_1 - Y_expanded_2, axis=2)

        # Avoid division by zero
        np.fill_diagonal(dist_projected, 1e-10)
        dist_projected[dist_projected == 0] = 1e-10

        # Calculate stress (error function)
        diff = dist_original - dist_projected
        stress = c * np.sum((diff ** 2) / dist_original) / 2  # Divide by 2 to avoid double counting

        # Check for convergence
        if abs(prev_stress - stress) < tol:
            break

        prev_stress = stress

        # Calculate gradients for all points (vectorized)
        # factor shape: (n_samples, n_samples)
        factor = -2 * c * diff / (dist_original * dist_projected)

        # Set diagonal to 0 to exclude self-interactions
        np.fill_diagonal(factor, 0)

        # Compute gradient: sum over j of factor[i,j] * (Y[i] - Y[j])
        # Shape broadcasting: (n_samples, n_samples, 1) * (n_samples, n_samples, n_components)
        gradient = np.sum(
            factor[:, :, np.newaxis] * (Y[:, np.newaxis, :] - Y[np.newaxis, :, :]),
            axis=1
        )

        # Update positions with gradient descent
        Y -= learning_rate * gradient

    return Y
