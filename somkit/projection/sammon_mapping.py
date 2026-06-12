from __future__ import annotations

import numpy as np

from somkit.decomposition import PCA
from somkit.exceptions import SomkitError


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
    :param learning_rate: Step scaling factor for the pseudo-Newton update,
        equivalent to MAGIC in SOM_PAK sammon.c (default: 0.2).
    :param init: Initialization method ('pca' or 'random') (default: 'pca').
    :param random_state: Random seed for reproducibility (default: None).
    :return: Projected data of shape (n_samples, n_components).
    """
    n_samples, n_features = data.shape

    if n_components >= n_features:
        raise ValueError(
            f"n_components ({n_components}) must be less than n_features ({n_features})"
        )

    # Collapse identical rows before optimizing: a zero original-space distance
    # makes the stress gradient blow up (SOM_PAK sammon.c removes duplicates
    # via remove_identicals). Duplicates are restored at the end so the output
    # keeps one row per input row. Note: np.unique sorts rows lexicographically,
    # so the PCA initialization sees a permuted input; the per-row result is
    # unaffected because inverse_index maps coordinates back to input order.
    data, inverse_index = np.unique(data, axis=0, return_inverse=True)
    n_samples = data.shape[0]

    if n_samples < n_components:
        raise SomkitError(
            f"After removing duplicate rows, only {n_samples} unique sample(s) "
            f"remain, which is fewer than n_components={n_components}. "
            "Provide data with more distinct rows."
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

    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_original, 1e-10)

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

        # Pseudo-Newton update for each point (SOM_PAK sammon.c sammon_iterate):
        # move each coordinate by MAGIC * e1 / |e2|, where e1 and e2 are the
        # first and second partial derivatives of the stress.
        Y_new = np.empty_like(Y)
        for i in range(n_samples):
            e1 = np.zeros(n_components)
            e2 = np.zeros(n_components)

            for j in range(n_samples):
                if i == j:
                    continue

                yd = Y[i] - Y[j]
                d_proj = dist_projected[i, j]
                dq = dist_original[i, j] - d_proj
                dr = dist_original[i, j] * d_proj
                e1 += yd * dq / dr
                e2 += (dq - yd**2 * (1.0 + dq / d_proj) / d_proj) / dr

            Y_new[i] = Y[i] + learning_rate * e1 / np.maximum(np.abs(e2), 1e-10)

        # Move the center of mass to the origin (as SOM_PAK sammon.c does)
        Y = Y_new - Y_new.mean(axis=0)

    return Y[inverse_index]


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
    :param learning_rate: Step scaling factor for the pseudo-Newton update,
        equivalent to MAGIC in SOM_PAK sammon.c (default: 0.2).
    :param init: Initialization method ('pca' or 'random') (default: 'pca').
    :param random_state: Random seed for reproducibility (default: None).
    :return: Projected data of shape (n_samples, n_components).
    """
    n_samples, n_features = data.shape

    if n_components >= n_features:
        raise ValueError(
            f"n_components ({n_components}) must be less than n_features ({n_features})"
        )

    # Collapse identical rows before optimizing (see sammon_mapping)
    data, inverse_index = np.unique(data, axis=0, return_inverse=True)
    n_samples = data.shape[0]

    if n_samples < n_components:
        raise SomkitError(
            f"After removing duplicate rows, only {n_samples} unique sample(s) "
            f"remain, which is fewer than n_components={n_components}. "
            "Provide data with more distinct rows."
        )

    # Set random seed
    rng = np.random.RandomState(random_state)

    # Calculate pairwise distances in the original space (vectorized)
    data_expanded_1 = data[:, np.newaxis, :]
    data_expanded_2 = data[np.newaxis, :, :]
    dist_original = np.linalg.norm(data_expanded_1 - data_expanded_2, axis=2)

    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist_original, 1e-10)

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
        stress = c * np.sum((diff ** 2) / dist_original)

        # Check for convergence
        if abs(prev_stress - stress) < tol:
            break

        prev_stress = stress

        # Pseudo-Newton update, vectorized (SOM_PAK sammon.c sammon_iterate):
        # move each coordinate by MAGIC * e1 / |e2|, where e1 and e2 are the
        # first and second partial derivatives of the stress.
        yd = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dr = dist_original * dist_projected

        ratio = diff / dr
        np.fill_diagonal(ratio, 0)
        e1 = np.sum(yd * ratio[:, :, np.newaxis], axis=1)

        curvature = (1.0 + diff / dist_projected) / dist_projected
        e2_terms = (diff[:, :, np.newaxis] - yd**2 * curvature[:, :, np.newaxis]) / dr[:, :, np.newaxis]
        # The i==j term must be excluded (as in the loop version): dr[i,i] is
        # 1e-20, so any non-zero numerator on the diagonal would explode.
        e2_terms[np.arange(n_samples), np.arange(n_samples), :] = 0
        e2 = np.sum(e2_terms, axis=1)

        Y_new = Y + learning_rate * e1 / np.maximum(np.abs(e2), 1e-10)

        # Move the center of mass to the origin (as SOM_PAK sammon.c does)
        Y = Y_new - Y_new.mean(axis=0)

    return Y[inverse_index]
