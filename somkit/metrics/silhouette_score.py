from __future__ import annotations

import numpy as np
from sklearn import metrics

silhouette_score = metrics.silhouette_score


# def euclidean_distances(X, Y=None):
#     """
#     Compute the euclidean distance matrix between each pair of samples in X and Y.

#     Args:
#     - X : array-like of shape (n_samples_X, n_features)
#         Feature matrix X.
#     - Y : array-like of shape (n_samples_Y, n_features), optional, default is None
#         Feature matrix Y. If None, use X itself.

#     Returns:
#     - distances : array-like of shape (n_samples_X, n_samples_Y)
#         Euclidean distances between samples in X and Y.
#     """
#     if Y is None:
#         Y = X

#     # Compute squared distances
#     distances = np.dot(X, Y.T)
#     distances *= -2
#     distances += np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
#     # Convert to actual distances
#     distances = np.sqrt(np.maximum(distances, 0))
#     return distances


# def silhouette_score(X, labels):
#     """
#     Calculate the silhouette score for each sample in X, given the labels, without using scikit-learn.

#     Args:
#     - X : array-like of shape (n_samples, n_features)
#         Feature matrix of the samples.
#     - labels : array-like of shape (n_samples,)
#         Cluster labels for each sample.

#     Returns:
#     - silhouette_score : float
#         Mean silhouette coefficient for all samples.
#     """

#     unique_labels = np.unique(labels)
#     n_clusters = len(unique_labels)

#     if n_clusters == 1 or n_clusters == len(labels):
#         return 0

#     dist_matrix = euclidean_distances(X)
#     silhouette_scores = np.zeros(len(X))

#     for i in range(len(X)):
#         same_cluster_mask = labels == labels[i]
#         different_cluster_mask = ~same_cluster_mask
#         a_i = np.mean(dist_matrix[i, same_cluster_mask])
#         b_i = np.inf
#         for label in unique_labels:
#             if label == labels[i]:
#                 continue
#             label_mask = labels == label
#             dist_to_other_cluster = np.mean(dist_matrix[i, label_mask])
#             b_i = min(b_i, dist_to_other_cluster)

#         silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

#     return np.mean(silhouette_scores)
