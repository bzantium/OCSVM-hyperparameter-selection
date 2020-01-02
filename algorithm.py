from sklearn.neighbors import NearestNeighbors
import numpy as np
import math


def edge_pattern_detection(X, k=None, threshold=0.01):
    edge_index = []
    normal_vector = []
    if k is None:
        k = math.ceil(5 * math.log10(len(X)))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, neighbor_indices = nbrs.kneighbors(X)
    for i in range(len(X)):
        neighbors = X[neighbor_indices[i][1:]]
        X_i = np.tile(X[i], (k, 1))
        v = (X_i - neighbors) / (np.linalg.norm(X_i - neighbors, axis=1, keepdims=True) + 1e-8)
        n = np.sum(v, axis=0, keepdims=True)
        normal_vector.append(n.squeeze())
        theta = np.dot(v, n.transpose())
        l = np.mean(theta >= 0)
        if l > 1 - threshold:
            edge_index.append(i)
    normal_vector = np.array(normal_vector)
    return edge_index, distances, neighbor_indices, normal_vector


def generate_pseudo_outlier(X_edge, ns_magnitude, edge_normal_vector):
    pseudo_outlier_X = X_edge + edge_normal_vector / \
                       (np.linalg.norm(edge_normal_vector, axis=1, keepdims=True) + 1e-8) * ns_magnitude

    peeudo_outlier_y = -np.ones(len(pseudo_outlier_X))
    return pseudo_outlier_X, peeudo_outlier_y


def generate_pseudo_target(X, normal_vector, neighbor_indices):
    pseudo_target_X = []
    for i in range(len(X)):
        x_ij_min = get_product_minimum(X[i], normal_vector[i], X[neighbor_indices[i][1:]])
        noramalized_direction = normal_vector[i] / (np.linalg.norm(normal_vector[i]) + 1e-8)
        pseudo_target_X.append(X[i] + np.dot(noramalized_direction, x_ij_min - X[i]) * noramalized_direction)
    pseudo_target_X = np.array(pseudo_target_X)
    peeudo_target_y = np.ones(len(pseudo_target_X))
    return pseudo_target_X, peeudo_target_y


def get_product_minimum(x_i, normal_vector_i, neighbors_i):
    normalized_normal_vector = normal_vector_i / (np.linalg.norm(normal_vector_i) + 1e-8)
    min_value = float("inf")
    x_ij_min = x_i
    for x_ij in neighbors_i:
        inp = np.dot(normalized_normal_vector, x_ij - x_i)
        if 0 < inp < min_value:
            min_value = inp
            x_ij_min = x_ij
    return x_ij_min
