from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import math


class SelfAdaptiveShifting:
    def __init__(self, data):
        self.data = data
        self.k = math.ceil(5 * math.log10(len(self.data)))
        self.distances = None
        self.all_neighbor_indices = None
        self.edge_indice = None
        self.normal_vectors = None
        self.pseudo_outliers = None
        self.pseudo_targets = None

    def edge_pattern_detection(self, threshold=0.01):
        edge_indice, normal_vectors = [], []
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(self.data)
        self.distances, self.all_neighbor_indices = nbrs.kneighbors(self.data)
        for i in range(len(self.data)):
            neighbors = self.data[self.all_neighbor_indices[i][1:]]
            x_i = np.tile(self.data[i], (self.k, 1))
            v = (x_i - neighbors) / (np.expand_dims(self.distances[i][1:], axis=1) + 1e-8)
            n = np.sum(v, axis=0, keepdims=True)
            normal_vectors.append(n.squeeze())
            theta = np.dot(v, n.transpose())
            l = np.mean(theta >= 0)
            if l > 1 - threshold:
                edge_indice.append(i)
        self.normal_vectors = np.array(normal_vectors)
        self.edge_indice = edge_indice

    def generate_pseudo_outliers(self):
        edge_data = self.data[self.edge_indice]
        l_ns = np.mean(self.distances[self.edge_indice][1:])
        edge_normal_vectors = self.normal_vectors[self.edge_indice]
        self.pseudo_outliers = edge_data + edge_normal_vectors / (np.linalg.norm(edge_normal_vectors, axis=1, keepdims=True) + 1e-8) * l_ns
        return self.pseudo_outliers

    def generate_pseudo_targets(self):
        shift_directions = -self.normal_vectors
        unit_shift_directions = shift_directions / (np.linalg.norm(shift_directions, axis=1, keepdims=True) + 1e-8)
        pseudo_targets = []
        for i in range(len(self.data)):
            min_product = self.get_product_minimum(self.data[i], unit_shift_directions[i], self.all_neighbor_indices[i])
            pseudo_targets.append(self.data[i] + min_product * unit_shift_directions[i])
            pseudo_targets.append(self.data[i] - min_product * unit_shift_directions[i])
        self.pseudo_targets = np.array(pseudo_targets)
        return self.pseudo_targets

    def get_product_minimum(self, x_i, unit_shift_direction, neighbor_indices):
        x_ij_min = x_i
        min_product = float("inf")
        for neighbor_index in neighbor_indices[1:]:
            x_ij = self.data[neighbor_index]
            inp = np.dot(unit_shift_direction, x_ij - x_i)
            if 0 < inp < min_product:
                min_product = inp
                x_ij_min = x_ij
        if x_ij_min is x_i:
            min_product = 0
        return min_product

    def visualize(self):
        plt.figure()
        if np.size(self.data, 1) > 2:
            all_data = np.concatenate((self.data, self.pseudo_outliers, self.pseudo_targets), axis=0)
            tsne_all_data = TSNE(n_components=2, random_state=42).fit_transform(all_data)
            tsne_data = tsne_all_data[:len(self.data)]
            tsne_pseudo_outliers = tsne_all_data[len(self.data):(len(self.data) + len(self.pseudo_outliers))]
            tsne_pseudo_targets = tsne_all_data[-len(self.pseudo_targets):]

            origin = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='k', lw=0)
            edge = plt.scatter(tsne_data[self.edge_indice][:,0], tsne_data[self.edge_indice][:, 1], c = 'b', lw=0)
            outlier = plt.scatter(tsne_pseudo_outliers[:, 0], tsne_pseudo_outliers[:, 1], c='r', lw=0)
            target = plt.scatter(tsne_pseudo_targets[:, 0], tsne_pseudo_targets[:, 1], c='g', lw=0)
        else:
            origin = plt.scatter(self.data[:, 0], self.data[:, 1], c='k', lw=0)
            edge = plt.scatter(self.data[self.edge_indice][:,0], self.data[self.edge_indice][:, 1], c = 'b', lw=0)
            outlier = plt.scatter(self.pseudo_outliers[:, 0], self.pseudo_outliers[:, 1], c='r', lw=0)
            target = plt.scatter(self.pseudo_targets[:, 0], self.pseudo_targets[:, 1], c='g', lw=0)

        plt.legend((origin, edge, outlier, target),
           ('Target data', 'Edge', 'Pseudo outliers', 'Pseudo target data'),
           loc='upper left',
           ncol=1,
           fontsize=12)
        plt.show()
