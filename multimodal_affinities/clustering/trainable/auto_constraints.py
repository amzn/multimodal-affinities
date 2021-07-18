# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import sys
import numpy as np
import torch

if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix, tril, find, triu, coo_matrix
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


class AutoConstraints(object):
    def __init__(self):
        pass

    def generate_auto_must_link_const_from_embeddings(self, embeddings, n_neighbors=2, dist_meas="cosine", w_multiplier=1):
        if len(embeddings) == 0:
            return []
        raw_embeddings = [embedding.detach().cpu().numpy() for embedding in embeddings]
        X = np.stack(raw_embeddings, axis=0)

        print("--- mutual Knn ---")
        E, W, connections_per_point, pairs_list = self.connectivity_structure_mknn(X=X,
                                                                                   n_neighbors=n_neighbors,
                                                                                   dist_meas=dist_meas,
                                                                                   w_multiplier=w_multiplier)
        return pairs_list


    def find_knn(self, X, n_neighbors, dist_meas, w_multiplier = 1):
        """
        :param X: dataset
        :param n_neighbours: number of neighbours for mknn calculation
        :return: P - mknn graph (csr matrix), Q - weights car matrix
        """
        samples = X.shape[0]
        batchsize = 10000
        b = np.arange(n_neighbors + 1)
        b = tuple(b[1:].ravel())

        z = np.zeros((samples, n_neighbors))
        weigh = np.zeros_like(z)
        X = np.reshape(X, (X.shape[0], -1))
        # This loop speeds up the computation by operating in batches
        # This can be parallelized to further utilize CPU/GPU resource
        for x in np.arange(0, samples, batchsize):
            start = x
            end = min(x + batchsize, samples)
            w = w_multiplier * cdist(X[start:end], X, dist_meas)
            # the first columns will be the indexes of the knn of each sample (the first column is always the same
            # index as the row)
            y = np.argpartition(w, b, axis=1)

            z[start:end, :] = y[:, 1:n_neighbors + 1]
            # the weights are the distances between the two samples
            weigh[start:end, :] = np.reshape(
                w[tuple(np.repeat(np.arange(end - start), n_neighbors)), tuple(
                    y[:, 1:n_neighbors + 1].ravel())], (end - start, n_neighbors))
            del (w)

        ind = np.repeat(np.arange(samples), n_neighbors)
        P = csr_matrix((np.ones((samples * n_neighbors)), (ind.ravel(), z.ravel())), shape=(samples, samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))
        return P, Q

    def generate_auto_cannot_link_const_from_ner_tags(self, entities, max_connections_per_point = 20):
        pairs_list = []
        tags = np.array([entity.ner_tag if entity.ner_tag is not None else -1 for entity in entities])
        indices_with_tags = np.where(tags > 0)
        counter_list = np.zeros((len(entities), 1))
        for ind_i in indices_with_tags[0].tolist():
            for ind_j in indices_with_tags[0].tolist():
                if tags[ind_i] == tags[ind_j]:
                    continue
                if counter_list[ind_i] <= max_connections_per_point and counter_list[ind_j] <= max_connections_per_point:
                    counter_list[ind_i] = counter_list[ind_i] + 1
                    counter_list[ind_j] = counter_list[ind_j] + 1
                    pairs_list.append([ind_i, ind_j])
        indices_upper_case = [i for i,entity in enumerate(entities) if entity.text.isupper()]
        for ind_i in indices_upper_case:
            for ind_j in range(len(entities)):
                if ind_j not in indices_upper_case:
                    if counter_list[ind_i] <= max_connections_per_point and counter_list[ind_j] <= max_connections_per_point:
                        counter_list[ind_i] = counter_list[ind_i] + 1
                        counter_list[ind_j] = counter_list[ind_j] + 1
                        pairs_list.append([ind_i, ind_j])
        return pairs_list

    def generate_auto_cannot_link_const_from_geometric_embeddings(self, embeddings, ratio_threshold = 1.2, max_connections_per_point = 15):
        """
        :param X: the dataset
        :param ratio_threshold
        :return: pairds_list
        """
        if len(embeddings) == 0:
            return []
        pairs_list = []
        height_vals = [embedding.detach().cpu().numpy()[0][2] for embedding in embeddings]
        counter_list = np.zeros((len(height_vals), 1))
        for i in range(len(height_vals)):
            height_i = height_vals[i]
            for j in range(i+1, len(height_vals)):
                height_j = height_vals[j]
                ratio_curr = float(height_i) / height_j
                if ratio_curr >= ratio_threshold or (1 / ratio_curr) >= ratio_threshold:
                    if counter_list[i] <= max_connections_per_point and counter_list[j] <= max_connections_per_point:
                        counter_list[i] = counter_list[i] + 1
                        counter_list[j] = counter_list[j] + 1
                        pairs_list.append([i,j])
        print("cannot_link_const_from_geometric_embeddings contributed %d new cannot-links" % len(pairs_list))
        return pairs_list


    def connectivity_structure_mknn(self, X, n_neighbors, dist_meas, w_multiplier = 1):
        """
        :param X: the dataset
        :param n_neighbours: the number of closest neighbours taken into account
        :param w_multiplier: if 1, obtains mutual nearest neighbors, if -1 obtains mutual farthest neighbors
        :return: matrix E with 1 where the two points are mknn. the matrix is lower triangular (zeros in the top triangular)
         so that each connection will be taken into account only once.
         W is a matrix of the weight of each connection. both are sparse matrices.
        """
        samples = X.shape[0]
        P, Q = self.find_knn(X, n_neighbors=n_neighbors, dist_meas=dist_meas, w_multiplier=w_multiplier)
        # Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) # + Tcsr.maximum(Tcsr.transpose())
        index = np.asarray(find(P)).T
        E = csr_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])), [samples, samples])
        connections_per_point = np.sum(E, 0)  # sum of each row
        E = triu(E, k=1)
        a = np.sum(connections_per_point) / samples  # calculating the averge number of connections
        w = \
            np.divide(a, np.sqrt(
                np.asarray(connections_per_point[0][0, E.row]) * np.asarray(connections_per_point[0][0, E.col])))[0]
        W = coo_matrix((w, (E.row, E.col)), [samples, samples])
        print('number of connections:', len(E.data), 'average connection per point', a)


        rows, cols, _ = find(P)
        pairs_list = []
        for i, j in zip(rows, cols):
            pairs_list.append([i,j])

        return E, W, connections_per_point, pairs_list