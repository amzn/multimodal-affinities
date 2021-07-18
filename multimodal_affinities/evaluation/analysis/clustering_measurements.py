# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from multimodal_affinities.blocks.document import Document


def cluster_accuracy(y_true, y_pred):
    """ Calculates the accuracy of the clustering using a linear assignment which maximizes the accuracy.
    Returns the accuracy percentage, ami, nmi and the matrix w of all the combinations of indexes of the
    original clusters and the calculated ones
    :param y_true: The ground-truth labels
    :param y_pred: The predicted labels
    :return the clustering accuracy metrics (acc, ami, nmi) and the predicted labels, permuted according to the ground-truth
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert y_pred.shape == y_true.shape, "y_pred.shape: " + str(y_pred.shape) + ", y_true.shape: " + str(y_true.shape)
    y_true_unique = np.unique(y_true)
    true_cluster_idx = np.nonzero(y_true[:, None] == y_true_unique)[1]
    distMatrix = max(y_pred.max() + 1, len(y_true_unique))
    w = np.zeros((distMatrix, len(y_true_unique)), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], true_cluster_idx[i]] += 1
    ind = linear_assignment(w.max() - w)
    y_pred_new = -1 * np.ones(len(y_pred), int)
    for i in range(0, len(y_pred)):
        j = np.argwhere(ind[:, 0] == y_pred[i])
        if j.shape[0] > 0:
            y_pred_new[i] = (ind[j[0], 1])
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    ami = adjusted_mutual_info_score(y_true, y_pred, average_method='max')
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return acc, ami, nmi, w, y_pred_new


def doc_to_labels(document):
    """
    Converts document to a label: list of cluster ids.
    :param document: Document object
    :return: list of cluster ids.
    """
    labels = []
    word_to_cluster = document.words_to_clusters()
    cluster_to_id = {cluster:cluster_id for cluster_id, cluster in enumerate(document.get_clusters())}

    for word in document.get_words():
        cluster = word_to_cluster[word]
        cluster_id = cluster_to_id[cluster] if cluster else -1
        labels.append(cluster_id)

    return labels


def measure_scores(pred, gt):
    """
    Measure clustering accuracy scores (acc, ami, nmi).
    :param pred: Document with clustering results of label as a list of clustering ids
    :param gt: Document with clustering results of label as a list of clustering ids
    :return: Tuple of (acc, ami, nmi)
    """
    y_pred = doc_to_labels(pred) if isinstance(pred, Document) else pred
    y_true = doc_to_labels(gt) if isinstance(gt, Document) else gt
    acc, ami, nmi, w, pred_labels_new = cluster_accuracy(y_true, y_pred)
    return acc, ami, nmi
