# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch
import torch.nn.functional as F
import numpy as np


def embedding_distance(document, height_ratio_cutoff_threshold, reduce):
    """
    Calculates L2 distance in embedding space between pair of entities.
    Assumes embedding have been pre-calculated.
    :return: L2 distance in embedding space
    """
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    def _embedding_distance(phrase_a, phrase_b):
        if phrase_a == phrase_b:
            return 0.0

        phrase_a_avg_height = sum([w.geometry.height for w in phrase_a.words]) / len(phrase_a.words)
        phrase_b_avg_height = sum([w.geometry.height for w in phrase_b.words]) / len(phrase_b.words)
        phrases_height_ratio = float(phrase_a_avg_height) / float(phrase_b_avg_height)
        high_threshold = 1.0 + height_ratio_cutoff_threshold
        low_threshold = 1.0 - height_ratio_cutoff_threshold
        if not high_threshold > phrases_height_ratio > low_threshold:   # Cutoff
            return 1.0

        with torch.no_grad():

            phrase_a_words = [w for w in phrase_a.words if len(w.text) >= 3]
            if len(phrase_a_words) == 0:
                phrase_a_words = phrase_a.words
            phrase_b_words = [w for w in phrase_b.words if len(w.text) >= 3]
            if len(phrase_b_words) == 0:
                phrase_b_words = phrase_b.words

            # words_a, words_b are tensors of BATCH X DIM
            words_a = torch.stack([w.embedding for w in phrase_a_words])
            words_b = torch.stack([w.embedding for w in phrase_b_words])

            # --- vec 1 ---
            # ...
            # --- vec 1 ---
            # --- vec 2 ---
            # ...
            # --- vec 2 ---
            words_a_batch = tile(words_a, dim=0, n_tile=words_b.shape[0])

            # --- vec 1 ---
            # --- vec 2 ---
            # --- vec 3 ---
            # ...
            # --- vec 1 ---
            # --- vec 2 ---
            # --- vec 3 ---
            # ...
            words_b_batch = words_b.repeat(words_a.shape[0], 1)

            euclidean_distance = F.pairwise_distance(words_a_batch, words_b_batch) / 2

            distance_tensor = 1.0 - torch.mean(euclidean_distance)
        return distance_tensor.numpy().item(0)
    return _embedding_distance


def neural_phrase_affinity_distance(document, height_ratio_cutoff_threshold, reduce):
    """
    Calculates neural distance between pair of phrases according to words voting mechanism:
    Assumes embedding have been pre-calculated for each word.
    1) Constructs all possible pairs of words between the 2 phrases.
    2) Calculates approximated contrastive loss between each pair of words (assuming they belong together).
    3) The affinity is calculated according to the "voting" of each pair.
    The affinity is stronger when more pairs of words agree with each other.
    :return: Distance = 1.0 - affinity
    """
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    def _neural_affinity_distance(phrase_a, phrase_b):
        if phrase_a == phrase_b:
            return 0.0

        phrase_a_avg_height = sum([w.geometry.height for w in phrase_a.words]) / len(phrase_a.words)
        phrase_b_avg_height = sum([w.geometry.height for w in phrase_b.words]) / len(phrase_b.words)
        phrases_height_ratio = float(phrase_a_avg_height) / float(phrase_b_avg_height)
        high_threshold = 1.0 + height_ratio_cutoff_threshold
        low_threshold = 1.0 - height_ratio_cutoff_threshold
        if not high_threshold > phrases_height_ratio > low_threshold:   # Cutoff
            return 1.0

        with torch.no_grad():

            phrase_a_words = [w for w in phrase_a.words if len(w.text) >= 3]
            if len(phrase_a_words) == 0:
                phrase_a_words = phrase_a.words
            phrase_b_words = [w for w in phrase_b.words if len(w.text) >= 3]
            if len(phrase_b_words) == 0:
                phrase_b_words = phrase_b.words

            # words_a, words_b are tensors of BATCH X DIM
            words_a = torch.stack([w.embedding for w in phrase_a_words])
            words_b = torch.stack([w.embedding for w in phrase_b_words])

            # --- vec 1 ---
            # ...
            # --- vec 1 ---
            # --- vec 2 ---
            # ...
            # --- vec 2 ---
            words_a_batch = tile(words_a, dim=0, n_tile=words_b.shape[0])

            # --- vec 1 ---
            # --- vec 2 ---
            # --- vec 3 ---
            # ...
            # --- vec 1 ---
            # --- vec 2 ---
            # --- vec 3 ---
            # ...
            words_b_batch = words_b.repeat(words_a.shape[0], 1)

            euclidean_distance = F.pairwise_distance(words_a_batch, words_b_batch)
            sigmoid = torch.exp(-torch.pow(euclidean_distance, 2) / 1.5)  # Approximation to sigmoid

            if reduce == 'mean':
                distance = F.binary_cross_entropy(input=sigmoid, target=torch.ones(words_a_batch.shape[0]))
            elif reduce == 'mean_without_log':
                distance = 1.0 - torch.mean(sigmoid)
            elif reduce == 'median':
                distance_samples = F.binary_cross_entropy(input=sigmoid, target=torch.ones(words_a_batch.shape[0]),
                                                          reduce=False)
                median_idx = len(distance_samples) // 2
                distance = distance_samples[median_idx]
            elif reduce == 'median_no_log':
                distance_samples = 1.0 - sigmoid
                median_idx = len(distance_samples) // 2
                distance = distance_samples[median_idx]
            else:
                raise ValueError('Unknown clustering reduce scheme: %r' % reduce)

        return distance.numpy().item(0)
    return _neural_affinity_distance
