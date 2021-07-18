# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import torch
from torch import nn


def weights_init(m):
    """
    changing the weights to a notmal distribution with mean=0 and std=0.01
    """
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.01)


class EmbeddingsProjection(nn.Module):
    """
    Projects each embedding branch separately and finally concats them, to allow finer mixture of
    multi-modal embeddings
    """

    def __init__(self, projections_desc, dropout):
        super(EmbeddingsProjection, self).__init__()
        self.dropout = dropout
        self.init_stddev = 0.01

        projection_blocks = []
        for embedding_entry in projections_desc:
            original_dim, projected_dim = embedding_entry

            single_embedding_projection_block = nn.Sequential(
                                                    nn.Dropout(self.dropout),
                                                    nn.Linear(original_dim, projected_dim),
                                                    nn.ReLU()
                                                )
            projection_blocks.append(single_embedding_projection_block)
        self.projections = nn.ModuleList(projection_blocks)

        self.n_layers = len(self.projections)
        for i in range(0, self.n_layers):
            self.projections[i].apply(weights_init)

    def forward(self, x):
        projected_embedding_concat = torch.cat([projection_block(embedding) for embedding, projection_block in zip(x, self.projections)], 1)
        return projected_embedding_concat


