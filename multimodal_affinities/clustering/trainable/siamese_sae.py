# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from torch import nn
from multimodal_affinities.clustering.trainable.embeddings_mlp import EmbeddingsProjection
from multimodal_affinities.clustering.trainable.sae import StackedAutoencoder


class SiameseStackedAutoencoder(nn.Module):

    def __init__(self, dims, dropout, projections_desc=None):
        super(SiameseStackedAutoencoder, self).__init__()
        self.autoencoder = StackedAutoencoder(dims, dropout)
        if projections_desc:
            self.projection_block = EmbeddingsProjection(projections_desc, dropout)
        else:
            self.projection_block = None
        self.train()

    def forward(self, x1, x2):
        if self.projection_block is not None:
            proj_x1 = self.projection_block(x1)
            proj_x2 = self.projection_block(x2)
        else:
            proj_x1, proj_x2 = x1, x2
        encoded_data1, decoded_data1 = self.autoencoder(proj_x1)
        encoded_data2, decoded_data2 = self.autoencoder(proj_x2)
        return proj_x1, proj_x2, encoded_data1, encoded_data2, decoded_data1, decoded_data2

