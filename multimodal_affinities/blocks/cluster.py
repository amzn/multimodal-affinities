# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(SRC_DIR)

from multimodal_affinities.blocks.document_entity import DocumentEntity
from multimodal_affinities.blocks.word import Word
from multimodal_affinities.blocks.geometry import Geometry


class Cluster(DocumentEntity):
    """ Represents a cluster of words """
    def __init__(self, words):

        text = [w.text for w in words]
        geometry = Geometry.union([w.geometry for w in words])
        super().__init__(text, geometry)

        # Sub clusters are broken into words
        self.words = []
        for entity in words:
            if isinstance(entity, Word):
                self.words.append(entity)
            elif isinstance(entity, Cluster):
                self.words.extend(entity.words)
