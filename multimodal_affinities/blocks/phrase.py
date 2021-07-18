# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(SRC_DIR)

from multimodal_affinities.blocks.document_entity import DocumentEntity
from multimodal_affinities.blocks.word import Word
from multimodal_affinities.blocks.geometry import Geometry


class Phrase(DocumentEntity):
    """ Represents a phrase in the document - a group of words with a semantic relationship """
    def __init__(self, words):

        text = ' '.join([w.text for w in words])
        geometry = Geometry.union([w.geometry for w in words])
        super().__init__(text, geometry)

        # Inner phrases are broken into words
        self.words = []
        for entity in words:
            if isinstance(entity, Word):
                self.words.append(entity)
            elif isinstance(entity, Phrase):
                self.words.extend(entity.words)

    def __key(self):
        return self.text, self.geometry

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()
