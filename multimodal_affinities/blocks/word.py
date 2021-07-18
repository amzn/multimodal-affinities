# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(SRC_DIR)

from multimodal_affinities.blocks.document_entity import DocumentEntity
from multimodal_affinities.blocks.geometry import Geometry


class Word(DocumentEntity):
    """ Represents a word in the document - the atomic unit the ocr emits """
    def __init__(self, word_info_dict):
        geometry_dict = word_info_dict["Geometry"]
        geometry = Geometry(geometry_dict)
        text = word_info_dict["DetectedText"]
        super().__init__(text, geometry)

    def __key(self):
        return self.text, self.geometry

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()
