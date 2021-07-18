# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from abc import ABC
import os, sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(SRC_DIR)


class DocumentEntity(ABC):
    """ Abstract object representing all entities contained within a document """
    def __init__(self, text, geometry):
        self.text = text
        self.geometry = geometry
        self.embedding = None
        self.ner_tag = -1

    def get_bbox(self):
        return self.geometry.get_left_top_width_height()

    def get_text(self):
        return self.text
