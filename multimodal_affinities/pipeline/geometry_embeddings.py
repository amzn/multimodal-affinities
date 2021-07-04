from __future__ import print_function
import torch
import math


class GeometryEmbeddings:
    """
    A class for generating a geometry descriptor for a document entity.
    """

    def __init__(self):
        pass

    def forward(self, document_entity):
        """
        :param document_entity: Document entity with geometry
        :return: A tensor of [1, #geometry_features]
        """
        left = document_entity.geometry.left
        top = document_entity.geometry.top
        right = document_entity.geometry.left + document_entity.geometry.width
        bottom = document_entity.geometry.top + document_entity.geometry.height
        radius_of_enclosing_circle = 0.5 * math.hypot(right - left, bottom - top)   # Half distance between tl and br
        center_x = document_entity.geometry.left + float(document_entity.geometry.width) / 2.0
        center_y = document_entity.geometry.top + float(document_entity.geometry.height) / 2.0
        return torch.Tensor([[
            document_entity.geometry.left,
            document_entity.geometry.top,
            document_entity.geometry.height,
            document_entity.geometry.width,
            # radius_of_enclosing_circle,
            # center_x,
            # center_y,
        ]])
