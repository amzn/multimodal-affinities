from scipy.spatial import distance
import numpy as np


def euclidean_metric(u,v):
    """
    :param u: Tuple of (x,y)
    :param v: Tuple of (x,y)
    :return: Euclidean distance between points u and v
    """
    u = np.array(u)
    v = np.array(v)
    return np.linalg.norm(u - v)


def bounding_box_to_top_left_bottom_right(entity):
    return entity.geometry.left,\
           entity.geometry.top, \
           entity.geometry.left + entity.geometry.width, \
           entity.geometry.top + entity.geometry.height


def rect_distance():
    """
    Returns the a distance functions between pairs of rectangles using the given dist_metric.
    :param dist_metric: A distance metric function of the form f(u,v) -> float where u and v are
    bounding boxes of the form [x, y, w, h]
    :return: Distance function f(u,v) for pairs of rectangles
    """

    dist_metric = euclidean_metric

    def _rect_distance(u, v):
        """
        Calculates the distance between 2 rects
        :param u: Rect 1 of the form [x, y, w, h]
        :param v: Rect 2 of the form [x, y, w, h]
        :return: Distance between u,v, using a predefined metric
        """
        x1a, y1a, x1b, y1b = u
        x2a, y2a, x2b, y2b = v

        left = x2b < x1a
        right = x1b < x2a
        bottom = y2b < y1a
        top = y1b < y2a
        if top and left:
            return dist_metric((x1a, y1b), (x2b, y2a))
        elif left and bottom:
            return dist_metric((x1a, y1a), (x2b, y2b))
        elif bottom and right:
            return dist_metric((x1b, y1a), (x2a, y2b))
        elif right and top:
            return dist_metric((x1b, y1b), (x2a, y2a))
        elif left:
            return dist_metric((x1a, 0), (x2b, 0))
        elif right:
            return dist_metric((x1b, 0), (x2a, 0))
        elif bottom:
            return dist_metric((0, y1a), (0, y2b))
        elif top:
            return dist_metric((0, y1b), (0, y2a))
        else:             # rectangles intersect
            return np.inf

    return _rect_distance

def entities_to_numpy(entities):

    return np.array([bounding_box_to_top_left_bottom_right(entity) for entity in entities])

def extract_nearest_neighbors_dists(entities):
    entities_np = entities_to_numpy(entities)
    distance_mat = distance.cdist(XA=entities_np, XB=entities_np, metric=rect_distance())
    return np.amin(a=distance_mat, axis=1)


def get_isolated_nodes(entities, threshold=2.0):

    nearest_neighbor_dists = extract_nearest_neighbors_dists(entities)
    mean_dist = np.mean(nearest_neighbor_dists)

    return [entity for entity, dist in zip(entities, nearest_neighbor_dists) if dist > mean_dist * threshold]
