def _extract_statistics(document_entities):

    min_width = 1.0
    min_height = 1.0
    for entity in document_entities:
        _, _, width, height = entity.get_bbox()
        min_width = min(min_width, width)
        min_height = min(min_height, height)

    return min_width, min_height

def _calculate_bbox_center(left, top, width, height):
    center_x = left + (width / 2)
    center_y = top + (height / 2)
    return center_x, center_y


def horizontal_metric(document_entities, document):
    _, min_height = _extract_statistics(document_entities)
    doc_width, doc_height = document.width, document.height
    doc_aspect_ratio = float(doc_width) / float(doc_height)

    print("Using horizontal metric v3")
    def _horizontal_metric_v3(u, v):
        """ Euclidean distance metric that considers only words in the same row. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = v.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        connection_ratio = (v_center_y - u_center_y) / max(u_height, v_height)

        u_right = u_left + u_width

        if u == v:
            return 0.0
        elif u_center_x > v_center_x or \
                abs(u_center_y - v_center_y) > (min_height / 2):
            return 1.0
        space = max(v_left - u_right, 0.0)
        h_box_ratio = max(u_height, v_height)
        dist = space * doc_aspect_ratio / h_box_ratio
        return dist

    return _horizontal_metric_v3


def vertical_metric(document_entities, document):

    min_width, _ = _extract_statistics(document_entities)

    print("Using vertical metric v3")
    def _vertical_metric_v3(u, v):
        """ Euclidean distance metric that considers only words in separate rows. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = v.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        u_right = u_left + u_width
        v_right = v_left + v_width
        u_bottom = u_top + u_height

        if u == v:
            return 0.0
        elif u_center_y > v_center_y or \
                v_left - u_right > (min_width / 2) or \
                u_left - v_right > (min_width / 2):
            return 1.0
        dist = max(v_top - u_bottom, 0.0)
        return dist / ((u_height + v_height) / 2.)

    return _vertical_metric_v3


def horizontal_symmetric_metric(document_entities):

    _, min_height = _extract_statistics(document_entities)

    def _horizontal_symmetric_metric(u, v):
        """ Euclidean distance metric that considers only words in the same row. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = v.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        if u == v:
            return 0.0
        elif abs(u_center_y - v_center_y) > (min_height / 2):
            return 1.0
        dist = abs(u_center_x - v_center_x)
        return dist

    return _horizontal_symmetric_metric


def vertical_symmetric_metric(document_entities):

    min_width, _ = _extract_statistics(document_entities)

    def _vertical_symmetric_metric(u, v):
        """ Euclidean distance metric that considers only words in separate rows. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = u.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        if u == v:
            return 0.0
        elif abs(u_center_x - v_center_x) > (min_width / 2):
            return 1.0
        dist = abs(u_center_y - v_center_y)
        return dist

    return _vertical_symmetric_metric
