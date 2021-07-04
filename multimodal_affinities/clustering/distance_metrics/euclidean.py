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

    def _horizontal_metric(u, v):
        """ Euclidean distance metric that considers only words in the same row. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = v.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        if u == v:
            return 0.0
        elif u_left > v_left or \
                abs(u_center_y - v_center_y) > (min_height / 2):
            return 1.0
        dist = abs(u_center_x - v_center_x)
        return dist

    return _horizontal_metric


def vertical_metric(document_entities, document):

    min_width, _ = _extract_statistics(document_entities)

    def _vertical_metric(u, v):
        """ Euclidean distance metric that considers only words in separate rows. """
        u_left, u_top, u_width, u_height = u.get_bbox()
        u_center_x, u_center_y = _calculate_bbox_center(u_left, u_top, u_width, u_height)
        v_left, v_top, v_width, v_height = u.get_bbox()
        v_center_x, v_center_y = _calculate_bbox_center(v_left, v_top, v_width, v_height)

        if u == v:
            return 0.0
        elif u_top > v_top or \
                abs(u_center_x - v_center_x) > (min_width / 2):
            return 1.0
        dist = abs(u_center_y - v_center_y)
        return dist

    return _vertical_metric


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
