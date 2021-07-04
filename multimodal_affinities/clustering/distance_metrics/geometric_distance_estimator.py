from distance_estimator_abstract import DistanceEstimatorAbstract


class GeometricDistanceEstimator(DistanceEstimatorAbstract):
    def __init__(self):
        pass

    def compute_dist(self, block_a, block_b):
        x0_a, y0_a, width_a, height_a = block_a.get_bbox()
        x0_b, y0_b, width_b, height_b = block_b.get_bbox()

        min_horizontal_dist = min(abs(x0_a + width_a - x0_b),
                                  abs(x0_b + width_b - x0_a))

        min_vertical_dist = min(abs(y0_a + height_a - y0_b),
                                abs(y0_b + height_b - y0_a))

        height_similarity = min(height_b / float(height_a)), height_a / float(height_b)

        dist = 0.4 * (1 - height_similarity) + \
               0.4 * min_horizontal_dist + \
               0.2 * min_vertical_dist

        return dist
