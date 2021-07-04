import copy

class Geometry(object):
    '''
    Bounding box:
    (0,0) ->
      |
      V     =================
            =               =
            =================
    '''

    def __init__(self, geometry_dict):

        self.bb_obj = geometry_dict["BoundingBox"]

        self.width = self.bb_obj["Width"]
        self.top = self.bb_obj["Top"]
        self.height = self.bb_obj["Height"]
        self.left = self.bb_obj["Left"]

        self.polygon = geometry_dict["Polygon"]

    @staticmethod
    def from_left_top_right_bottom(left, top, right, bottom):
        geometry_dict = {
            "BoundingBox": {
                "Width": right-left,
                "Top": top,
                "Height": bottom - top,
                "Left": left
            },
            "Polygon": [
                {
                    "Y": top,
                    "X": left
                },
                {
                    "Y": top,
                    "X": right
                },
                {
                    "Y": bottom,
                    "X": right
                },
                {
                    "Y": bottom,
                    "X": left
                }
            ]
        }

        return Geometry(geometry_dict)

    def get_left_top_width_height(self):
        return [self.left, self.top, self.width, self.height]

    def get_left_top_right_bottom(self):
        return [self.left, self.top, self.left + self.width, self.top + self.height]

    def get_bb_corners(self):
        pt1 = (self.left, self.top)
        pt2 = (self.left + self.width, self.top)
        pt3 = (self.left + self.width, self.top + self.height)
        pt4 = (self.left, self.top + self.height)
        return [pt1, pt2, pt3, pt4]

    def area(self):
        return self.width * self.height

    @staticmethod
    def union(geometries):
        if not geometries:
            return None
        elif len(geometries) == 1:
            return copy.deepcopy(geometries[0])

        left = min(*[geom.left for geom in geometries])
        top = min(*[geom.top for geom in geometries])
        right = max(*[geom.left + geom.width for geom in geometries])
        bottom = max(*[geom.top + geom.height for geom in geometries])

        return Geometry.from_left_top_right_bottom(left, top, right, bottom)
