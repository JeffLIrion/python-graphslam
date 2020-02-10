"""A ``Vertex`` class.

"""


# pylint: disable=too-few-public-methods
class Vertex:
    """A class for representing a vertex in Graph SLAM.

    Parameters
    ----------
    vertex_id : int
        The vertex's unique ID
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex

    Attributes
    ----------
    vertex_id : int
        The vertex's unique ID
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex

    """
    def __init__(self, vertex_id, pose):
        self.vertex_id = vertex_id
        self.pose = pose
