# Copyright (c) 2020 Jeff Irion and contributors

"""A ``Vertex`` class.

"""

from .pose.se2 import PoseSE2
from .pose.se3 import PoseSE3


# pylint: disable=too-few-public-methods
class Vertex:
    """A class for representing a vertex in Graph SLAM.

    Parameters
    ----------
    vertex_id : int
        The vertex's unique ID
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex
    vertex_index : int, None
        The vertex's index in the graph's ``vertices`` list

    Attributes
    ----------
    id : int
        The vertex's unique ID
    index : int, None
        The vertex's index in the graph's ``vertices`` list
    pose : graphslam.pose.base_pose.BasePose
        The pose associated with the vertex

    """
    def __init__(self, vertex_id, pose, vertex_index=None):
        self.id = vertex_id
        self.pose = pose
        self.index = vertex_index

    def to_g2o(self):
        """Export the vertex to the .g2o format.

        Returns
        -------
        str
            The vertex in .g2o format

        """
        if isinstance(self.pose, PoseSE2):
            return "VERTEX_SE2 {} {} {} {}\n".format(self.id, self.pose[0], self.pose[1], self.pose[2])

        if isinstance(self.pose, PoseSE3):
            return "VERTEX_SE3:QUAT {} {} {} {} {} {} {} {}\n".format(self.id, self.pose[0], self.pose[1], self.pose[2], self.pose[3], self.pose[4], self.pose[5], self.pose[6])

        raise NotImplementedError
