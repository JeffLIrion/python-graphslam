# Copyright (c) 2020 Jeff Irion and contributors

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
