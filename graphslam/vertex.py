# Copyright (c) 2020 Jeff Irion and contributors

"""A ``Vertex`` class.

"""

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from .pose.r2 import PoseR2
from .pose.r3 import PoseR3
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

    def plot(self, color='r', marker='o', markersize=3):
        """Plot the vertex.

        Parameters
        ----------
        color : str
            The color that will be used to plot the vertex
        marker : str
            The marker that will be used to plot the vertex
        markersize : int
            The size of the plotted vertex

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        if isinstance(self.pose, (PoseR2, PoseSE2)):
            x, y = self.pose.position
            plt.plot(x, y, color=color, marker=marker, markersize=markersize)

        elif isinstance(self.pose, (PoseR3, PoseSE3)):
            x, y, z = self.pose.position
            plt.plot([x], [y], [z], markerfacecolor=color, markeredgecolor=color, marker=marker, markersize=markersize)

        else:
            raise NotImplementedError
