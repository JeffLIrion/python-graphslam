# Copyright (c) 2020 Jeff Irion and contributors

r"""A class for landmark edges.

"""


import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from .base_edge import BaseEdge

from ..pose.r2 import PoseR2
from ..pose.se2 import PoseSE2
from ..pose.r3 import PoseR3
from ..pose.se3 import PoseSE3


class EdgeLandmark(BaseEdge):
    r"""A class for representing landmark edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`
    offset : BasePose
        The offset

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`
    offset : BasePose
        The offset

    """

    def __init__(self, vertex_ids, information, estimate, vertices=None, offset=None):
        super().__init__(vertex_ids, information, estimate, vertices)
        self.offset = offset

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        .. math::

           \mathbf{e}_j = \mathbf{z}_j - ((p_1 \oplus p_{\text{offset}})^{-1} \oplus p_2)


        Returns
        -------
        np.ndarray
            The error for the edge

        """
        return (self.estimate - ((self.vertices[0].pose + self.offset).inverse + self.vertices[1].pose)).to_compact()

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        pose_oplus_offset = self.vertices[0].pose + self.offset
        # fmt: off
        return [-np.dot(np.dot(np.dot(pose_oplus_offset.inverse.jacobian_self_oplus_point_wrt_self(self.vertices[1].pose), pose_oplus_offset.jacobian_inverse()), self.vertices[0].pose.jacobian_self_oplus_point_wrt_self(self.offset)), self.vertices[0].pose.jacobian_boxplus()),
                -np.dot(pose_oplus_offset.inverse.jacobian_self_oplus_point_wrt_point(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on

    def to_g2o(self):
        """Export the edge to the .g2o format.

        Returns
        -------
        str
            The edge in .g2o format

        """
        # https://docs.ros.org/en/kinetic/api/rtabmap/html/OptimizerG2O_8cpp_source.html
        # fmt: off
        if isinstance(self.vertices[0].pose, PoseSE2):
            return "EDGE_SE2_XY {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1]) + " ".join([str(x) for x in self.information[np.triu_indices(2, 0)]]) + "\n"

        if isinstance(self.vertices[0].pose, PoseSE3):
            offset_parameter = 0
            return "EDGE_SE3_TRACKXYZ {} {} {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], offset_parameter, self.estimate[0], self.estimate[1], self.estimate[2]) + " ".join([str(x) for x in self.information[np.triu_indices(3, 0)]]) + "\n"
        # fmt: on

        raise NotImplementedError

    def plot(self, color="b"):
        """Plot the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        if isinstance(self.vertices[0].pose, (PoseR2, PoseSE2)):
            xy = np.array([v.pose.position for v in self.vertices])
            plt.plot(xy[:, 0], xy[:, 1], color=color)

        elif isinstance(self.vertices[0].pose, (PoseR3, PoseSE3)):
            xyz = np.array([v.pose.position for v in self.vertices])
            plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color)

        else:
            raise NotImplementedError
