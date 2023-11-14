# Copyright (c) 2020 Jeff Irion and contributors

r"""A class for landmark edges.

"""


import numpy as np

from .base_edge import BaseEdge


class EdgeLandmark(BaseEdge):
    r"""A class for representing landmark edges in Graph SLAM.

    Parameters
    ----------
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose, np.array
        The expected measurement :math:`\mathbf{z}_j`; this should be the same type as ``self.vertices[1].pose``
        or a numpy array that is the same length and behaves in the same way (e.g., an array of length 2 instead
        of a `PoseSE2` object)
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge
    offset : BasePose, None
        The offset that is applied to the first pose; this should be the same type as ``self.vertices[0].pose``

    Attributes
    ----------
    estimate : BasePose, np.array
        The expected measurement :math:`\mathbf{z}_j`; this should be the same type as ``self.vertices[1].pose``
        or a numpy array that is the same length and behaves in the same way (e.g., an array of length 2 instead
        of a `PoseSE2` object)
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    offset : BasePose, None
        The offset that is applied to the first pose; this should be the same type as ``self.vertices[0].pose``
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    """

    def __init__(self, vertex_ids, information, estimate, vertices=None, offset=None):
        super().__init__(vertex_ids, information, estimate, vertices)
        self.offset = offset

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        .. math::

           \mathbf{e}_j =((p_1 \oplus p_{\text{offset}})^{-1} \oplus p_2) - \mathbf{z}_j


        :math:`SE(2)` landmark edges in g2o
        -----------------------------------

        - https://github.com/RainerKuemmerle/g2o/blob/c422dcc0a92941a0dfedd8531cb423138c5181bd/g2o/types/slam2d/edge_se2_pointxy.h#L44-L48


        :math:`SE(3)` landmark edges in g2o
        -----------------------------------

        - https://github.com/RainerKuemmerle/g2o/blob/c422dcc0a92941a0dfedd8531cb423138c5181bd/g2o/types/slam3d/edge_se3_pointxyz.cpp#L81-L92
           - https://github.com/RainerKuemmerle/g2o/blob/c422dcc0a92941a0dfedd8531cb423138c5181bd/g2o/types/slam3d/parameter_se3_offset.h#L76-L82
           - https://github.com/RainerKuemmerle/g2o/blob/c422dcc0a92941a0dfedd8531cb423138c5181bd/g2o/types/slam3d/parameter_se3_offset.cpp#L70


        Returns
        -------
        np.ndarray
            The error for the edge

        """
        return (((self.vertices[0].pose + self.offset).inverse + self.vertices[1].pose) - self.estimate).to_compact()

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
        return [np.dot(np.dot(np.dot(pose_oplus_offset.inverse.jacobian_self_oplus_point_wrt_self(self.vertices[1].pose), pose_oplus_offset.jacobian_inverse()), self.vertices[0].pose.jacobian_self_oplus_other_wrt_self(self.offset)), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(pose_oplus_offset.inverse.jacobian_self_oplus_point_wrt_point(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on

    def to_g2o(self):
        """Export the edge to the .g2o format.

        Returns
        -------
        str
            The edge in .g2o format

        """
        # Not yet implemented

    @classmethod
    def from_g2o(cls, line, g2o_params_or_none=None):
        """Load an edge from a line in a .g2o file.

        Parameters
        ----------
        line : str
            The line from the .g2o file
        g2o_params_or_none : dict, None
            A dictionary where the values are `graphslam.g2o_parameters.BaseG2OParameters` objects, or
            ``None`` if there are no such parameters

        Returns
        -------
        EdgeLandmark, None
            The instantiated edge object, or ``None`` if ``line`` does not correspond to a landmark edge

        """
        # Not yet implemented

    def plot(self, color="g"):
        """Plot the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """
        # Not yet implemented

    def equals(self, other, tol=1e-6):
        """Check whether two edges are equal.

        Parameters
        ----------
        other : BaseEdge
            The edge to which we are comparing
        tol : float
            The tolerance

        Returns
        -------
        bool
            Whether the two edges are equal

        """
        if not type(self.offset) is type(other.offset):  # noqa
            return False

        if not self.offset.equals(other.offset, tol):
            return False

        return BaseEdge.equals(self, other, tol)
