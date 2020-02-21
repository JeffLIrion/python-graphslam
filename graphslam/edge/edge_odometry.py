# Copyright (c) 2020 Jeff Irion and contributors

r"""A class for odometry edges.

"""


import numpy as np

from .base_edge import BaseEdge


class EdgeOdometry(BaseEdge):
    r"""A class for representing odometry edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : np.ndarray, float
        The expected measurement :math:`\mathbf{z}_j`

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : np.ndarray, float
        The expected measurement :math:`\mathbf{z}_j`

    """

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        .. math::

           \mathbf{e}_j = \mathbf{z}_j - (p_2 \ominus p_1)


        Returns
        -------
        np.ndarray
            The error for the edge

        """
        return self.estimate - (self.vertices[1].pose - self.vertices[0].pose).to_compact()

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        return [-np.dot(self.vertices[1].pose.jacobian_self_ominus_other_wrt_other_compact(self.vertices[0].pose), self.vertices[0].pose.jacobian_boxplus()),
                -np.dot(self.vertices[1].pose.jacobian_self_ominus_other_wrt_self_compact(self.vertices[0].pose), self.vertices[1].pose.jacobian_boxplus())]
