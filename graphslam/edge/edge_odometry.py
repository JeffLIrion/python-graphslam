# Copyright (c) 2020 Jeff Irion and contributors

r"""A class for odometry edges.

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
from ..util import upper_triangular_matrix_to_full_matrix


class EdgeOdometry(BaseEdge):
    r"""A class for representing odometry edges in Graph SLAM.

    Parameters
    ----------
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    Attributes
    ----------
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    """

    def is_valid(self):
        """Check that the edge is valid.

        Returns
        -------
        bool
            Whether the edge is valid

        """
        # This will make sure that `len(self.vertices) == len(self.vertex_ids)`
        if not self._is_valid() or len(self.vertices) != 2:
            return False

        # The poses and the estimate must all be the same type
        pose_type = type(self.vertices[0].pose)
        if not isinstance(self.vertices[1].pose, pose_type) or not isinstance(self.estimate, pose_type):
            return False

        # The information matrix must be the correct size
        n = pose_type.COMPACT_DIMENSIONALITY
        return self.information.shape == (n, n)

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        .. math::

           \mathbf{e}_j = \mathbf{z}_j - (p_2 \ominus p_1)


        Returns
        -------
        np.ndarray
            The error for the edge

        """
        return (self.estimate - (self.vertices[1].pose - self.vertices[0].pose)).to_compact()

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        # fmt: off
        return [np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_other(self.vertices[0].pose)), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_self(self.vertices[0].pose)), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on

    def to_g2o(self):
        """Export the edge to the .g2o format.

        Returns
        -------
        str
            The edge in .g2o format

        """
        # fmt: off
        if isinstance(self.vertices[0].pose, PoseSE2):
            return "EDGE_SE2 {} {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1], self.estimate[2]) + " ".join([str(x) for x in self.information[np.triu_indices(3, 0)]]) + "\n"

        if isinstance(self.vertices[0].pose, PoseSE3):
            return "EDGE_SE3:QUAT {} {} {} {} {} {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1], self.estimate[2], self.estimate[3], self.estimate[4], self.estimate[5], self.estimate[6]) + " ".join([str(x) for x in self.information[np.triu_indices(6, 0)]]) + "\n"
        # fmt: on

        raise NotImplementedError

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
        EdgeOdometry, None
            The instantiated edge object, or ``None`` if ``line`` does not correspond to an odometry edge

        """
        if line.startswith("EDGE_SE2 "):
            numbers = line[len("EDGE_SE2 "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
            vertex_ids = [int(numbers[0]), int(numbers[1])]
            estimate = PoseSE2(arr[:2], arr[2])
            information = upper_triangular_matrix_to_full_matrix(arr[3:], 3)
            return EdgeOdometry(vertex_ids, information, estimate)

        if line.startswith("EDGE_SE3:QUAT "):
            numbers = line[len("EDGE_SE3:QUAT "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
            vertex_ids = [int(numbers[0]), int(numbers[1])]
            estimate = PoseSE3(arr[:3], arr[3:7])
            estimate.normalize()
            information = upper_triangular_matrix_to_full_matrix(arr[7:], 6)
            return EdgeOdometry(vertex_ids, information, estimate)

        return None

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
