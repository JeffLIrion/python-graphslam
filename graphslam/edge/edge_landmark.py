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
from ..util import upper_triangular_matrix_to_full_matrix


class EdgeLandmark(BaseEdge):
    r"""A class for representing landmark edges in Graph SLAM.

    Parameters
    ----------
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`; this should be the same type as ``self.vertices[1].pose``
    offset : BasePose, None
        The offset that is applied to the first pose; this should be the same type as ``self.vertices[0].pose``
    offset_id : int, None
        The ID of the offset; this is only used for writing to .g2o format
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    Attributes
    ----------
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`; this should be the same type as ``self.vertices[1].pose``
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    offset : BasePose, None
        The offset that is applied to the first pose; this should be the same type as ``self.vertices[0].pose``
    offset_id : int, None
        The ID of the offset; this is only used for writing to .g2o format
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    """

    def __init__(self, vertex_ids, information, estimate, offset, offset_id=None, vertices=None):
        super().__init__(vertex_ids, information, estimate, vertices)
        self.offset = offset
        self.offset_id = offset_id

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
        # https://docs.ros.org/en/kinetic/api/rtabmap/html/OptimizerG2O_8cpp_source.html
        # fmt: off
        if isinstance(self.vertices[0].pose, PoseSE2):
            return "EDGE_SE2_XY {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1]) + " ".join([str(x) for x in self.information[np.triu_indices(2, 0)]]) + "\n"

        if isinstance(self.vertices[0].pose, PoseSE3):
            return "EDGE_SE3_TRACKXYZ {} {} {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.offset_id, self.estimate[0], self.estimate[1], self.estimate[2]) + " ".join([str(x) for x in self.information[np.triu_indices(3, 0)]]) + "\n"
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
        EdgeLandmark, None
            The instantiated edge object, or ``None`` if ``line`` does not correspond to a landmark edge

        """
        if line.startswith("EDGE_SE2_XY "):
            numbers = line[len("EDGE_SE2_XY "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
            vertex_ids = [int(numbers[0]), int(numbers[1])]
            estimate = PoseR2(arr[:2])
            information = upper_triangular_matrix_to_full_matrix(arr[2:], 2)
            # 2-D landmark edges in g2o don't support an offset, so just use the identity
            return EdgeLandmark(vertex_ids, information, estimate, offset=PoseSE2.identity(), offset_id=0)

        if line.startswith("EDGE_SE3_TRACKXYZ "):
            assert g2o_params_or_none is not None
            numbers = line[len("EDGE_SE3_TRACKXYZ "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[3:]], dtype=np.float64)
            vertex_ids = [int(numbers[0]), int(numbers[1])]
            offset_id = int(numbers[2])
            offset = g2o_params_or_none[("PARAMS_SE3OFFSET", offset_id)].value
            estimate = PoseR3(arr[:3])
            information = upper_triangular_matrix_to_full_matrix(arr[3:], 3)
            return EdgeLandmark(vertex_ids, information, estimate, offset=offset, offset_id=offset_id)

        return None

    def plot(self, color="g"):
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

        if ((self.offset_id is None) ^ (other.offset_id is None)) or (
            self.offset_id is not None and self.offset_id != other.offset_id
        ):
            return False

        return BaseEdge.equals(self, other, tol)
