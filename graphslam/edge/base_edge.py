# Copyright (c) 2020 Jeff Irion and contributors

r"""A base class for edges.

"""


from abc import ABC, abstractmethod

import numpy as np

from graphslam.pose.base_pose import BasePose


class BaseEdge(ABC):
    r"""A class for representing edges in Graph SLAM.

    Parameters
    ----------
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose, np.ndarray, float
        The expected measurement :math:`\mathbf{z}_j`
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    Attributes
    ----------
    estimate : BasePose, np.ndarray, float
        The expected measurement :math:`\mathbf{z}_j`
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    vertex_ids : list[int]
        The IDs of all vertices constrained by this edge
    vertices : list[graphslam.vertex.Vertex], None
        A list of the vertices constrained by the edge

    """

    #: The difference that will be used for numerical differentiation
    _NUMERICAL_DIFFERENTIATION_EPSILON = 1e-6

    def __init__(self, vertex_ids, information, estimate, vertices=None):
        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices

    def _is_valid(self):
        """Check some basic criteria for the edge.

        Returns
        -------
        bool
            Whether the basic validity criteria for the edge are satisfied

        """
        # Make sure the `self.vertices` list has been populated and that it is the same length as `self.vertex_ids`
        if self.vertices is None or len(self.vertices) != len(self.vertex_ids):
            return False

        for vertex, v_id in zip(self.vertices, self.vertex_ids):
            if vertex.id != v_id:
                return False

        return True

    @abstractmethod
    def is_valid(self):
        """Check that the edge is valid.

        - The `vertices` attribute is populated, it is the correct length, and the poses are the correct types
        - The `estimate` attribute is the correct type and length
        - The `information` attribute is the right shape
        - Any other checks

        Returns
        -------
        bool
            Whether the edge is valid

        """

    @abstractmethod
    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        Returns
        -------
        np.ndarray, float
            The error for the edge

        """

    def calc_chi2(self):
        r"""Calculate the :math:`\chi^2` error for the edge.

        .. math::

           \mathbf{e}_j^T \Omega_j \mathbf{e}_j


        Returns
        -------
        float
            The :math:`\chi^2` error for the edge

        """
        err = self.calc_error()

        return np.dot(np.dot(np.transpose(err), self.information), err)

    def calc_chi2_gradient_hessian(self):
        r"""Calculate the edge's contributions to the graph's :math:`\chi^2` error, gradient (:math:`\mathbf{b}`), and Hessian (:math:`H`).

        Returns
        -------
        float
            The :math:`\chi^2` error for the edge
        list[tuple[int, np.ndarray]]
            The edge's contribution(s) to the gradient
        list[tuple[tuple[int, int], np.ndarray]]
            The edge's contribution(s) to the Hessian

        """
        chi2 = self.calc_chi2()

        err = self.calc_error()

        jacobians = self.calc_jacobians()

        # fmt: off
        return (
            chi2,
            [(v.gradient_index, np.dot(np.dot(np.transpose(err), self.information), jacobian)) for v, jacobian in zip(self.vertices, jacobians)],
            [((self.vertices[i].gradient_index, self.vertices[j].gradient_index), np.dot(np.dot(np.transpose(jacobians[i]), self.information), jacobians[j])) for i in range(len(jacobians)) for j in range(i, len(jacobians))],
        )
        # fmt: on

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        err = self.calc_error()

        return [self._calc_jacobian(err, v.pose.COMPACT_DIMENSIONALITY, i) for i, v in enumerate(self.vertices)]

    def _calc_jacobian(self, err, dim, vertex_index):
        r"""Calculate the Jacobian of the edge with respect to the specified vertex's pose.

        Parameters
        ----------
        err : np.ndarray
            The current error for the edge (see :math:`BaseEdge.calc_error`)
        dim : int
            The dimensionality of the compact pose representation
        vertex_index : int
            The index of the vertex (pose) for which we are computing the Jacobian

        Returns
        -------
        np.ndarray
            The Jacobian of the edge with respect to the specified vertex's pose

        """
        jacobian = np.zeros(err.shape + (dim,))
        p0 = self.vertices[vertex_index].pose.copy()

        for d in range(dim):
            # update the pose
            delta_pose = np.zeros(dim)
            delta_pose[d] = self._NUMERICAL_DIFFERENTIATION_EPSILON
            self.vertices[vertex_index].pose += delta_pose

            # compute the numerical derivative
            jacobian[:, d] = (self.calc_error() - err) / self._NUMERICAL_DIFFERENTIATION_EPSILON

            # restore the pose
            self.vertices[vertex_index].pose = p0.copy()

        return jacobian

    def to_g2o(self):
        """Export the edge to the .g2o format.

        .. note:: Overload this method to support writing to .g2o files.

        Returns
        -------
        str, None
            The edge in .g2o format, or ``None`` if writing to g2o format is not supported

        """
        return None

    @classmethod
    def from_g2o(cls, line, g2o_params_or_none=None):  # pylint: disable=unused-argument
        """Load an edge from a line in a .g2o file.

        .. note:: Overload this method to support loading from .g2o files.

        Parameters
        ----------
        line : str
            The line from the .g2o file
        g2o_params_or_none : dict, None
            A dictionary where the values are `graphslam.g2o_parameters.BaseG2OParameters` objects, or
            ``None`` if there are no such parameters

        Returns
        -------
        BaseEdge, None
            The instantiated edge object, or ``None`` if ``line`` does not correspond to this edge type
            (or if this edge type does not support loading from g2o)

        """
        return None

    def plot(self, color=""):
        """Plot the edge.

        .. note:: Overload this method to support plotting the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """

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
        if not type(self) is type(other):
            return False

        if len(self.vertex_ids) != len(other.vertex_ids):
            return False

        if any(v_id1 != v_id2 for v_id1, v_id2 in zip(self.vertex_ids, other.vertex_ids)):
            return False

        # fmt: off
        if self.information.shape != other.information.shape or np.linalg.norm(self.information - other.information) / max(np.linalg.norm(self.information), tol) >= tol:
            return False
        # fmt: on

        if isinstance(self.estimate, BasePose):
            return isinstance(other.estimate, BasePose) and self.estimate.equals(other.estimate, tol)

        # fmt: off
        return not isinstance(other.estimate, BasePose) and np.linalg.norm(self.estimate - other.estimate) / max(np.linalg.norm(self.estimate), tol) < tol
        # fmt: on
