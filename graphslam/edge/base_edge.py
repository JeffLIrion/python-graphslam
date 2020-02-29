# Copyright (c) 2020 Jeff Irion and contributors

r"""A base class for edges.

"""


import numpy as np


#: The difference that will be used for numerical differentiation
EPSILON = 1e-6


class BaseEdge:
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
    def __init__(self, vertex_ids, information, estimate, vertices=None):
        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        Returns
        -------
        np.ndarray, float
            The error for the edge

        """
        raise NotImplementedError

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
        dict
            The edge's contribution(s) to the gradient
        dict
            The edge's contribution(s) to the Hessian

        """
        chi2 = self.calc_chi2()

        err = self.calc_error()

        jacobians = self.calc_jacobians()

        return chi2, {v.index: np.dot(np.dot(np.transpose(err), self.information), jacobian) for v, jacobian in zip(self.vertices, jacobians)}, {(self.vertices[i].index, self.vertices[j].index): np.dot(np.dot(np.transpose(jacobians[i]), self.information), jacobians[j]) for i in range(len(jacobians)) for j in range(i, len(jacobians))}

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

        # The dimensionality of the compact pose representation
        dim = len(self.vertices[0].pose.to_compact())

        return [self._calc_jacobian(err, dim, i) for i in range(len(self.vertices))]

    def _calc_jacobian(self, err, dim, vertex_index):
        r"""Calculate the Jacobian of the edge with respect to the specified vertex's pose.

        Parameters
        ----------
        err : np.ndarray
            The current error for the edge (see :meth:`BaseEdge.calc_error`)
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
            delta_pose[d] = EPSILON
            self.vertices[vertex_index].pose += delta_pose

            # compute the numerical derivative
            jacobian[:, d] = (self.calc_error() - err) / EPSILON

            # restore the pose
            self.vertices[vertex_index].pose = p0.copy()

        return jacobian

    def to_g2o(self):
        """Export the edge to the .g2o format.

        Returns
        -------
        str
            The edge in .g2o format

        """
        raise NotImplementedError

    def plot(self, color=''):
        """Plot the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """
        raise NotImplementedError
