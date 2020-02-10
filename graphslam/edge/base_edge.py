r"""A base class for edges.

"""


import numpy as np


class BaseEdge:
    r"""A class for representing edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge

    """
    def __init__(self, vertices, information):
        self.vertices = vertices
        self.information = information

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
        e_j = self.calc_error()

        return np.dot(np.dot(np.transpose(e_j), self.information), e_j)
