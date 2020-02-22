# Copyright (c) 2020 Jeff Irion and contributors

r"""Representation of a pose in :math:`SE(2)`.

"""

import math

import numpy as np

from .base_pose import BasePose

from ..util import neg_pi_to_pi


class PoseSE2(BasePose):
    r"""A representation of a pose in :math:`SE(2)`.

    Parameters
    ----------
    position : np.ndarray, list
        The position in :math:`\mathbb{R}^2`
    orientation : float
        The angle of the pose (in radians)

    """
    def __new__(cls, position, orientation):
        obj = np.array([position[0], position[1], neg_pi_to_pi(orientation)], dtype=np.float64).view(cls)
        return obj

    def copy(self):
        """Return a copy of the pose.

        Returns
        -------
        PoseSE2
            A copy of the pose

        """
        return PoseSE2(self[:2], self[2])

    def to_array(self):
        """Return the pose as a numpy array.

        Returns
        -------
        np.ndarray
            The pose as a numpy array

        """
        return np.array(self)

    def to_compact(self):
        """Return the pose as a compact numpy array.

        Returns
        -------
        np.ndarray
            The pose as a compact numpy array

        """
        return np.array(self)

    def to_matrix(self):
        """Return the pose as an :math:`SE(2)` matrix.

        Returns
        -------
        np.ndarray
            The pose as an :math:`SE(2)` matrix

        """
        return np.array([[np.cos(self[2]), -np.sin(self[2]), self[0]], [np.sin(self[2]), np.cos(self[2]), self[1]], [0., 0., 1.]], dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix):
        """Return the pose as an :math:`SE(2)` matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The :math:`SE(2)` matrix that will be converted to a `PoseSE2` instance

        Returns
        -------
        PoseSE2
            The matrix as a `PoseSE2` object

        """
        return cls([matrix[0, 2], matrix[1, 2]], math.atan2(matrix[1, 0], matrix[0, 0]))

    # ======================================================================= #
    #                                                                         #
    #                                Properties                               #
    #                                                                         #
    # ======================================================================= #
    @property
    def position(self):
        """Return the pose's position.

        Returns
        -------
        np.ndarray
            The position portion of the pose

        """
        return np.array(self[:2])

    @property
    def orientation(self):
        """Return the pose's orientation.

        Returns
        -------
        float
            The angle of the pose

        """
        return self[2]

    @property
    def inverse(self):
        """Return the pose's inverse.

        Returns
        -------
        PoseSE2
            The pose's inverse

        """
        return PoseSE2([-self[0] * np.cos(self[2]) - self[1] * np.sin(self[2]), self[0] * np.sin(self[2]) - self[1] * np.cos([self[2]])], -self[2])

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        r"""Add poses (i.e., pose composition): :math:`p_1 \oplus p_2`.

        Parameters
        ----------
        other : PoseSE2
            The other pose

        Returns
        -------
        PoseSE2
            The result of pose composition

        """
        return PoseSE2([self[0] + other[0] * np.cos(self[2]) - other[1] * np.sin(self[2]), self[1] + other[0] * np.sin(self[2]) + other[1] * np.cos(self[2])], neg_pi_to_pi(self[2] + other[2]))

    def __sub__(self, other):
        r"""Subtract poses (i.e., inverse pose composition): :math:`p_1 \ominus p_2`.

        Parameters
        ----------
        other : PoseSE2
            The other pose

        Returns
        -------
        PoseSE2
            The result of inverse pose composition

        """
        return PoseSE2([(self[0] - other[0]) * np.cos(other[2]) + (self[1] - other[1]) * np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])], neg_pi_to_pi(self[2] - other[2]))

    # ======================================================================= #
    #                                                                         #
    #                                Jacobians                                #
    #                                                                         #
    # ======================================================================= #
    def jacobian_self_oplus_other_wrt_self(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        """
        return np.array([[1., 0., -other[0] * np.sin(self[2]) - other[1] * np.cos(self[2])],
                         [0., 1., other[0] * np.cos(self[2]) - other[1] * np.sin(self[2])],
                         [0., 0., 1.]])

    def jacobian_self_oplus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        """
        return np.array([[1., 0., -other[0] * np.sin(self[2]) - other[1] * np.cos(self[2])],
                         [0., 1., other[0] * np.cos(self[2]) - other[1] * np.sin(self[2])],
                         [0., 0., 1.]])

    def jacobian_self_oplus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])

    def jacobian_self_oplus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])

    def jacobian_self_ominus_other_wrt_self(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        """
        return np.array([[np.cos(other[2]), np.sin(other[2]), 0.],
                         [-np.sin(other[2]), np.cos(other[2]), 0.],
                         [0., 0., 1.]])

    def jacobian_self_ominus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        """
        return np.array([[np.cos(other[2]), np.sin(other[2]), 0.],
                         [-np.sin(other[2]), np.cos(other[2]), 0.],
                         [0., 0., 1.]])

    def jacobian_self_ominus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[-np.cos(other[2]), -np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])],
                         [np.sin(other[2]), -np.cos(other[2]), (other[0] - self[0]) * np.cos(other[2]) + (other[1] - self[1]) * np.sin(other[2])],
                         [0., 0., -1.]])

    def jacobian_self_ominus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[-np.cos(other[2]), -np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])],
                         [np.sin(other[2]), -np.cos(other[2]), (other[0] - self[0]) * np.cos(other[2]) + (other[1] - self[1]) * np.sin(other[2])],
                         [0., 0., -1.]])

    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`

        """
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])
