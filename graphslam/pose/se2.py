# Copyright (c) 2020 Jeff Irion and contributors

r"""Representation of a pose in :math:`SE(2)`.

"""

import math

import numpy as np

from .base_pose import BasePose
from .r2 import PoseR2

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

    #: The compact dimensionality
    COMPACT_DIMENSIONALITY = 3

    def __new__(cls, position, orientation):
        obj = np.array([position[0], position[1], neg_pi_to_pi(orientation)], dtype=np.float64).view(cls)
        return obj

    @classmethod
    def identity(cls):
        """Return the identity pose.

        Returns
        -------
        PoseSE2
            The identity pose

        """
        return PoseSE2([0.0, 0.0], 0.0)

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
        # fmt: off
        return np.array([[np.cos(self[2]), -np.sin(self[2]), self[0]],
                         [np.sin(self[2]), np.cos(self[2]), self[1]],
                         [0., 0., 1.]],
                        dtype=np.float64)
        # fmt: on

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
        # fmt: off
        return PoseSE2([-self[0] * np.cos(self[2]) - self[1] * np.sin(self[2]),
                        self[0] * np.sin(self[2]) - self[1] * np.cos(self[2])],
                       -self[2])
        # fmt: on

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
        PoseSE2, PoseR2
            The result of pose composition

        """
        if isinstance(other, PoseSE2) or (isinstance(other, np.ndarray) and len(other) == 3):
            # fmt: off
            return PoseSE2([self[0] + other[0] * np.cos(self[2]) - other[1] * np.sin(self[2]),
                            self[1] + other[0] * np.sin(self[2]) + other[1] * np.cos(self[2])],
                           self[2] + other[2])
            # fmt: on

        if isinstance(other, PoseR2) or (isinstance(other, np.ndarray) and len(other) == 2):
            # pose (+) point
            # fmt: off
            return PoseR2([self[0] + other[0] * np.cos(self[2]) - other[1] * np.sin(self[2]),
                           self[1] + other[0] * np.sin(self[2]) + other[1] * np.cos(self[2])])
            # fmt: on

        raise NotImplementedError

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
        # fmt: off
        return PoseSE2([(self[0] - other[0]) * np.cos(other[2]) + (self[1] - other[1]) * np.sin(other[2]),
                        (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])],
                       self[2] - other[2])
        # fmt: on

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
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[1., 0., -other[0] * np.sin(self[2]) - other[1] * np.cos(self[2])],
                         [0., 1., other[0] * np.cos(self[2]) - other[1] * np.sin(self[2])],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_oplus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[1., 0., -other[0] * np.sin(self[2]) - other[1] * np.cos(self[2])],
                         [0., 1., other[0] * np.cos(self[2]) - other[1] * np.sin(self[2])],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_oplus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_oplus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_ominus_other_wrt_self(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[np.cos(other[2]), np.sin(other[2]), 0.],
                         [-np.sin(other[2]), np.cos(other[2]), 0.],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_ominus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[np.cos(other[2]), np.sin(other[2]), 0.],
                         [-np.sin(other[2]), np.cos(other[2]), 0.],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_ominus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[-np.cos(other[2]), -np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])],
                         [np.sin(other[2]), -np.cos(other[2]), (other[0] - self[0]) * np.cos(other[2]) + (other[1] - self[1]) * np.sin(other[2])],
                         [0., 0., -1.]])
        # fmt: on

    def jacobian_self_ominus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[-np.cos(other[2]), -np.sin(other[2]), (other[0] - self[0]) * np.sin(other[2]) + (self[1] - other[1]) * np.cos(other[2])],
                         [np.sin(other[2]), -np.cos(other[2]), (other[0] - self[0]) * np.cos(other[2]) + (other[1] - self[1]) * np.sin(other[2])],
                         [0., 0., -1.]])
        # fmt: on

    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[np.cos(self[2]), -np.sin(self[2]), 0.],
                         [np.sin(self[2]), np.cos(self[2]), 0.],
                         [0., 0., 1.]])
        # fmt: on

    def jacobian_self_oplus_point_wrt_self(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`, where `:math:p_2` is a point.

        Parameters
        ----------
        point : PoseR2
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1` (shape: ``2 x 3``)

        """
        # fmt: off
        return np.array([[1., 0., -point[0] * np.sin(self[2]) - point[1] * np.cos(self[2])],
                         [0., 1., point[0] * np.cos(self[2]) - point[1] * np.sin(self[2])]])
        # fmt: on

    def jacobian_self_oplus_point_wrt_point(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`, where `:math:p_2` is a point.

        Parameters
        ----------
        point : PoseR2
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2` (shape: ``2 x 2``)

        """
        # fmt: off
        return np.array([[np.cos(self[2]), -np.sin(self[2])],
                         [np.sin(self[2]), np.cos(self[2])]])
        # fmt: on

    def jacobian_inverse(self):
        r"""Compute the Jacobian of :math:`p^{-1}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p^{-1}` (shape: ``3 x 3``)

        """
        # fmt: off
        return np.array([[-np.cos(self[2]), -np.sin(self[2]), self[0] * np.sin(self[2]) - self[1] * np.cos(self[2])],
                         [np.sin(self[2]), -np.cos(self[2]), self[0] * np.cos(self[2]) + self[1] * np.sin(self[2])],
                         [0., 0., -1.]])
        # fmt: on
