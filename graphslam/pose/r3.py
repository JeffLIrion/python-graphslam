# Copyright (c) 2020 Jeff Irion and contributors

r"""Representation of a point in :math:`\mathbb{R}^3`.

"""

import numpy as np

from .base_pose import BasePose


class PoseR3(BasePose):
    r"""A representation of a 3-D point.

    Parameters
    ----------
    position : np.ndarray, list
        The position in :math:`\mathbb{R}^3`

    """

    #: The compact dimensionality
    COMPACT_DIMENSIONALITY = 3

    def __new__(cls, position):
        obj = np.asarray(position, dtype=np.float64).view(cls)
        return obj

    @classmethod
    def identity(cls):
        """Return the identity pose.

        Returns
        -------
        PoseR3
            The identity pose

        """
        return PoseR3([0.0, 0.0, 0.0])

    def copy(self):
        """Return a copy of the pose.

        Returns
        -------
        PoseR3
            A copy of the pose

        """
        return PoseR3([self[0], self[1], self[2]])

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
        return np.array(self)

    @property
    def orientation(self):
        """Return the pose's orientation.

        Returns
        -------
        float
            A ``PoseR3`` object has no orientation, so this will always return 0.

        """
        return 0.0

    @property
    def inverse(self):
        """Return the pose's inverse.

        Returns
        -------
        PoseR3
            The pose's inverse

        """
        return PoseR3([-self[0], -self[1], -self[2]])

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        """Add poses (i.e., pose composition).

        Parameters
        ----------
        other : PoseR3
            The other pose

        Returns
        -------
        PoseR3
            The result of pose composition

        """
        return PoseR3(np.add(self, other))

    def __sub__(self, other):
        """Subtract poses (i.e., inverse pose composition).

        Parameters
        ----------
        other : PoseR3
            The other pose

        Returns
        -------
        PoseR3
            The result of inverse pose composition

        """
        return PoseR3(np.subtract(self, other))

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
        return np.eye(3)

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
        return np.eye(3)

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
        return np.eye(3)

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
        return np.eye(3)

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
        return np.eye(3)

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
        return np.eye(3)

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
        return -np.eye(3)

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
        return -np.eye(3)

    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}` (shape: ``3 x 3``)

        """
        return np.eye(3)

    def jacobian_self_oplus_point_wrt_self(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`, where `:math:p_2` is a point.

        Parameters
        ----------
        point : PoseR3
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1` (shape: ``3 x 3``)

        """
        return np.eye(3)

    def jacobian_self_oplus_point_wrt_point(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`, where `:math:p_2` is a point.

        Parameters
        ----------
        point : PoseR3
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2` (shape: ``3 x 3``)

        """
        return np.eye(3)

    def jacobian_inverse(self):
        r"""Compute the Jacobian of :math:`p^{-1}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p^{-1}` (shape: ``3 x 3``)

        """
        return -np.eye(3)
