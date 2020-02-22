# Copyright (c) 2020 Jeff Irion and contributors

r"""Representation of a pose in :math:`SE(3)`.

"""

import numpy as np

from .base_pose import BasePose


class PoseSE3(BasePose):
    r"""A representation of a pose in :math:`SE(3)`.

    Parameters
    ----------
    position : np.ndarray, list
        The position in :math:`\mathbb{R}^3`
    orientation : np.ndarray, list
        The orientation of the pose as a unit quaternion: :math:`[q_x, q_y, q_z, q_w]`

    """
    def __new__(cls, position, orientation):
        obj = np.array([position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]], dtype=np.float64).view(cls)
        return obj

    def normalize(self):
        """Normalize the quaternion portion of the pose.

        """
        sgn = 1. if self[6] >= 0. else -1.
        self[3:] /= (sgn * np.linalg.norm(self[3:]))

    def copy(self):
        """Return a copy of the pose.

        Returns
        -------
        PoseSE3
            A copy of the pose

        """
        return PoseSE3(self[:3], self[3:])

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
        return np.array(self[:6])

    def to_matrix(self):
        """Return the pose as an :math:`SE(3)` matrix.

        Returns
        -------
        np.ndarray
            The pose as an :math:`SE(3)` matrix

        """
        return np.array([[self[6]**2 + self[3]**2 - self[4]**2 - self[5]**2, 2. * (self[3] * self[4] - self[5] * self[6]), 2. * (self[3] * self[5] + self[4] * self[6]), self[0]],
                         [2. * (self[3] * self[4] + self[6] * self[5]), self[6]**2 - self[3]**2 + self[4]**2 - self[5]**2, 2. * (self[4] * self[5] - self[3] * self[6]), self[1]],
                         [2. * (self[3] * self[5] - self[4] * self[6]), 2. * (self[3] * self[6] + self[4] * self[5]), self[6]**2 - self[3]**2 - self[4]**2 + self[5]**2, self[2]],
                         [0., 0., 0., 1.]], dtype=np.float64)

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
        return np.array(self[:3])

    @property
    def orientation(self):
        """Return the pose's orientation.

        Returns
        -------
        float
            The pose's quaternion

        """
        return np.array(self[3:])

    @property
    def inverse(self):
        """Return the pose's inverse.

        Returns
        -------
        PoseSE3
            The pose's inverse

        """
        return PoseSE3([-self[0] + 2. * ((self[4]**2 + self[5]**2) * self[0] - (self[3] * self[4] + self[5] * self[6]) * self[1] + (self[4] * self[6] - self[3] * self[5]) * self[2]),
                        -self[1] + 2. * ((self[5] * self[6] - self[3] * self[4]) * self[0] + (self[3]**2 + self[5]**2) * self[1] - (self[3] * self[6] + self[4] * self[5]) * self[2]),
                        -self[2] + 2. * (-(self[3] * self[5] + self[4] * self[6]) * self[0] + (self[3] * self[6] - self[4] * self[5]) * self[1] + (self[3]**2 + self[4]**2) * self[2])],
                       [-self[3], -self[4], -self[5], self[6]])

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        r"""Add poses (i.e., pose composition): :math:`p_1 \oplus p_2`.

        Parameters
        ----------
        other : PoseSE3
            The other pose

        Returns
        -------
        PoseSE3
            The result of pose composition

        """
        if isinstance(other, PoseSE3):
            # oplus: self (+) other
            return PoseSE3([self[0] + other[0] + 2. * (-(self[4]**2 + self[5]**2) * other[0] + (self[3] * self[4] - self[5] * self[6]) * other[1] + (self[3] * self[5] + self[4] * self[6]) * other[2]),
                            self[1] + other[1] + 2. * ((self[3] * self[4] + self[5] * self[6]) * other[0] - (self[3]**2 + self[5]**2) * other[1] + (self[4] * self[5] - self[3] * self[6]) * other[2]),
                            self[2] + other[2] + 2. * ((self[3] * self[5] - self[4] * self[6]) * other[0] + (self[3] * self[6] + self[4] * self[5]) * other[1] - (self[3]**2 + self[4]**2) * other[2])],
                           [self[6] * other[3] + self[3] * other[6] + self[4] * other[5] - self[5] * other[4],
                            self[6] * other[4] - self[3] * other[5] + self[4] * other[6] + self[5] * other[3],
                            self[6] * other[5] + self[3] * other[4] - self[4] * other[3] + self[5] * other[6],
                            self[6] * other[6] - self[3] * other[3] - self[4] * other[4] - self[5] * other[5]])

        if isinstance(other, np.ndarray):
            if len(other) == 6:
                # boxplus: self [+] other
                qnorm = np.linalg.norm(other[3:])
                if qnorm > 1.0:
                    qw, qx, qy, qz = 1., 0., 0., 0.
                else:
                    qw = np.sqrt(1. - qnorm**2)
                    qx, qy, qz = other[3:]

                return PoseSE3([self[0] + other[0] + 2. * (-(self[4]**2 + self[5]**2) * other[0] + (self[3] * self[4] - self[5] * self[6]) * other[1] + (self[3] * self[5] + self[4] * self[6]) * other[2]),
                                self[1] + other[1] + 2. * ((self[3] * self[4] + self[5] * self[6]) * other[0] - (self[3]**2 + self[5]**2) * other[1] + (self[4] * self[5] - self[3] * self[6]) * other[2]),
                                self[2] + other[2] + 2. * ((self[3] * self[5] - self[4] * self[6]) * other[0] + (self[3] * self[6] + self[4] * self[5]) * other[1] - (self[3]**2 + self[4]**2) * other[2])],
                               [self[6] * qx + self[3] * qw + self[4] * qz - self[5] * qy,
                                self[6] * qy - self[3] * qz + self[4] * qw + self[5] * qx,
                                self[6] * qz + self[3] * qy - self[4] * qx + self[5] * qw,
                                self[6] * qw - self[3] * qx - self[4] * qy - self[5] * qz])

        raise NotImplementedError

    def __sub__(self, other):
        r"""Subtract poses (i.e., inverse pose composition): :math:`p_1 \ominus p_2`.

        Parameters
        ----------
        other : PoseSE3
            The other pose

        Returns
        -------
        PoseSE3
            The result of inverse pose composition

        """
        return PoseSE3([self[0] - other[0] + 2. * (-(other[4]**2 + other[5]**2) * (self[0] - other[0]) + (other[3] * other[4] + other[5] * other[6]) * (self[1] - other[1]) + (other[3] * other[5] - other[4] * other[6]) * (self[2] - other[2])),
                        self[1] - other[1] + 2. * ((other[3] * other[4] - other[5] * other[6]) * (self[0] - other[0]) - (other[3]**2 + other[5]**2) * (self[1] - other[1]) + (other[4] * other[5] + other[3] * other[6]) * (self[2] - other[2])),
                        self[2] - other[2] + 2. * ((other[3] * other[5] + other[4] * other[6]) * (self[0] - other[0]) + (other[4] * other[5] - other[3] * other[6]) * (self[1] - other[1]) - (other[3]**2 + other[4]**2) * (self[2] - other[2]))],
                       [other[6] * self[3] - other[3] * self[6] - other[4] * self[5] + other[5] * self[4],
                        other[6] * self[4] + other[3] * self[5] - other[4] * self[6] - other[5] * self[3],
                        other[6] * self[5] - other[3] * self[4] + other[4] * self[3] - other[5] * self[6],
                        other[6] * self[6] + other[3] * self[3] + other[4] * self[4] + other[5] * self[5]])

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
        return np.array([[1., 0., 0., 2. * self[4] * other[1] + 2. * self[5] * other[2], -4. * self[4] * other[0] + 2. * self[3] * other[1] + 2. * self[6] * other[2], -4. * self[5] * other[0] - 2. * self[6] * other[1] + 2. * self[3] * other[2], -2. * self[5] * other[1] + 2. * self[4] * other[2]],
                         [0., 1., 0., 2. * self[4] * other[0] - 4. * self[3] * other[1] - 2. * self[6] * other[2], 2. * self[3] * other[0] + 2. * self[5] * other[2], -2. * self[6] * other[0] - 4. * self[5] * other[1] + self[4] * other[2], 2. * self[5] * other[0] - 2. * self[3] * other[2]],
                         [0., 0., 1., 2. * self[5] * other[0] + 2. * self[6] * other[1] - 4. * self[3] * other[2], -2. * self[6] * other[0] + 2. * self[5] * other[1] - 4. * self[4] * other[2], 2. * self[3] * other[0] + 2. * self[4] * other[1], -2. * self[4] * other[0] + 2. * self[3] * other[1]],
                         [0., 0., 0., other[6], other[5], -other[4], other[3]],
                         [0., 0., 0., -other[5], other[6], other[3], other[4]],
                         [0., 0., 0., other[4], -other[3], other[6], other[5]],
                         [0., 0., 0., -other[3], -other[4], -other[5], other[6]]], dtype=np.float64)

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
        return np.array([[1., 0., 0., 2. * self[4] * other[1] + 2. * self[5] * other[2], -4. * self[4] * other[0] + 2. * self[3] * other[1] + 2. * self[6] * other[2], -4. * self[5] * other[0] - 2. * self[6] * other[1] + 2. * self[3] * other[2], -2. * self[5] * other[1] + 2. * self[4] * other[2]],
                         [0., 1., 0., 2. * self[4] * other[0] - 4. * self[3] * other[1] - 2. * self[6] * other[2], 2. * self[3] * other[0] + 2. * self[5] * other[2], -2. * self[6] * other[0] - 4. * self[5] * other[1] + self[4] * other[2], 2. * self[5] * other[0] - 2. * self[3] * other[2]],
                         [0., 0., 1., 2. * self[5] * other[0] + 2. * self[6] * other[1] - 4. * self[3] * other[2], -2. * self[6] * other[0] + 2. * self[5] * other[1] - 4. * self[4] * other[2], 2. * self[3] * other[0] + 2. * self[4] * other[1], -2. * self[4] * other[0] + 2. * self[3] * other[1]],
                         [0., 0., 0., other[6], other[5], -other[4], other[3]],
                         [0., 0., 0., -other[5], other[6], other[3], other[4]],
                         [0., 0., 0., other[4], -other[3], other[6], other[5]]], dtype=np.float64)

    def jacobian_self_oplus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            (Unused) The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[1. - 2. * (self[4]**2 + self[5]**2), 2. * (self[3] * self[4] - self[5] * self[6]), 2. * (self[3] * self[5] + self[4] * self[6]), 0., 0., 0., 0.],
                         [2. * (self[3] * self[4] + self[5] * self[6]), 1. - 2. * (self[3]**2 + self[5]**2), 2. * (self[4] * self[5] - self[3] * self[6]), 0., 0., 0., 0.],
                         [2. * (self[3] * self[5] - self[4] * self[6]), 2. * (self[3] * self[6] + self[4] * self[5]), 1. - 2. * (self[3]**2 + self[4]**2), 0., 0., 0., 0.],
                         [0., 0., 0., self[6], -self[5], self[4], self[3]],
                         [0., 0., 0., self[5], self[6], -self[3], self[4]],
                         [0., 0., 0., -self[4], self[3], self[6], self[5]],
                         [0., 0., 0., -self[3], -self[4], -self[5], self[6]]], dtype=np.float64)

    def jacobian_self_oplus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Parameters
        ----------
        other : BasePose
            (Unused) The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        return np.array([[1. - 2. * (self[4]**2 + self[5]**2), 2. * (self[3] * self[4] - self[5] * self[6]), 2. * (self[3] * self[5] + self[4] * self[6]), 0., 0., 0., 0.],
                         [2. * (self[3] * self[4] + self[5] * self[6]), 1. - 2. * (self[3]**2 + self[5]**2), 2. * (self[4] * self[5] - self[3] * self[6]), 0., 0., 0., 0.],
                         [2. * (self[3] * self[5] - self[4] * self[6]), 2. * (self[3] * self[6] + self[4] * self[5]), 1. - 2. * (self[3]**2 + self[4]**2), 0., 0., 0., 0.],
                         [0., 0., 0., self[6], -self[5], self[4], self[3]],
                         [0., 0., 0., self[5], self[6], -self[3], self[4]],
                         [0., 0., 0., -self[4], self[3], self[6], self[5]]], dtype=np.float64)

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
        return np.array([[1. - 2. * (other[4]**2 + other[5]**2), 2. * (other[3] * other[4] + other[5] * other[6]), 2. * (other[3] * other[5] - other[4] * other[6]), 0., 0., 0., 0.],
                         [2. * (other[3] * other[4] - other[5] * other[6]), 1. - 2. * (other[3]**2 + other[5]**2), 2. * (other[4] * other[5] + other[3] * other[6]), 0., 0., 0., 0.],
                         [2. * (other[3] * other[5] + other[4] * other[6]), 2. * (other[4] * other[5] - other[3] * other[6]), 1. - 2. * (other[3]**2 + other[4]**2), 0., 0., 0., 0.],
                         [0., 0., 0., other[6], other[5], -other[4], -other[3]],
                         [0., 0., 0., -other[5], other[6], other[3], -other[4]],
                         [0., 0., 0., other[4], -other[3], other[6], -other[5]],
                         [0., 0., 0., other[3], other[4], other[5], other[6]]], dtype=np.float64)

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
        return np.array([[1. - 2. * (other[4]**2 + other[5]**2), 2. * (other[3] * other[4] + other[5] * other[6]), 2. * (other[3] * other[5] - other[4] * other[6]), 0., 0., 0., 0.],
                         [2. * (other[3] * other[4] - other[5] * other[6]), 1. - 2. * (other[3]**2 + other[5]**2), 2. * (other[4] * other[5] + other[3] * other[6]), 0., 0., 0., 0.],
                         [2. * (other[3] * other[5] + other[4] * other[6]), 2. * (other[4] * other[5] - other[3] * other[6]), 1. - 2. * (other[3]**2 + other[4]**2), 0., 0., 0., 0.],
                         [0., 0., 0., other[6], other[5], -other[4], -other[3]],
                         [0., 0., 0., -other[5], other[6], other[3], -other[4]],
                         [0., 0., 0., other[4], -other[3], other[6], -other[5]]], dtype=np.float64)

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
        return np.array([[-1. + 2. * (other[4]**2 + other[5]**2), -2. * (other[3] * other[4] + other[5] * other[6]), -2. * (other[3] * other[5] - other[4] * other[6]), 2. * (other[4] * (self[1] - other[1]) + other[5] * (self[2] - other[2])), 2. * (-2. * other[4] * (self[0] - other[0]) + other[3] * (self[1] - other[1]) - other[6] * (self[2] - other[2])), 2. * (-2. * other[5] * (self[0] - other[0]) + other[6] * (self[1] - other[1]) + other[3] * (self[2] - other[2])), 2. * (other[5] * (self[1] - other[1]) - other[4] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[4] - other[5] * other[6]), -1. + 2. * (other[3]**2 + other[5]**2), -2. * (other[4] * other[5] + other[3] * other[6]), 2. * (other[4] * (self[0] - other[0]) - 2. * other[3] * (self[1] - other[1]) + other[6] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[5] * (self[2] - other[2])), 2. * (-other[6] * (self[0] - other[0]) - 2. * other[5] * (self[1] - other[1]) + other[4] * (self[2] - other[2])), 2. * (-other[5] * (self[0] - other[0]) + other[3] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[5] + other[4] * other[6]), -2. * (other[4] * other[5] - other[3] * other[6]), -1. + 2. * (other[3]**2 + other[4]**2), 2. * (other[5] * (self[0] - other[0]) - other[6] * (self[1] - other[1]) - 2. * other[3] * (self[2] - other[2])), 2. * (other[6] * (self[0] - other[0]) + other[5] * (self[1] - other[1]) - 2. * other[4] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[4] * (self[1] - other[1])), 2. * (other[4] * (self[0] - other[0]) - other[3] * (self[1] - other[1]))],
                         [0., 0., 0., -self[6], -self[5], self[4], self[3]],
                         [0., 0., 0., self[5], -self[6], -self[3], self[4]],
                         [0., 0., 0., -self[4], self[3], -self[6], self[5]],
                         [0., 0., 0., self[3], self[4], self[5], self[6]]], dtype=np.float64)

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
        return np.array([[-1. + 2. * (other[4]**2 + other[5]**2), -2. * (other[3] * other[4] + other[5] * other[6]), -2. * (other[3] * other[5] - other[4] * other[6]), 2. * (other[4] * (self[1] - other[1]) + other[5] * (self[2] - other[2])), 2. * (-2. * other[4] * (self[0] - other[0]) + other[3] * (self[1] - other[1]) - other[6] * (self[2] - other[2])), 2. * (-2. * other[5] * (self[0] - other[0]) + other[6] * (self[1] - other[1]) + other[3] * (self[2] - other[2])), 2. * (other[5] * (self[1] - other[1]) - other[4] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[4] - other[5] * other[6]), -1. + 2. * (other[3]**2 + other[5]**2), -2. * (other[4] * other[5] + other[3] * other[6]), 2. * (other[4] * (self[0] - other[0]) - 2. * other[3] * (self[1] - other[1]) + other[6] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[5] * (self[2] - other[2])), 2. * (-other[6] * (self[0] - other[0]) - 2. * other[5] * (self[1] - other[1]) + other[4] * (self[2] - other[2])), 2. * (-other[5] * (self[0] - other[0]) + other[3] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[5] + other[4] * other[6]), -2. * (other[4] * other[5] - other[3] * other[6]), -1. + 2. * (other[3]**2 + other[4]**2), 2. * (other[5] * (self[0] - other[0]) - other[6] * (self[1] - other[1]) - 2. * other[3] * (self[2] - other[2])), 2. * (other[6] * (self[0] - other[0]) + other[5] * (self[1] - other[1]) - 2. * other[4] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[4] * (self[1] - other[1])), 2. * (other[4] * (self[0] - other[0]) - other[3] * (self[1] - other[1]))],
                         [0., 0., 0., -self[6], -self[5], self[4], self[3]],
                         [0., 0., 0., self[5], -self[6], -self[3], self[4]],
                         [0., 0., 0., -self[4], self[3], -self[6], self[5]]], dtype=np.float64)

    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`

        """
        return np.array([[1. - 2. * (self[4]**2 + self[5]**2), 2. * (self[3] * self[4] - self[5] * self[6]), 2. * (self[3] * self[5] + self[4] * self[6]), 0., 0., 0.],
                         [2. * (self[3] * self[4] + self[5] * self[6]), 1. - 2. * (self[3]**2 + self[5]**2), 2. * (self[4] * self[5] - self[3] * self[6]), 0., 0., 0.],
                         [2. * (self[3] * self[5] - self[4] * self[6]), 2. * (self[3] * self[6] + self[4] * self[5]), 1. - 2. * (self[3]**2 + self[4]**2), 0., 0., 0.],
                         [0., 0., 0., self[6], -self[5], self[4]],
                         [0., 0., 0., self[5], self[6], -self[3]],
                         [0., 0., 0., -self[4], self[3], self[6]],
                         [0., 0., 0., -self[3], -self[4], -self[5]]], dtype=np.float64)
