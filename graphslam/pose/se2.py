r"""Representation of a pose in `SE(2)`.

"""

import numpy as np

from .base_pose import BasePose


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
        obj = np.array([position[0], position[1], orientation], dtype=np.float64).view(cls)
        return obj

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
            A ``PoseSE2`` object has no orientation, so this will always return 0.

        """
        return self[2]

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        """Add poses (i.e., pose composition).

        Parameters
        ----------
        other : PoseSE2
            The other pose

        Returns
        -------
        PoseSE2
            The result of pose composition

        """
        return PoseSE2([0, 0], 0)
        # return PoseSE2(np.add(self, other))

    def __sub__(self, other):
        """Subtract poses (i.e., inverse pose composition).

        Parameters
        ----------
        other : PoseSE2
            The other pose

        Returns
        -------
        PoseSE2
            The result of inverse pose composition

        """
        return PoseSE2([0, 0], 0)
        # return PoseSE2(np.subtract(self, other))
