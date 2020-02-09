r"""Representation of a point in `\mathbb{R}^2`.

"""

import numpy as np

from .base_pose import BasePose


class PoseR2(BasePose):
    r"""A representation of a 2-D point.

    Parameters
    ----------
    position : np.ndarray, list
        The position in :math:`\mathbb{R}^2`

    """
    def __new__(cls, position):
        obj = np.asarray(position, dtype=np.float64).view(cls)
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
            A ``PoseR2`` object has no orientation, so this will always return 0.

        """
        return 0.

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        """Add poses (i.e., pose composition).

        Parameters
        ----------
        other : PoseR2
            The other pose

        Returns
        -------
        PoseR2
            The result of pose composition

        """
        return PoseR2(np.add(self, other))

    def __sub__(self, other):
        """Subtract poses (i.e., inverse pose composition).

        Parameters
        ----------
        other : PoseR2
            The other pose

        Returns
        -------
        PoseR2
            The result of inverse pose composition

        """
        return PoseR2(np.subtract(self, other))
