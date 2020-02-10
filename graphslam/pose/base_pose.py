r"""A base class for poses.

"""

import numpy as np


class BasePose(np.ndarray):
    """A base class for poses.

    """

    def copy(self):
        """Return a copy of the pose.

        Returns
        -------
        BasePose
            A copy of the pose

        """
        raise NotImplementedError

    def to_array(self):
        """Return the pose as a numpy array.

        Returns
        -------
        np.ndarray
            The pose as a numpy array

        """
        raise NotImplementedError

    def to_compact(self):
        """Return the pose as a compact numpy array.

        Returns
        -------
        np.ndarray
            The pose as a compact numpy array

        """
        raise NotImplementedError

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
            The pose's position

        """
        raise NotImplementedError

    @property
    def orientation(self):
        """Return the pose's orientation.

        Returns
        -------
        float, np.ndarray
            The pose's orientation

        """
        raise NotImplementedError

    @property
    def inverse(self):
        """Return the pose's inverse.

        Returns
        -------
        BasePose
            The pose's inverse

        """
        raise NotImplementedError

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def __add__(self, other):
        """Add poses (i.e., pose composition).

        Parameters
        ----------
        other : BasePose
            The other pose

        Returns
        -------
        BasePose
            The result of pose composition

        """
        raise NotImplementedError

    def __sub__(self, other):
        """Subtract poses (i.e., inverse pose composition).

        Parameters
        ----------
        other : BasePose
            The other pose

        Returns
        -------
        BasePose
            The result of inverse pose composition

        """
        raise NotImplementedError
