r"""A base class for poses.

"""

from abc import ABC, abstractmethod

import numpy as np


class BasePose(ABC, np.ndarray):
    """A base class for poses.

    """

    @abstractmethod
    def to_array(self):
        """Return the pose as a numpy array.

        Returns
        -------
        np.ndarray
            The pose as a numpy array

        """

    @abstractmethod
    def to_compact(self):
        """Return the pose as a compact numpy array.

        Returns
        -------
        np.ndarray
            The pose as a compact numpy array

        """

    @property
    @abstractmethod
    def position(self):
        """Return the pose's position.

        Returns
        -------
        np.ndarray
            The pose's position

        """

    @property
    @abstractmethod
    def orientation(self):
        """Return the pose's orientation.

        Returns
        -------
        float, np.ndarray
            The pose's orientation

        """

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    @abstractmethod
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

    @abstractmethod
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
