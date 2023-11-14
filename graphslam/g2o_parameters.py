# Copyright (c) 2020 Jeff Irion and contributors

"""Classes for storing parameters from .g2o files.

"""


from abc import ABC, abstractmethod

# from graphslam.pose.se2 import PoseSE2
# from graphslam.pose.se3 import PoseSE3


class BaseG2OParameter(ABC):
    """A base class for representing parameters from .g2o files.

    Parameters
    ----------
    key
        A key that will be used to lookup this parameter in a dictionary
    value
        This parameter's value

    Attributes
    ----------
    key
        A key that will be used to lookup this parameter in a dictionary
    value
        This parameter's value

    """

    def __init__(self, key, value):
        self.key = key
        self.value = value

    @abstractmethod
    def to_g2o(self):
        """Export the parameter to the .g2o format.

        Returns
        -------
        str, None
            The parameter in .g2o format

        """

    @classmethod
    @abstractmethod
    def from_g2o(cls, line):
        """Load a parameter from a line in a .g2o file.

        Parameters
        ----------
        line : str
            The line from the .g2o file

        Returns
        -------
        BaseParameter, None
            The instantiated parameter object, or ``None`` if ``line`` does not correspond to this parameter type

        """
