# Copyright (c) 2020 Jeff Irion and contributors

"""Classes for storing parameters from .g2o files.

"""


from abc import ABC, abstractmethod

from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


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
        BaseG2OParameter, None
            The instantiated parameter object, or ``None`` if ``line`` does not correspond to this parameter type

        """


class G2OParameterSE2Offset(BaseG2OParameter):
    """A class for storing a g2o :math:`SE(2)` offset parameter.

    Attributes
    ----------
    key : tuple[str, int]
        A tuple of the form ``("PARAMS_SE2OFFSET", id)``
    value : graphslam.pose.se2.PoseSE2
        The offset

    """

    def to_g2o(self):
        """Export the :math:`SE(2)` offset parameter to the .g2o format.

        Returns
        -------
        str
            The parameter in .g2o format

        """
        return "PARAMS_SE2OFFSET {} {} {} {}\n".format(self.key[1], self.value[0], self.value[1], self.value[2])

    @classmethod
    def from_g2o(cls, line):
        """Load an :math:`SE(2)` offset parameter from a line in a .g2o file.

        Parameters
        ----------
        line : str
            The line from the .g2o file

        Returns
        -------
        G2OParameterSE2Offset, None
            The instantiated parameter object, or ``None`` if ``line`` does not correspond to an :math:`SE(2)` offset parameter

        """
        if not line.startswith("PARAMS_SE2OFFSET "):
            return None

        numbers = line[len("PARAMS_SE2OFFSET "):].split()  # fmt: skip
        arr = [float(number) for number in numbers[1:]]
        return cls(("PARAMS_SE2OFFSET", int(numbers[0])), PoseSE2([arr[0], arr[1]], arr[2]))


class G2OParameterSE3Offset(BaseG2OParameter):
    """A class for storing a g2o :math:`SE(3)` offset parameter.

    Attributes
    ----------
    key : tuple[str, int]
        A tuple of the form ``("PARAMS_SE3OFFSET", id)``
    value : graphslam.pose.se3.PoseSE3
        The offset

    """

    def to_g2o(self):
        """Export the :math:`SE(3)` offset parameter to the .g2o format.

        Returns
        -------
        str
            The parameter in .g2o format

        """
        return "PARAMS_SE3OFFSET {} {} {} {} {} {} {} {}\n".format(
            self.key[1],
            self.value[0],
            self.value[1],
            self.value[2],
            self.value[3],
            self.value[4],
            self.value[5],
            self.value[6],
        )

    @classmethod
    def from_g2o(cls, line):
        """Load an :math:`SE(3)` offset parameter from a line in a .g2o file.

        Parameters
        ----------
        line : str
            The line from the .g2o file

        Returns
        -------
        G2OParameterSE3Offset, None
            The instantiated parameter object, or ``None`` if ``line`` does not correspond to an :math:`SE(3)` offset parameter

        """
        if not line.startswith("PARAMS_SE3OFFSET "):
            return None

        # // PARAMS_SE3OFFSET id x y z qw qx qy qz
        numbers = line[len("PARAMS_SE3OFFSET "):].split()  # fmt: skip
        arr = [float(number) for number in numbers[1:]]
        return cls(("PARAMS_SE3OFFSET", int(numbers[0])), PoseSE3(arr[:3], arr[3:]))
