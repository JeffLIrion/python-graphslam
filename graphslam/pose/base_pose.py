# Copyright (c) 2020 Jeff Irion and contributors

r"""A base class for poses.

"""

from abc import ABC, abstractmethod

import numpy as np


class BasePose(ABC):
    """A base class for poses.

    Parameters
    ----------
    arr : np.ndarray, list
        The array that will be stored as `self._data`
    dtype : type
        The type for the numpy array `self._data`

    """

    def __init__(self, arr, dtype=np.float64):
        self._data = np.array(arr, dtype=dtype)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    @abstractmethod
    def copy(self):
        """Return a copy of the pose.

        Returns
        -------
        BasePose
            A copy of the pose

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

    def approx_equal(self, other, tol=1e-6):
        """Check whether two poses are approximately equal.

        Parameters
        ----------
        other : BasePose
            The pose to which we are comparing
        tol : float
            The tolerance

        Returns
        -------
        bool
            Whether the two poses are approximately equal

        """
        # pylint: disable=protected-access
        return np.linalg.norm(self._data - other._data) / max(np.linalg.norm(self._data), tol) < tol

    # ======================================================================= #
    #                                                                         #
    #                                Properties                               #
    #                                                                         #
    # ======================================================================= #
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

    @property
    @abstractmethod
    def inverse(self):
        """Return the pose's inverse.

        Returns
        -------
        BasePose
            The pose's inverse

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

    def __iadd__(self, other):
        """Add poses in-place (i.e., pose composition).

        Parameters
        ----------
        other : BasePose
            The other pose

        Returns
        -------
        BasePose
            The result of pose composition

        """
        return self + other

    # ======================================================================= #
    #                                                                         #
    #                                Jacobians                                #
    #                                                                         #
    # ======================================================================= #
    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`

        """
