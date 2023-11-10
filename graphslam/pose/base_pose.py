# Copyright (c) 2020 Jeff Irion and contributors

r"""A base class for poses.

"""

import numpy as np


class BasePose(np.ndarray):
    """A base class for poses."""

    @classmethod
    def identity(cls):
        """Return the identity pose.

        Returns
        -------
        BasePose
            The identity pose

        """
        raise NotImplementedError

    # pylint: disable=arguments-differ
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

    def equals(self, other, tol=1e-6):
        """Check whether two poses are equal.

        Parameters
        ----------
        other : BasePose
            The pose to which we are comparing
        tol : float
            The tolerance

        Returns
        -------
        bool
            Whether the two poses are equal

        """
        return np.linalg.norm(self.to_array() - other.to_array()) / max(np.linalg.norm(self.to_array()), tol) < tol

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
    def jacobian_self_oplus_other_wrt_self(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

           # The dimensionality of `self + other`
           n_oplus = len((self + other).to_array())

        Then the shape of the Jacobian will be ``n_oplus x n_self``.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        """
        raise NotImplementedError

    def jacobian_self_oplus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

           # The compact dimensionality of `self + other`
           n_compact = (self + other).COMPACT_DIMENSIONALITY

        Then the shape of the Jacobian will be ``n_compact x n_self``.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        """
        raise NotImplementedError

    def jacobian_self_oplus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Let

        .. code::

           # The dimensionality of `other`
           n_other = len(other.to_array())

           # The dimensionality of `self + other`
           n_oplus = len((self + other).to_array())

        Then the shape of the Jacobian will be ``n_oplus x n_other``.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        raise NotImplementedError

    def jacobian_self_oplus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        Let

        .. code::

           # The dimensionality of `other`
           n_other = len(other.to_array())

           # The compact dimensionality of `self + other`
           n_compact = (self + other).COMPACT_DIMENSIONALITY

        Then the shape of the Jacobian will be ``n_compact x n_other``.

        Parameters
        ----------
        other : BasePose
            The pose that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        raise NotImplementedError

    def jacobian_self_ominus_other_wrt_self(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

           # The dimensionality of `self - other`
           n_ominus = len((self - other).to_array())

        Then the shape of the Jacobian will be ``n_ominus x n_self``.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        """
        raise NotImplementedError

    def jacobian_self_ominus_other_wrt_self_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

           # The compact dimensionality of `self - other`
           n_compact = (self - other).COMPACT_DIMENSIONALITY

        Then the shape of the Jacobian will be ``n_compact x n_self``.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_1`.

        """
        raise NotImplementedError

    def jacobian_self_ominus_other_wrt_other(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Let

        .. code::

           # The dimensionality of `other`
           n_other = len(other.to_array())

           # The dimensionality of `self - other`
           n_ominus = len((self - other).to_array())

        Then the shape of the Jacobian will be ``n_ominus x n_other``.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        """
        raise NotImplementedError

    def jacobian_self_ominus_other_wrt_other_compact(self, other):
        r"""Compute the Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        Let

        .. code::

           # The dimensionality of `other`
           n_other = len(other.to_array())

           # The compact dimensionality of `self - other`
           n_compact = (self - other).COMPACT_DIMENSIONALITY

        Then the shape of the Jacobian will be ``n_compact x n_other``.

        Parameters
        ----------
        other : BasePose
            The pose that is being subtracted from ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \ominus p_2` w.r.t. :math:`p_2`.

        """
        raise NotImplementedError

    def jacobian_boxplus(self):
        r"""Compute the Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`.

        Let

        .. code::

           # The dimensionality of :math:`\Delta \mathbf{x}`, which should be the same as
           # the compact dimensionality of `self`
           n_dx = self.COMPACT_DIMENSIONALITY

           # The dimensionality of :math:`p_1 \boxplus \Delta \mathbf{x}`, which should be
           # the same as the dimensionality of `self`
           n_boxplus = len(self.to_array())

        Then the shape of the Jacobian will be ``n_boxplus x n_dx``.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`

        """
        raise NotImplementedError

    def jacobian_self_oplus_point_wrt_self(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`, where `:math:p_2` is a point.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

           # The dimensionality of `self + point`
           n_oplus = len((self + point).to_array())

        Then the shape of the Jacobian will be ``n_oplus x n_self``.

        Parameters
        ----------
        point : BasePose
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_1`.

        """
        raise NotImplementedError

    def jacobian_self_oplus_point_wrt_point(self, point):
        r"""Compute the Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`, where `:math:p_2` is a point.

        Let

        .. code::

           # The dimensionality of `point`
           n_point = len(point.to_array())

           # The dimensionality of `self + point`
           n_oplus = len((self + point).to_array())

        Then the shape of the Jacobian will be ``n_oplus x n_point``.

        Parameters
        ----------
        point : BasePose
            The point that is being added to ``self``

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \oplus p_2` w.r.t. :math:`p_2`.

        """
        raise NotImplementedError

    def jacobian_inverse(self):
        r"""Compute the Jacobian of :math:`p^{-1}`.

        Let

        .. code::

           # The dimensionality of `self`
           n_self = len(self.to_array())

        Then the shape of the Jacobian will be ``n_self x n_self``.

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p^{-1}`

        """
        raise NotImplementedError
