# Copyright (c) 2020 Jeff Irion and contributors

r"""A base class for poses.

"""

import numpy as np


class BasePose(np.ndarray):
    """A base class for poses.

    """

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

        Returns
        -------
        np.ndarray
            The Jacobian of :math:`p_1 \boxplus \Delta \mathbf{x}` w.r.t. :math:`\Delta \mathbf{x}` evaluated at :math:`\Delta \mathbf{x} = \mathbf{0}`

        """
        raise NotImplementedError
