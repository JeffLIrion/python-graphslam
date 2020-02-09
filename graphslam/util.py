"""Utility functions used throughout the package.

"""

import numpy as np


TWO_PI = 2 * np.pi


def neg_pi_to_pi(angle):
    r"""Normalize ``angle`` to be in :math:`[-\pi, \pi)`.

    Parameters
    ----------
    angle : float
        An angle (in radians)

    Returns
    -------
    float
        The angle normalized to :math:`[-\pi, \pi)`

    """
    return (angle + np.pi) % (TWO_PI) - np.pi
