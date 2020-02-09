"""Unit tests for the util.py module.

"""


import unittest

import numpy as np

from graphslam import util


class TestUtil(unittest.TestCase):
    """Tests for the functions in util.py.

    """

    def test_neg_pi_to_pi(self):
        """Test the ``neg_pi_to_pi()`` function.

        """
        for angle in range(-10, 10):
            new_angle = util.neg_pi_to_pi(angle)
            self.assertGreaterEqual(new_angle, -np.pi)
            self.assertLess(new_angle, np.pi)
            self.assertAlmostEqual(np.cos(angle), np.cos(new_angle))
            self.assertAlmostEqual(np.sin(angle), np.sin(new_angle))


if __name__ == '__main__':
    unittest.main()
