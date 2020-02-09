"""Unit tests for the pose/pose_r2.py module.

"""


import unittest

import numpy as np

from graphslam.pose.se2 import PoseSE2


class TestPoseSE2(unittest.TestCase):
    """Tests for the ``PoseSE2`` class.

    """

    def test_constructor(self):
        """Test that a ``PoseSE2`` instance can be created.

        """
        r2a = PoseSE2([1, 2], 3)
        r2b = PoseSE2(np.array([3, 4]), 5)
        self.assertIsInstance(r2a, PoseSE2)
        self.assertIsInstance(r2b, PoseSE2)

    def test_to_array(self):
        """Test that the ``to_array`` method works as expected.

        """
        r2 = PoseSE2([1, 2], 1)
        arr = r2.to_array()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseSE2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 1])), 0.)

    def test_to_compact(self):
        """Test that the ``to_compact`` method works as expected.

        """
        r2 = PoseSE2([1, 2], 1)
        arr = r2.to_compact()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseSE2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 1])), 0.)

    def test_position(self):
        """Test that the ``position`` property works as expected.

        """
        r2 = PoseSE2([1, 2], 3)
        pos = r2.position

        true_pos = np.array([1, 2])
        self.assertIsInstance(pos, np.ndarray)
        self.assertNotIsInstance(pos, PoseSE2)
        self.assertAlmostEqual(np.linalg.norm(true_pos - pos), 0.)

    def test_orientation(self):
        """Test that the ``orientation`` property works as expected.

        """
        r2 = PoseSE2([1, 2], 1.5)

        self.assertEqual(r2.orientation, 1.5)

    @unittest.skip("Not properly implemented yet")
    def test_add(self):
        """Test that the overloaded ``__add__`` method works as expected.

        """
        r2a = PoseSE2([1, 2], 3)
        r2b = PoseSE2([3, 4], 3)

        expected = PoseSE2([4, 6], 3)
        self.assertAlmostEqual(np.linalg.norm((r2a + r2b).to_array() - expected), 0.)

    @unittest.skip("Not properly implemented yet")
    def test_sub(self):
        """Test that the overloaded ``__sub__`` method works as expected.

        """
        r2a = PoseSE2([1, 2], 3)
        r2b = PoseSE2([3, 4], 3)

        expected = PoseSE2([2, 2], 3)
        self.assertAlmostEqual(np.linalg.norm((r2b - r2a).to_array() - expected), 0.)


if __name__ == '__main__':
    unittest.main()
