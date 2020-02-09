"""Unit tests for the pose/pose_r2.py module.

"""


import unittest

import numpy as np

from graphslam.pose.r2 import PoseR2


class TestPoseR2(unittest.TestCase):
    """Tests for the ``PoseR2`` class.

    """

    def setUp(self):
        # TODO
        pass

    def test_constructor(self):
        """Test that a ``PoseR2`` instance can be created.

        """
        r2a = PoseR2([1, 2])
        r2b = PoseR2(np.array([3, 4]))
        self.assertIsInstance(r2a, PoseR2)
        self.assertIsInstance(r2b, PoseR2)

    def test_to_array(self):
        """Test that the ``to_array`` method works as expected.

        """
        r2 = PoseR2([1, 2])
        arr = r2.to_array()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2])), 0.)

    def test_to_compact(self):
        """Test that the ``to_compact`` method works as expected.

        """
        r2 = PoseR2([1, 2])
        arr = r2.to_compact()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2])), 0.)

    def test_position(self):
        """Test that the ``position`` property works as expected.

        """
        r2 = PoseR2([1, 2])
        pos = r2.position

        true_pos = np.array([1, 2])
        self.assertIsInstance(pos, np.ndarray)
        self.assertNotIsInstance(pos, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(true_pos - pos), 0.)

    def test_orientation(self):
        """Test that the ``orientation`` property works as expected.

        """
        r2 = PoseR2([1, 2])

        self.assertEqual(r2.orientation, 0.)

    def test_add(self):
        """Test that the overloaded ``__add__`` method works as expected.

        """
        r2a = PoseR2([1, 2])
        r2b = PoseR2([3, 4])

        expected = PoseR2([4, 6])
        self.assertAlmostEqual(np.linalg.norm((r2a + r2b).position - expected), 0.)

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method works as expected.

        """
        r2a = PoseR2([1, 2])
        r2b = PoseR2([3, 4])

        expected = PoseR2([2, 2])
        self.assertAlmostEqual(np.linalg.norm((r2b - r2a).position - expected), 0.)


if __name__ == '__main__':
    unittest.main()
