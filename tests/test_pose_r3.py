"""Unit tests for the pose/pose_r3.py module.

"""


import unittest

import numpy as np

from graphslam.pose.r3 import PoseR3


class TestPoseR3(unittest.TestCase):
    """Tests for the ``PoseR3`` class.

    """

    def setUp(self):
        # TODO
        pass

    def test_constructor(self):
        """Test that a ``PoseR3`` instance can be created.

        """
        r3a = PoseR3([1, 2, 3])
        r3b = PoseR3(np.array([3, 4, 5]))
        self.assertIsInstance(r3a, PoseR3)
        self.assertIsInstance(r3b, PoseR3)

    def test_to_array(self):
        """Test that the ``to_array`` method works as expected.

        """
        r3 = PoseR3([1, 2, 3])
        arr = r3.to_array()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR3)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 3])), 0.)

    def test_to_compact(self):
        """Test that the ``to_compact`` method works as expected.

        """
        r3 = PoseR3([1, 2, 3])
        arr = r3.to_compact()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR3)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 3])), 0.)

    def test_position(self):
        """Test that the ``position`` property works as expected.

        """
        r3 = PoseR3([1, 2, 3])
        pos = r3.position

        true_pos = np.array([1, 2, 3])
        self.assertIsInstance(pos, np.ndarray)
        self.assertNotIsInstance(pos, PoseR3)
        self.assertAlmostEqual(np.linalg.norm(true_pos - pos), 0.)

    def test_orientation(self):
        """Test that the ``orientation`` property works as expected.

        """
        r3 = PoseR3([1, 2, 3])

        self.assertEqual(r3.orientation, 0.)

    def test_add(self):
        """Test that the overloaded ``__add__`` method works as expected.

        """
        r3a = PoseR3([1, 2, 3])
        r3b = PoseR3([3, 4, 5])

        expected = PoseR3([4, 6, 8])
        self.assertAlmostEqual(np.linalg.norm((r3a + r3b).to_array() - expected), 0.)

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method works as expected.

        """
        r3a = PoseR3([1, 2, 3])
        r3b = PoseR3([3, 4, 5])

        expected = PoseR3([2, 2, 2])
        self.assertAlmostEqual(np.linalg.norm((r3b - r3a).to_array() - expected), 0.)


if __name__ == '__main__':
    unittest.main()
