"""Unit tests for the pose/pose_r2.py module.

"""


import unittest

from graphslam.pose.base_pose import BasePose


class TestBasePose(unittest.TestCase):
    """Tests for the ``PoseR2`` class.

    """

    def test_to_array(self):
        """Test that the ``to_array`` method is not implemented.

        """
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            p.to_array()

    def test_to_compact(self):
        """Test that the ``to_compact`` method is not implemented.

        """
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            p.to_compact()

    def test_position(self):
        """Test that the ``position`` property is not implemented.

        """
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.position

    def test_orientation(self):
        """Test that the ``orientation`` property is not implemented.

        """
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.orientation

    def test_inverse(self):
        """Test that the ``inverse`` property is not implemented.

        """
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.inverse

    def test_add(self):
        """Test that the overloaded ``__add__`` method is not implemented.

        """
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1 + p2

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method is not implemented.

        """
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p2 - p1


if __name__ == '__main__':
    unittest.main()
