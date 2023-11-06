# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the pose/pose_r2.py module.

"""


import unittest

from graphslam.pose.base_pose import BasePose


class TestBasePose(unittest.TestCase):
    """Tests for the ``PoseR2`` class."""

    def test_identity(self):
        """Test that the ``identity`` method is not implemented."""
        with self.assertRaises(NotImplementedError):
            BasePose.identity()

    def test_to_array(self):
        """Test that the ``to_array`` method is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            p.to_array()

    def test_to_compact(self):
        """Test that the ``to_compact`` method is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            p.to_compact()

    def test_copy(self):
        """Test that the ``copy`` method is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            p.copy()

    # ======================================================================= #
    #                                                                         #
    #                                Properties                               #
    #                                                                         #
    # ======================================================================= #
    def test_position(self):
        """Test that the ``position`` property is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.position

    def test_orientation(self):
        """Test that the ``orientation`` property is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.orientation

    def test_inverse(self):
        """Test that the ``inverse`` property is not implemented."""
        p = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p.inverse

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def test_add(self):
        """Test that the overloaded ``__add__`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1 + p2

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p2 - p1

    # ======================================================================= #
    #                                                                         #
    #                                Jacobians                                #
    #                                                                         #
    # ======================================================================= #
    def test_jacobian_self_oplus_other_wrt_self(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_other_wrt_self(p2)

    def test_jacobian_self_oplus_other_wrt_self_compact(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self_compact`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_other_wrt_self_compact(p2)

    def test_jacobian_self_oplus_other_wrt_other(self):
        """Test that the ``jacobian_self_oplus_other_wrt_other`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_other_wrt_other(p2)

    def test_jacobian_self_oplus_other_wrt_other_compact(self):
        """Test that the ``jacobian_self_oplus_other_wrt_other_compact`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_other_wrt_other_compact(p2)

    def test_jacobian_self_ominus_other_wrt_self(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_ominus_other_wrt_self(p2)

    def test_jacobian_self_ominus_other_wrt_self_compact(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self_compact`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_ominus_other_wrt_self_compact(p2)

    def test_jacobian_self_ominus_other_wrt_other(self):
        """Test that the ``jacobian_self_ominus_other_wrt_other`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_ominus_other_wrt_other(p2)

    def test_jacobian_self_ominus_other_wrt_other_compact(self):
        """Test that the ``jacobian_self_ominus_other_wrt_other_compact`` method is not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_ominus_other_wrt_other_compact(p2)

    def test_jacobian_boxplus(self):
        """Test that the ``jacobian_boxplus`` method is not implemented."""
        p1 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_boxplus()

    def test_jacobian_self_oplus_point(self):
        """Test that the ``jacobian_self_oplus_point_wrt_self`` and ``jacobian_self_oplus_point_wrt_point`` methods are not implemented."""
        p1 = BasePose([])
        p2 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_point_wrt_self(p2)

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_self_oplus_point_wrt_point(p2)

    def test_jacobian_inverse(self):
        """Test that the ``jacobian_inverse`` method is not implemented."""
        p1 = BasePose([])

        with self.assertRaises(NotImplementedError):
            _ = p1.jacobian_inverse()


if __name__ == "__main__":
    unittest.main()
