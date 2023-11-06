# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the pose/pose_r2.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.pose.r2 import PoseR2
from graphslam.edge.base_edge import BaseEdge
from .edge_types import EdgeInverse, EdgeOMinus, EdgeOMinusCompact, EdgeOPlus, EdgeOPlusCompact, EdgeOPlusPoint


class TestPoseR2(unittest.TestCase):
    """Tests for the ``PoseR2`` class."""

    def test_constructor(self):
        """Test that a ``PoseR2`` instance can be created."""
        r2a = PoseR2([1, 2])
        r2b = PoseR2(np.array([3, 4]))
        self.assertIsInstance(r2a, PoseR2)
        self.assertIsInstance(r2b, PoseR2)

    def test_copy(self):
        """Test that the ``copy`` method works as expected."""
        p1 = PoseR2([1, 2])
        p2 = p1.copy()

        p2[0] = 0
        self.assertEqual(p1[0], 1)

    def test_to_array(self):
        """Test that the ``to_array`` method works as expected."""
        r2 = PoseR2([1, 2])
        arr = r2.to_array()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2])), 0.0)

    def test_to_compact(self):
        """Test that the ``to_compact`` method works as expected."""
        r2 = PoseR2([1, 2])
        arr = r2.to_compact()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2])), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                                Properties                               #
    #                                                                         #
    # ======================================================================= #
    def test_position(self):
        """Test that the ``position`` property works as expected."""
        r2 = PoseR2([1, 2])
        pos = r2.position

        true_pos = np.array([1, 2])
        self.assertIsInstance(pos, np.ndarray)
        self.assertNotIsInstance(pos, PoseR2)
        self.assertAlmostEqual(np.linalg.norm(true_pos - pos), 0.0)

    def test_orientation(self):
        """Test that the ``orientation`` property works as expected."""
        r2 = PoseR2([1, 2])

        self.assertEqual(r2.orientation, 0.0)

    def test_inverse(self):
        """Test that the ``inverse`` property works as expected."""
        r2 = PoseR2([1, 2])

        true_inv = np.array([-1, -2])

        self.assertAlmostEqual(np.linalg.norm(r2.inverse.to_array() - true_inv), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def test_add(self):
        """Test that the overloaded ``__add__`` method works as expected."""
        r2a = PoseR2([1, 2])
        r2b = PoseR2([3, 4])

        expected = PoseR2([4, 6])
        self.assertAlmostEqual(np.linalg.norm(((r2a + r2b) - expected).to_array()), 0.0)

        r2a += r2b
        self.assertAlmostEqual(np.linalg.norm((r2a - expected).to_array()), 0.0)

        expected2 = expected + PoseR2.identity()
        self.assertAlmostEqual(np.linalg.norm((expected2 - expected).to_array()), 0.0)

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method works as expected."""
        r2a = PoseR2([1, 2])
        r2b = PoseR2([3, 4])

        expected = PoseR2([2, 2])
        self.assertAlmostEqual(np.linalg.norm(((r2b - r2a) - expected).to_array()), 0.0)

        expected2 = expected - PoseR2.identity()
        self.assertAlmostEqual(np.linalg.norm((expected2 - expected).to_array()), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                                Jacobians                                #
    #                                                                         #
    # ======================================================================= #
    def test_jacobian_self_oplus_other(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self`` and ``jacobian_self_oplus_other_wrt_other`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlus([1, 2], np.eye(2), np.zeros(2), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)

    def test_jacobian_self_ominus_other(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self`` and ``jacobian_self_ominus_other_wrt_other`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOMinus([1, 2], np.eye(2), np.zeros(2), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)

    def test_jacobian_self_oplus_other_compact(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self_compact`` and ``jacobian_self_oplus_other_wrt_other_compact`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlusCompact([1, 2], np.eye(2), np.zeros(2), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)

    def test_jacobian_self_ominus_other_compact(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self_compact`` and ``jacobian_self_ominus_other_wrt_other_compact`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOMinusCompact([1, 2], np.eye(2), np.zeros(2), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)

    def test_jacobian_self_oplus_point(self):
        """Test that the ``jacobian_self_oplus_point_wrt_self`` and ``jacobian_self_oplus_point_wrt_point`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlusPoint([1, 2], np.eye(2), np.zeros(2), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)

    def test_jacobian_inverse(self):
        """Test that the ``jacobian_inverse`` method is correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)

            e = EdgeInverse([1], np.eye(2), np.zeros(2), [v1])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0)


if __name__ == "__main__":
    unittest.main()
