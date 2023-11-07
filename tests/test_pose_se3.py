# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the pose/pose_se3.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se3 import PoseSE3
from graphslam.edge.base_edge import BaseEdge
from .edge_types import EdgeInverse, EdgeOMinus, EdgeOMinusCompact, EdgeOPlus, EdgeOPlusCompact, EdgeOPlusPoint


class TestPoseSE3(unittest.TestCase):
    """Tests for the ``PoseSE3`` class."""

    def test_constructor(self):
        """Test that a ``PoseSE3`` instance can be created."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])
        p2 = PoseSE3(np.array([4, 5, 6]), np.array([1, 0, 0, 0]))
        self.assertIsInstance(p1, PoseSE3)
        self.assertIsInstance(p2, PoseSE3)

    def test_normalize(self):
        """Test that the ``normalize`` method works as expected."""
        p1 = PoseSE3([1, 2, 3], [2, 2, 2, 2])
        p2 = PoseSE3(np.array([4, 5, 6]), np.array([2, 0, 0, 0]))

        p1.normalize()
        p2.normalize()

        self.assertAlmostEqual(np.linalg.norm(p1.to_array() - np.array([1, 2, 3, 0.5, 0.5, 0.5, 0.5])), 0.0)
        self.assertAlmostEqual(np.linalg.norm(p2.to_array() - np.array([4, 5, 6, 1, 0, 0, 0])), 0.0)

    def test_copy(self):
        """Test that the ``copy`` method works as expected."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])
        p2 = p1.copy()

        p2[0] = 0
        self.assertEqual(p1[0], 1)

    def test_to_array(self):
        """Test that the ``to_array`` method works as expected."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])
        arr = p1.to_array()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseSE3)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 3, 0, 0, 0, 1])), 0.0)

    def test_to_compact(self):
        """Test that the ``to_compact`` method works as expected."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])
        arr = p1.to_compact()

        self.assertIsInstance(arr, np.ndarray)
        self.assertNotIsInstance(arr, PoseSE3)
        self.assertAlmostEqual(np.linalg.norm(arr - np.array([1, 2, 3, 0, 0, 0])), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                                Properties                               #
    #                                                                         #
    # ======================================================================= #
    def test_position(self):
        """Test that the ``position`` property works as expected."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])
        pos = p1.position

        true_pos = np.array([1, 2, 3])
        self.assertIsInstance(pos, np.ndarray)
        self.assertNotIsInstance(pos, PoseSE3)
        self.assertAlmostEqual(np.linalg.norm(true_pos - pos), 0.0)

    def test_orientation(self):
        """Test that the ``orientation`` property works as expected."""
        p1 = PoseSE3([1, 2, 3], [0, 0, 0, 1])

        self.assertAlmostEqual(np.linalg.norm(p1.orientation - np.array([0, 0, 0, 1])), 0.0)

    def test_inverse(self):
        """Test that the ``inverse`` property works as expected."""
        np.random.seed(0)

        for _ in range(10):
            p = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p.normalize()

            expected = np.linalg.inv(p.to_matrix())
            self.assertAlmostEqual(np.linalg.norm(p.inverse.to_matrix() - expected), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                              Magic Methods                              #
    #                                                                         #
    # ======================================================================= #
    def test_add(self):
        """Test that the overloaded ``__add__`` method works as expected."""
        np.random.seed(0)

        # PoseSE3 (+) PoseSE3
        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            expected = np.dot(p1.to_matrix(), p2.to_matrix())
            self.assertAlmostEqual(np.linalg.norm((p1 + p2).to_matrix() - expected), 0.0)

            expected2 = (p1 + p2 + PoseSE3.identity()).to_matrix()
            self.assertAlmostEqual(np.linalg.norm(expected2 - expected), 0.0)

        # PoseSE3 [+] numpy.ndarray
        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2_compact = p2.to_compact()

            if np.linalg.norm(p2.orientation[:3]) > 1.0:
                p2[3:] = [0.0, 0.0, 0.0, 1.0]
            else:
                p2.normalize()
                p2_compact[3:] = p2.orientation[:3]

            p1.normalize()

            expected = np.dot(p1.to_matrix(), p2.to_matrix())
            self.assertAlmostEqual(np.linalg.norm((p1 + p2_compact).to_matrix() - expected), 0.0)

            p1 += p2_compact
            self.assertAlmostEqual(np.linalg.norm(p1.to_matrix() - expected), 0.0)

        # PoseSE3 (+) PoseR3
        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseR3(np.random.random_sample(3))

            p1.normalize()

            expected = np.dot(p1.to_matrix(), np.array([p2[0], p2[1], p2[2], 1.0]))
            self.assertAlmostEqual(np.linalg.norm(p1 + p2 - expected[:3]), 0.0)

        with self.assertRaises(NotImplementedError):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            _ = p1 + 5

    def test_sub(self):
        """Test that the overloaded ``__sub__`` method works as expected."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            expected = np.dot(np.linalg.inv(p2.to_matrix()), p1.to_matrix())
            self.assertAlmostEqual(np.linalg.norm((p1 - p2).to_matrix() - expected), 0.0)

            expected2 = (p1 - p2 - PoseSE3.identity()).to_matrix()
            self.assertAlmostEqual(np.linalg.norm(expected2 - expected), 0.0)

    # ======================================================================= #
    #                                                                         #
    #                                Jacobians                                #
    #                                                                         #
    # ======================================================================= #
    def test_jacobian_self_oplus_other(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self`` and ``jacobian_self_oplus_other_wrt_other`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlus([1, 2], np.eye(7), np.zeros(7), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_jacobian_self_ominus_other(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self`` and ``jacobian_self_ominus_other_wrt_other`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOMinus([1, 2], np.eye(7), np.zeros(7), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, 5)

    def test_jacobian_self_oplus_other_compact(self):
        """Test that the ``jacobian_self_oplus_other_wrt_self_compact`` and ``jacobian_self_oplus_other_wrt_other_compact`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlusCompact([1, 2], np.eye(7), np.zeros(7), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_jacobian_self_ominus_other_compact(self):
        """Test that the ``jacobian_self_ominus_other_wrt_self_compact`` and ``jacobian_self_ominus_other_wrt_other_compact`` methods are correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOMinusCompact([1, 2], np.eye(7), np.zeros(7), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, 5)

    def test_jacobian_self_oplus_point(self):
        """Test that the ``jacobian_self_oplus_point_wrt_self`` and ``jacobian_self_oplus_point_wrt_point`` methods are correctly implemented."""
        np.random.seed(0)

        for a in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseR3(np.random.random_sample(3))

            p1.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOPlusPoint([1, 2], np.eye(3), np.zeros(3), [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_jacobian_inverse(self):
        """Test that the ``jacobian_inverse`` method is correctly implemented."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()

            v1 = Vertex(1, p1)

            e = EdgeInverse([1], np.eye(3), np.zeros(3), [v1])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
