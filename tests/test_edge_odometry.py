# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the ``EdgeOdometry`` class.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


class TestEdgeOdometry(unittest.TestCase):
    """Tests for the ``EdgeOdometry`` class."""

    def test_plot(self):
        """Test that the ``plot`` method is not implemented."""
        v_none = Vertex(0, None)
        v_r2 = Vertex(1, PoseR2([1, 2]))
        v_se2 = Vertex(2, PoseSE2([1, 2], 3))
        v_r3 = Vertex(3, PoseR3([1, 2, 3]))
        v_se3 = Vertex(4, PoseSE3([1, 2, 3], [0.5, 0.5, 0.5, 0.5]))

        with self.assertRaises(NotImplementedError):
            e = EdgeOdometry(0, 1, 0, [v_none, v_none])
            e.plot()

        for v in [v_r2, v_se2, v_r3, v_se3]:
            e = EdgeOdometry(0, 1, 0, [v, v])
            e.plot()

    def test_calc_jacobians_r2(self):
        """Test that the ``calc_jacobians`` method gives the correct results."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR2(np.random.random_sample(2))
            p2 = PoseR2(np.random.random_sample(2))
            estimate = PoseR2(np.random.random_sample(2))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOdometry([1, 2], np.eye(2), estimate, [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_calc_jacobians_r3(self):
        """Test that the ``calc_jacobians`` method gives the correct results."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseR3(np.random.random_sample(3))
            p2 = PoseR3(np.random.random_sample(3))
            estimate = PoseR3(np.random.random_sample(3))

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOdometry([1, 2], np.eye(3), estimate, [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_calc_jacobians_se2(self):
        """Test that the ``calc_jacobians`` method gives the correct results."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
            p2 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
            estimate = PoseSE2(np.random.random_sample(2), np.random.random_sample())

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOdometry([1, 2], np.eye(3), estimate, [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_calc_jacobians_se3(self):
        """Test that the ``calc_jacobians`` method gives the correct results."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            estimate = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))

            p1.normalize()
            p2.normalize()
            estimate.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeOdometry([1, 2], np.eye(6), estimate, [v1, v2])

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
