# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the edge_landmark.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.edge.edge_landmark import EdgeLandmark
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


class TestEdgeLandmark(unittest.TestCase):
    """Tests for the ``EdgeLandmark`` class."""

    def test_calc_jacobians3d(self):
        """Test that the ``calc_jacobians`` method is correctly implemented."""
        np.random.seed(0)

        for a in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseR3(np.random.random_sample(3))
            offset = PoseSE3(np.random.random_sample(3), [0.0, 0.0, 0.0, 1.0])

            p1.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], np.eye(3), np.zeros(3), [v1, v2], offset)

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_calc_jacobians2d(self):
        """Test that the ``calc_jacobians`` method is correctly implemented."""
        np.random.seed(0)

        for a in range(10):
            p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
            p2 = PoseR2(np.random.random_sample(2))
            offset = PoseSE2(np.random.random_sample(2), 0.0)

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], np.eye(2), np.zeros(2), [v1, v2], offset)

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=5)

    def test_to_g2o_and_from_g2o_2d(self):
        """Test that the `to_g2o` and `from_g2o` methods work correctly for 2D landmark edges."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
            p2 = PoseR2(np.random.random_sample(2))
            offset = PoseR2(np.random.random_sample(2))
            estimate = PoseSE2(np.random.random_sample(2), 0.0)

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], np.eye(2), estimate, [v1, v2], offset)

            self.assertTrue(e.equals(EdgeLandmark.from_g2o(e.to_g2o())))

    def test_to_g2o_and_from_g2o_3d(self):
        """Test that the `to_g2o` and `from_g2o` methods work correctly for 3D landmark edges."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseR3(np.random.random_sample(3))
            offset = PoseR3(np.random.random_sample(3))
            estimate = PoseSE3(np.random.random_sample(3), [0.0, 0.0, 0.0, 1.0])

            p1.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], np.eye(3), estimate, [v1, v2], offset)

            self.assertTrue(e.equals(EdgeLandmark.from_g2o(e.to_g2o())))


if __name__ == "__main__":
    unittest.main()
