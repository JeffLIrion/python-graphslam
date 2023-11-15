# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the edge_landmark.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.edge.edge_landmark import EdgeLandmark
from graphslam.g2o_parameters import G2OParameterSE3Offset
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
            information = np.eye(3)
            estimate = np.zeros(3)

            p1.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], information, estimate, offset, offset_id=0, vertices=[v1, v2])

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
            information = np.eye(2)
            estimate = np.zeros(2)

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], information, estimate, offset, offset_id=0, vertices=[v1, v2])

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
            information = np.eye(2)
            estimate = PoseR2(np.random.random_sample(2))

            # 2-D landmark edges in .g2o format don't support an offset, so use the default
            offset = PoseSE2.identity()
            offset_id = 0

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            e = EdgeLandmark([1, 2], information, estimate, offset, offset_id, [v1, v2])
            e2 = EdgeLandmark.from_g2o(e.to_g2o())

            self.assertTrue(e.equals(e2))

            # Set the `offset` to something different (then restore it)
            e2.offset = None
            self.assertFalse(e.equals(e2))
            e2.offset = offset

            self.assertTrue(e.equals(e2))

            e2.offset_id += 1
            self.assertFalse(e.equals(e2))

    def test_to_g2o_and_from_g2o_3d(self):
        """Test that the `to_g2o` and `from_g2o` methods work correctly for 3D landmark edges."""
        np.random.seed(0)

        for _ in range(10):
            p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
            p2 = PoseR3(np.random.random_sample(3))
            offset = PoseSE3(np.random.random_sample(3), [0.0, 0.0, 0.0, 1.0])
            offset_id = 0
            information = np.eye(3)
            estimate = PoseR3(np.random.random_sample(3))

            p1.normalize()

            v1 = Vertex(1, p1)
            v2 = Vertex(2, p2)

            g2o_params_or_none = {
                ("PARAMS_SE3OFFSET", offset_id): G2OParameterSE3Offset(("PARAMS_SE3OFFSET", offset_id), offset)
            }

            e = EdgeLandmark([1, 2], information, estimate, offset, offset_id, [v1, v2])
            e2 = EdgeLandmark.from_g2o(e.to_g2o(), g2o_params_or_none)

            self.assertTrue(e.equals(e2))

            # Set the `offset` to something different (then restore it)
            e2.offset = None
            self.assertFalse(e.equals(e2))
            e2.offset = offset
            self.assertTrue(e.equals(e2))

            e2.offset_id += 1
            self.assertFalse(e.equals(e2))

    def test_to_g2o_and_from_g2o_edge_cases(self):
        """Test edge cases for the `to_g2o` and `from_g2o` methods."""
        offset = PoseSE3.identity()
        information = np.eye(3)
        estimate = PoseR3.identity()

        v = Vertex(0, None)
        e = EdgeLandmark([1, 2], information, estimate, offset, offset_id=0, vertices=[v, v])

        # v.pose = None is not supported
        with self.assertRaises(NotImplementedError):
            e.to_g2o()

        edge_or_none = EdgeLandmark.from_g2o("bologna")
        self.assertIsNone(edge_or_none)

    def test_equals(self):
        """Test that the `equals` method works correctly."""
        np.random.seed(0)

        p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p2 = PoseR2(np.random.random_sample(2))
        offset = PoseSE2.identity()
        information = np.eye(2)
        estimate = PoseR2([0.0, 0.0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)

        e = EdgeLandmark([1, 2], information, estimate, offset, vertices=[v1, v2])
        e2 = EdgeLandmark([1, 2], information, estimate, offset=None, vertices=[v1, v2])

        # Different offset pose types
        self.assertFalse(e.equals(e2))

        # Different offsets
        e2.offset = PoseSE2([1.0, 2.0], 3.0)
        self.assertFalse(e.equals(e2))

        # Same offset = they are equal
        e2.offset = e.offset
        self.assertTrue(e.equals(e2))

        # Different offset IDs
        e.offset_id = 5
        self.assertFalse(e.equals(e2))

    def test_plot_failure(self):
        """Test what happens when trying to plot an unsupported pose type."""
        with self.assertRaises(NotImplementedError):
            v = Vertex(0, None)
            e = EdgeLandmark([1, 2], np.eye(2), PoseR2([0.0, 0.0]), offset=PoseR2.identity(), vertices=[v, v])
            e.plot()


if __name__ == "__main__":
    unittest.main()
