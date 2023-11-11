# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for creating a custom edge type.

"""


import unittest
from unittest import mock

import numpy as np

from graphslam.edge.base_edge import BaseEdge
from graphslam.graph import Graph
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3
from graphslam.vertex import Vertex

from .patchers import FAKE_FILE, open_fake_file


class DistanceEdgeNumericalJacobians(BaseEdge):
    """A custom edge type that constrains the distance between two poses.

    This is all that is needed to implement a custom edge type.

    """

    def to_g2o(self):
        """Not supported, so return ``None``."""
        return None

    @classmethod
    def from_g2o(cls, line):
        """Not supported, so return ``None``."""
        return None

    def plot(self, color=""):
        """Not supported, so don't do anything."""

    def calc_error(self):
        """Calculate the error, which is the distance between the two poses minus the estimate."""
        return np.array([np.linalg.norm((self.vertices[0].pose - self.vertices[1].pose).position) - self.estimate])


class DistanceEdgeAnalyticalJacobians(DistanceEdgeNumericalJacobians):
    """A custom edge type with analytical Jacobians, as well as `to_g2o` and `from_g2o` methods.

    - Analytical Jacobians make the optimization faster and more accurate
      - See the test `test_analytical_jacobians` below for an example of how to validate that the
        analytical Jacobians are correct by comparing them to their numerical counterparts
    - `to_g2o` and `from_g2o` methods allow reading and writing the edge in .g2o files
      - See the test `TestDistanceEdgeSE3::test_to_g2o_and_from_g2o` below for an example

    """

    def calc_jacobians(self):
        """Calculate the Jacobian of the edge's error with respect to each constrained pose.

        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        p0 = self.vertices[0].pose
        p1 = self.vertices[1].pose
        t = (p0 - p1).position

        dim_point = len(t)

        # The error for the edge is computed as:
        #
        #   err = np.linalg.norm((p0 - p1).position) - self.estimate
        #
        # Let
        #
        #   dim_pose = len(p0.to_array())
        #
        # The Jacobians of `(p0 - p1).position` are `dim_point x dim_pose` matrices:
        #
        #   jacobian_diff_p0 = p0.jacobian_self_ominus_other_wrt_self(p1)[:dim_point, :]
        #   jacobian_diff_p1 = p0.jacobian_self_ominus_other_wrt_other(p1)[:dim_point, :]
        #
        # The Jacobian of `np.linalg.norm(t)` is
        #
        #   jacobian_norm = t / np.linalg.norm(t)
        def jacobian_norm(translation):
            """The Jacobian of taking the norm of `translation`."""
            return translation / np.linalg.norm(translation)

        # fmt: off
        # Use the chain rule to compute the Jacobians
        return [np.dot(np.dot(jacobian_norm(t), p0.jacobian_self_ominus_other_wrt_self(p1)[:dim_point, :]), p0.jacobian_boxplus()),
                np.dot(np.dot(jacobian_norm(t), p0.jacobian_self_ominus_other_wrt_other(p1)[:dim_point, :]), p1.jacobian_boxplus())]
        # fmt: on

    def to_g2o(self):
        """Write to a .g2o format."""
        # Only implemented for SE(3) poses
        if isinstance(self.vertices[0].pose, PoseSE3):
            return "EDGE_DISTANCE_SE3 {} {} {} {}\n".format(
                self.vertex_ids[0], self.vertex_ids[1], self.estimate, self.information[0][0]
            )

        return None

    @classmethod
    def from_g2o(cls, line):
        """Load from a line in a .g2o file."""
        # Only implemented for SE(3) poses
        if line.startswith("EDGE_DISTANCE_SE3 "):
            numbers = line[len("EDGE_DISTANCE_SE3 "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
            vertex_ids = [int(numbers[0]), int(numbers[1])]
            estimate = arr[0]
            information = np.array([[arr[1]]])
            return DistanceEdgeAnalyticalJacobians(vertex_ids, information, estimate)

        return None


class TestDistanceEdgeR2(unittest.TestCase):
    r"""Test that the analytical Jacobians in `DistanceEdgeAnalyticalJacobians` are correct for :math:`\mathbb{R}^2` poses."""

    def setUp(self):
        np.random.seed(0)

        self.p0_list = [PoseR2(np.random.random_sample(2)) for _ in range(10)]
        self.p1_list = [PoseR2(np.random.random_sample(2)) for _ in range(10)]
        self.estimate_list = [np.random.random_sample() for _ in range(10)]

    def test_analytical_jacobians(self):
        """Test that the analytical Jacobians are correct."""
        for p0, p1, estimate in zip(self.p0_list, self.p1_list, self.estimate_list):
            v0 = Vertex(0, p0)
            v1 = Vertex(1, p1)
            e = DistanceEdgeAnalyticalJacobians([0, 1], np.eye(1), estimate, [v0, v1])

            self.assertIsInstance(e.calc_error(), np.ndarray)

            numerical_jacobians = BaseEdge.calc_jacobians(e)

            analytical_jacobians = e.calc_jacobians()

            self.assertEqual(len(numerical_jacobians), len(analytical_jacobians))
            for n, a in zip(numerical_jacobians, analytical_jacobians):
                places = 5 if not isinstance(p0, PoseSE3) else 4
                self.assertAlmostEqual(np.linalg.norm(n - a), 0.0, places=places)


class TestDistanceEdgeR3(TestDistanceEdgeR2):
    r"""Test that the analytical Jacobians in `DistanceEdgeAnalyticalJacobians` are correct for :math:`\mathbb{R}^3` poses."""

    def setUp(self):
        np.random.seed(0)

        self.p0_list = [PoseR3(np.random.random_sample(3)) for _ in range(10)]
        self.p1_list = [PoseR3(np.random.random_sample(3)) for _ in range(10)]
        self.estimate_list = [np.random.random_sample() for _ in range(10)]


class TestDistanceEdgeSE2(TestDistanceEdgeR2):
    """Test that the analytical Jacobians in `DistanceEdgeAnalyticalJacobians` are correct for :math:`SE(2)` poses."""

    def setUp(self):
        np.random.seed(0)

        self.p0_list = [PoseSE2(np.random.random_sample(2), np.random.random_sample()) for _ in range(10)]
        self.p1_list = [PoseSE2(np.random.random_sample(2), np.random.random_sample()) for _ in range(10)]
        self.estimate_list = [np.random.random_sample() for _ in range(10)]


class TestDistanceEdgeSE3(TestDistanceEdgeR2):
    """Test that the analytical Jacobians in `DistanceEdgeAnalyticalJacobians` are correct for :math:`SE(3)` poses."""

    def setUp(self):
        np.random.seed(0)

        self.p0_list = [PoseSE3(np.random.random_sample(3), np.random.random_sample(4)) for _ in range(10)]
        self.p1_list = [PoseSE3(np.random.random_sample(3), np.random.random_sample(4)) for _ in range(10)]
        self.estimate_list = [np.random.random_sample() for _ in range(10)]

        for idx in range(10):
            self.p0_list[idx].normalize()
            self.p1_list[idx].normalize()

    def test_to_g2o_and_from_g2o(self):
        """Test that the `to_g2o` and `from_g2o` allow reading and writing this edge in .g2o files."""
        FAKE_FILE.clear()

        # Build a graph with one edge
        v0 = Vertex(0, self.p0_list[0])
        v1 = Vertex(1, self.p1_list[0])
        estimate = self.estimate_list[0]

        vertices = [v0, v1]
        edges = [DistanceEdgeAnalyticalJacobians([0, 1], np.eye(1), estimate, [v0, v1])]
        g = Graph(edges, vertices)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.graph.open", open_fake_file):
            # Without providing the custom edge type, the graphs will not be the same
            g2 = Graph.load_g2o("test.g2o")
            self.assertFalse(g.equals(g2))

            # When provided with the custom edge type, the graphs will be the same
            g3 = Graph.load_g2o("test.g2o", [DistanceEdgeAnalyticalJacobians])
            self.assertTrue(g.equals(g3))


if __name__ == "__main__":
    unittest.main()
