# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the graph.py module.

"""


import unittest
from unittest.mock import mock_open, patch

import numpy as np

from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.graph import Graph
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3
from graphslam.vertex import Vertex


class TestGraphR2(unittest.TestCase):
    r"""Tests for the ``Graph`` class with :math:`\mathbb{R}^3` poses.

    """

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`\mathbb{R}^2`.

        """
        np.random.seed(0)

        p1 = PoseR2(np.random.random_sample(2))
        p2 = PoseR2(np.random.random_sample(2))
        p3 = PoseR2(np.random.random_sample(2))
        estimate = PoseR2([0, 0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(2), estimate, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(2), estimate, [v3, v2])

        self.g = Graph([e1, e2], [v1, v2, v3])

    def test_calc_chi2(self):
        r"""Test that the :math:`\chi^2` for a ``Graph`` can be computed.

        """
        chi2 = self.g._edges[0].calc_chi2() + self.g._edges[1].calc_chi2()  # pylint: disable=protected-access

        self.assertAlmostEqual(chi2, self.g.calc_chi2())

    def test_optimize(self):
        """Test that a ``Graph`` can be optimized.

        """
        chi2_orig = self.g.calc_chi2()

        p0 = self.g._vertices[0].pose.to_array()  # pylint: disable=protected-access
        self.g.optimize()
        self.assertLess(self.g.calc_chi2(), chi2_orig)

        # Make sure the first pose was held fixed
        self.assertAlmostEqual(np.linalg.norm(p0 - self.g._vertices[0].pose.to_array()), 0.)  # pylint: disable=protected-access

    # pylint: disable=protected-access
    def test_to_g2o(self):
        """Test that the ``to_g2o`` method is implemented correctly, or raises ``NotImplementedError``.

        """
        # Supported types
        if isinstance(self.g._vertices[0].pose, (PoseSE2, PoseSE3)):
            print(self.g._vertices[0].to_g2o())
            print(self.g._edges[0].to_g2o())

            with patch("graphslam.graph.open", mock_open()):
                self.g.to_g2o("test.g2o")
        # Unsupported types
        else:
            with self.assertRaises(NotImplementedError):
                print(self.g._vertices[0].to_g2o())

            with self.assertRaises(NotImplementedError):
                print(self.g._edges[0].to_g2o())

            with patch("graphslam.graph.open", mock_open()):
                with self.assertRaises(NotImplementedError):
                    self.g.to_g2o("test.g2o")


class TestGraphR3(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`\mathbb{R}^3` poses.

    """

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`\mathbb{R}^3`.

        """
        np.random.seed(0)

        p1 = PoseR3(np.random.random_sample(3))
        p2 = PoseR3(np.random.random_sample(3))
        p3 = PoseR3(np.random.random_sample(3))
        estimate = PoseR3([0, 0, 0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(3), estimate, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(3), estimate, [v3, v2])

        self.g = Graph([e1, e2], [v1, v2, v3])


class TestGraphSE2(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`SE(2)` poses.

    """

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`SE(2)`.

        """
        np.random.seed(0)

        p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p2 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p3 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        estimate = PoseSE2([0, 0], 0.)

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(3), estimate, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(3), estimate, [v3, v2])

        self.g = Graph([e1, e2], [v1, v2, v3])


class TestGraphSE3(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`SE(3)` poses.

    """

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`SE(3)`.

        """
        np.random.seed(0)

        p1 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        p2 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        p3 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        estimate = PoseSE3([0, 0, 0], [0, 0, 0, 1])

        p1.normalize()
        p2.normalize()
        p3.normalize()

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(6), estimate, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(6), estimate, [v3, v2])

        self.g = Graph([e1, e2], [v1, v2, v3])


if __name__ == '__main__':
    unittest.main()
