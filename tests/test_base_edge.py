# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the graph.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.r2 import PoseR2
from graphslam.pose.se2 import PoseSE2


# pylint: disable=abstract-method
class SimpleEdge(BaseEdge):
    """A simple edge class for testing.

    """
    def calc_error(self):
        """A simple "error" method."""
        return len(self.vertices)


class TestBaseEdge(unittest.TestCase):
    """Tests for the ``BaseEdge`` class.

    """

    def test_constructor(self):
        """Test that a ``BaseEdge`` object can be created.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge(0, 1, 0, [v])

        self.assertEqual(e.vertices[0].id, 0)
        self.assertEqual(e.information, 1)

    def test_calc_error(self):
        """Test that the ``calc_error`` method is not implemented.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge(0, 1, 0, [v])

        with self.assertRaises(NotImplementedError):
            _ = e.calc_error()

    def test_calc_chi2(self):
        """Test that the ``calc_chi2`` method works as expected.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = SimpleEdge(0, 1, 0, [v])

        self.assertEqual(e.calc_chi2(), 1)

    def test_calc_jacobians(self):
        """Test that the ``calc_jacobians`` method works as expected.

        """
        p1 = PoseR2([1, 2])
        p2 = PoseR2([3, 4])
        estimate = PoseR2([0, 0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)

        e = EdgeOdometry([1, 2], np.eye(2), estimate, [v1, v2])

        jacobians = e.calc_jacobians()

        self.assertAlmostEqual(np.linalg.norm(jacobians[0] - np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(jacobians[1] + np.eye(2)), 0.)

    def test_calc_chi2_gradient_hessian(self):
        """Test that the ``calc_chi2_gradient_hessian`` method works as expected.

        """
        p1 = PoseR2([1, 3])
        p2 = PoseR2([2, 4])
        estimate = PoseR2([0, 0])

        v1 = Vertex(0, p1, 0)
        v2 = Vertex(1, p2, 1)

        e = EdgeOdometry([0, 1], np.eye(2), estimate, [v1, v2])

        chi2, gradient, hessian = e.calc_chi2_gradient_hessian()

        self.assertEqual(chi2, 2.)

        self.assertAlmostEqual(np.linalg.norm(gradient[0] + np.ones(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(gradient[1] - np.ones(2)), 0.)

        self.assertAlmostEqual(np.linalg.norm(hessian[(0, 0)] - np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(hessian[(0, 1)] + np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(hessian[(1, 1)] - np.eye(2)), 0.)

    def test_to_g2o(self):
        """Test that the ``to_g2o`` method is not implemented.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge(0, 1, 0, [v])

        with self.assertRaises(NotImplementedError):
            _ = e.to_g2o()

    def test_plot(self):
        """Test that the ``plot`` method is not implemented.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge(0, 1, 0, [v])

        with self.assertRaises(NotImplementedError):
            e.plot()


if __name__ == '__main__':
    unittest.main()
