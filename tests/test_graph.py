# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the graph.py module.

"""


import os
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.graph import Graph
from graphslam.load import load_g2o
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3
from graphslam.vertex import Vertex


class TestGraphR2(unittest.TestCase):
    r"""Tests for the ``Graph`` class with :math:`\mathbb{R}^2` poses."""

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`\mathbb{R}^2`."""
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
        r"""Test that the :math:`\chi^2` for a ``Graph`` can be computed."""
        chi2 = self.g._edges[0].calc_chi2() + self.g._edges[1].calc_chi2()  # pylint: disable=protected-access

        self.assertAlmostEqual(chi2, self.g.calc_chi2())

    def test_optimize(self):
        """Test that a ``Graph`` can be optimized."""
        chi2_orig = self.g.calc_chi2()

        p0 = self.g._vertices[0].pose.to_array()  # pylint: disable=protected-access
        result = self.g.optimize()
        self.assertLess(self.g.calc_chi2(), chi2_orig)
        self.assertTrue(result.converged)
        self.assertNotEqual(result.initial_chi2, result.iteration_results[0].chi2)
        self.assertEqual(result.final_chi2, result.iteration_results[-2].chi2)

        # Make sure the first pose was held fixed
        # fmt: off
        self.assertAlmostEqual(np.linalg.norm(p0 - self.g._vertices[0].pose.to_array()), 0.)  # pylint: disable=protected-access
        # fmt: on

    def _test_optimize_fixed_vertices(self, fixed_indices):
        """Test that a ``Graph`` can be optimized with vertices held fixed."""
        chi2_orig = self.g.calc_chi2()

        poses_before = [self.g._vertices[i].pose.to_array() for i in fixed_indices]  # pylint: disable=protected-access
        for i in fixed_indices:
            self.g._vertices[i].fixed = True  # pylint: disable=protected-access

        result = self.g.optimize(fix_first_pose=False)
        self.assertLess(self.g.calc_chi2(), chi2_orig)
        self.assertTrue(result.converged)
        self.assertNotEqual(result.initial_chi2, result.iteration_results[0].chi2)
        self.assertEqual(result.final_chi2, result.iteration_results[-2].chi2)
        print(result)

        # Make sure the poses were held fixed
        poses_after = [self.g._vertices[i].pose.to_array() for i in fixed_indices]  # pylint: disable=protected-access
        for before, after in zip(poses_before, poses_after):
            self.assertAlmostEqual(np.linalg.norm(before - after), 0.0)

    def test_optimize_fix_1(self):
        """Test that the ``optimize`` method works correctly when fixing vertex 1."""
        self._test_optimize_fixed_vertices([1])

    def test_optimize_fix_2(self):
        """Test that the ``optimize`` method works correctly when fixing vertex 2."""
        self._test_optimize_fixed_vertices([2])

    def test_optimize_fix_01(self):
        """Test that the ``optimize`` method works correctly when fixing vertices 0 and 1."""
        self._test_optimize_fixed_vertices([0, 1])

    def test_optimize_fix_02(self):
        """Test that the ``optimize`` method works correctly when fixing vertices 0 and 2."""
        self._test_optimize_fixed_vertices([0, 2])

    def test_optimize_fix_12(self):
        """Test that the ``optimize`` method works correctly when fixing vertices 1 and 2."""
        self._test_optimize_fixed_vertices([1, 2])

    # pylint: disable=protected-access
    def test_to_g2o(self):
        """Test that the ``to_g2o`` method is implemented correctly, or raises ``NotImplementedError``."""
        # Fully supported types
        if isinstance(self.g._vertices[0].pose, (PoseSE2, PoseSE3)):
            print(self.g._vertices[0].to_g2o())
            print(self.g._edges[0].to_g2o())

            with patch("graphslam.graph.open", mock_open()):
                self.g.to_g2o("test.g2o")

        # Unsupported edges
        if isinstance(self.g._vertices[0].pose, (PoseR2, PoseR3)):
            with self.assertRaises(NotImplementedError):
                print(self.g._edges[0].to_g2o())

            with patch("graphslam.graph.open", mock_open()):
                with self.assertRaises(NotImplementedError):
                    self.g.to_g2o("test.g2o")

    def test_plot(self):
        """Test that the ``plot`` method does not raise an exception."""
        # avoid showing the plots
        with patch("graphslam.graph.plt.show"):
            self.g.plot(title="Title")

    def test_equals(self):
        """Test that the ``equals`` method returns False when comparing to an empty graph."""
        g = Graph([], [])
        self.assertFalse(g.equals(self.g))


class TestGraphR3(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`\mathbb{R}^3` poses."""

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`\mathbb{R}^3`."""
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
    r"""Tests for the ``Graph`` class with :math:`SE(2)` poses."""

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`SE(2)`."""
        np.random.seed(0)

        p1 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p2 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p3 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        estimate = PoseSE2([0, 0], 0.0)

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(3), estimate, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(3), estimate, [v3, v2])

        self.g = Graph([e1, e2], [v1, v2, v3])


class TestGraphSE3(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`SE(3)` poses."""

    def setUp(self):
        r"""Setup a simple ``Graph`` in :math:`SE(3)`."""
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


class TestGraphR2SE2(unittest.TestCase):
    r"""Test optimizing a graph with :math:`\mathbb{R}^2` poses and :math:`SE(2)` vertices."""

    def test_optimize(self):
        """Test that optimization works."""
        np.random.seed(0)

        # R^2 poses and edges
        p1 = PoseR2(np.random.random_sample(2))
        p2 = PoseR2(np.random.random_sample(2))
        p3 = PoseR2(np.random.random_sample(2))
        estimate_r2 = PoseR2([0, 0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(2), estimate_r2, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(2), estimate_r2, [v3, v2])

        # SE(2) poses and edges
        p4 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p5 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        p6 = PoseSE2(np.random.random_sample(2), np.random.random_sample())
        estimate_se2 = PoseSE2([0, 0], 0.0)

        v4 = Vertex(4, p4)
        v5 = Vertex(5, p5)
        v6 = Vertex(6, p6)

        e3 = EdgeOdometry([4, 5], np.eye(3), estimate_se2, [v4, v5])
        e4 = EdgeOdometry([6, 5], 2 * np.eye(3), estimate_se2, [v6, v5])

        v1.fixed = True
        v4.fixed = True

        g = Graph([e1, e2, e3, e4], [v1, v2, v3, v4, v5, v6])

        g.optimize()


class TestGraphR3SE3(TestGraphR2):
    r"""Tests for the ``Graph`` class with :math:`\mathbb{R}^3` poses and :math:`SE(3)` vertices."""

    def test_optimize(self):
        """Test that optimization works."""
        np.random.seed(0)

        # R^3 poses and edges
        p1 = PoseR3(np.random.random_sample(3))
        p2 = PoseR3(np.random.random_sample(3))
        p3 = PoseR3(np.random.random_sample(3))
        estimate_r3 = PoseR3([0, 0, 0])

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOdometry([1, 2], np.eye(3), estimate_r3, [v1, v2])
        e2 = EdgeOdometry([3, 2], 2 * np.eye(3), estimate_r3, [v3, v2])

        p4 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        p5 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        p6 = PoseSE3(np.random.random_sample(3), np.random.random_sample(4))
        estimate = PoseSE3([0, 0, 0], [0, 0, 0, 1])

        p4.normalize()
        p5.normalize()
        p6.normalize()

        v4 = Vertex(4, p4)
        v5 = Vertex(5, p5)
        v6 = Vertex(6, p6)

        e3 = EdgeOdometry([4, 5], np.eye(6), estimate, [v4, v5])
        e4 = EdgeOdometry([6, 5], 2 * np.eye(6), estimate, [v6, v5])

        v1.fixed = True
        v4.fixed = True

        g = Graph([e1, e2, e3, e4], [v1, v2, v3, v4, v5, v6])

        g.optimize()


class TestGraphOptimization(unittest.TestCase):
    """Tests the optimizations for specific graphs."""

    def test_intel(self):
        """Test for optimizing the Intel dataset."""
        intel = os.path.join(os.path.dirname(__file__), "..", "data", "input_INTEL.g2o")

        g = load_g2o(intel)
        result = g.optimize()
        print(result)
        self.assertTrue(result.converged)
        self.assertEqual(result.num_iterations + 1, len(result.iteration_results))

        optimized = os.path.join(os.path.dirname(__file__), "input_INTEL_optimized.g2o")

        # Uncomment this line to write the output file
        # g.to_g2o(optimized)

        g2 = load_g2o(optimized)
        self.assertTrue(g.equals(g2))

    def test_intel_two_iterations(self):
        """Test for optimizing the Intel dataset."""
        intel = os.path.join(os.path.dirname(__file__), "..", "data", "input_INTEL.g2o")

        g = load_g2o(intel)
        result = g.optimize(max_iter=2)
        self.assertFalse(result.converged)
        print(result)

    def test_intel_max_iterations(self):
        """Test for optimizing the Intel dataset."""
        intel = os.path.join(os.path.dirname(__file__), "..", "data", "input_INTEL.g2o")

        g = load_g2o(intel)
        result = g.optimize()
        self.assertTrue(result.converged)
        self.assertEqual(result.num_iterations + 1, len(result.iteration_results))

        g2 = load_g2o(intel)
        result2 = g2.optimize(max_iter=result.num_iterations)
        self.assertTrue(result2.converged)
        self.assertEqual(result2.num_iterations, len(result2.iteration_results))

    def test_parking_garage(self):
        """Test for optimizing the parking garage dataset."""
        parking_garage = os.path.join(os.path.dirname(__file__), "..", "data", "parking-garage.g2o")

        g = load_g2o(parking_garage)
        result = g.optimize()
        print(result)
        self.assertTrue(result.converged)
        self.assertEqual(result.num_iterations + 1, len(result.iteration_results))

        optimized = os.path.join(os.path.dirname(__file__), "parking-garage_optimized.g2o")

        # Uncomment this line to write the output file
        # g.to_g2o(optimized)

        g2 = load_g2o(optimized)
        self.assertTrue(g.equals(g2))


if __name__ == "__main__":
    unittest.main()
