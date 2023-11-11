# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the load.py module.

"""


import os
import unittest
from unittest import mock

import numpy as np

from graphslam.graph import Graph
from graphslam.load import load_g2o, load_g2o_r2, load_g2o_r3, load_g2o_se2, load_g2o_se3
from graphslam.pose.r2 import PoseR2
from graphslam.util import upper_triangular_matrix_to_full_matrix
from graphslam.vertex import Vertex

from .edge_types import BaseEdgeForTests
from .patchers import FAKE_FILE, open_fake_file


class EdgeWithoutToG2OWithoutFromG2O(BaseEdgeForTests):
    """An edge class without ``to_g2o`` and ``from_g2o`` support.

    This class is only compatible with ``PoseR2`` poses.

    """

    def calc_error(self):
        """Return an error vector."""
        return np.array([1.0, 2.0])


class EdgeWithToG2OWithoutFromG2O(EdgeWithoutToG2OWithoutFromG2O):
    """An edge class with a ``to_g2o`` method but not a ``from_g2o`` method."""

    def to_g2o(self):
        """Write to g2o format."""
        # fmt: off
        return "TestEdge {} {} {} ".format(self.vertex_ids[0], self.estimate[0], self.estimate[1]) + " ".join([str(x) for x in self.information[np.triu_indices(2, 0)]]) + "\n"
        # fmt: on


class EdgeWithoutToG2OWithFromG2O(EdgeWithoutToG2OWithoutFromG2O):
    """An edge class with a ``from_g2o`` method but not a ``to_g2o`` method."""

    @classmethod
    def from_g2o(cls, line):
        """Write to g2o format."""
        if line.startswith("TestEdge "):
            numbers = line[len("TestEdge "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
            vertex_ids = [int(numbers[0])]
            estimate = arr[:2]
            information = upper_triangular_matrix_to_full_matrix(arr[2:], 2)
            return cls(vertex_ids, information, estimate)

        return None


class EdgeWithToG2OWithFromG2O(EdgeWithToG2OWithoutFromG2O, EdgeWithoutToG2OWithFromG2O):
    """An edge class with ``to_g2o`` and ``from_g2o`` methods."""


class TestLoad(unittest.TestCase):
    """Tests for the ``load`` functions."""

    def setUp(self):
        """Clear ``FAKE_FILE``."""
        FAKE_FILE.clear()

    def test_load_g2o_r2(self):
        """Test the ``load_g2o_r2()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_r2.g2o")
        g = load_g2o_r2(infile)
        chi2 = g.calc_chi2()

        # There are currently no edges in this graph
        self.assertEqual(chi2, 0.0)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o_r2("test.g2o")
            self.assertAlmostEqual(chi2, g2.calc_chi2())

    def test_load_g2o_r3(self):
        """Test the ``load_g2o_r3()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_r3.g2o")
        g = load_g2o_r3(infile)
        chi2 = g.calc_chi2()

        # There are currently no edges in this graph
        self.assertEqual(chi2, 0.0)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o_r3("test.g2o")
            self.assertAlmostEqual(chi2, g2.calc_chi2())

    def test_load_g2o_se2(self):
        """Test the ``load_g2o_se2()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_se2.g2o")
        g = load_g2o_se2(infile)
        chi2 = g.calc_chi2()

        self.assertGreater(chi2, 0.0)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o_se2("test.g2o")
            self.assertAlmostEqual(chi2, g2.calc_chi2())

    def test_load_g2o_se3(self):
        """Test the ``load_g2o_se3()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_se3.g2o")
        g = load_g2o_se3(infile)
        chi2 = g.calc_chi2()

        self.assertGreater(chi2, 0.0)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o_se3("test.g2o")
            self.assertAlmostEqual(chi2, g2.calc_chi2())

    def test_load_custom_edge_without_to_g2o_without_from_g2o(self):
        """Test that loading a graph with an edge type that does not have ``to_g2o`` and ``from_g2o`` methods works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithoutToG2OWithoutFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o("test.g2o")
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_with_to_g2o_without_from_g2o(self):
        """Test that loading a graph with an edge type that has a ``to_g2o`` method but not a ``from_g2o`` method works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithToG2OWithoutFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o("test.g2o", [EdgeWithToG2OWithoutFromG2O])
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_without_to_g2o_with_from_g2o(self):
        """Test that loading a graph with an edge type that has a ``from_g2o`` method but not a ``to_g2o`` method works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithoutToG2OWithFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o("test.g2o", [EdgeWithoutToG2OWithFromG2O])
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_with_to_g2o_with_from_g2o(self):
        """Test that loading a graph with an edge type that has ``to_g2o`` and ``from_g2o`` methods works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithToG2OWithFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.load.open", open_fake_file):
            g2 = load_g2o("test.g2o", [EdgeWithToG2OWithFromG2O])
            self.assertTrue(g.equals(g2))
