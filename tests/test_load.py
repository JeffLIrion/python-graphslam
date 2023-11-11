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
from graphslam.vertex import Vertex

from .edge_types import (
    EdgeWithoutToG2OWithoutFromG2O,
    EdgeWithToG2OWithoutFromG2O,
    EdgeWithoutToG2OWithFromG2O,
    EdgeWithToG2OWithFromG2O,
)
from .patchers import FAKE_FILE, open_fake_file


class TestLoad(unittest.TestCase):
    """Tests for the ``load`` functions."""

    def setUp(self):
        """Clear ``FAKE_FILE``."""
        FAKE_FILE.clear()

    def test_load_g2o(self):
        """Test the ``load_g2o()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_r2.g2o")
        g = load_g2o(infile)
        g2 = load_g2o_r2(infile)
        self.assertTrue(g.equals(g2))

    def test_load_g2o_r2(self):
        """Test the ``load_g2o_r2()`` function."""
        infile = os.path.join(os.path.dirname(__file__), "test_r2.g2o")
        g = load_g2o_r2(infile)
        chi2 = g.calc_chi2()

        # There are currently no edges in this graph
        self.assertEqual(chi2, 0.0)

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.graph.open", open_fake_file):
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

        with mock.patch("graphslam.graph.open", open_fake_file):
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

        with mock.patch("graphslam.graph.open", open_fake_file):
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

        with mock.patch("graphslam.graph.open", open_fake_file):
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

        with mock.patch("graphslam.graph.open", open_fake_file):
            g2 = Graph.load_g2o("test.g2o", [EdgeWithoutToG2OWithFromG2O])
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_with_to_g2o_without_from_g2o(self):
        """Test that loading a graph with an edge type that has a ``to_g2o`` method but not a ``from_g2o`` method works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithToG2OWithoutFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.graph.open", open_fake_file):
            g2 = Graph.load_g2o("test.g2o", [EdgeWithToG2OWithoutFromG2O])
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_without_to_g2o_with_from_g2o(self):
        """Test that loading a graph with an edge type that has a ``from_g2o`` method but not a ``to_g2o`` method works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithoutToG2OWithFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.graph.open", open_fake_file):
            g2 = Graph.load_g2o("test.g2o", [EdgeWithoutToG2OWithFromG2O])
            self.assertFalse(g.equals(g2))

    def test_load_custom_edge_with_to_g2o_with_from_g2o(self):
        """Test that loading a graph with an edge type that has ``to_g2o`` and ``from_g2o`` methods works as expected."""
        p0 = PoseR2([1.0, 2.0])
        v0 = Vertex(0, p0)

        e = EdgeWithToG2OWithFromG2O([0], np.eye(2), np.ones(2))

        g = Graph([e], [v0])

        with mock.patch("graphslam.graph.open", open_fake_file):
            g.to_g2o("test.g2o")

        with mock.patch("graphslam.graph.open", open_fake_file):
            g2 = Graph.load_g2o("test.g2o", [EdgeWithToG2OWithFromG2O])
            self.assertTrue(g.equals(g2))
