# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the load.py module.

"""


import os
import unittest
from unittest import mock

from graphslam.load import load_g2o_r2, load_g2o_r3, load_g2o_se2, load_g2o_se3

from .patchers import FAKE_FILE, open_fake_file


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
