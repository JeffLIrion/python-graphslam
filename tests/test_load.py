# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the load.py module.

"""


import os
import unittest

from graphslam.load import load_g2o_se2, load_g2o_se3


class TestLoad(unittest.TestCase):
    """Tests for the ``load`` functions.

    """
    def test_load_g2o_se2(self):
        """Test the ``load_g2o_se2()`` function.

        """
        infile = os.path.join(os.path.dirname(__file__), 'test_se2.g2o')
        g = load_g2o_se2(infile)

        self.assertGreater(g.calc_chi2(), 0.)

    def test_load_g2o_se3(self):
        """Test the ``load_g2o_se3()`` function.

        """
        infile = os.path.join(os.path.dirname(__file__), 'test_se3.g2o')
        g = load_g2o_se3(infile)

        self.assertGreater(g.calc_chi2(), 0.)
