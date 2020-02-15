"""Unit tests for the load.py module.

"""


import os
import unittest

from graphslam.load import load_g2o_se2


class TestLoad(unittest.TestCase):
    """Tests for the ``load`` functions.

    """
    def test_load_g2o_se2(self):
        """Test the ``load_g2o_se2()`` function.

        """
        infile = os.path.join(os.path.dirname(__file__), 'test_se2.g2o')
        g = load_g2o_se2(infile)

        self.assertGreater(g.calc_chi2(), 0.)
