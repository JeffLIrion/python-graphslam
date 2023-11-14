# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the g2o parameters classes.

"""


import unittest

from graphslam.g2o_parameters import G2OParameterSE2Offset, G2OParameterSE3Offset
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


class TestG2OParameterSE2Offset(unittest.TestCase):
    """Tests for the ``G2OParameterSE2Offset`` class."""

    def test_to_g2o_from_g2o(self):
        """Test the `to_g2o` and `from_g2o` methods."""
        param = G2OParameterSE2Offset(("PARAMS_SE2OFFSET", 2), PoseSE2([1.0, 2.0], 3.0))

        param2 = G2OParameterSE2Offset.from_g2o(param.to_g2o())

        self.assertTupleEqual(param.key, param2.key)
        self.assertTrue(param.value.equals(param2.value))

        self.assertIsNone(G2OParameterSE2Offset.from_g2o("bologna"))


class TestG2OParameterSE3Offset(unittest.TestCase):
    """Tests for the ``G2OParameterSE3Offset`` class."""

    def test_to_g2o_from_g2o(self):
        """Test the `to_g2o` and `from_g2o` methods."""
        param = G2OParameterSE3Offset(("PARAMS_SE3OFFSET", 4), PoseSE3([1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4]))
        param.value.normalize()

        param2 = G2OParameterSE3Offset.from_g2o(param.to_g2o())

        self.assertTupleEqual(param.key, param2.key)
        self.assertTrue(param.value.equals(param2.value))

        self.assertIsNone(G2OParameterSE3Offset.from_g2o("bologna"))
