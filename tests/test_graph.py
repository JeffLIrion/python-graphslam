"""Unit tests for the graph.py module.

"""


import unittest

import numpy as np

from graphslam.graph import Graph
from graphslam.pose.r3 import PoseR3
from graphslam.vertex import Vertex

from .edge_oplus_ominus import EdgeOMinus


class TestGraph(unittest.TestCase):
    """Tests for the ``Graph`` class.

    """

    def setUp(self):
        """Setup a simple ``Graph``.

        """
        np.random.seed(0)

        p1 = PoseR3(np.random.random_sample(3))
        p2 = PoseR3(np.random.random_sample(3))
        p3 = PoseR3(np.random.random_sample(3))

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOMinus([1, 2], np.eye(3), np.zeros(3), [v1, v2])
        e2 = EdgeOMinus([3, 2], 2 * np.eye(3), np.zeros(3), [v1, v3])

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


if __name__ == '__main__':
    unittest.main()
