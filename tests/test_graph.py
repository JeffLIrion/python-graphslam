"""Unit tests for the graph.py module.

"""


import unittest

import numpy as np

from graphslam.graph import Graph
from graphslam.pose.r3 import PoseR3
from graphslam.vertex import Vertex

from .edge_oplus_ominus import EdgeOPlus


class TestGraph(unittest.TestCase):
    """Tests for the ``Graph`` class.

    """

    def test_calc_chi2(self):
        r"""Test that the :math:`\chi^2` for a ``Graph`` can be computed.

        """
        np.random.seed(0)

        p1 = PoseR3(np.random.random_sample(3))
        p2 = PoseR3(np.random.random_sample(3))
        p3 = PoseR3(np.random.random_sample(3))

        v1 = Vertex(1, p1)
        v2 = Vertex(2, p2)
        v3 = Vertex(3, p3)

        e1 = EdgeOPlus([1, 2], np.eye(3), np.zeros(3), [v1, v2])
        e2 = EdgeOPlus([2, 3], np.eye(3), np.zeros(3), [v2, v3])

        chi2 = e1.calc_chi2() + e2.calc_chi2()

        g = Graph([e1, e2], [v1, v2, v3])

        self.assertAlmostEqual(chi2, g.calc_chi2())

    def test_optimize(self):
        """Test that a ``Graph`` can be optimized.

        """
        g = Graph([], [])
        g.optimize()
        self.assertIsNotNone(g)


if __name__ == '__main__':
    unittest.main()
