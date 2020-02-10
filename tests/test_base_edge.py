"""Unit tests for the graph.py module.

"""


import unittest

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.pose.se2 import PoseSE2


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
        e = BaseEdge([v], 1)

        self.assertEqual(e.vertices[0].vertex_id, 0)
        self.assertEqual(e.information, 1)

    def test_calc_error(self):
        """Test that the ``calc_error`` method is not implemented.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge([v], 1)

        with self.assertRaises(NotImplementedError):
            _ = e.calc_error()

    def test_calc_chi2(self):
        """Test that the ``calc_chi2`` method works as expected.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = SimpleEdge([v], 1)

        self.assertEqual(e.calc_chi2(), 1)


if __name__ == '__main__':
    unittest.main()
