"""Unit tests for the graph.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.pose.se2 import PoseSE2


class TestVertex(unittest.TestCase):
    """Tests for the ``Vetex`` class.

    """

    def test_constructor(self):
        """Test that a ``Vertex`` object can be created.

        """
        v = Vertex(1, PoseSE2([1, 2], 3))

        self.assertEqual(v.id, 1)
        self.assertAlmostEqual(np.linalg.norm(v.pose.to_array() - np.array([1., 2., 3.])), 0.)


if __name__ == '__main__':
    unittest.main()
