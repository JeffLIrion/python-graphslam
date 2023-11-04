# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the graph.py module.

"""


import unittest

from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


class TestEdgeOdometry(unittest.TestCase):
    """Tests for the ``EdgeOdometry`` class."""

    def test_plot(self):
        """Test that the ``plot`` method is not implemented."""
        v_none = Vertex(0, None)
        v_r2 = Vertex(1, PoseR2([1, 2]))
        v_se2 = Vertex(2, PoseSE2([1, 2], 3))
        v_r3 = Vertex(3, PoseR3([1, 2, 3]))
        v_se3 = Vertex(4, PoseSE3([1, 2, 3], [0.5, 0.5, 0.5, 0.5]))

        with self.assertRaises(NotImplementedError):
            e = EdgeOdometry(0, 1, 0, [v_none, v_none])
            e.plot()

        for v in [v_r2, v_se2, v_r3, v_se3]:
            e = EdgeOdometry(0, 1, 0, [v, v])
            e.plot()


if __name__ == "__main__":
    unittest.main()
