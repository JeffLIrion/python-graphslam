# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the ``Vertex`` class.

"""


import unittest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa pylint: disable=unused-import

from graphslam.vertex import Vertex
from graphslam.pose.r2 import PoseR2
from graphslam.pose.r3 import PoseR3
from graphslam.pose.se2 import PoseSE2
from graphslam.pose.se3 import PoseSE3


class TestVertex(unittest.TestCase):
    """Tests for the ``Vertex`` class."""

    def test_constructor(self):
        """Test that a ``Vertex`` object can be created."""
        v = Vertex(1, PoseSE2([1, 2], 3))

        self.assertEqual(v.id, 1)
        self.assertAlmostEqual(np.linalg.norm(v.pose.to_array() - np.array([1.0, 2.0, 3.0])), 0.0)

    def test_plot(self):
        """Test that a ``Vertex`` can be plotted."""
        v_none = Vertex(0, None)
        v_r2 = Vertex(1, PoseR2([1, 2]))
        v_se2 = Vertex(2, PoseSE2([1, 2], 3))
        v_r3 = Vertex(3, PoseR3([1, 2, 3]))
        v_se3 = Vertex(4, PoseSE3([1, 2, 3], [0.5, 0.5, 0.5, 0.5]))

        with self.assertRaises(NotImplementedError):
            v_none.plot()

        for v in [v_r2, v_se2, v_r3, v_se3]:
            fig = plt.figure()
            if len(v.pose.position) == 3:
                fig.add_subplot(111, projection="3d")
            v.plot()

    def test_equals(self):
        """Test that the ``equals`` method works correctly."""
        v0 = Vertex(1, PoseR2([1, 2]))

        v1 = Vertex(1, PoseR2([1, 2]))
        self.assertTrue(v0.equals(v1))

        v1.id = 2
        self.assertFalse(v0.equals(v1))

        v2 = Vertex(1, PoseR3([1, 2, 3]))
        self.assertFalse(v0.equals(v2))

    def test_to_g2o_from_g2o(self):
        """Test that the ``to_g2o`` and ``from_g2o`` methods work correctly."""
        v_r2 = Vertex(1, PoseR2([1, 2]))
        v_se2 = Vertex(2, PoseSE2([1, 2], 3))
        v_r3 = Vertex(3, PoseR3([1, 2, 3]))
        v_se3 = Vertex(4, PoseSE3([1, 2, 3], [0.5, 0.5, 0.5, 0.5]))

        for v in [v_r2, v_se2, v_r3, v_se3]:
            self.assertTrue(v.equals(Vertex.from_g2o(v.to_g2o())))

    def test_to_g2o_unsupported_type(self):
        """Test that an unsupported pose type cannot be written to a .g2o file."""
        # Use `None` in lieue of an actual unsupported pose
        v = Vertex(0, None)

        with self.assertRaises(NotImplementedError):
            v.to_g2o()


if __name__ == "__main__":
    unittest.main()
