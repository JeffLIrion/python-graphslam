"""Unit tests for the graph.py module.

"""


import numpy as np

from graphslam.edge.base_edge import BaseEdge


class EdgeOPlus(BaseEdge):
    """A simple edge class for testing.

    """
    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose + self.vertices[1].pose).to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        return [np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_self(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_other(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]


class EdgeOMinus(BaseEdge):
    """A simple edge class for testing.

    """
    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose - self.vertices[1].pose).to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        return [np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_self(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_other(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
