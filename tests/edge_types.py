# Copyright (c) 2020 Jeff Irion and contributors

"""Edge types used for unit tests.

"""


import numpy as np

from graphslam.edge.base_edge import BaseEdge


class BaseEdgeForTests(BaseEdge):
    """A base edge class for tests."""

    def to_g2o(self):
        """Not supported, so don't do anything."""

    @classmethod
    def from_g2o(cls, line):
        """Not supported, so return ``None``."""
        return None

    def plot(self, color=""):
        """Not supported, so don't do anything."""


class EdgeOPlus(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose + self.vertices[1].pose).to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        # fmt: off
        return [np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_self(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_other(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on


class EdgeOMinus(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose - self.vertices[1].pose).to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        # fmt: off
        return [np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_self(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_other(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on


class EdgeOPlusCompact(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose + self.vertices[1].pose).to_compact()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        # fmt: off
        return [np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_self_compact(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_oplus_other_wrt_other_compact(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on


class EdgeOMinusCompact(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose - self.vertices[1].pose).to_compact()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        # fmt: off
        return [np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_self_compact(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on


class EdgeOPlusPoint(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return (self.vertices[0].pose + self.vertices[1].pose).to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        # fmt: off
        return [np.dot(self.vertices[0].pose.jacobian_self_oplus_point_wrt_self(self.vertices[1].pose), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(self.vertices[0].pose.jacobian_self_oplus_point_wrt_point(self.vertices[1].pose), self.vertices[1].pose.jacobian_boxplus())]
        # fmt: on


class EdgeInverse(BaseEdgeForTests):
    """A simple edge class for testing."""

    def calc_error(self):
        """A simple "error" method."""
        return self.vertices[0].pose.inverse.to_array()

    def calc_jacobians(self):
        """Calculate the Jacobians."""
        return [np.dot(self.vertices[0].pose.jacobian_inverse(), self.vertices[0].pose.jacobian_boxplus())]  # fmt: skip
