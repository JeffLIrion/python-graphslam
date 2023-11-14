# Copyright (c) 2020 Jeff Irion and contributors

"""Edge types used for unit tests.

"""


import numpy as np

from graphslam.edge.base_edge import BaseEdge
from graphslam.util import upper_triangular_matrix_to_full_matrix


class BaseEdgeForTests(BaseEdge):
    """A base edge class for tests."""

    def to_g2o(self):
        """Not supported, so return ``None``."""
        return None

    @classmethod
    def from_g2o(cls, line, g2o_params_or_none=None):
        """Not supported, so return ``None``."""
        return None

    def plot(self, color=""):
        """Not supported, so don't do anything."""


class EdgeWithoutToG2OWithoutFromG2O(BaseEdgeForTests):
    """An edge class without ``to_g2o`` and ``from_g2o`` support.

    This class is only compatible with ``PoseR2`` poses.

    """

    def calc_error(self):
        """Return an error vector."""
        return np.array([1.0, 2.0])


class EdgeWithToG2OWithoutFromG2O(EdgeWithoutToG2OWithoutFromG2O):
    """An edge class with a ``to_g2o`` method but not a ``from_g2o`` method."""

    def to_g2o(self):
        """Write to g2o format."""
        # fmt: off
        return "TestEdge {} {} {} ".format(self.vertex_ids[0], self.estimate[0], self.estimate[1]) + " ".join([str(x) for x in self.information[np.triu_indices(2, 0)]]) + "\n"
        # fmt: on


class EdgeWithoutToG2OWithFromG2O(EdgeWithoutToG2OWithoutFromG2O):
    """An edge class with a ``from_g2o`` method but not a ``to_g2o`` method."""

    @classmethod
    def from_g2o(cls, line, g2o_params_or_none=None):
        """Write to g2o format."""
        if line.startswith("TestEdge "):
            numbers = line[len("TestEdge "):].split()  # fmt: skip
            arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
            vertex_ids = [int(numbers[0])]
            estimate = arr[:2]
            information = upper_triangular_matrix_to_full_matrix(arr[2:], 2)
            return cls(vertex_ids, information, estimate)

        return None


class EdgeWithToG2OWithFromG2O(EdgeWithToG2OWithoutFromG2O, EdgeWithoutToG2OWithFromG2O):
    """An edge class with ``to_g2o`` and ``from_g2o`` methods."""


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
