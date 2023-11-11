# Copyright (c) 2020 Jeff Irion and contributors

"""Functions for loading graphs.

"""


import logging

from .edge.base_edge import BaseEdge
from .edge.edge_odometry import EdgeOdometry
from .graph import Graph
from .vertex import Vertex


_LOGGER = logging.getLogger(__name__)


def load_g2o(infile, custom_edge_types=None):
    r"""Load a graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file
    custom_edge_types : list[type], None
        A list of custom edge types, which must be subclasses of ``BaseEdge``

    Returns
    -------
    Graph
        The loaded graph

    """
    edges = []
    vertices = []

    custom_edge_types = custom_edge_types or []
    for edge_type in custom_edge_types:
        assert issubclass(edge_type, BaseEdge)

    def custom_edge_from_g2o(line, custom_edge_types):
        """Load a custom edge from a .g2o line.

        Parameters
        ----------
        line : str
            A line from a .g2o file
        custom_edge_types : list[type]
            A list of custom edge types, which must be subclasses of ``BaseEdge``

        Returns
        -------
        BaseEdge, None
            The instantiated edge object, or ``None`` if the line does not correspond to any of the custom edge types

        """
        for custom_edge_type in custom_edge_types:
            edge_or_none = custom_edge_type.from_g2o(line)
            if edge_or_none:
                return edge_or_none

        return None

    with open(infile) as f:
        for line in f.readlines():
            if line.strip():
                # Vertex
                vertex_or_none = Vertex.from_g2o(line)
                if vertex_or_none:
                    vertices.append(vertex_or_none)
                    continue

                # Custom edge types
                custom_edge_or_none = custom_edge_from_g2o(line, custom_edge_types)
                if custom_edge_or_none:
                    edges.append(custom_edge_or_none)
                    continue

                # Odometry Edge
                edge_or_none = EdgeOdometry.from_g2o(line)
                if edge_or_none:
                    edges.append(edge_or_none)
                    continue

                _LOGGER.warning("Line not supported -- '%s'", line.rstrip())

    return Graph(edges, vertices)


def load_g2o_r2(infile):
    r"""Load an :math:`\mathbb{R}^2` graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file

    Returns
    -------
    Graph
        The loaded graph

    """
    _LOGGER.warning("load_g2o_r2 is deprecated; use load_g2o instead")
    return load_g2o(infile)


def load_g2o_r3(infile):
    r"""Load an :math:`\mathbb{R}^3` graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file

    Returns
    -------
    Graph
        The loaded graph

    """
    _LOGGER.warning("load_g2o_r3 is deprecated; use load_g2o instead")
    return load_g2o(infile)


def load_g2o_se2(infile):
    """Load an :math:`SE(2)` graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file

    Returns
    -------
    Graph
        The loaded graph

    """
    _LOGGER.warning("load_g2o_se2 is deprecated; use load_g2o instead")
    return load_g2o(infile)


def load_g2o_se3(infile):
    """Load an :math:`SE(3)` graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file

    Returns
    -------
    Graph
        The loaded graph

    """
    _LOGGER.warning("load_g2o_se3 is deprecated; use load_g2o instead")
    return load_g2o(infile)
