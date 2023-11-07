# Copyright (c) 2020 Jeff Irion and contributors

"""Functions for loading graphs.

"""


import logging

import numpy as np

from .edge.edge_odometry import EdgeOdometry
from .graph import Graph
from .pose.r2 import PoseR2
from .pose.r3 import PoseR3
from .pose.se2 import PoseSE2
from .pose.se3 import PoseSE3
from .util import upper_triangular_matrix_to_full_matrix
from .vertex import Vertex


_LOGGER = logging.getLogger(__name__)


def load_g2o(infile):
    r"""Load a graph from a .g2o file.

    Parameters
    ----------
    infile : str
        The path to the .g2o file

    Returns
    -------
    Graph
        The loaded graph

    """
    edges = []
    vertices = []

    with open(infile) as f:
        for line in f.readlines():
            # R^2
            if line.startswith("VERTEX_XY "):
                numbers = line[len("VERTEX_XY "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
                p = PoseR2(arr)
                v = Vertex(int(numbers[0]), p)
                vertices.append(v)
                continue

            # R^3
            if line.startswith("VERTEX_TRACKXYZ "):
                numbers = line[len("VERTEX_TRACKXYZ "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
                p = PoseR3(arr)
                v = Vertex(int(numbers[0]), p)
                vertices.append(v)
                continue

            # SE(2)
            if line.startswith("VERTEX_SE2 "):
                numbers = line[len("VERTEX_SE2 "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
                p = PoseSE2(arr[:2], arr[2])
                v = Vertex(int(numbers[0]), p)
                vertices.append(v)
                continue

            if line.startswith("EDGE_SE2 "):
                numbers = line[len("EDGE_SE2 "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
                vertex_ids = [int(numbers[0]), int(numbers[1])]
                estimate = PoseSE2(arr[:2], arr[2])
                information = upper_triangular_matrix_to_full_matrix(arr[3:], 3)
                e = EdgeOdometry(vertex_ids, information, estimate)
                edges.append(e)
                continue

            # SE(3)
            if line.startswith("VERTEX_SE3:QUAT "):
                numbers = line[len("VERTEX_SE3:QUAT "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
                p = PoseSE3(arr[:3], arr[3:])
                v = Vertex(int(numbers[0]), p)
                vertices.append(v)
                continue

            if line.startswith("EDGE_SE3:QUAT "):
                numbers = line[len("EDGE_SE3:QUAT "):].split()  # fmt: skip
                arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
                vertex_ids = [int(numbers[0]), int(numbers[1])]
                estimate = PoseSE3(arr[:3], arr[3:7])
                estimate.normalize()
                information = upper_triangular_matrix_to_full_matrix(arr[7:], 6)
                e = EdgeOdometry(vertex_ids, information, estimate)
                edges.append(e)
                continue

            if line.strip():
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
