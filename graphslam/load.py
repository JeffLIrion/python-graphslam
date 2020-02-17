# Copyright (c) 2020 Jeff Irion and contributors

"""Functions for loading graphs.

"""


import numpy as np

from .edge.edge_odometry import EdgeOdometry
from .graph import Graph
from .pose.se2 import PoseSE2
from .util import solve_for_edge_dimensionality, upper_triangular_matrix_to_full_matrix
from .vertex import Vertex


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
    edges = []
    vertices = []

    with open(infile) as f:
        for line in f.readlines():
            if line.startswith("VERTEX_SE2"):
                numbers = line[10:].split()
                arr = np.array([float(number) for number in numbers[1:]], dtype=np.float64)
                p = PoseSE2(arr[:2], arr[2])
                v = Vertex(int(numbers[0]), p)
                vertices.append(v)

            elif line.startswith("EDGE_SE2"):
                numbers = line[9:].split()
                arr = np.array([float(number) for number in numbers[2:]], dtype=np.float64)
                n = solve_for_edge_dimensionality(len(arr) - 2)
                vertex_ids = [int(numbers[0]), int(numbers[1])]
                estimate = arr[:n]
                information = upper_triangular_matrix_to_full_matrix(arr[n:], n)
                e = EdgeOdometry(vertex_ids, information, estimate)
                edges.append(e)

    return Graph(edges, vertices)