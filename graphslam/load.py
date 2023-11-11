# Copyright (c) 2020 Jeff Irion and contributors

"""Functions for loading graphs.

"""


import logging

from .graph import Graph


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
    _LOGGER.warning("load_g2o is deprecated; use Graph.load_g2o instead")
    return Graph.load_g2o(infile)


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
    _LOGGER.warning("load_g2o_r2 is deprecated; use Graphload_g2o instead")
    return Graph.load_g2o(infile)


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
    _LOGGER.warning("load_g2o_r3 is deprecated; use Graph.load_g2o instead")
    return Graph.load_g2o(infile)


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
    _LOGGER.warning("load_g2o_se2 is deprecated; use Graph.load_g2o instead")
    return Graph.load_g2o(infile)


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
    _LOGGER.warning("load_g2o_se3 is deprecated; use Graph.load_g2o instead")
    return Graph.load_g2o(infile)
