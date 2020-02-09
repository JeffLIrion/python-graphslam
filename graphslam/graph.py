"""A ``Graph`` class that stores the edges and vertices required for Graph SLAM.

"""


class Graph(object):
    """A graph that will be optimized via Graph SLAM.

    Parameters
    ----------
    edges : list
        TODO
    vertices : list
        TODO

    Attributes
    ----------
    TODO
        TODO

    """
    def __init__(self):
        self.data = 5

    def optimize(self):
        r"""Optimize the :math:`\chi^2` error for the ``Graph``.

        """
