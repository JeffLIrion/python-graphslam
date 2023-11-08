# Copyright (c) 2020 Jeff Irion and contributors

r"""A ``Graph`` class that stores the edges and vertices required for Graph SLAM.

Given:

* A set of :math:`M` edges (i.e., constraints) :math:`\mathcal{E}`

  * :math:`e_j \in \mathcal{E}` is an edge
  * :math:`\mathbf{e}_j \in \mathbb{R}^\bullet` is the error associated with that edge, where :math:`\bullet` is a scalar that depends on the type of edge
  * :math:`\Omega_j` is the :math:`\bullet \times \bullet` information matrix associated with edge :math:`e_j`

* A set of :math:`N` vertices :math:`\mathcal{V}`

  * :math:`v_i \in \mathcal{V}` is a vertex
  * :math:`\mathbf{x}_i \in \mathbb{R}^c` is the compact pose associated with :math:`v_i`
  * :math:`\boxplus` is the pose composition operator that yields a (non-compact) pose that lies in (a subspace of) :math:`\mathbb{R}^d`

We want to optimize

.. math::

   \chi^2 = \sum_{e_j \in \mathcal{E}} \mathbf{e}_j^T \Omega_j \mathbf{e}_j.

Let

.. math::

   \mathbf{x} := \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \\ \vdots \\ \mathbf{x}_N \end{bmatrix} \in \mathbb{R}^{cN}.

We will solve this optimization problem iteratively.  Let

.. math::

   \mathbf{x}^{k+1} := \mathbf{x}^k \boxplus \Delta \mathbf{x}^k.

The :math:`\chi^2` error at iteration :math:`k+1` is

.. math::

   \chi_{k+1}^2 = \sum_{e_j \in \mathcal{E}} \underbrace{\left[ \mathbf{e}_j(\mathbf{x}^{k+1}) \right]^T}_{1 \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\mathbf{e}_j(\mathbf{x}^{k+1})}_{\bullet \times 1}.

We will linearize the errors as:

.. math::

   \mathbf{e}_j(\mathbf{x}^{k+1}) &= \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \\
   &\approx \mathbf{e}_j(\mathbf{x}^{k}) + \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right] \Delta \mathbf{x}^k \\
   &= \mathbf{e}_j(\mathbf{x}^{k}) + \left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right) \frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k} \Delta \mathbf{x}^k.

Plugging this into the formula for :math:`\chi^2`, we get:

.. math::

   \chi_{k+1}^2 &\approx \ \ \ \ \ \sum_{e_j \in \mathcal{E}} \underbrace{[ \mathbf{e}_j(\mathbf{x}^k)]^T}_{1 \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\mathbf{e}_j(\mathbf{x}^k)}_{\bullet \times 1} \\
   &\hphantom{\approx} \ \ \ + \sum_{e_j \in \mathcal{E}} \underbrace{[ \mathbf{e}_j(\mathbf{x^k}) ]^T }_{1 \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)}_{\bullet \times dN} \underbrace{\frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k}}_{dN \times cN} \underbrace{\Delta \mathbf{x}^k}_{cN \times 1} \\
   &\hphantom{\approx} \ \ \ + \sum_{e_j \in \mathcal{E}} \underbrace{(\Delta \mathbf{x}^k)^T}_{1 \times cN} \underbrace{ \left( \frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k} \right)^T}_{cN \times dN} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)^T}_{dN \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)}_{\bullet \times dN} \underbrace{\frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k}}_{dN \times cN} \underbrace{\Delta \mathbf{x}^k}_{cN \times 1} \\
   &= \chi_k^2 + 2 \mathbf{b}^T \Delta \mathbf{x}^k + (\Delta \mathbf{x}^k)^T H \Delta \mathbf{x}^k,

where

.. math::

   \mathbf{b}^T &= \sum_{e_j \in \mathcal{E}} \underbrace{[ \mathbf{e}_j(\mathbf{x^k}) ]^T }_{1 \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)}_{\bullet \times dN} \underbrace{\frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k}}_{dN \times cN} \\
   H &= \sum_{e_j \in \mathcal{E}} \underbrace{ \left( \frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k} \right)^T}_{cN \times dN} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)^T}_{dN \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\left( \left. \frac{\partial \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)} \right|_{\Delta \mathbf{x}^k = \mathbf{0}} \right)}_{\bullet \times dN} \underbrace{\frac{\partial (\mathbf{x}^k \boxplus \Delta \mathbf{x}^k)}{\partial \Delta \mathbf{x}^k}}_{dN \times cN}.

Using this notation, we obtain the optimal update as

.. math::

   \Delta \mathbf{x}^k = -H^{-1} \mathbf{b}.

We apply this update to the poses and repeat until convergence.

"""


from collections import defaultdict
from functools import reduce
import time
import warnings

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, lil_matrix
from scipy.sparse.linalg import spsolve

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa pylint: disable=unused-import
except ImportError:  # pragma: no cover
    plt = None


warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


# pylint: disable=too-few-public-methods
class OptimizationResult:
    r"""A class for storing information about a graph optimization; see `Graph.optimize`.

    Attributes
    ----------
    converged : bool
        Whether the optimization converged
    duration_s : float, None
        The total time for the optimization (in seconds)
    final_chi2 : float, None
        The final :math:`\chi^2` error
    initial_chi2 : float, None
        The initial :math:`\chi^2` error
    iteration_results : list[IterationResult]
        Information about each iteration
    num_iterations : int, None
        The number of iterations that were performed

    """

    # pylint: disable=too-few-public-methods
    class IterationResult:
        r"""A class for storing information about a single graph optimization iteration; see `Graph.optimize`.

        Attributes
        ----------
        calc_chi2_gradient_hessian_duration_s : float, None
            The time to compute :math:`\chi^2`, the gradient, and the Hessian (in seconds); see `Graph._calc_chi2_gradient_hessian`
        chi2 : float, None
            The :math:`\chi^2` of the graph after performing this iteration's update
        duration_s : float, None
            The total time for this iteration (in seconds)
        rel_diff : float, None
            The relative difference in the :math:`\chi^2` as a result of this iteration
        solve_duration_s : float, None
            The time to solve :math:`H \Delta \mathbf{x}^k = -\mathbf{b}` (in seconds)
        update_duration_s : float, None
            The time to update the poses (in seconds)

        """

        def __init__(self):
            self.calc_chi2_gradient_hessian_duration_s = None
            self.chi2 = None
            self.duration_s = None
            self.rel_diff = None
            self.solve_duration_s = None
            self.update_duration_s = None

        def is_complete_iteration(self):
            r"""Whether this was a full iteration.

            At iteration ``i``, we compute the :math:`\chi^2` error for iteration ``i-1`` (see `Graph.optimize`).  If
            this meets the convergence criteria, then we do not solve the linear system and update the poses, and so
            this is not a complete iteration.

            Returns
            -------
            bool
                Whether this was a complete iteration (i.e., we solve the linear system and updated the poses)

            """
            return self.solve_duration_s is not None

    def __init__(self):
        self.converged = False
        self.duration_s = None
        self.final_chi2 = None
        self.initial_chi2 = None
        self.iteration_results = []
        self.num_iterations = None

    def __str__(self):
        """Format the optimization results in a string table.

        Returns
        -------
        str
            The formatted optimization results

        """
        initial_chi2_str = "{:.4f}".format(self.initial_chi2)
        final_chi2_str = "{:.4f}".format(self.final_chi2)
        chi2_str_len = max(len(initial_chi2_str), len(final_chi2_str))

        lines = [
            "Initial chi^2 = {}{}".format(" " * (chi2_str_len - len(initial_chi2_str)), initial_chi2_str),
            "Final chi^2   = {}{}".format(" " * (chi2_str_len - len(final_chi2_str)), final_chi2_str),
            "",
            "Converged = {}".format(self.converged),
            "Iterations = {}".format(self.num_iterations),
            "Duration = {:.3f} s".format(self.duration_s),
            "",
            "Iteration                chi^2        rel. change        duration (s)        calc_chi2_gradient_hessian (s)        solve (s)        update (s)",
            "---------                -----        -----------        ------------        ------------------------------        ---------        ----------",
        ]

        for i, iter_result in enumerate(self.iteration_results):
            if iter_result.is_complete_iteration():
                lines.append(
                    "{:9d} {:20.4f} {:18.6f}        {:12.3f}        {:30.3f}        {:9.3f}        {:10.3f}".format(
                        i + 1,
                        iter_result.chi2,
                        iter_result.rel_diff,
                        iter_result.duration_s,
                        iter_result.calc_chi2_gradient_hessian_duration_s,
                        iter_result.solve_duration_s,
                        iter_result.update_duration_s,
                    )
                )

        return "\n".join(lines)


# pylint: disable=too-few-public-methods
class _Chi2GradientHessian:
    r"""A class that is used to aggregate the :math:`\chi^2` error, gradient, and Hessian.

    Attributes
    ----------
    chi2 : float
        The :math:`\chi^2` error
    gradient : defaultdict
        The contributions to the gradient vector
    hessian : defaultdict
        The contributions to the Hessian matrix

    """

    class DefaultArray:
        """A class for use in a `defaultdict`."""

        def __iadd__(self, other):
            """Add `other` to `self` and return `other`.

            Parameters
            ----------
            other : np.ndarray
                The numpy array that is being added to `self`

            Returns
            -------
            np.ndarray
                `other`

            """
            return other

    def __init__(self):
        self.chi2 = 0.0
        self.gradient = defaultdict(_Chi2GradientHessian.DefaultArray)
        self.hessian = defaultdict(_Chi2GradientHessian.DefaultArray)

    @staticmethod
    def update(chi2_grad_hess, incoming):
        r"""Update the :math:`\chi^2` error and the gradient and Hessian dictionaries.

        Parameters
        ----------
        chi2_grad_hess : _Chi2GradientHessian
            The ``_Chi2GradientHessian`` that will be updated
        incoming : tuple
            The return value from `BaseEdge.calc_chi2_gradient_hessian`

        """
        chi2_grad_hess.chi2 += incoming[0]

        for idx, contrib in incoming[1]:
            chi2_grad_hess.gradient[idx] += contrib

        for (idx1, idx2), contrib in incoming[2]:
            if idx1 <= idx2:
                chi2_grad_hess.hessian[idx1, idx2] += contrib
            else:
                chi2_grad_hess.hessian[idx2, idx1] += np.transpose(contrib)

        return chi2_grad_hess


class Graph(object):
    r"""A graph that will be optimized via Graph SLAM.

    Parameters
    ----------
    edges : list[graphslam.edge.base_edge.BaseEdge]
        A list of the vertices in the graph
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices in the graph

    Attributes
    ----------
    _chi2 : float, None
        The current :math:`\chi^2` error, or ``None`` if it has not yet been computed
    _edges : list[graphslam.edge.base_edge.BaseEdge]
        A list of the edges (i.e., constraints) in the graph
    _fixed_gradient_indices : set[int]
        The set of gradient indices (i.e., `Vertex.gradient_index`) for vertices that are fixed
    _gradient : numpy.ndarray, None
        The gradient :math:`\mathbf{b}` of the :math:`\chi^2` error, or ``None`` if it has not yet been computed
    _hessian : scipy.sparse.lil_matrix, None
        The Hessian matrix :math:`H`, or ``None`` if it has not yet been computed
    _len_gradient : int, None
        The length of the gradient vector (and the Hessian matrix)
    _vertices : list[graphslam.vertex.Vertex]
        A list of the vertices in the graph

    """

    def __init__(self, edges, vertices):
        # The vertices and edges lists
        self._edges = edges
        self._vertices = vertices
        self._fixed_gradient_indices = set()

        # The chi^2 error, gradient, and Hessian
        self._chi2 = None
        self._gradient = None
        self._hessian = None

        self._len_gradient = None

        self._initialize()

    def _initialize(self):
        """Fill in the ``vertices`` attributes for the graph's edges, and other necessary preparations."""
        # Fill in the vertices' `gradient_index` attribute
        gradient_index = 0
        for v in self._vertices:
            v.gradient_index = gradient_index
            gradient_index += v.pose.COMPACT_DIMENSIONALITY

        # The length of the gradient vector (and the shape of the Hessian matrix)
        self._len_gradient = gradient_index

        index_id_dict = {i: v.id for i, v in enumerate(self._vertices)}
        id_index_dict = {v_id: v_index for v_index, v_id in index_id_dict.items()}

        # Fill in the `vertices` attributes for the edges
        for e in self._edges:
            e.vertices = [self._vertices[id_index_dict[v_id]] for v_id in e.vertex_ids]

    def calc_chi2(self):
        r"""Calculate the :math:`\chi^2` error for the ``Graph``.

        Returns
        -------
        float
            The :math:`\chi^2` error

        """
        self._chi2 = sum((e.calc_chi2() for e in self._edges))
        return self._chi2

    def _calc_chi2_gradient_hessian(self):
        r"""Calculate the :math:`\chi^2` error, the gradient :math:`\mathbf{b}`, and the Hessian :math:`H`."""
        # fmt: off
        chi2_gradient_hessian = reduce(_Chi2GradientHessian.update, (e.calc_chi2_gradient_hessian() for e in self._edges), _Chi2GradientHessian())
        # fmt: on

        self._chi2 = chi2_gradient_hessian.chi2

        # Fill in the gradient vector
        self._gradient = np.zeros(self._len_gradient, dtype=np.float64)
        for gradient_idx, contrib in chi2_gradient_hessian.gradient.items():
            # If a vertex is fixed, its block in the gradient vector is zero and so there is nothing to do
            if gradient_idx not in self._fixed_gradient_indices:
                # fmt: off
                self._gradient[gradient_idx: gradient_idx + len(contrib)] += contrib
                # fmt: on

        # Fill in the Hessian matrix
        self._hessian = lil_matrix((self._len_gradient, self._len_gradient), dtype=np.float64)
        for (hessian_row_idx, hessian_col_idx), contrib in chi2_gradient_hessian.hessian.items():
            dim = contrib.shape[0]
            if hessian_row_idx in self._fixed_gradient_indices or hessian_col_idx in self._fixed_gradient_indices:
                # For fixed vertices, the diagonal block is the identity matrix and the off-diagonal blocks are zero
                if hessian_row_idx == hessian_col_idx:
                    # fmt: off
                    self._hessian[hessian_row_idx: hessian_row_idx + dim, hessian_col_idx: hessian_col_idx + dim] = np.eye(dim)
                    # fmt: on
                continue

            # fmt: off
            self._hessian[hessian_row_idx: hessian_row_idx + dim, hessian_col_idx: hessian_col_idx + dim] = contrib
            # fmt: on

            if hessian_row_idx != hessian_col_idx:
                # fmt: off
                self._hessian[hessian_col_idx: hessian_col_idx + dim, hessian_row_idx: hessian_row_idx + dim] = np.transpose(contrib)
                # fmt: on

    def optimize(self, tol=1e-4, max_iter=20, fix_first_pose=True, verbose=True):
        r"""Optimize the :math:`\chi^2` error for the ``Graph``.

        Parameters
        ----------
        tol : float
            If the relative decrease in the :math:`\chi^2` error between iterations is less than ``tol``, we will stop
        max_iter : int
            The maximum number of iterations
        fix_first_pose : bool
            If ``True``, we will fix the first pose
        verbose : bool
            Whether to print information about the optimization

        Returns
        -------
        ret : OptimizationResult
            Information about this optimization

        """
        start_time = time.time()

        ret = OptimizationResult()

        if fix_first_pose:
            self._vertices[0].fixed = True

        # Populate the set of fixed gradient indices
        self._fixed_gradient_indices = {v.gradient_index for v in self._vertices if v.fixed}

        # Previous iteration's chi^2 error
        chi2_prev = -1.0

        # For displaying the optimization progress
        if verbose:
            print("\nIteration                chi^2        rel. change")
            print("---------                -----        -----------")

        for i in range(max_iter):
            ret.iteration_results.append(OptimizationResult.IterationResult())
            iteration_start_time = time.time()

            # Calculate chi^2, the gradient, and the Hessian
            calc_chi2_gradient_hessian_start_time = time.time()
            self._calc_chi2_gradient_hessian()
            ret.iteration_results[-1].calc_chi2_gradient_hessian_duration_s = time.time() - calc_chi2_gradient_hessian_start_time  # fmt: skip

            # Check for convergence (from the previous iteration); this avoids having to calculate chi^2 twice
            if i > 0:
                rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
                if verbose:
                    print("{:9d} {:20.4f} {:18.6f}".format(i, self._chi2, -rel_diff))

                # Update the previous iteration's chi^2 and relative difference
                ret.iteration_results[-2].chi2 = self._chi2
                ret.iteration_results[-2].rel_diff = -rel_diff

                if self._chi2 <= chi2_prev and rel_diff < tol:
                    # Record information about this iteration and the optimization as a whole
                    ret.converged = True
                    ret.num_iterations = i
                    ret.final_chi2 = self._chi2
                    ret.iteration_results[-1].duration_s = time.time() - iteration_start_time
                    ret.duration_s = time.time() - start_time
                    return ret

            else:
                ret.initial_chi2 = self._chi2
                if verbose:
                    print("{:9d} {:20.4f}".format(i, self._chi2))

            # Update the previous iteration's chi^2 error
            chi2_prev = self._chi2

            # Solve for the updates
            solve_start_time = time.time()
            dx = spsolve(self._hessian, -self._gradient)  # pylint: disable=invalid-unary-operand-type
            ret.iteration_results[-1].solve_duration_s = time.time() - solve_start_time

            # Apply the updates
            update_start_time = time.time()
            for v in self._vertices:
                # fmt: off
                v.pose += dx[v.gradient_index: v.gradient_index + v.pose.COMPACT_DIMENSIONALITY]
                # fmt: on
            ret.iteration_results[-1].update_duration_s = time.time() - update_start_time

            # Record the duration for this iteration
            ret.iteration_results[-1].duration_s = time.time() - iteration_start_time

        # If we reached the maximum number of iterations, print out the final iteration's results
        self.calc_chi2()
        rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
        if verbose:
            print("{:9d} {:20.4f} {:18.6f}".format(max_iter, self._chi2, -rel_diff))

        # Update the final iteration's chi^2 and relative difference
        ret.iteration_results[-1].chi2 = self._chi2
        ret.iteration_results[-1].rel_diff = -rel_diff

        # Record information about the optimization as a whole
        ret.converged = self._chi2 <= chi2_prev and rel_diff < tol
        ret.num_iterations = max_iter
        ret.final_chi2 = self._chi2
        ret.duration_s = time.time() - start_time
        return ret

    def to_g2o(self, outfile):
        """Save the graph in .g2o format.

        Parameters
        ----------
        outfile : str
            The path where the graph will be saved

        """
        with open(outfile, "w") as f:
            for v in self._vertices:
                f.write(v.to_g2o())

            for e in self._edges:
                f.write(e.to_g2o())

    def plot(self, vertex_color="r", vertex_marker="o", vertex_markersize=3, edge_color="b", title=None):
        """Plot the graph.

        Parameters
        ----------
        vertex_color : str
            The color that will be used to plot the vertices
        vertex_marker : str
            The marker that will be used to plot the vertices
        vertex_markersize : int
            The size of the plotted vertices
        edge_color : str
            The color that will be used to plot the edges
        title : str, None
            The title that will be used for the plot

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        fig = plt.figure()
        if any(len(v.pose.position) == 3 for v in self._vertices):
            fig.add_subplot(111, projection="3d")

        for e in self._edges:
            e.plot(edge_color)

        for v in self._vertices:
            v.plot(vertex_color, vertex_marker, vertex_markersize)

        if title:
            plt.title(title)

        plt.show()

    def equals(self, other, tol=1e-6):
        """Check whether two graphs are equal.

        Parameters
        ----------
        other : Graph
            The graph to which we are comparing
        tol : float
            The tolerance

        Returns
        -------
        bool
            Whether the two graphs are equal

        """
        # pylint: disable=protected-access
        if len(self._edges) != len(other._edges) or len(self._vertices) != len(other._vertices):
            return False

        # fmt: off
        return all(e1.equals(e2, tol) for e1, e2 in zip(self._edges, other._edges)) and all(v1.pose.equals(v2.pose, tol) for v1, v2 in zip(self._vertices, other._vertices))
        # fmt: on
