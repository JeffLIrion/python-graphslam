r"""A ``Graph`` class that stores the edges and vertices required for Graph SLAM.

Given:

* A set of :math:`M` edges (i.e., constraints) :math:`\mathcal{E}`

  * :math:`e_j \in \mathcal{E}` is an edge
  * :math:`\mathbf{e}_j \in \mathbb{R}^\bullet` is the error associated with that edge, where :math:`\bullet` is a scalar that depends on the type of edge
  * :math:`\Omega_j` is the :math:`\bullet \times \bullet` information matrix associated with edge :math:`e_j`

* A set of :math:`N` vertices :math:`\mathcal{V}`

  * :math:`v_i \in \mathcal{V}` is a vertex
  * :math:`\mathbb{x}_i \in \mathbb{R}^c` is the compact pose associated with :math:`v_i`
  * :math:`\boxplus` is the pose composition operator that yields a (non-compact) pose that lies in (a subspace of) :math:`\mathbb{R}^d`

We want to optimize

.. math::

   \chi^2 = \sum_{e_j \in \mathcal{E}} \mathbb{e}_j^T \Omega_j \mathbb{e}_j.

Let

.. math::

   \mathbf{x} := \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \\ \vdots \\ \mathbf{x}_N \end{bmatrix} \in \mathbb{R}^{cN}.

We will solve this optimization problem iteratively.  We will need to represent each pose Let

.. math::

   \mathbf{x}^{k+1} = \mathbf{x}^k \boxplus \Delta \mathbf{x}^k.

The :math:`\chi^2` error at iteration :math:`k+1` is

.. math::

   \chi_{k+1} = \sum_{e_j \in \mathcal{E}} \underbrace{\left[ \mathbf{e}_j(\mathbf{x}^{k+1}) \right]^T}_{1 \times \bullet} \underbrace{\Omega_j}_{\bullet \times \bullet} \underbrace{\mathbf{e}_j(\mathbf{x}^{k+1})}_{\bullet \times 1}.

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
