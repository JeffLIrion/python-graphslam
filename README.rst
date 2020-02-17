graphslam
=========

.. image:: https://travis-ci.com/JeffLIrion/python-graphslam.svg?branch=master
   :target: https://travis-ci.com/JeffLIrion/python-graphslam

.. image:: https://coveralls.io/repos/github/JeffLIrion/python-graphslam/badge.svg?branch=master
   :target: https://coveralls.io/github/JeffLIrion/python-graphslam?branch=master


Documentation for this package can be found at https://python-graphslam.readthedocs.io/.


This package implements a Graph SLAM solver in Python.

Features
--------

- Optimize :math:`\mathbb{R}^2`, :math:`\mathbb{R}^3`, and :math:`SE(2)` datasets
- Analytic Jacobians
- Supports odometry edges
- Load :math:`SE(2)` datasets from .g2o files

Planned Features
^^^^^^^^^^^^^^^^

- Optimize :math:`SE(3)` datasets
- Load :math:`SE(3)` datasets from .g2o files


Installation
------------

.. code-block::

   pip install graphslam


Example Usage
-------------

.. code-block::

   >>> from graphslam.load import load_g2o_se2

   >>> g = load_g2o_se2("input_INTEL_g2o")

   >>> g.calc_chi2()

   10140102.260977369

   >>> g.optimize()

   Iteration  1: chi2_prev = 10140102.2610, self._chi2 = 20807915286.0808
   Iteration  2: chi2_prev = 20807915286.0808, self._chi2 = 17138938.1459
   Iteration  3: chi2_prev = 17138938.1459, self._chi2 = 8449141792.5765
   Iteration  4: chi2_prev = 8449141792.5765, self._chi2 = 227464854.3710
   Iteration  5: chi2_prev = 227464854.3710, self._chi2 = 24133028.4628
   Iteration  6: chi2_prev = 24133028.4628, self._chi2 = 1971385.3240
   Iteration  7: chi2_prev = 1971385.3240, self._chi2 = 3381916.5105
   Iteration  8: chi2_prev = 3381916.5105, self._chi2 = 772245.1031
   Iteration  9: chi2_prev = 772245.1031, self._chi2 = 453384.0468
   Iteration 10: chi2_prev = 453384.0468, self._chi2 = 182479.5075
   Iteration 11: chi2_prev = 182479.5075, self._chi2 = 172322.2547
   Iteration 12: chi2_prev = 172322.2547, self._chi2 = 157810.0204
   Iteration 13: chi2_prev = 157810.0204, self._chi2 = 158375.4994
   Iteration 14: chi2_prev = 158375.4994, self._chi2 = 157012.2347
   Iteration 15: chi2_prev = 157012.2347, self._chi2 = 156990.8030
   Iteration 16: chi2_prev = 156990.8030, self._chi2 = 156860.7980
   Iteration 17: chi2_prev = 156860.7980, self._chi2 = 156856.9133


References and Links
--------------------

* `A tutorial on graph-based SLAM <http://domino.informatik.uni-freiburg.de/teaching/ws10/praktikum/slamtutorial.pdf>`_
* `A tutorial on SE(3) transformation parameterizations and on-manifold optimization <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.5407&rep=rep1&type=pdf>`_
* `Datasets from Luca Carlone <https://lucacarlone.mit.edu/datasets/>`_


Live Coding Graph SLAM in Python
--------------------------------

If you're interested, you can watch as I coded this up.

1. `Live coding Graph SLAM in Python (Part 1) <https://youtu.be/yXWkNC_A_YE>`_
2. `Live coding Graph SLAM in Python (Part 2) <https://youtu.be/M2udkF0UNUg>`_
3. `Live coding Graph SLAM in Python (Part 3) <https://youtu.be/CiBdVcIObVU>`_
4. `Live coding Graph SLAM in Python (Part 4) <https://youtu.be/GBAThis-_wM>`_
5. `Live coding Graph SLAM in Python (Part 5) <https://youtu.be/J3NyieGVwIw>`_
