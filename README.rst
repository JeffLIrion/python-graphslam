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

   >>> g = load_g2o_se2("input_INTEL.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.calc_chi2()

   10140102.260977369

   >>> g.optimize()

   Iteration  1: chi2_prev = 10140102.2610, self._chi2 = 20833919847.7259
   Iteration  2: chi2_prev = 20833919847.7259, self._chi2 = 17362991.8231
   Iteration  3: chi2_prev = 17362991.8231, self._chi2 = 8602802245.3984
   Iteration  4: chi2_prev = 8602802245.3984, self._chi2 = 235948977.0316
   Iteration  5: chi2_prev = 235948977.0316, self._chi2 = 24157387.7109
   Iteration  6: chi2_prev = 24157387.7109, self._chi2 = 1951370.9006
   Iteration  7: chi2_prev = 1951370.9006, self._chi2 = 3316955.3135
   Iteration  8: chi2_prev = 3316955.3135, self._chi2 = 756557.7506
   Iteration  9: chi2_prev = 756557.7506, self._chi2 = 444315.7353
   Iteration 10: chi2_prev = 444315.7353, self._chi2 = 181332.8638
   Iteration 11: chi2_prev = 181332.8638, self._chi2 = 171861.6236
   Iteration 12: chi2_prev = 171861.6236, self._chi2 = 157799.2821
   Iteration 13: chi2_prev = 157799.2821, self._chi2 = 158330.1207
   Iteration 14: chi2_prev = 158330.1207, self._chi2 = 157011.2435
   Iteration 15: chi2_prev = 157011.2435, self._chi2 = 156985.9860
   Iteration 16: chi2_prev = 156985.9860, self._chi2 = 156860.6274
   Iteration 17: chi2_prev = 156860.6274, self._chi2 = 156856.5443


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
