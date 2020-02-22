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

- Optimize :math:`\mathbb{R}^2`, :math:`\mathbb{R}^3`, :math:`SE(2)`, and :math:`SE(3)` datasets
- Analytic Jacobians
- Supports odometry edges
- Load :math:`SE(2)` and :math:`SE(3)` datasets from .g2o files


Installation
------------

.. code-block::

   pip install graphslam


Example Usage
-------------

SE(3) Dataset
^^^^^^^^^^^^^

.. code-block::

   >>> from graphslam.load import load_g2o_se3

   >>> g = load_g2o_se3("parking-garage.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.calc_chi2()

   16720.020602489112

   >>> g.optimize()

   Iteration  1: chi2_prev = 16720.0206, self._chi2 = 26.5495
   Iteration  2: chi2_prev = 26.5495, self._chi2 = 1.2712
   Iteration  3: chi2_prev = 1.2712, self._chi2 = 1.2402
   Iteration  4: chi2_prev = 1.2402, self._chi2 = 1.2396
   Iteration  5: chi2_prev = 1.2396, self._chi2 = 1.2395


SE(2) Dataset
^^^^^^^^^^^^^

.. code-block::

   >>> from graphslam.load import load_g2o_se2

   >>> g = load_g2o_se2("input_INTEL.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.calc_chi2()

   7191686.382493544

   >>> g.optimize()

   Iteration  1: chi2_prev = 7191686.3825, self._chi2 = 319915276.1284
   Iteration  2: chi2_prev = 319915276.1284, self._chi2 = 124894535.1749
   Iteration  3: chi2_prev = 124894535.1749, self._chi2 = 338185.8171
   Iteration  4: chi2_prev = 338185.8171, self._chi2 = 734.5142
   Iteration  5: chi2_prev = 734.5142, self._chi2 = 215.8405
   Iteration  6: chi2_prev = 215.8405, self._chi2 = 215.8405


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
