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

   17425.89298299299

   >>> g.optimize()

   Iteration  1: chi2_prev = 17425.8930, self._chi2 = 2101.3908
   Iteration  2: chi2_prev = 2101.3908, self._chi2 = 695.2287
   Iteration  3: chi2_prev = 695.2287, self._chi2 = 685.6427
   Iteration  4: chi2_prev = 685.6427, self._chi2 = 691.8391
   Iteration  5: chi2_prev = 691.8391, self._chi2 = 691.4596
   Iteration  6: chi2_prev = 691.4596, self._chi2 = 686.1112
   Iteration  7: chi2_prev = 686.1112, self._chi2 = 685.2138
   Iteration  8: chi2_prev = 685.2138, self._chi2 = 685.2582
   Iteration  9: chi2_prev = 685.2582, self._chi2 = 685.3748
   Iteration 10: chi2_prev = 685.3748, self._chi2 = 685.5076
   Iteration 11: chi2_prev = 685.5076, self._chi2 = 685.5009


SE(2) Dataset
^^^^^^^^^^^^^

.. code-block::

   >>> from graphslam.load import load_g2o_se2

   >>> g = load_g2o_se2("input_INTEL.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.calc_chi2()

   10140102.260977369

   >>> g.optimize()

   Iteration  1: chi2_prev = 10140102.2610, self._chi2 = 20788949397.2203
   Iteration  2: chi2_prev = 20788949397.2203, self._chi2 = 16923475.8850
   Iteration  3: chi2_prev = 16923475.8850, self._chi2 = 8294793755.7228
   Iteration  4: chi2_prev = 8294793755.7228, self._chi2 = 220115513.6180
   Iteration  5: chi2_prev = 220115513.6180, self._chi2 = 24117440.3125
   Iteration  6: chi2_prev = 24117440.3125, self._chi2 = 1990004.8692
   Iteration  7: chi2_prev = 1990004.8692, self._chi2 = 3445068.7836
   Iteration  8: chi2_prev = 3445068.7836, self._chi2 = 788043.5452
   Iteration  9: chi2_prev = 788043.5452, self._chi2 = 462337.4617
   Iteration 10: chi2_prev = 462337.4617, self._chi2 = 183661.3263
   Iteration 11: chi2_prev = 183661.3263, self._chi2 = 172777.5398
   Iteration 12: chi2_prev = 172777.5398, self._chi2 = 157818.2026
   Iteration 13: chi2_prev = 157818.2026, self._chi2 = 158420.4379
   Iteration 14: chi2_prev = 158420.4379, self._chi2 = 157013.3727
   Iteration 15: chi2_prev = 157013.3727, self._chi2 = 156995.5912
   Iteration 16: chi2_prev = 156995.5912, self._chi2 = 156861.0154
   Iteration 17: chi2_prev = 156861.0154, self._chi2 = 156857.2851


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
