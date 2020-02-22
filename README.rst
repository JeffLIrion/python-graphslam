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

   Iteration                chi^2        rel. change
   ---------                -----        -----------
           0           16720.0206
           1              26.5495            -0.9984
           2               1.2712            -0.9521
           3               1.2402            -0.0244
           4               1.2396            -0.0005
           5               1.2395            -0.0001


SE(2) Dataset
^^^^^^^^^^^^^

.. code-block::

   >>> from graphslam.load import load_g2o_se2

   >>> g = load_g2o_se2("input_INTEL.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.calc_chi2()

   7191686.382493544

   >>> g.optimize()

   Iteration                chi^2        rel. change
   ---------                -----        -----------
           0         7191686.3825
           1       319915276.1284            43.4840
           2       124894535.1749            -0.6096
           3          338185.8171            -0.9973
           4             734.5142            -0.9978
           5             215.8405            -0.7061
           6             215.8405            -0.0000


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
