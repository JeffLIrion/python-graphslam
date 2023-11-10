graphslam
=========

.. image:: https://github.com/JeffLIrion/python-graphslam/actions/workflows/python-package.yml/badge.svg?branch=master
   :target: https://github.com/JeffLIrion/python-graphslam/actions/workflows/python-package.yml

.. image:: https://coveralls.io/repos/github/JeffLIrion/python-graphslam/badge.svg?branch=master
   :target: https://coveralls.io/github/JeffLIrion/python-graphslam?branch=master


Documentation for this package can be found at https://python-graphslam.readthedocs.io/.


This package implements a Graph SLAM solver in Python.

Features
--------

- Optimize :math:`\mathbb{R}^2`, :math:`\mathbb{R}^3`, :math:`SE(2)`, and :math:`SE(3)` datasets
- Analytic Jacobians
- Supports odometry edges
- Supports custom edge types (see `tests/test_custom_edge.py <tests/test_custom_edge.py>`_ for an example)
- Import and export .g2o files for :math:`SE(2)` and :math:`SE(3)` datasets


Installation
------------

.. code-block::

   pip install graphslam


Example Usage
-------------

SE(3) Dataset
^^^^^^^^^^^^^

.. code-block:: python

   >>> from graphslam.load import load_g2o

   >>> g = load_g2o("data/parking-garage.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.plot(vertex_markersize=1)

   >>> g.calc_chi2()

   16720.02100546733

   >>> g.optimize()

   >>> g.plot(vertex_markersize=1)


**Output:**

::

   Iteration                chi^2        rel. change
   ---------                -----        -----------
           0           16720.0210
           1              45.6644          -0.997269
           2               1.2936          -0.971671
           3               1.2387          -0.042457
           4               1.2387          -0.000001


+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| **Original**                                                                                                          | **Optimized**                                                                                                                   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| .. image::                                                                                  images/parking-garage.png | .. image::                                                                                  images/parking-garage-optimized.png |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------+


SE(2) Dataset
^^^^^^^^^^^^^

.. code-block:: python

   >>> from graphslam.load import load_g2o

   >>> g = load_g2o("data/input_INTEL.g2o")  # https://lucacarlone.mit.edu/datasets/

   >>> g.plot()

   >>> g.calc_chi2()

   7191686.382493544

   >>> g.optimize()

   >>> g.plot()


**Output:**

::

   Iteration                chi^2        rel. change
   ---------                -----        -----------
           0         7191686.3825
           1       319950425.6477          43.488929
           2       124950341.8035          -0.609470
           3          338165.0770          -0.997294
           4             734.7343          -0.997827
           5             215.8405          -0.706233
           6             215.8405          -0.000000


+--------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+
| **Original**                                                                                                       | **Optimized**                                                                                                                |
+--------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+
| .. image::                                                                                  images/input_INTEL.png | .. image::                                                                                  images/input_INTEL-optimized.png |
+--------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+

References and Acknowledgments
------------------------------


1. Grisetti, G., Kummerle, R., Stachniss, C. and Burgard, W., 2010. `A tutorial on graph-based SLAM <http://domino.informatik.uni-freiburg.de/teaching/ws10/praktikum/slamtutorial.pdf>`_. IEEE Intelligent Transportation Systems Magazine, 2(4), pp.31-43.
2. Blanco, J.L., 2010. `A tutorial on SE(3) transformation parameterizations and on-manifold optimization <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.5407&rep=rep1&type=pdf>`_. University of Malaga, Tech. Rep, 3.
3. Carlone, L., Tron, R., Daniilidis, K. and Dellaert, F., 2015, May. `Initialization techniques for 3D SLAM: a survey on rotation estimation and its use in pose graph optimization <https://smartech.gatech.edu/bitstream/handle/1853/53710/Carlone15icra.pdf>`_. In 2015 IEEE international conference on robotics and automation (ICRA) (pp. 4597-4604). IEEE.
4. Carlone, L. and Censi, A., 2014. `From angular manifolds to the integer lattice: Guaranteed orientation estimation with application to pose graph optimization <https://arxiv.org/pdf/1211.3063.pdf>`_. IEEE Transactions on Robotics, 30(2), pp.475-492.


Thanks to Luca Larlone for allowing inclusion of the `Intel and parking garage datasets <https://lucacarlone.mit.edu/datasets/>`_ in this repo.


Live Coding Graph SLAM in Python
--------------------------------

If you're interested, you can watch as I coded this up.

1. `Live coding Graph SLAM in Python (Part 1) <https://youtu.be/yXWkNC_A_YE>`_
2. `Live coding Graph SLAM in Python (Part 2) <https://youtu.be/M2udkF0UNUg>`_
3. `Live coding Graph SLAM in Python (Part 3) <https://youtu.be/CiBdVcIObVU>`_
4. `Live coding Graph SLAM in Python (Part 4) <https://youtu.be/GBAThis-_wM>`_
5. `Live coding Graph SLAM in Python (Part 5) <https://youtu.be/J3NyieGVwIw>`_
