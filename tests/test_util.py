# Copyright (c) 2020 Jeff Irion and contributors

"""Unit tests for the util.py module.

"""


import unittest

import numpy as np

from graphslam import util


class TestUtil(unittest.TestCase):
    """Tests for the functions in util.py."""

    def test_neg_pi_to_pi(self):
        """Test the ``neg_pi_to_pi()`` function."""
        for angle in range(-10, 10):
            new_angle = util.neg_pi_to_pi(angle)
            self.assertGreaterEqual(new_angle, -np.pi)
            self.assertLess(new_angle, np.pi)
            self.assertAlmostEqual(np.cos(angle), np.cos(new_angle))
            self.assertAlmostEqual(np.sin(angle), np.sin(new_angle))

    def test_solve_for_edge_dimensionality(self):
        """Test the ``solve_for_edge_dimensionality()`` function."""
        d = [1, 2, 3, 4]
        n = [2, 5, 9, 14]

        for d_i, n_i in zip(d, n):
            self.assertEqual(util.solve_for_edge_dimensionality(n_i), d_i)

    def test_upper_triangular_matrix_to_full_matrix(self):
        """Test the ``upper_triangular_matrix_to_full_matrix()`` function."""
        arrays = [np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        answers = [
            np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64),
            np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]]),
        ]
        dims = [3, 4]

        for array, answer, dim in zip(arrays, answers, dims):
            mat = util.upper_triangular_matrix_to_full_matrix(array, dim)
            self.assertAlmostEqual(np.linalg.norm(mat - answer), 0.0)


if __name__ == "__main__":
    unittest.main()
