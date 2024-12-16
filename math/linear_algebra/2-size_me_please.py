#!/usr/bin/env python3

"""
This module defines a function matrix_shape that calculates the shape
of a matrix
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix (list of lists).

    Args:
        matrix (list): A 2D list (matrix) whose shape we want to calculate.

    Returns:
        list: A list representing the shape of the matrix (its dimensions).
              Each entry in the list represents the size of each dimension.
              For example, for a 3x2 matrix, it returns [3, 2].
    """
    if not isinstance(matrix, list):
        return []
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
