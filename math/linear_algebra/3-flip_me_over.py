#!/usr/bin/env python3
"""
A function that transposes a 2D matrix
"""


def matrix_transpose(matrix):
    """
    transposes a 2D matrix
    """
    return [list(row) for row in zip(*matrix)]
