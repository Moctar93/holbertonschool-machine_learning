#!/usr/bin/env python3

"""Compute the determinant of a matrix"""


def determinant(matrix):
    """
    Compute the determinant of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    for factor_idx, factor in enumerate(matrix[0]):
        sub_matrix = [elem[:factor_idx] + elem[factor_idx + 1:]
                      for elem in matrix[1:]]
        det += (-1)**factor_idx * factor * determinant(sub_matrix)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            sub_matrix_value = [
                    row[:j] + row[j+1:]
                    for row_idx, row in enumerate(matrix)
                    if row_idx != i
                    ]
            det_sub_matrix = determinant(sub_matrix_value)
            minor_row.append(det_sub_matrix)
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """Calculates the cofactor matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor = [
                row[:j] + row[j+1:]
                for idx, row in enumerate(matrix)
                if idx != i
            ]
            cofactor_value = ((-1) ** (i + j)) * determinant(minor)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
