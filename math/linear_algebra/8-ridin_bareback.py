#!/usr/bin/env python3
"""
A function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.
    """
    if len(mat1[0]) != len(mat2):
        return None

    newmat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            result = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row.append(result)
        newmat.append(row)
    return newmat
