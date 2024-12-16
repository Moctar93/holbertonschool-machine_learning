#!/usr/bin/env python3
""" Afunction taht that concatenates two matrices
along a specific axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """  return a new numpy.ndarray."""
    return np.concatenate((mat1, mat2), axis=axis)
