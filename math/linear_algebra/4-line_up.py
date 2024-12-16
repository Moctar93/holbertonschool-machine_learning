#!/usr/bin/env python3
"""
Afunction that adds two arrays
"""


def add_arrays(arr1, arr2):
    """
    Adding two arrays-wise
    """

    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
