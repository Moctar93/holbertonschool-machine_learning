#!/usr/bin/env python3
"""
Module containing a function that converts DataFrame data to a NumPy array.
"""


def array(df):
    """
    Selects the last 10 rows of High and Close columns.

    Args:
        df: pandas DataFrame containing High and Close columns.

    Returns:
        A numpy.ndarray containing the selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
