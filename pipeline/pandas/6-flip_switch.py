#!/usr/bin/env python3
"""
Module for reversing and transposing a pandas DataFrame.
"""


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Args:
        df: pandas DataFrame.

    Returns:
        The sorted and transposed pandas DataFrame.
    """
    return df.sort_index(ascending=False).T
