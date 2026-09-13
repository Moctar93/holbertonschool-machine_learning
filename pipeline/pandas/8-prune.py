#!/usr/bin/env python3
"""
Module for removing NaN values from the Close column.
"""


def prune(df):
    """
    Removes entries where Close contains NaN values.

    Args:
        df: pandas DataFrame containing a Close column.

    Returns:
        The modified pandas DataFrame.
    """
    return df.dropna(subset=["Close"])
